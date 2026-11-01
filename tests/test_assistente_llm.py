"""Testes da integração LLM com respostas SIMULADAS (sem chamar a OpenAI).

A chamada real à OpenAI é validada separadamente em tests/test_openai_real.py
(executado somente com RUN_OPENAI_TESTS=1) e no notebook assistente_laudo_medico.ipynb.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import requests
from fastapi.testclient import TestClient

import assistente_llm as llm
import main

EXPLICACAO_BOA = """**Dados analisados**
Pontos côncavos (média): 0.052; pontos côncavos (pior): 0.146; perímetro (pior): 108.4; raio (pior): 16.9.
**Resultado do modelo**
O modelo retornou a classe 1, que corresponde a Maligno.
**Como interpretar**
É uma classificação automática baseada em quatro medidas.
**Limitações**
Protótipo educacional treinado em dados públicos.
**Próximos passos**
O resultado não é diagnóstico e precisa de avaliação por profissional de saúde habilitado."""


class ClienteFalso:
    """Imita client.chat.completions.create e registra as chamadas."""

    def __init__(self, respostas: list[str]):
        self.respostas = list(respostas)
        self.chamadas: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.chamadas.append(kwargs)
        conteudo = self.respostas.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=conteudo))])


def json_extracao(**valores):
    base = {
        "concave_points_mean": 0.052,
        "concave_points_worst": 0.146,
        "perimeter_worst": 108.4,
        "radius_worst": 16.9,
    }
    base.update(valores)
    return json.dumps(base)


@pytest.fixture
def api_em_memoria(monkeypatch):
    """Redireciona requests.post para a API real executando em memória (TestClient)."""
    with TestClient(main.app) as cliente_api:
        chamadas = []

        def post_falso(url, json=None, timeout=None):
            chamadas.append(json)
            return cliente_api.post("/predict", json=json)

        monkeypatch.setattr(llm.requests, "post", post_falso)
        yield chamadas


# ---------------- Extração ----------------

def test_extracao_completa_valida_com_pydantic():
    cliente = ClienteFalso([json_extracao()])
    parametros = llm.extrair_parametros("texto sintético", cliente, "modelo-teste")
    assert parametros.ausentes() == []
    assert parametros.validar_para_predicao()["perimeter_worst"] == 108.4
    chamada = cliente.chamadas[0]
    assert chamada["temperature"] == 0
    assert chamada["response_format"] == {"type": "json_object"}
    assert chamada["model"] == "modelo-teste"


def test_extracao_incompleta_marca_ausentes():
    cliente = ClienteFalso([json_extracao(radius_worst=None)])
    parametros = llm.extrair_parametros("texto", cliente)
    assert parametros.ausentes() == ["radius_worst"]
    with pytest.raises(ValueError, match="radius_worst"):
        parametros.validar_para_predicao()


def test_extracao_negativa_e_invalida():
    cliente = ClienteFalso([json_extracao(concave_points_mean=-0.03)])
    with pytest.raises(llm.ExtracaoInvalida):
        llm.extrair_parametros("texto", cliente)


def test_extracao_ignora_chaves_inventadas():
    conteudo = json.loads(json_extracao())
    conteudo["probabilidade_maligno"] = 0.9
    parametros = llm.extrair_parametros("texto", ClienteFalso([json.dumps(conteudo)]))
    assert "probabilidade_maligno" not in parametros.model_dump()


def test_extracao_json_invalido_gera_erro():
    with pytest.raises(RuntimeError, match="JSON"):
        llm.extrair_parametros("texto", ClienteFalso(["isto não é json"]))


def test_texto_vazio_nao_chama_llm():
    cliente = ClienteFalso([])
    with pytest.raises(ValueError):
        llm.extrair_parametros("   ", cliente)
    assert cliente.chamadas == []


# ---------------- Fluxo completo ----------------

def test_fluxo_completo_usa_classe_da_api(api_em_memoria):
    cliente = ClienteFalso([json_extracao(), EXPLICACAO_BOA])
    resultado = llm.processar_exame("texto sintético", cliente, url_api="http://api-falsa/predict")

    assert resultado["status"] == "sucesso"
    assert api_em_memoria == [json.loads(json_extracao())]  # a API recebeu os valores extraídos
    assert resultado["retorno_api"]["model_file"] == "modelo_pipeline_genetico.pkl"

    # o prompt da explicação recebe a classe e o rótulo retornados pela API
    contexto = json.loads(cliente.chamadas[1]["messages"][1]["content"])
    assert str(contexto["classe_retornada_pelo_modelo"]) == resultado["retorno_api"]["output"]
    assert contexto["significado_da_classe"] == resultado["retorno_api"]["label"]


def test_fluxo_incompleto_nao_chama_api_nem_explicacao(api_em_memoria):
    cliente = ClienteFalso([json_extracao(perimeter_worst=None)])
    resultado = llm.processar_exame("texto", cliente)
    assert resultado["status"] == "revisao_necessaria"
    assert resultado["parametros_ausentes"] == ["perimeter_worst"]
    assert api_em_memoria == []
    assert len(cliente.chamadas) == 1


def test_fluxo_invalido_nao_chama_api(api_em_memoria):
    cliente = ClienteFalso([json_extracao(radius_worst=-5)])
    resultado = llm.processar_exame("texto", cliente)
    assert resultado["status"] == "entrada_invalida"
    assert api_em_memoria == []


def test_api_fora_do_ar_gera_mensagem_clara(monkeypatch):
    def post_com_erro(*args, **kwargs):
        raise requests.ConnectionError("recusado")

    monkeypatch.setattr(llm.requests, "post", post_com_erro)
    parametros = llm.ParametrosExame.model_validate(json.loads(json_extracao()))
    with pytest.raises(RuntimeError, match="uvicorn main:app"):
        llm.chamar_api_predicao(parametros, "http://127.0.0.1:9/predict")


# ---------------- Configuração ----------------

def test_cliente_sem_chave_falha_com_mensagem_clara():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        llm.criar_cliente_openai("")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        llm.criar_cliente_openai("sua_chave_aqui")


# ---------------- Avaliação das explicações ----------------

def test_avaliacao_aprova_explicacao_fiel():
    parametros = json.loads(json_extracao())
    avaliacao = llm.avaliar_explicacao(EXPLICACAO_BOA, parametros, "Maligno")
    assert all(avaliacao.values()), avaliacao


def test_avaliacao_detecta_alucinacoes():
    parametros = json.loads(json_extracao())
    ruim = (
        "**Resultado do modelo** O resultado é benigno com 97% de probabilidade de acerto. "
        "Recomenda-se cirurgia."
    )
    avaliacao = llm.avaliar_explicacao(ruim, parametros, "Maligno")
    assert not avaliacao["fidelidade_rotulo"]
    assert not avaliacao["sem_rotulo_oposto_afirmado"]
    assert not avaliacao["sem_numeros_inventados"]
    assert not avaliacao["sem_tratamento"]
    assert not avaliacao["estrutura"]
    assert not avaliacao["aviso_medico"]


def test_avaliacao_aceita_somente_porcentagens_das_metricas_fornecidas():
    parametros = json.loads(json_extracao())
    metricas = {"accuracy": 0.9590643, "recall": 0.953125}
    fiel = EXPLICACAO_BOA + "\nNo teste, o modelo teve acurácia de 95,91% e recall de 95,31%."
    inventada = EXPLICACAO_BOA + "\nO modelo tem 99% de acerto."
    assert llm.avaliar_explicacao(fiel, parametros, "Maligno", metricas)["sem_numeros_inventados"]
    assert not llm.avaliar_explicacao(inventada, parametros, "Maligno", metricas)["sem_numeros_inventados"]
    # sem métricas fornecidas, qualquer porcentagem é considerada inventada
    assert not llm.avaliar_explicacao(fiel, parametros, "Maligno")["sem_numeros_inventados"]


def test_explicacao_recebe_nomes_em_portugues():
    cliente = ClienteFalso([EXPLICACAO_BOA])
    parametros = llm.ParametrosExame.model_validate(json.loads(json_extracao()))
    llm.gerar_explicacao(parametros, {"output": "1", "label": "Maligno"}, cliente)
    contexto = json.loads(cliente.chamadas[0]["messages"][1]["content"])
    assert contexto["valores_de_entrada"]["raio (pior valor)"] == 16.9
    assert contexto["significado_da_classe"] == "Maligno"


def test_cenarios_de_avaliacao_cobrem_casos_exigidos():
    esperados = {c["esperado"] for c in llm.CENARIOS_AVALIACAO}
    assert {"sucesso", "revisao_necessaria", "entrada_invalida"} <= esperados
    assert len(llm.CENARIOS_AVALIACAO) >= 3
