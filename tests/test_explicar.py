"""Testes do endpoint POST /explicar com a LLM SIMULADA (sem chamar a OpenAI).

A chamada real ao endpoint é validada em tests/test_openai_real.py
(executado somente com RUN_OPENAI_TESTS=1).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import assistente_llm as llm
import main
from tests.test_assistente_llm import EXPLICACAO_BOA, ClienteFalso, json_extracao

TEXTO = "Exame sintético: pontos côncavos (média) 0,052; (pior) 0,146; perímetro 108,4; raio 16,9."


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def usar_llm(monkeypatch):
    """Substitui o cliente OpenAI da API por um cliente com respostas simuladas."""

    def configurar(respostas):
        cliente = ClienteFalso(respostas)
        monkeypatch.setattr(main, "obter_cliente_llm", lambda: (cliente, "modelo-teste"))
        return cliente

    return configurar


def contador(client, nome_metrica):
    """Soma das linhas de /metrics que começam com o nome (e rótulos) informados."""
    return sum(
        float(linha.rsplit(" ", 1)[1])
        for linha in client.get("/metrics").text.splitlines()
        if linha.startswith(nome_metrica)
    )


def test_explicar_sucesso_classifica_com_o_modelo_e_explica(client, usar_llm):
    cliente = usar_llm([json_extracao(), EXPLICACAO_BOA])
    antes = contador(client, 'llm_explanations_total{status="sucesso"}')

    resposta = client.post("/explicar", json={"texto_exame": TEXTO})

    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["status"] == "sucesso"
    assert corpo["parametros"]["perimeter_worst"] == 108.4
    assert corpo["predicao"]["label"] == "Maligno"
    assert corpo["predicao"]["model_file"] == "modelo_pipeline_genetico.pkl"
    assert corpo["explicacao"] == EXPLICACAO_BOA
    assert corpo["modelo_llm"] == "modelo-teste"
    assert corpo["request_id"] == resposta.headers["X-Request-ID"]
    assert len(cliente.chamadas) == 2  # extração + explicação
    assert contador(client, 'llm_explanations_total{status="sucesso"}') == antes + 1


def test_explicar_usa_a_mesma_classificacao_do_predict(client, usar_llm):
    usar_llm([json_extracao(), EXPLICACAO_BOA])
    explicado = client.post("/explicar", json={"texto_exame": TEXTO}).json()
    direto = client.post("/predict", json=explicado["parametros"]).json()
    assert explicado["predicao"]["output"] == direto["output"]


def test_explicar_valor_ausente_pede_revisao_sem_classificar(client, usar_llm):
    cliente = usar_llm([json_extracao(radius_worst=None)])
    antes = contador(client, "model_predictions_total")

    corpo = client.post("/explicar", json={"texto_exame": "Raio não medido."}).json()

    assert corpo["status"] == "revisao_necessaria"
    assert corpo["parametros_ausentes"] == ["radius_worst"]
    assert corpo["predicao"] is None and corpo["explicacao"] is None
    assert len(cliente.chamadas) == 1  # só a extração
    assert contador(client, "model_predictions_total") == antes


def test_explicar_valor_negativo_e_entrada_invalida(client, usar_llm):
    cliente = usar_llm([json_extracao(concave_points_mean=-0.03)])
    corpo = client.post("/explicar", json={"texto_exame": "Valor -0,03."}).json()
    assert corpo["status"] == "entrada_invalida"
    assert corpo["predicao"] is None
    assert len(cliente.chamadas) == 1


@pytest.mark.parametrize("payload", [{"texto_exame": "   "}, {}, {"texto_exame": "x", "extra": 1}])
def test_explicar_rejeita_corpo_invalido_sem_chamar_a_llm(client, usar_llm, payload):
    cliente = usar_llm([])
    resposta = client.post("/explicar", json=payload)
    assert resposta.status_code == 422
    assert cliente.chamadas == []


def test_explicar_sem_chave_retorna_503(client, monkeypatch):
    monkeypatch.setattr(main, "cliente_llm", None)
    monkeypatch.setattr(
        llm,
        "carregar_configuracao",
        lambda: {"OPENAI_API_KEY": "", "OPENAI_MODEL": "x", "PREDICTION_API_URL": ""},
    )
    resposta = client.post("/explicar", json={"texto_exame": TEXTO})
    assert resposta.status_code == 503
    assert "OPENAI_API_KEY" in resposta.json()["detail"]
    # A predição continua funcionando sem a chave da OpenAI.
    assert client.post("/predict", json={
        "concave_points_mean": 0.052, "concave_points_worst": 0.146,
        "perimeter_worst": 108.4, "radius_worst": 16.9,
    }).status_code == 200


def test_explicar_falha_da_openai_retorna_502(client, monkeypatch):
    class ClienteComErro(ClienteFalso):
        def _create(self, **kwargs):
            raise ConnectionError("sem conexão")

    monkeypatch.setattr(main, "obter_cliente_llm", lambda: (ClienteComErro([]), "modelo-teste"))
    antes = contador(client, 'llm_errors_total{error_type="ConnectionError"}')

    resposta = client.post("/explicar", json={"texto_exame": TEXTO})

    assert resposta.status_code == 502
    assert contador(client, 'llm_errors_total{error_type="ConnectionError"}') == antes + 1
