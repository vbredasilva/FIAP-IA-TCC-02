"""Integração com LLM (OpenAI) para interpretar o resultado do modelo.

Fluxo:
1. Texto sintético do exame -> LLM extrai os 4 valores (JSON, sem inventar).
2. Pydantic valida os valores (ausentes e inválidos interrompem o fluxo).
3. Valores válidos -> POST /predict na API de classificação (ou o modelo já
   carregado, quando o fluxo roda dentro da API pelo endpoint POST /explicar).
4. Classe retornada pelo modelo -> LLM gera explicação objetiva em português,
   com limitações e necessidade de avaliação médica.

O texto do exame NUNCA é classificado pela LLM: a classificação vem sempre da API.
O módulo não depende de variáveis globais: o cliente OpenAI e a URL da API são
parâmetros, o que permite testes automatizados com respostas simuladas.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Optional

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from modelos import MAPA_CLASSES, NOMES_FEATURES, PASTA_PROJETO

MODELO_PADRAO = "gpt-4.1-mini"
URL_API_PADRAO = "http://127.0.0.1:8000/predict"

# Nomes em português usados nas explicações (mais legíveis para o médico).
NOMES_PT = {
    "concave_points_mean": "pontos côncavos (média)",
    "concave_points_worst": "pontos côncavos (pior valor)",
    "perimeter_worst": "perímetro (pior valor)",
    "radius_worst": "raio (pior valor)",
}


# ============================================================
# Configuração
# ============================================================

def carregar_configuracao() -> dict[str, str]:
    """Lê o arquivo .env da pasta do projeto (se existir) e as variáveis de ambiente."""
    from dotenv import load_dotenv

    load_dotenv(PASTA_PROJETO / ".env")
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        "OPENAI_MODEL": os.getenv("OPENAI_MODEL", MODELO_PADRAO),
        "PREDICTION_API_URL": os.getenv("PREDICTION_API_URL", URL_API_PADRAO),
    }


def criar_cliente_openai(api_key: str):
    """Cria o cliente oficial da OpenAI; falha com mensagem clara sem chave."""
    if not api_key or api_key == "sua_chave_aqui":
        raise RuntimeError(
            "OPENAI_API_KEY não configurada. Copie .env.example para .env e informe a chave."
        )
    from openai import OpenAI

    return OpenAI(api_key=api_key)


# ============================================================
# Validação dos parâmetros
# ============================================================

class ParametrosExame(BaseModel):
    """Valores extraídos pela LLM. None = ausente, ilegível ou ambíguo no texto."""

    model_config = ConfigDict(extra="ignore")

    concave_points_mean: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    concave_points_worst: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    perimeter_worst: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    radius_worst: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)

    def ausentes(self) -> list[str]:
        return [nome for nome, valor in self.model_dump().items() if valor is None]

    def validar_para_predicao(self) -> dict[str, float]:
        faltantes = self.ausentes()
        if faltantes:
            raise ValueError("Parâmetros ausentes ou ambíguos: " + ", ".join(faltantes))
        return {nome: float(valor) for nome, valor in self.model_dump().items()}


class ExtracaoInvalida(ValueError):
    """A LLM devolveu valores fora do domínio permitido (ex.: negativos)."""


# ============================================================
# Prompts
# ============================================================

PROMPT_EXTRACAO = """
Você é um extrator de dados. Não faça diagnóstico e não estime valores.
Extraia somente valores explicitamente informados no texto recebido.
Responda exclusivamente com um objeto JSON válido contendo exatamente as chaves:
concave_points_mean, concave_points_worst, perimeter_worst, radius_worst.
Correspondência dos nomes: "pontos côncavos (média)" = concave_points_mean;
"pontos côncavos (pior valor)" = concave_points_worst; "perímetro (pior valor)" =
perimeter_worst; "raio (pior valor)" = radius_worst.
Use número decimal com ponto. Se um valor estiver ausente, ilegível ou ambíguo,
use null. Copie o número exatamente como está no texto, inclusive se for negativo.
Não converta unidades sem que a conversão esteja explicitamente indicada.
""".strip()

PROMPT_EXPLICACAO = """
Você apoia médicos na leitura do resultado de um modelo de machine learning educacional.
Redija, em português do Brasil, uma explicação objetiva usando APENAS os dados do JSON.
Regras obrigatórias:
- Não invente fatos, diagnóstico definitivo, prognóstico, tratamento, probabilidade,
  porcentagem, sensibilidade, especificidade, valores de referência ou evidências clínicas.
- O modelo retorna somente uma classe (sem probabilidade); não sugira grau de certeza.
- Diferencie claramente os valores de entrada e a classe retornada pelo modelo.
- Use as métricas de desempenho informadas no JSON somente se existirem e cite-as
  como desempenho no conjunto de teste, não como chance deste caso.
Estruture o texto exatamente com estas seções (títulos em negrito):
**Dados analisados**; **Resultado do modelo**; **Como interpretar**;
**Limitações**; **Próximos passos**.
Em "Limitações" explique que é um protótipo educacional treinado em dados públicos
(Breast Cancer Wisconsin) com apenas quatro características.
Em "Próximos passos" indique que o resultado não é diagnóstico e exige avaliação e
confirmação por profissional de saúde habilitado. Máximo de 250 palavras.
""".strip()


# ============================================================
# Etapas do fluxo
# ============================================================

def extrair_parametros(texto_exame: str, cliente, modelo: str = MODELO_PADRAO) -> ParametrosExame:
    """Etapa 1 e 2: LLM extrai os valores e Pydantic valida."""
    if not texto_exame or not texto_exame.strip():
        raise ValueError("O texto do exame não pode estar vazio.")

    resposta = cliente.chat.completions.create(
        model=modelo,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PROMPT_EXTRACAO},
            {"role": "user", "content": texto_exame},
        ],
    )

    conteudo = resposta.choices[0].message.content
    if not conteudo:
        raise RuntimeError("A OpenAI retornou uma resposta vazia.")

    try:
        dados = json.loads(conteudo)
    except json.JSONDecodeError as erro:
        raise RuntimeError(f"A LLM não retornou JSON válido: {erro}") from erro

    # Considera somente as quatro chaves esperadas.
    dados = {nome: dados.get(nome) for nome in NOMES_FEATURES}
    try:
        return ParametrosExame.model_validate(dados)
    except ValidationError as erro:
        raise ExtracaoInvalida(f"Valores extraídos inválidos: {erro.errors()}") from erro


def chamar_api_predicao(
    parametros: ParametrosExame,
    url: str = URL_API_PADRAO,
    timeout: float = 30,
) -> dict[str, Any]:
    """Etapa 3: envia os parâmetros validados para POST /predict."""
    payload = parametros.validar_para_predicao()

    try:
        resposta = requests.post(url, json=payload, timeout=timeout)
        resposta.raise_for_status()
    except requests.Timeout as erro:
        raise RuntimeError("A API de predição excedeu o tempo limite.") from erro
    except requests.RequestException as erro:
        raise RuntimeError(
            f"Falha ao chamar a API de predição ({url}). "
            "Verifique se ela está rodando: uvicorn main:app"
        ) from erro

    dados = resposta.json()
    if dados.get("output") not in {"0", "1"}:
        raise RuntimeError(f"Resposta inesperada da API: {dados}")
    return dados


def gerar_explicacao(
    parametros: ParametrosExame,
    retorno_api: dict[str, Any],
    cliente,
    modelo: str = MODELO_PADRAO,
    metricas_teste: Optional[dict[str, float]] = None,
) -> str:
    """Etapa 4: explicação em linguagem natural do resultado retornado pela API."""
    classe = int(retorno_api["output"])
    contexto = {
        "valores_de_entrada": {
            NOMES_PT[nome]: valor for nome, valor in parametros.validar_para_predicao().items()
        },
        "classe_retornada_pelo_modelo": classe,
        "significado_da_classe": MAPA_CLASSES[classe],
        "arquivo_do_modelo": retorno_api.get("model_file"),
    }
    if metricas_teste:
        contexto["metricas_do_modelo_no_conjunto_de_teste"] = metricas_teste

    resposta = cliente.chat.completions.create(
        model=modelo,
        temperature=0.2,
        messages=[
            {"role": "system", "content": PROMPT_EXPLICACAO},
            {"role": "user", "content": json.dumps(contexto, ensure_ascii=False)},
        ],
    )

    conteudo = resposta.choices[0].message.content
    if not conteudo:
        raise RuntimeError("A OpenAI retornou uma explicação vazia.")
    return conteudo.strip()


def processar_exame(
    texto_exame: str,
    cliente,
    modelo: str = MODELO_PADRAO,
    url_api: str = URL_API_PADRAO,
    metricas_teste: Optional[dict[str, float]] = None,
    predizer: Optional[Callable[[ParametrosExame], dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Fluxo completo. A predição só é feita quando os 4 valores são válidos.

    Por padrão a predição é feita via HTTP (POST /predict em url_api). O endpoint
    POST /explicar da própria API passa `predizer` para usar o modelo já carregado.
    """
    try:
        parametros = extrair_parametros(texto_exame, cliente, modelo)
    except ExtracaoInvalida as erro:
        return {
            "status": "entrada_invalida",
            "mensagem": "Há valores fora do domínio permitido; a predição não foi feita.",
            "detalhe": str(erro),
        }

    faltantes = parametros.ausentes()
    if faltantes:
        return {
            "status": "revisao_necessaria",
            "mensagem": "Complete os parâmetros antes da predição.",
            "parametros_extraidos": parametros.model_dump(),
            "parametros_ausentes": faltantes,
        }

    if predizer is None:
        retorno_api = chamar_api_predicao(parametros, url_api)
    else:
        retorno_api = predizer(parametros)
    explicacao = gerar_explicacao(parametros, retorno_api, cliente, modelo, metricas_teste)

    return {
        "status": "sucesso",
        "parametros": parametros.model_dump(),
        "retorno_api": retorno_api,
        "explicacao": explicacao,
    }


# ============================================================
# Avaliação das explicações
# ============================================================

SECOES_OBRIGATORIAS = [
    "Dados analisados",
    "Resultado do modelo",
    "Como interpretar",
    "Limitações",
    "Próximos passos",
]

TERMOS_PROIBIDOS = [
    "quimioterapia",
    "radioterapia",
    "mastectomia",
    "cirurgia",
    "prognóstico de",
    "sobrevida",
]


def _normalizar_numero(texto: str) -> str:
    return texto.replace(",", ".")


def _porcentagens_inventadas(explicacao: str, metricas_teste: Optional[dict[str, float]]) -> list[str]:
    """Porcentagens do texto que NÃO correspondem a nenhuma métrica fornecida."""
    permitidas = [100 * valor for valor in (metricas_teste or {}).values()]
    inventadas = []
    for numero in re.findall(r"(\d+(?:[.,]\d+)?)\s*%", explicacao):
        valor = float(_normalizar_numero(numero))
        if not any(abs(valor - p) <= 0.051 for p in permitidas):
            inventadas.append(numero + "%")
    return inventadas


def avaliar_explicacao(
    explicacao: str,
    parametros: dict[str, float],
    rotulo_esperado: str,
    metricas_teste: Optional[dict[str, float]] = None,
) -> dict[str, bool]:
    """Checklist automático e verificável de qualidade da explicação.

    - fidelidade: cita o rótulo retornado pela API e todos os valores de entrada;
    - sem_rotulo_oposto_afirmado: não afirma a classe oposta como resultado;
    - estrutura: contém as cinco seções pedidas no prompt;
    - sem_numeros_inventados: nenhuma porcentagem fora das métricas fornecidas
      e nenhuma "probabilidade de" (o modelo não produz probabilidade);
    - sem_tratamento: não sugere tratamentos ou prognóstico;
    - aviso_medico: menciona avaliação por profissional/médico;
    - tamanho_adequado: até ~300 palavras.
    """
    texto = explicacao.lower()
    texto_numeros = _normalizar_numero(explicacao)
    oposto = "benigno" if rotulo_esperado.lower() == "maligno" else "maligno"

    valores_citados = all(
        re.search(rf"(?<![\d.]){re.escape(f'{valor:g}')}(?![\d])", texto_numeros)
        or re.search(rf"(?<![\d.]){re.escape(str(valor))}(?![\d])", texto_numeros)
        for valor in parametros.values()
    )

    return {
        "fidelidade_rotulo": rotulo_esperado.lower() in texto,
        "fidelidade_valores": bool(valores_citados),
        "sem_rotulo_oposto_afirmado": not re.search(
            rf"(classificad[oa]|resultado|classe)[^.\n]{{0,40}}\b{oposto}\b", texto
        ),
        "estrutura": all(secao.lower() in texto for secao in SECOES_OBRIGATORIAS),
        "sem_numeros_inventados": not _porcentagens_inventadas(explicacao, metricas_teste)
        and "probabilidade de" not in texto,
        "sem_tratamento": not any(termo in texto for termo in TERMOS_PROIBIDOS),
        "aviso_medico": any(
            termo in texto for termo in ["profissional de saúde", "médic", "profissional habilitado"]
        ),
        "tamanho_adequado": len(explicacao.split()) <= 300,
    }


# Cenários sintéticos usados na avaliação qualitativa (valores dentro das faixas
# observadas no dataset público, mas que não pertencem a nenhum paciente).
CENARIOS_AVALIACAO = [
    {
        "nome": "1. Completo - perfil benigno",
        "esperado": "sucesso",
        "texto": (
            "Laudo sintético para teste. Paciente fictício A. Medidas do exame: "
            "pontos côncavos (média) 0,021; pontos côncavos (pior valor) 0,072; "
            "perímetro (pior valor) 84,5; raio (pior valor) 13,2."
        ),
    },
    {
        "nome": "2. Completo - perfil maligno",
        "esperado": "sucesso",
        "texto": (
            "Exame sintético B: concave points mean = 0.121, concave points worst = 0.245, "
            "o pior perímetro registrado foi 165.3 e o pior raio foi 24.8."
        ),
    },
    {
        "nome": "3. Incompleto - falta o raio",
        "esperado": "revisao_necessaria",
        "texto": (
            "Exame sintético C: pontos côncavos médios 0,05; pontos côncavos no pior valor "
            "0,14; perímetro pior 108,4. O raio não foi medido."
        ),
    },
    {
        "nome": "4. Inválido - valor negativo",
        "esperado": "entrada_invalida",
        "texto": (
            "Exame sintético D: pontos côncavos (média) -0,03; pontos côncavos (pior valor) "
            "0,10; perímetro (pior valor) 95,0; raio (pior valor) 14,0."
        ),
    },
    {
        "nome": "5. Sem dados numéricos",
        "esperado": "revisao_necessaria",
        "texto": "Paciente sintético E relata desconforto. Exame de imagem ainda não realizado.",
    },
]
