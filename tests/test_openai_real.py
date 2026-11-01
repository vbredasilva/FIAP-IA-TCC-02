"""Teste de ponta a ponta com chamada REAL à OpenAI (opcional).

Não roda por padrão, para que `pytest` funcione sem chave e sem internet.
Para executar (PowerShell):

    $env:RUN_OPENAI_TESTS = "1"; python -m pytest tests/test_openai_real.py -v

A chave é lida do arquivo .env (OPENAI_API_KEY). A API de predição roda em
memória (TestClient), então não é preciso subir o uvicorn para este teste.
"""

from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

import assistente_llm as llm
import main
from modelos import PASTA_RESULTADOS

METRICAS_TESTE = json.loads((PASTA_RESULTADOS / "resultados_ag.json").read_text(encoding="utf-8"))[
    "modelo_final"
]["metricas_teste"]

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_OPENAI_TESTS") != "1",
    reason="Chamada real à OpenAI desativada (defina RUN_OPENAI_TESTS=1).",
)


@pytest.fixture(scope="module")
def configuracao():
    config = llm.carregar_configuracao()
    if not config["OPENAI_API_KEY"] or config["OPENAI_API_KEY"] == "sua_chave_aqui":
        pytest.fail("RUN_OPENAI_TESTS=1, mas OPENAI_API_KEY não está configurada no .env")
    return config


@pytest.fixture(scope="module")
def cliente(configuracao):
    return llm.criar_cliente_openai(configuracao["OPENAI_API_KEY"])


@pytest.fixture
def api_em_memoria(monkeypatch):
    with TestClient(main.app) as cliente_api:
        monkeypatch.setattr(
            llm.requests,
            "post",
            lambda url, json=None, timeout=None: cliente_api.post("/predict", json=json),
        )
        yield


@pytest.mark.parametrize("cenario", llm.CENARIOS_AVALIACAO, ids=lambda c: c["nome"])
def test_cenarios_reais(cenario, cliente, configuracao, api_em_memoria):
    resultado = llm.processar_exame(
        cenario["texto"], cliente, configuracao["OPENAI_MODEL"], metricas_teste=METRICAS_TESTE
    )
    assert resultado["status"] == cenario["esperado"], resultado

    if resultado["status"] == "sucesso":
        avaliacao = llm.avaliar_explicacao(
            resultado["explicacao"],
            resultado["parametros"],
            resultado["retorno_api"]["label"],
            METRICAS_TESTE,
        )
        falhas = [criterio for criterio, ok in avaliacao.items() if not ok]
        assert not falhas, (falhas, resultado["explicacao"])


def test_endpoint_explicar_real(configuracao, monkeypatch):
    """POST /explicar de ponta a ponta: OpenAI extrai, modelo classifica, OpenAI explica."""
    monkeypatch.setattr(main, "cliente_llm", None)
    with TestClient(main.app) as cliente_api:
        resposta = cliente_api.post(
            "/explicar", json={"texto_exame": llm.CENARIOS_AVALIACAO[1]["texto"]}
        )
    assert resposta.status_code == 200, resposta.text
    corpo = resposta.json()
    assert corpo["status"] == "sucesso", corpo
    assert corpo["predicao"]["label"] == "Maligno"
    avaliacao = llm.avaliar_explicacao(
        corpo["explicacao"], corpo["parametros"], corpo["predicao"]["label"], METRICAS_TESTE
    )
    falhas = [criterio for criterio, ok in avaliacao.items() if not ok]
    assert not falhas, (falhas, corpo["explicacao"])
