"""Testes do contrato HTTP da API FastAPI de predição."""

from __future__ import annotations

import pickle

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main
from modelos import CAMINHO_MODELO, NOMES_FEATURES, carregar_dados, dividir_treino_teste

AMOSTRA_VALIDA = {
    "concave_points_mean": 0.052,
    "concave_points_worst": 0.146,
    "perimeter_worst": 108.4,
    "radius_worst": 16.9,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def modelo_direto():
    with CAMINHO_MODELO.open("rb") as arquivo:
        return pickle.load(arquivo)


# ---------------- Carregamento do modelo ----------------

def test_api_carrega_o_modelo_otimizado(client):
    resposta = client.get("/health/ready")
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert corpo["status"] == "ready"
    assert corpo["model_loaded"] is True
    assert corpo["model_file"] == "modelo_pipeline_genetico.pkl"
    assert corpo["model_steps"] == ["scaler", "modelo"]
    assert corpo["features"] == NOMES_FEATURES
    assert corpo["classes"] == {"0": "Benigno", "1": "Maligno"}


@pytest.fixture
def preservar_estado(client):
    """Restaura o modelo da API depois de testes que forçam falha na inicialização."""
    estado = (main.model, main.model_path)
    yield
    main.model, main.model_path = estado
    main.MODEL_READY.set(1)


@pytest.mark.usefixtures("preservar_estado")
def test_model_path_inexistente_impede_inicializacao(monkeypatch):
    """Sem fallback silencioso: arquivo errado faz a API falhar ao subir."""
    monkeypatch.setenv("MODEL_PATH", "nao_existe.pkl")
    with pytest.raises(FileNotFoundError):
        with TestClient(main.app):
            pass


@pytest.mark.usefixtures("preservar_estado")
def test_model_path_sem_scaler_e_rejeitado(monkeypatch, tmp_path):
    from sklearn.svm import LinearSVC

    x, y = carregar_dados()
    sem_scaler = LinearSVC().fit(x, y)
    caminho = tmp_path / "sem_scaler.pkl"
    caminho.write_bytes(pickle.dumps(sem_scaler))
    monkeypatch.setenv("MODEL_PATH", str(caminho))
    with pytest.raises(ValueError, match="scaler"):
        with TestClient(main.app):
            pass


# ---------------- Health, raiz e métricas ----------------

def test_liveness(client):
    resposta = client.get("/health/live")
    assert resposta.status_code == 200
    assert resposta.json() == {"status": "alive"}


def test_raiz_informa_endpoints(client):
    corpo = client.get("/").json()
    assert corpo["endpoints"]["predict"] == "POST /predict"


def test_endpoint_health_antigo_nao_existe(client):
    assert client.get("/health").status_code == 404


def test_metrics_expoe_contadores(client):
    client.post("/predict", json=AMOSTRA_VALIDA)
    resposta = client.get("/metrics")
    assert resposta.status_code == 200
    assert "api_requests_total" in resposta.text
    assert "model_predictions_total" in resposta.text
    assert "model_ready 1.0" in resposta.text


# ---------------- Predição ----------------

def test_predict_contrato_de_resposta(client):
    resposta = client.post("/predict", json=AMOSTRA_VALIDA, headers={"X-Request-ID": "teste-123"})
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert set(corpo) == {"output", "label", "model_file", "processing_time_ms", "request_id"}
    assert corpo["output"] in {"0", "1"}
    assert corpo["label"] == {"0": "Benigno", "1": "Maligno"}[corpo["output"]]
    assert corpo["model_file"] == "modelo_pipeline_genetico.pkl"
    assert corpo["request_id"] == "teste-123"
    assert corpo["processing_time_ms"] >= 0
    assert resposta.headers["X-Request-ID"] == "teste-123"
    assert float(resposta.headers["X-Process-Time"]) >= 0


def test_predict_via_get_nao_e_permitido(client):
    assert client.get("/predict", params=AMOSTRA_VALIDA).status_code == 405


def test_api_igual_a_predicao_direta_no_conjunto_de_teste(client, modelo_direto):
    """Para todas as 171 amostras de teste, API e pipeline direto concordam."""
    _, x_teste, _, _ = dividir_treino_teste(*carregar_dados())
    esperado = modelo_direto.predict(x_teste)

    for (_, linha), classe in zip(x_teste.iterrows(), esperado):
        resposta = client.post("/predict", json={k: float(v) for k, v in linha.items()})
        assert resposta.status_code == 200
        assert resposta.json()["output"] == str(int(classe))


def test_casos_reais_do_dataset_sao_classificados_coerentemente(client):
    """Amostras extremas do dataset (valores originais) -> classes esperadas."""
    benigno = {"concave_points_mean": 0.0, "concave_points_worst": 0.0,
               "perimeter_worst": 59.16, "radius_worst": 9.456}
    maligno = {"concave_points_mean": 0.1471, "concave_points_worst": 0.2654,
               "perimeter_worst": 184.6, "radius_worst": 25.38}
    assert client.post("/predict", json=benigno).json()["label"] == "Benigno"
    assert client.post("/predict", json=maligno).json()["label"] == "Maligno"


# ---------------- Entradas inválidas ----------------

@pytest.mark.parametrize(
    "alteracao",
    [
        {"concave_points_mean": -1},
        {"perimeter_worst": 0},
        {"radius_worst": "abc"},
        {"concave_points_worst": None},
    ],
)
def test_predict_rejeita_valores_invalidos(client, alteracao):
    resposta = client.post("/predict", json={**AMOSTRA_VALIDA, **alteracao})
    assert resposta.status_code == 422


def test_predict_rejeita_campo_ausente(client):
    incompleto = dict(AMOSTRA_VALIDA)
    incompleto.pop("radius_worst")
    resposta = client.post("/predict", json=incompleto)
    assert resposta.status_code == 422
    assert "radius_worst" in resposta.text


def test_predict_rejeita_campo_extra(client):
    resposta = client.post("/predict", json={**AMOSTRA_VALIDA, "idade": 50})
    assert resposta.status_code == 422


def test_predict_rejeita_corpo_que_nao_e_json(client):
    resposta = client.post("/predict", content="texto", headers={"Content-Type": "text/plain"})
    assert resposta.status_code == 422


def test_predict_rejeita_infinito(client):
    resposta = client.post(
        "/predict",
        content='{"concave_points_mean": 0.05, "concave_points_worst": 0.1, '
        '"perimeter_worst": Infinity, "radius_worst": 10}',
        headers={"Content-Type": "application/json"},
    )
    assert resposta.status_code == 422


def test_predict_dataframe_tem_colunas_na_ordem_do_treino(client, modelo_direto, monkeypatch):
    recebido = {}

    class Espiao:
        named_steps = modelo_direto.named_steps

        def predict(self, x):
            recebido["x"] = x
            return modelo_direto.predict(x)

    monkeypatch.setattr(main, "model", Espiao())
    client.post("/predict", json=dict(reversed(list(AMOSTRA_VALIDA.items()))))
    assert isinstance(recebido["x"], pd.DataFrame)
    assert list(recebido["x"].columns) == NOMES_FEATURES


def test_respostas_json_declaram_charset_utf8(client):
    """Sem charset, o PowerShell 5.1 exibe os acentos das respostas corrompidos."""
    respostas = [
        client.post("/predict", json=AMOSTRA_VALIDA),
        client.post("/predict", json={}),
        client.get("/predict"),
        client.get("/health/ready"),
    ]
    for resposta in respostas:
        assert resposta.headers["content-type"] == "application/json; charset=utf-8"
    assert respostas[1].status_code == 422 and respostas[2].status_code == 405
