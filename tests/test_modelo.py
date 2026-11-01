"""Testes dos dados, do pipeline e do modelo exportado pelo notebook."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modelos import (
    CAMINHO_DADOS,
    CAMINHO_MODELO,
    FABRICAS_MODELOS,
    MAPA_CLASSES,
    NOMES_FEATURES,
    PASTA_RESULTADOS,
    calcular_metricas,
    carregar_dados,
    criar_baseline,
    dividir_treino_teste,
)

CAMINHO_RESULTADOS = PASTA_RESULTADOS / "resultados_ag.json"


@pytest.fixture(scope="module")
def modelo_exportado():
    with CAMINHO_MODELO.open("rb") as arquivo:
        return pickle.load(arquivo)


@pytest.fixture(scope="module")
def divisao():
    return dividir_treino_teste(*carregar_dados())


# ---------------- Dados e classes ----------------

def test_mapeamento_das_classes_a_partir_do_csv():
    bruto = pd.read_csv(CAMINHO_DADOS)
    x, y = carregar_dados()
    assert set(bruto["diagnosis"]) == {"B", "M"}
    assert (y[bruto["diagnosis"] == "B"] == 0).all()
    assert (y[bruto["diagnosis"] == "M"] == 1).all()
    assert MAPA_CLASSES == {0: "Benigno", 1: "Maligno"}
    # proporções do dataset público Breast Cancer Wisconsin
    assert (y == 0).sum() == 357 and (y == 1).sum() == 212


def test_mapeamento_igual_ao_label_encoder_da_fase_1():
    from sklearn.preprocessing import LabelEncoder

    bruto = pd.read_csv(CAMINHO_DADOS)
    _, y = carregar_dados()
    assert (LabelEncoder().fit_transform(bruto["diagnosis"]) == y.to_numpy()).all()


def test_ordem_e_valores_das_features():
    bruto = pd.read_csv(CAMINHO_DADOS)
    x, _ = carregar_dados()
    assert list(x.columns) == NOMES_FEATURES == [
        "concave_points_mean",
        "concave_points_worst",
        "perimeter_worst",
        "radius_worst",
    ]
    assert np.allclose(x["concave_points_mean"], bruto["concave points_mean"])
    assert np.allclose(x["radius_worst"], bruto["radius_worst"])


def test_divisao_reproduzivel_e_estratificada():
    x, y = carregar_dados()
    x_tr1, x_te1, y_tr1, y_te1 = dividir_treino_teste(x, y)
    x_tr2, x_te2, _, _ = dividir_treino_teste(x, y)
    assert list(x_te1.index) == list(x_te2.index)
    assert len(x_te1) == 171 and len(x_tr1) == 398
    assert abs(y_te1.mean() - y_tr1.mean()) < 0.01


# ---------------- Pipelines ----------------

@pytest.mark.parametrize("nome_modelo", ["KNN", "SVM", "Random Forest"])
def test_pipelines_contem_scaler_antes_do_classificador(nome_modelo):
    for pipeline in (criar_baseline(nome_modelo), FABRICAS_MODELOS[nome_modelo]({})):
        assert isinstance(pipeline, Pipeline)
        assert list(pipeline.named_steps) == ["scaler", "modelo"]
        assert isinstance(pipeline.named_steps["scaler"], StandardScaler)


def test_pipeline_recebe_valores_originais(divisao):
    """O pipeline com scaler aplicado em dados brutos == scaler manual + modelo."""
    x_treino, x_teste, y_treino, _ = divisao
    pipeline = FABRICAS_MODELOS["SVM"]({"C": 1}).fit(x_treino, y_treino)

    scaler = StandardScaler().fit(x_treino)
    from sklearn.svm import LinearSVC

    manual = LinearSVC(C=1, random_state=42).fit(scaler.transform(x_treino), y_treino)
    assert (pipeline.predict(x_teste) == manual.predict(scaler.transform(x_teste))).all()


# ---------------- Modelo exportado ----------------

def test_modelo_exportado_e_pipeline_otimizado(modelo_exportado):
    assert CAMINHO_MODELO.name == "modelo_pipeline_genetico.pkl"
    assert isinstance(modelo_exportado, Pipeline)
    assert list(modelo_exportado.named_steps) == ["scaler", "modelo"]
    assert list(modelo_exportado.classes_) == [0, 1]
    assert list(modelo_exportado.feature_names_in_) == NOMES_FEATURES


def test_modelo_exportado_corresponde_ao_relatorio(modelo_exportado, divisao):
    """As métricas de teste do arquivo .pkl são as mesmas salvas pelo notebook."""
    resultados = json.loads(CAMINHO_RESULTADOS.read_text(encoding="utf-8"))
    final = resultados["modelo_final"]
    _, x_teste, _, y_teste = divisao

    assert type(modelo_exportado.named_steps["modelo"]).__name__ == final["classe_sklearn"]
    assert modelo_exportado.named_steps["modelo"].get_params()  # parâmetros acessíveis
    metricas = calcular_metricas(y_teste, modelo_exportado.predict(x_teste))
    for nome, valor in final["metricas_teste"].items():
        assert metricas[nome] == pytest.approx(valor, abs=1e-9)


def test_modelo_exportado_reproduzivel_pelos_hiperparametros(modelo_exportado, divisao):
    """Retreinar com os hiperparâmetros registrados gera as mesmas predições."""
    resultados = json.loads(CAMINHO_RESULTADOS.read_text(encoding="utf-8"))
    final = resultados["modelo_final"]
    x_treino, x_teste, y_treino, _ = divisao

    retreinado = FABRICAS_MODELOS[final["modelo"]](final["hiperparametros"]).fit(x_treino, y_treino)
    assert (retreinado.predict(x_teste) == modelo_exportado.predict(x_teste)).all()
