"""Configuração central dos dados e dos modelos de diagnóstico.

Este módulo é a fonte única de verdade para:
- a ordem e o significado das quatro características usadas pelo modelo;
- o mapeamento das classes (0 = Benigno, 1 = Maligno);
- a divisão reproduzível entre treino e teste;
- os pipelines (StandardScaler + classificador) usados no notebook, na API e nos testes;
- os espaços de busca do algoritmo genético.

Assim, notebook, API e testes utilizam exatamente as mesmas definições.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# ============================================================
# Caminhos (sempre relativos à pasta do projeto)
# ============================================================

PASTA_PROJETO = Path(__file__).resolve().parent
CAMINHO_DADOS = PASTA_PROJETO / "dados" / "dados.csv"
CAMINHO_MODELO = PASTA_PROJETO / "modelo_pipeline_genetico.pkl"
PASTA_RESULTADOS = PASTA_PROJETO / "resultados"

# ============================================================
# Dados
# ============================================================

# Ordem das características usada no treino e na predição.
# Chave: nome usado no código/API. Valor: nome da coluna no CSV.
FEATURES = {
    "concave_points_mean": "concave points_mean",
    "concave_points_worst": "concave points_worst",
    "perimeter_worst": "perimeter_worst",
    "radius_worst": "radius_worst",
}
NOMES_FEATURES = list(FEATURES)

# Mapeamento explícito do alvo (o LabelEncoder da Fase 1 gerava o mesmo
# resultado por ordem alfabética: B -> 0, M -> 1).
MAPA_DIAGNOSTICO = {"B": 0, "M": 1}
MAPA_CLASSES = {0: "Benigno", 1: "Maligno"}

SEMENTE = 42
PROPORCAO_TESTE = 0.3


def carregar_dados(caminho: Path | str = CAMINHO_DADOS) -> tuple[pd.DataFrame, pd.Series]:
    """Lê o CSV e devolve X (4 features renomeadas) e y (0/1)."""
    dados = pd.read_csv(caminho)
    x = dados[list(FEATURES.values())].rename(
        columns={coluna: nome for nome, coluna in FEATURES.items()}
    )
    y = dados["diagnosis"].map(MAPA_DIAGNOSTICO)
    if y.isna().any():
        raise ValueError("Coluna 'diagnosis' contém valores diferentes de 'B' e 'M'.")
    return x, y.astype(int).rename("diagnosis")


def dividir_treino_teste(x: pd.DataFrame, y: pd.Series):
    """Divisão 70/30 estratificada e reproduzível (a mesma da Fase 1)."""
    return train_test_split(
        x, y, test_size=PROPORCAO_TESTE, random_state=SEMENTE, stratify=y
    )


# ============================================================
# Pipelines
# ============================================================

def _pipeline(classificador) -> Pipeline:
    """O scaler fica dentro do pipeline: na validação cruzada ele é ajustado
    apenas nos dados de treino de cada fold, e o modelo exportado recebe
    valores originais (sem escalonamento manual)."""
    return Pipeline([("scaler", StandardScaler()), ("modelo", classificador)])


def criar_knn(parametros: dict[str, Any]) -> Pipeline:
    return _pipeline(KNeighborsClassifier(**parametros))


def criar_svm(parametros: dict[str, Any]) -> Pipeline:
    return _pipeline(LinearSVC(random_state=SEMENTE, **parametros))


def criar_random_forest(parametros: dict[str, Any]) -> Pipeline:
    return _pipeline(RandomForestClassifier(random_state=SEMENTE, n_jobs=1, **parametros))


FABRICAS_MODELOS = {
    "KNN": criar_knn,
    "SVM": criar_svm,
    "Random Forest": criar_random_forest,
}

# Hiperparâmetros definidos manualmente pelo grupo (modelos "originais").
PARAMETROS_BASELINE = {
    "KNN": {"n_neighbors": 5},
    "SVM": {"C": 1},
    "Random Forest": {},
}


def criar_baseline(nome_modelo: str) -> Pipeline:
    """Recria os modelos originais com os mesmos hiperparâmetros manuais."""
    if nome_modelo == "Random Forest":
        # A Fase 1 usava random_state=7 no Random Forest original.
        return _pipeline(RandomForestClassifier(random_state=7))
    return FABRICAS_MODELOS[nome_modelo](PARAMETROS_BASELINE[nome_modelo])


# Espaço de busca do algoritmo genético (cada gene = um hiperparâmetro).
ESPACOS_BUSCA = {
    "KNN": {
        "n_neighbors": list(range(1, 26, 2)),
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "SVM": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "class_weight": [None, "balanced"],
        "max_iter": [2000, 5000, 10000],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [None, 3, 5, 8, 12, 16, 20],
        "min_samples_split": [2, 4, 6, 8, 10],
        "min_samples_leaf": [1, 2, 3, 4, 5],
        "max_features": ["sqrt", "log2", None],
        "class_weight": [None, "balanced"],
    },
}


# ============================================================
# Métricas
# ============================================================

def calcular_metricas(y_real, y_pred) -> dict[str, float]:
    """Métricas de teste; a classe positiva é 1 (Maligno)."""
    return {
        "accuracy": float(accuracy_score(y_real, y_pred)),
        "precision": float(precision_score(y_real, y_pred, zero_division=0)),
        "recall": float(recall_score(y_real, y_pred, zero_division=0)),
        "f1": float(f1_score(y_real, y_pred, zero_division=0)),
    }
