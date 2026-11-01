"""Testes do algoritmo genético e dos operadores genéticos."""

from __future__ import annotations

import random

import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from algoritmo_genetico import (
    EXPERIMENTOS_AG,
    aplicar_crossover,
    aplicar_mutacao,
    avaliar_individuo,
    chave_individuo,
    criar_individuo,
    criar_populacao,
    escolher_melhor,
    executar_algoritmo_genetico,
    selecionar_por_torneio,
)
from modelos import ESPACOS_BUSCA, FABRICAS_MODELOS


@pytest.fixture
def espaco_busca():
    return {
        "n_neighbors": [1, 3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    }


@pytest.fixture
def dados_sinteticos():
    return make_classification(
        n_samples=120,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )


def fabrica_knn(parametros):
    return KNeighborsClassifier(**parametros)


def executar_curto(espaco, x, y, semente=42, **extra):
    parametros = dict(
        fabrica_modelo=fabrica_knn,
        espaco_busca=espaco,
        x_treino=x,
        y_treino=y,
        tamanho_populacao=6,
        numero_geracoes=4,
        taxa_crossover=0.8,
        taxa_mutacao=0.2,
        tamanho_torneio=2,
        quantidade_elites=1,
        scoring="recall",
        semente=semente,
        n_jobs=1,
        verbose=False,
    )
    parametros.update(extra)
    return executar_algoritmo_genetico(**parametros)


# ---------------- Experimentos e espaço de busca ----------------

def test_existem_tres_experimentos_com_configuracoes_distintas():
    assert len(EXPERIMENTOS_AG) >= 3
    nomes = {exp["nome"] for exp in EXPERIMENTOS_AG}
    assert len(nomes) == len(EXPERIMENTOS_AG)
    configs = {
        (e["tamanho_populacao"], e["taxa_mutacao"], e["taxa_crossover"]) for e in EXPERIMENTOS_AG
    }
    assert len(configs) == len(EXPERIMENTOS_AG)


@pytest.mark.parametrize("nome_modelo", ["KNN", "SVM", "Random Forest"])
def test_individuos_dos_tres_modelos_tem_genes_validos(nome_modelo):
    espaco = ESPACOS_BUSCA[nome_modelo]
    rng = random.Random(0)
    for _ in range(20):
        individuo = criar_individuo(espaco, rng)
        assert set(individuo) == set(espaco)
        assert all(valor in espaco[gene] for gene, valor in individuo.items())
        # o indivíduo precisa gerar um modelo válido
        FABRICAS_MODELOS[nome_modelo](individuo)


# ---------------- Operadores ----------------

def test_criar_individuo_respeita_espaco(espaco_busca):
    individuo = criar_individuo(espaco_busca, random.Random(0))
    assert set(individuo) == set(espaco_busca)
    for parametro, valor in individuo.items():
        assert valor in espaco_busca[parametro]


def test_criar_populacao_tamanho(espaco_busca):
    populacao = criar_populacao(espaco_busca, tamanho_populacao=5, rng=random.Random(0))
    assert len(populacao) == 5


def test_chave_individuo_independe_da_ordem():
    assert chave_individuo({"a": 1, "b": 2}) == chave_individuo({"b": 2, "a": 1})


def test_crossover_combina_genes_dos_pais(espaco_busca):
    pai_1 = {"n_neighbors": 1, "weights": "uniform", "p": 1}
    pai_2 = {"n_neighbors": 9, "weights": "distance", "p": 2}
    filho_1, filho_2 = aplicar_crossover(pai_1, pai_2, 1.0, random.Random(1))

    for gene in pai_1:
        # crossover uniforme: cada gene vem de um dos pais e os filhos são complementares
        assert {filho_1[gene], filho_2[gene]} == {pai_1[gene], pai_2[gene]}


def test_crossover_com_taxa_zero_copia_os_pais():
    pai_1 = {"a": 1, "b": 1}
    pai_2 = {"a": 2, "b": 2}
    filho_1, filho_2 = aplicar_crossover(pai_1, pai_2, 0.0, random.Random(1))
    assert (filho_1, filho_2) == (pai_1, pai_2)
    assert filho_1 is not pai_1  # cópia, não referência


def test_mutacao_mantem_genes_no_dominio(espaco_busca):
    original = {"n_neighbors": 1, "weights": "uniform", "p": 1}
    mutado = aplicar_mutacao(original, espaco_busca, 1.0, random.Random(2))
    assert set(mutado) == set(original)
    for parametro, valor in mutado.items():
        assert valor in espaco_busca[parametro]
    assert original == {"n_neighbors": 1, "weights": "uniform", "p": 1}  # sem efeito colateral


def test_mutacao_com_taxa_zero_nao_altera(espaco_busca):
    original = {"n_neighbors": 1, "weights": "uniform", "p": 1}
    assert aplicar_mutacao(original, espaco_busca, 0.0, random.Random(2)) == original


def test_mutacao_altera_genes_com_taxa_alta(espaco_busca):
    rng = random.Random(3)
    original = {"n_neighbors": 1, "weights": "uniform", "p": 1}
    alteracoes = sum(aplicar_mutacao(original, espaco_busca, 1.0, rng) != original for _ in range(50))
    assert alteracoes > 0


def test_selecao_torneio_escolhe_melhor_do_torneio():
    populacao = [({"a": 1}, 0.1), ({"a": 2}, 0.9), ({"a": 3}, 0.2)]
    assert selecionar_por_torneio(populacao, 3, random.Random(3)) == {"a": 2}


def test_selecao_torneio_favorece_individuos_melhores():
    populacao = [({"a": i}, i / 10) for i in range(10)]
    rng = random.Random(4)
    escolhidos = [selecionar_por_torneio(populacao, 3, rng)["a"] for _ in range(500)]
    media = sum(escolhidos) / len(escolhidos)
    assert media > 4.5  # aleatório puro teria média 4.5


# ---------------- Fitness e cache ----------------

def test_fitness_igual_ao_recall_da_validacao_cruzada(dados_sinteticos):
    x, y = dados_sinteticos
    individuo = {"n_neighbors": 3, "weights": "uniform", "p": 2}
    validacao = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    esperado = cross_val_score(fabrica_knn(individuo), x, y, cv=validacao, scoring="recall").mean()

    cache: dict = {}
    obtido = avaliar_individuo(individuo, fabrica_knn, x, y, validacao, cache, n_jobs=1)
    assert obtido == pytest.approx(esperado)
    assert 0.0 <= obtido <= 1.0


def test_cache_evita_reavaliar_individuo_repetido(dados_sinteticos):
    x, y = dados_sinteticos
    chamadas = []

    def fabrica_contadora(parametros):
        chamadas.append(parametros)
        return fabrica_knn(parametros)

    validacao = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cache: dict = {}
    individuo = {"n_neighbors": 3, "weights": "uniform", "p": 2}
    primeiro = avaliar_individuo(individuo, fabrica_contadora, x, y, validacao, cache, n_jobs=1)
    segundo = avaliar_individuo(dict(individuo), fabrica_contadora, x, y, validacao, cache, n_jobs=1)
    assert primeiro == segundo
    assert len(chamadas) == 1


# ---------------- Execução completa ----------------

def test_execucao_retorna_historico_e_elitismo_nao_piora(espaco_busca, dados_sinteticos):
    x, y = dados_sinteticos
    melhor, fitness, historico = executar_curto(espaco_busca, x, y)

    assert set(melhor) == set(espaco_busca)
    assert 0.0 <= fitness <= 1.0
    assert len(historico) == 4
    melhores = [h["melhor_fitness"] for h in historico]
    # com elitismo, o melhor fitness nunca diminui entre gerações
    assert all(b >= a - 1e-12 for a, b in zip(melhores, melhores[1:]))
    assert fitness >= melhores[-1] - 1e-12


def test_execucao_reproduzivel_com_mesma_semente(espaco_busca, dados_sinteticos):
    x, y = dados_sinteticos
    random.seed(123)  # estado global não deve influenciar
    resultado_1 = executar_curto(espaco_busca, x, y, semente=7)
    random.seed(999)
    resultado_2 = executar_curto(espaco_busca, x, y, semente=7)
    assert resultado_1 == resultado_2


def test_ag_evolui_a_populacao(dados_sinteticos):
    """O fitness médio da população sobe: há pressão seletiva, não busca aleatória."""
    x, y = dados_sinteticos
    espaco = {
        "n_neighbors": list(range(1, 60, 2)),
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    }
    _, _, historico = executar_curto(
        espaco, x, y, tamanho_populacao=10, numero_geracoes=6, taxa_mutacao=0.1, tamanho_torneio=3
    )
    assert historico[-1]["fitness_medio"] > historico[0]["fitness_medio"]


# ---------------- Desempate ----------------

def test_escolher_melhor_usa_recall_depois_f1_depois_custo():
    resultados = [
        {"nome": "A", "recall_cv": 0.90, "f1_cv": 0.99, "avaliacoes": 5},
        {"nome": "B", "recall_cv": 0.95, "f1_cv": 0.90, "avaliacoes": 50},
        {"nome": "C", "recall_cv": 0.95, "f1_cv": 0.92, "avaliacoes": 40},
        {"nome": "D", "recall_cv": 0.95, "f1_cv": 0.92, "avaliacoes": 30},
    ]
    assert escolher_melhor(resultados)["nome"] == "D"


def test_escolher_melhor_empate_total_mantem_ordem():
    resultados = [
        {"nome": "A", "recall_cv": 0.95, "f1_cv": 0.92, "avaliacoes": 30},
        {"nome": "B", "recall_cv": 0.95, "f1_cv": 0.92, "avaliacoes": 30},
    ]
    assert escolher_melhor(resultados)["nome"] == "A"
