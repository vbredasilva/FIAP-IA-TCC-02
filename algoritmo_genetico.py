"""Algoritmo genético para otimização de hiperparâmetros de classificadores.

Única implementação do AG no projeto: o notebook e os testes importam daqui.

Representação: cada indivíduo é um dicionário {hiperparâmetro: valor}; cada
hiperparâmetro é um gene cujo domínio é a lista de valores do espaço de busca.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate

Individuo = dict[str, Any]
EspacoBusca = dict[str, list[Any]]


def criar_individuo(espaco_busca: EspacoBusca, rng: random.Random | None = None) -> Individuo:
    """Cria uma combinação aleatória de hiperparâmetros."""
    rng = rng or random
    return {
        parametro: rng.choice(valores)
        for parametro, valores in espaco_busca.items()
    }


def criar_populacao(
    espaco_busca: EspacoBusca,
    tamanho_populacao: int,
    rng: random.Random | None = None,
) -> list[Individuo]:
    """Cria a população inicial do algoritmo genético."""
    return [criar_individuo(espaco_busca, rng) for _ in range(tamanho_populacao)]


def chave_individuo(individuo: Individuo) -> tuple:
    """Transforma o indivíduo em chave para reutilizar avaliações anteriores."""
    return tuple(sorted(individuo.items(), key=lambda item: item[0]))


def avaliar_individuo(
    individuo: Individuo,
    fabrica_modelo: Callable[[Individuo], Any],
    x_treino,
    y_treino,
    validacao,
    cache: dict,
    scoring: str = "recall",
    n_jobs: int | None = -1,
) -> float:
    """Fitness: média do score na validação cruzada (somente dados de treino).

    O resultado fica em cache, então indivíduos repetidos não são retreinados.
    """
    chave = chave_individuo(individuo)

    if chave not in cache:
        modelo = fabrica_modelo(individuo)
        resultados = cross_val_score(
            modelo,
            x_treino,
            y_treino,
            cv=validacao,
            scoring=scoring,
            n_jobs=n_jobs,
            error_score="raise",
        )
        cache[chave] = float(resultados.mean())

    return cache[chave]


def selecionar_por_torneio(
    populacao_avaliada: list[tuple[Individuo, float]],
    tamanho_torneio: int,
    rng: random.Random | None = None,
) -> Individuo:
    """Seleciona o melhor indivíduo de uma amostra aleatória da população."""
    rng = rng or random
    competidores = rng.sample(
        populacao_avaliada,
        k=min(tamanho_torneio, len(populacao_avaliada)),
    )
    vencedor = max(competidores, key=lambda item: item[1])
    return deepcopy(vencedor[0])


def aplicar_crossover(
    pai_1: Individuo,
    pai_2: Individuo,
    taxa_crossover: float,
    rng: random.Random | None = None,
) -> tuple[Individuo, Individuo]:
    """Combina os hiperparâmetros dos pais com crossover uniforme."""
    rng = rng or random
    if rng.random() >= taxa_crossover:
        return deepcopy(pai_1), deepcopy(pai_2)

    filho_1: Individuo = {}
    filho_2: Individuo = {}

    for parametro in pai_1:
        if rng.random() < 0.5:
            filho_1[parametro] = pai_1[parametro]
            filho_2[parametro] = pai_2[parametro]
        else:
            filho_1[parametro] = pai_2[parametro]
            filho_2[parametro] = pai_1[parametro]

    return filho_1, filho_2


def aplicar_mutacao(
    individuo: Individuo,
    espaco_busca: EspacoBusca,
    taxa_mutacao: float,
    rng: random.Random | None = None,
) -> Individuo:
    """Sorteia um novo valor (do domínio do gene) com probabilidade taxa_mutacao."""
    rng = rng or random
    individuo_mutado = deepcopy(individuo)

    for parametro, valores in espaco_busca.items():
        if rng.random() < taxa_mutacao:
            individuo_mutado[parametro] = rng.choice(valores)

    return individuo_mutado


def executar_algoritmo_genetico(
    fabrica_modelo: Callable[[Individuo], Any],
    espaco_busca: EspacoBusca,
    x_treino,
    y_treino,
    tamanho_populacao: int = 12,
    numero_geracoes: int = 8,
    taxa_crossover: float = 0.8,
    taxa_mutacao: float = 0.15,
    tamanho_torneio: int = 3,
    quantidade_elites: int = 2,
    scoring: str = "recall",
    semente: int = 42,
    n_jobs: int | None = -1,
    verbose: bool = True,
) -> tuple[Individuo, float, list[dict[str, float]]]:
    """Executa a busca genética e retorna (melhor indivíduo, fitness, histórico).

    Cada geração: avaliação -> elitismo -> torneio -> crossover -> mutação.
    A semente controla população inicial, seleção, crossover, mutação e folds.
    """
    rng = random.Random(semente)

    validacao = StratifiedKFold(n_splits=5, shuffle=True, random_state=semente)
    cache_avaliacoes: dict = {}
    populacao = criar_populacao(espaco_busca, tamanho_populacao, rng)
    historico: list[dict[str, float]] = []

    def avaliar(individuo: Individuo) -> float:
        return avaliar_individuo(
            individuo,
            fabrica_modelo,
            x_treino,
            y_treino,
            validacao,
            cache_avaliacoes,
            scoring=scoring,
            n_jobs=n_jobs,
        )

    for geracao in range(1, numero_geracoes + 1):
        populacao_avaliada = [(individuo, avaliar(individuo)) for individuo in populacao]
        # sort é estável: em caso de empate, mantém a ordem da população.
        populacao_avaliada.sort(key=lambda item: item[1], reverse=True)

        melhor_fitness = populacao_avaliada[0][1]
        fitness_medio = float(np.mean([fitness for _, fitness in populacao_avaliada]))
        historico.append(
            {
                "geracao": geracao,
                "melhor_fitness": melhor_fitness,
                "fitness_medio": fitness_medio,
                "avaliacoes_unicas": len(cache_avaliacoes),
            }
        )

        if verbose:
            print(
                f"Geracao {geracao:02d} | "
                f"Melhor {scoring}: {melhor_fitness:.4f} | "
                f"{scoring} medio: {fitness_medio:.4f} | "
                f"avaliacoes unicas: {len(cache_avaliacoes)}"
            )

        # Elitismo: os melhores passam sem alteração para a próxima geração.
        nova_populacao = [
            deepcopy(individuo)
            for individuo, _ in populacao_avaliada[:quantidade_elites]
        ]

        while len(nova_populacao) < tamanho_populacao:
            pai_1 = selecionar_por_torneio(populacao_avaliada, tamanho_torneio, rng)
            pai_2 = selecionar_por_torneio(populacao_avaliada, tamanho_torneio, rng)
            filho_1, filho_2 = aplicar_crossover(pai_1, pai_2, taxa_crossover, rng)
            filho_1 = aplicar_mutacao(filho_1, espaco_busca, taxa_mutacao, rng)
            filho_2 = aplicar_mutacao(filho_2, espaco_busca, taxa_mutacao, rng)
            nova_populacao.extend([filho_1, filho_2])

        populacao = nova_populacao[:tamanho_populacao]

    # A última população foi criada no fim do laço e ainda precisa ser avaliada.
    populacao_final = [(individuo, avaliar(individuo)) for individuo in populacao]
    melhor_individuo, melhor_fitness = max(populacao_final, key=lambda item: item[1])
    return deepcopy(melhor_individuo), melhor_fitness, historico


def metricas_validacao_cruzada(
    modelo,
    x_treino,
    y_treino,
    semente: int = 42,
    n_jobs: int | None = -1,
) -> dict[str, float]:
    """Accuracy, precision, recall e F1 médios nos mesmos 5 folds do AG.

    Usado para documentar e desempatar resultados sem tocar no conjunto de teste.
    """
    validacao = StratifiedKFold(n_splits=5, shuffle=True, random_state=semente)
    resultados = cross_validate(
        modelo,
        x_treino,
        y_treino,
        cv=validacao,
        scoring=["accuracy", "precision", "recall", "f1"],
        n_jobs=n_jobs,
    )
    return {
        metrica: float(resultados[f"test_{metrica}"].mean())
        for metrica in ["accuracy", "precision", "recall", "f1"]
    }


def escolher_melhor(resultados: list[dict[str, Any]]) -> dict[str, Any]:
    """Critério de escolha (somente dados de treino/validação cruzada):

    1. maior recall médio na validação cruzada (fitness);
    2. empate -> maior F1 médio na validação cruzada;
    3. novo empate -> menor número de avaliações únicas (menor custo);
    4. novo empate -> ordem de definição (primeiro da lista).

    Cada item precisa das chaves "recall_cv", "f1_cv" e "avaliacoes".
    """
    return min(
        enumerate(resultados),
        key=lambda par: (
            -round(par[1]["recall_cv"], 10),
            -round(par[1]["f1_cv"], 10),
            par[1]["avaliacoes"],
            par[0],
        ),
    )[1]


# Três configurações distintas exigidas pelo Tech Challenge (definidas pelo grupo).
EXPERIMENTOS_AG = [
    {
        "nome": "Conservador",
        "descricao": "População menor, baixa mutação e torneio pequeno (explora pouco).",
        "tamanho_populacao": 8,
        "numero_geracoes": 10,
        "taxa_crossover": 0.7,
        "taxa_mutacao": 0.05,
        "tamanho_torneio": 2,
        "quantidade_elites": 1,
    },
    {
        "nome": "Equilibrado",
        "descricao": "Configuração intermediária entre exploração e explotação.",
        "tamanho_populacao": 12,
        "numero_geracoes": 10,
        "taxa_crossover": 0.8,
        "taxa_mutacao": 0.15,
        "tamanho_torneio": 3,
        "quantidade_elites": 2,
    },
    {
        "nome": "Exploratorio",
        "descricao": "População maior e mutação alta para diversificar a busca.",
        "tamanho_populacao": 20,
        "numero_geracoes": 10,
        "taxa_crossover": 0.9,
        "taxa_mutacao": 0.30,
        "tamanho_torneio": 4,
        "quantidade_elites": 2,
    },
]
