"""
API de predição para modelo de diagnóstico.

Decisões de implementação:
1. O modelo otimizado pelo algoritmo genético (modelo_pipeline_genetico.pkl)
   é carregado uma única vez durante a inicialização da aplicação.
   O caminho pode ser alterado pela variável MODEL_PATH; não existe fallback:
   se o arquivo não carregar, a API não sobe.
2. O arquivo é um sklearn Pipeline (StandardScaler + classificador), então a
   API recebe os valores originais do exame, sem escalonamento manual.
3. As predições são recebidas por POST em formato JSON.
4. Logs registram request_id, rota, status e tempo de processamento.
5. Métricas são expostas no endpoint /metrics no formato do Prometheus.
6. Endpoints /health/live e /health/ready suportam health checks.
7. POST /explicar recebe o texto do exame: a LLM (OpenAI) extrai os valores,
   o modelo carregado classifica e a LLM explica o resultado. A classificação
   nunca é feita pela LLM. Sem OPENAI_API_KEY, só este endpoint fica indisponível.
"""

import logging
import os
import pickle
import time
import json
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

import assistente_llm as llm
from modelos import CAMINHO_MODELO, MAPA_CLASSES, NOMES_FEATURES, PASTA_PROJETO, PASTA_RESULTADOS


# ============================================================
# Configuração
# ============================================================

def resolver_caminho_modelo() -> Path:
    """MODEL_PATH (opcional) relativo à pasta do projeto ou absoluto."""
    caminho = Path(os.getenv("MODEL_PATH", str(CAMINHO_MODELO)))
    if not caminho.is_absolute():
        caminho = PASTA_PROJETO / caminho
    return caminho


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
API_VERSION = "2.0.0"


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=LOG_LEVEL,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("diagnostico-api")


# ============================================================
# Métricas Prometheus
# ============================================================

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Quantidade total de requisições recebidas",
    ["method", "endpoint", "status_code"],
)

REQUEST_DURATION = Histogram(
    "api_request_duration_seconds",
    "Tempo de processamento das requisições",
    ["method", "endpoint"],
)

PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Quantidade total de predições realizadas",
    ["prediction"],
)

PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total",
    "Quantidade de erros durante a predição",
    ["error_type"],
)

LLM_EXPLANATIONS = Counter(
    "llm_explanations_total",
    "Quantidade de chamadas ao POST /explicar, por resultado do fluxo",
    ["status"],
)

LLM_ERRORS = Counter(
    "llm_errors_total",
    "Quantidade de falhas na integração com a LLM",
    ["error_type"],
)

MODEL_READY = Gauge(
    "model_ready",
    "Indica se o modelo foi carregado corretamente",
)


# ============================================================
# Estado da aplicação
# ============================================================

model = None
model_path: Path | None = None
cliente_llm = None  # criado na primeira chamada a POST /explicar


def obter_cliente_llm():
    """Cliente OpenAI e nome do modelo LLM, lidos do .env na primeira chamada."""
    global cliente_llm
    config = llm.carregar_configuracao()
    if cliente_llm is None:
        cliente_llm = llm.criar_cliente_openai(config["OPENAI_API_KEY"])
    return cliente_llm, config["OPENAI_MODEL"]


@lru_cache(maxsize=1)
def metricas_teste_modelo() -> dict[str, float] | None:
    """Métricas de teste do modelo final, usadas como contexto na explicação."""
    try:
        resultados = json.loads((PASTA_RESULTADOS / "resultados_ag.json").read_text(encoding="utf-8"))
        return {k: round(v, 4) for k, v in resultados["modelo_final"]["metricas_teste"].items()}
    except (OSError, KeyError, ValueError):
        return None


def carregar_modelo(caminho: Path):
    """Carrega o pipeline treinado no notebook e confere o formato esperado."""
    with caminho.open("rb") as model_file:
        modelo = pickle.load(model_file)

    if not hasattr(modelo, "named_steps") or "scaler" not in modelo.named_steps:
        raise ValueError(
            f"{caminho.name} não é um Pipeline com a etapa 'scaler'. "
            "Execute o notebook diagnosticos_algoritmo_genetico.ipynb para gerá-lo."
        )
    if list(modelo.classes_) != [0, 1]:
        raise ValueError(f"Classes inesperadas no modelo: {list(modelo.classes_)}")
    if list(getattr(modelo, "feature_names_in_", [])) != NOMES_FEATURES:
        raise ValueError("Ordem das características diferente da esperada.")
    return modelo


# ============================================================
# Inicialização e encerramento
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Executa tarefas de inicialização e encerramento da API.

    O modelo é carregado uma única vez quando a aplicação sobe,
    evitando leitura do arquivo em cada requisição.
    """

    global model, model_path

    caminho = resolver_caminho_modelo()

    try:
        logger.info("Carregando modelo | caminho=%s", caminho)
        model = carregar_modelo(caminho)
        model_path = caminho
        MODEL_READY.set(1)
        logger.info("Modelo carregado com sucesso | caminho=%s", caminho)

    except Exception:
        model = None
        model_path = None
        MODEL_READY.set(0)
        logger.exception("Falha ao carregar o modelo | caminho=%s", caminho)
        raise

    yield

    model = None
    model_path = None
    MODEL_READY.set(0)
    logger.info("Encerrando API de diagnóstico")


# ============================================================
# Aplicação
# ============================================================

class JSONUTF8Response(JSONResponse):
    """JSON com charset explícito: clientes como o PowerShell 5.1 exibem os acentos corretamente."""

    media_type = "application/json; charset=utf-8"


app = FastAPI(
    default_response_class=JSONUTF8Response,
    title="API de Predição de Diagnóstico",
    description=(
        "Recebe as quatro características numéricas do exame e retorna a "
        "classificação do modelo otimizado por algoritmo genético "
        "(0 = Benigno, 1 = Maligno). O endpoint POST /explicar recebe o texto "
        "do exame e devolve a classificação com uma explicação gerada por LLM. "
        "Protótipo educacional: não substitui avaliação médica."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)


# ============================================================
# Schemas
# ============================================================

class PredictionInput(BaseModel):
    """
    Dados de entrada esperados pelo modelo (valores originais, sem escalonamento).
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "concave_points_mean": 0.052,
                    "concave_points_worst": 0.146,
                    "perimeter_worst": 108.4,
                    "radius_worst": 16.9,
                }
            ]
        },
    )

    concave_points_mean: float = Field(
        ...,
        ge=0,
        allow_inf_nan=False,
        description="Média dos pontos côncavos",
    )

    concave_points_worst: float = Field(
        ...,
        ge=0,
        allow_inf_nan=False,
        description="Pior valor dos pontos côncavos",
    )

    perimeter_worst: float = Field(
        ...,
        gt=0,
        allow_inf_nan=False,
        description="Pior valor do perímetro",
    )

    radius_worst: float = Field(
        ...,
        gt=0,
        allow_inf_nan=False,
        description="Pior valor do raio",
    )


class PredictionOutput(BaseModel):
    """
    Resposta retornada pela API.
    """

    output: str = Field(description="Classe prevista: '0' ou '1'")
    label: str = Field(description="Significado da classe: Benigno ou Maligno")
    model_file: str = Field(description="Arquivo do modelo que gerou a predição")
    processing_time_ms: float
    request_id: str


TextoExame = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=5000)]


class ExplanationInput(BaseModel):
    """
    Texto livre do exame (laudo sintético) com as quatro medidas.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "texto_exame": (
                        "Exame sintético: pontos côncavos (média) 0,121; pontos côncavos "
                        "(pior valor) 0,245; perímetro (pior valor) 165,3; raio (pior valor) 24,8."
                    )
                }
            ]
        },
    )

    texto_exame: TextoExame = Field(description="Texto do exame com as quatro medidas")


class ExplanationOutput(BaseModel):
    """
    Resultado do fluxo LLM -> modelo -> LLM.
    """

    status: Literal["sucesso", "revisao_necessaria", "entrada_invalida"] = Field(
        description="sucesso: classificado e explicado; revisao_necessaria: faltam valores; "
        "entrada_invalida: há valor fora do domínio (ex.: negativo)"
    )
    mensagem: str
    parametros: dict[str, float | None] | None = Field(
        default=None, description="Valores extraídos do texto pela LLM"
    )
    parametros_ausentes: list[str] = Field(default_factory=list)
    predicao: PredictionOutput | None = Field(
        default=None, description="Classificação do modelo (somente quando status = sucesso)"
    )
    explicacao: str | None = Field(default=None, description="Explicação em português gerada pela LLM")
    detalhe: str | None = None
    modelo_llm: str
    processing_time_ms: float
    request_id: str


# ============================================================
# Tratamento de erros de validação
# ============================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Retorna 422 com campo, mensagem e valor recebido.

    O valor é convertido em texto porque entradas como Infinity/NaN não são
    serializáveis em JSON e fariam a resposta de erro falhar com 500.
    """

    erros = [
        {
            "campo": ".".join(str(parte) for parte in erro.get("loc", [])),
            "mensagem": erro.get("msg"),
            "tipo": erro.get("type"),
            "valor_recebido": repr(erro.get("input")),
        }
        for erro in exc.errors()
    ]

    logger.warning(
        "Entrada inválida | request_id=%s | erros=%s",
        getattr(request.state, "request_id", "-"),
        erros,
    )

    return JSONUTF8Response(status_code=422, content={"detail": erros})


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Erros 4xx/5xx (ex.: 503, 502) no mesmo formato, com charset UTF-8."""

    return JSONUTF8Response(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=getattr(exc, "headers", None),
    )


# ============================================================
# Middleware de monitoramento
# ============================================================

@app.middleware("http")
async def monitoring_middleware(
    request: Request,
    call_next,
):
    """
    Registra o tempo, o status e um identificador para cada requisição.
    """

    request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4()),
    )

    request.state.request_id = request_id
    start_time = time.perf_counter()

    try:
        response = await call_next(request)

    except Exception:
        duration = time.perf_counter() - start_time

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code="500",
        ).inc()

        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)

        logger.exception(
            "Erro não tratado | request_id=%s | method=%s | "
            "path=%s | duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration * 1000,
        )

        raise

    duration = time.perf_counter() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=str(response.status_code),
    ).inc()

    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path,
    ).observe(duration)

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{duration:.6f}"

    logger.info(
        "Requisição finalizada | request_id=%s | method=%s | "
        "path=%s | status=%s | duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration * 1000,
    )

    return response


# ============================================================
# Endpoints
# ============================================================

@app.get("/", tags=["Application"])
def read_root():
    return {
        "application": "API de Predição de Diagnóstico",
        "version": API_VERSION,
        "status": "running",
        "documentation": "/docs",
        "endpoints": {
            "predict": "POST /predict",
            "explain": "POST /explicar",
            "liveness": "GET /health/live",
            "readiness": "GET /health/ready",
            "metrics": "GET /metrics",
        },
    }


@app.get("/health/live", tags=["Health"])
def liveness():
    """
    Informa se o processo da API está em execução.
    """

    return {"status": "alive"}


@app.get("/health/ready", tags=["Health"])
def readiness():
    """
    Informa se a API está pronta para receber predições.
    """

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo ainda não está disponível",
        )

    return {
        "status": "ready",
        "model_loaded": True,
        "model_file": model_path.name,
        "model_steps": list(model.named_steps),
        "classifier": type(model.named_steps["modelo"]).__name__,
        "features": NOMES_FEATURES,
        "classes": {str(k): v for k, v in MAPA_CLASSES.items()},
    }


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """
    Expõe métricas no padrão Prometheus (texto).
    """

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


def executar_predicao(dados: dict[str, float], request_id: str) -> PredictionOutput:
    """Classifica os quatro valores com o modelo carregado (usado por /predict e /explicar)."""

    start_time = time.perf_counter()

    try:
        # DataFrame com as mesmas colunas, na mesma ordem, usadas no treino.
        novo_input = pd.DataFrame(
            [dados],
            columns=NOMES_FEATURES,
        )

        prediction = int(model.predict(novo_input)[0])
        prediction_value = str(prediction)

        processing_time_ms = (
            time.perf_counter() - start_time
        ) * 1000

        PREDICTION_COUNT.labels(
            prediction=prediction_value
        ).inc()

        logger.info(
            "Predição realizada | request_id=%s | "
            "prediction=%s | processing_time_ms=%.2f",
            request_id,
            prediction_value,
            processing_time_ms,
        )

        return PredictionOutput(
            output=prediction_value,
            label=MAPA_CLASSES[prediction],
            model_file=model_path.name,
            processing_time_ms=round(
                processing_time_ms,
                2,
            ),
            request_id=request_id,
        )

    except Exception as error:
        PREDICTION_ERRORS.labels(
            error_type=type(error).__name__
        ).inc()

        logger.exception(
            "Erro durante predição | request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Erro interno durante a predição",
        ) from error


@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["Prediction"],
)
def make_predict(
    prediction_input: PredictionInput,
    request: Request,
):
    """
    Executa a predição utilizando o modelo carregado.
    """

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo indisponível",
        )

    return executar_predicao(prediction_input.model_dump(), request.state.request_id)


@app.post(
    "/explicar",
    response_model=ExplanationOutput,
    tags=["Explanation"],
)
def explicar(
    explanation_input: ExplanationInput,
    request: Request,
):
    """
    Recebe o texto do exame, extrai os valores com a LLM, classifica com o
    modelo carregado e devolve a classificação com uma explicação em português.
    """

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo indisponível",
        )

    request_id = request.state.request_id
    start_time = time.perf_counter()

    try:
        cliente, modelo_llm = obter_cliente_llm()
    except RuntimeError as error:
        LLM_ERRORS.labels(error_type="configuracao").inc()
        logger.warning("LLM indisponível | request_id=%s | motivo=%s", request_id, error)
        raise HTTPException(status_code=503, detail=str(error)) from error

    def predizer(parametros: llm.ParametrosExame) -> dict:
        return executar_predicao(parametros.validar_para_predicao(), request_id).model_dump()

    try:
        resultado = llm.processar_exame(
            explanation_input.texto_exame,
            cliente,
            modelo_llm,
            metricas_teste=metricas_teste_modelo(),
            predizer=predizer,
        )
    except HTTPException:
        raise
    except Exception as error:
        LLM_ERRORS.labels(error_type=type(error).__name__).inc()
        logger.exception("Falha na integração com a LLM | request_id=%s", request_id)
        raise HTTPException(
            status_code=502,
            detail="Falha ao consultar a LLM. Verifique a chave, o modelo e a conexão com a OpenAI.",
        ) from error

    status = resultado["status"]
    LLM_EXPLANATIONS.labels(status=status).inc()
    processing_time_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Explicação processada | request_id=%s | status=%s | modelo_llm=%s | processing_time_ms=%.2f",
        request_id,
        status,
        modelo_llm,
        processing_time_ms,
    )

    if status == "sucesso":
        return ExplanationOutput(
            status=status,
            mensagem="Exame classificado pelo modelo e explicado pela LLM.",
            parametros=resultado["parametros"],
            predicao=resultado["retorno_api"],
            explicacao=resultado["explicacao"],
            modelo_llm=modelo_llm,
            processing_time_ms=round(processing_time_ms, 2),
            request_id=request_id,
        )

    return ExplanationOutput(
        status=status,
        mensagem=resultado["mensagem"],
        parametros=resultado.get("parametros_extraidos"),
        parametros_ausentes=resultado.get("parametros_ausentes", []),
        detalhe=resultado.get("detalhe"),
        modelo_llm=modelo_llm,
        processing_time_ms=round(processing_time_ms, 2),
        request_id=request_id,
    )
