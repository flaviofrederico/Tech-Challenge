"""API de inferência do modelo de churn (FastAPI).

Endpoints:
- GET  /health   -> verifica se a API e o modelo estão operacionais.
- POST /predict  -> retorna a probabilidade de churn para um cliente.

Para rodar localmente:
    uvicorn churn_prediction.api.main:app --reload --port 8000

Documentação interativa (Swagger): http://localhost:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from churn_prediction.api.middleware import LatencyLoggingMiddleware
from churn_prediction.api.schemas import (
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
)
from churn_prediction.config import ALL_FEATURES, DEFAULT_DECISION_THRESHOLD
from churn_prediction.logging_config import get_logger
from churn_prediction.models.persistence import load_artifacts
from churn_prediction.models.training import predict_proba

logger = get_logger(__name__)

# Threshold de produção: usa o threshold ótimo de custo encontrado em
# treino, se disponível nos metadados; cai para o default caso contrário.
_model_state: dict = {
    "pipeline": None,
    "model": None,
    "metadata": None,
    "production_threshold": DEFAULT_DECISION_THRESHOLD,
}


def _risk_tier(probability: float) -> str:
    if probability < 0.3:
        return "low"
    if probability < 0.6:
        return "medium"
    return "high"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo e o pipeline uma única vez, na inicialização."""
    try:
        pipeline, model, metadata = load_artifacts()
        _model_state["pipeline"] = pipeline
        _model_state["model"] = model
        _model_state["metadata"] = metadata
        _model_state["production_threshold"] = metadata.get("test_metrics", {}).get(
            "optimal_threshold", DEFAULT_DECISION_THRESHOLD
        )
        logger.info(
            "model_loaded_successfully",
            extra={
                "model_type": metadata.get("model_type"),
                "production_threshold": _model_state["production_threshold"],
            },
        )
    except FileNotFoundError as exc:
        logger.error("model_artifacts_not_found", extra={"error": str(exc)})
        # A API ainda sobe, mas /health reportará "degraded" e /predict
        # retornará 503 — preferível a falhar o processo inteiro no boot,
        # o que facilita diagnosticar o problema em produção.
    yield
    _model_state.clear()


app = FastAPI(
    title="Churn Prediction API",
    description="API de inferência para o modelo MLP de previsão de churn.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(LatencyLoggingMiddleware)


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health() -> HealthResponse:
    """Verifica se a API está operacional e se o modelo foi carregado."""
    model_loaded = _model_state["model"] is not None
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_type=(_model_state["metadata"] or {}).get("model_type")
        if model_loaded
        else None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Prediz a probabilidade de churn para um cliente.

    O threshold de decisão usado é o threshold ótimo de custo de negócio
    encontrado durante o treino (ver Model Card), não necessariamente 0.5.
    """
    if _model_state["model"] is None or _model_state["pipeline"] is None:
        logger.error("predict_called_without_model_loaded")
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Verifique se os artefatos existem em models/.",
        )

    try:
        input_df = pd.DataFrame([customer.model_dump()])[ALL_FEATURES]
        X_transformed = _model_state["pipeline"].transform(input_df)
        proba = float(
            predict_proba(_model_state["model"], np.asarray(X_transformed))[0]
        )
    except Exception as exc:  # noqa: BLE001 - log e converte em erro HTTP 500
        logger.error("predict_failed", extra={"error": str(exc)})
        raise HTTPException(
            status_code=500, detail="Erro interno ao gerar a predição."
        ) from exc

    threshold = _model_state["production_threshold"]
    prediction = "Yes" if proba >= threshold else "No"

    logger.info(
        "prediction_made",
        extra={
            "churn_probability": round(proba, 4),
            "churn_prediction": prediction,
            "threshold_used": threshold,
        },
    )

    return PredictionResponse(
        churn_probability=proba,
        churn_prediction=prediction,
        threshold_used=threshold,
        risk_tier=_risk_tier(proba),
    )
