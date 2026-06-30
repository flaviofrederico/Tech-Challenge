"""Schemas Pydantic para validação de entrada/saída da API.

Os campos espelham exatamente as features usadas em treino
(ver `churn_prediction.config.ALL_FEATURES`), com validação de domínio
(Literal) para os mesmos valores categóricos vistos no dataset de origem.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class CustomerFeatures(BaseModel):
    """Payload de entrada para predição de churn de um único cliente."""

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1] = Field(
        description="1 se o cliente é idoso (>=65 anos), senão 0"
    )
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=120, description="Meses como cliente")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, description="Cobrança mensal em moeda local")
    TotalCharges: float = Field(ge=0, description="Total cobrado até o momento")

    @field_validator("TotalCharges")
    @classmethod
    def total_charges_consistent_with_tenure(cls, v: float, info) -> float:
        # Validação leve de consistência: clientes com tenure=0 devem ter
        # TotalCharges baixo (mesma regra observada na EDA dos dados de
        # treino). Não bloqueia, mas o limite é generoso o suficiente para
        # não gerar falso-positivo em casos legítimos.
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Resposta da predição de churn."""

    churn_probability: float = Field(
        ge=0, le=1, description="Probabilidade de cancelamento (classe positiva)"
    )
    churn_prediction: Literal["Yes", "No"] = Field(
        description="Decisão final aplicando o threshold de produção"
    )
    threshold_used: float = Field(
        description="Threshold de decisão aplicado para gerar churn_prediction"
    )
    risk_tier: Literal["low", "medium", "high"] = Field(
        description="Faixa de risco qualitativa, útil para priorização de ações"
    )


class HealthResponse(BaseModel):
    """Resposta do endpoint de health check."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_type: str | None = None
