"""Testes da API FastAPI: health check, predição válida e validação Pydantic."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churn_prediction.api.main import app


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint_returns_ok_when_model_loaded(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_type"] == "mlp_pytorch"


def test_predict_returns_valid_probability(
    client: TestClient, valid_customer_payload: dict
) -> None:
    response = client.post("/predict", json=valid_customer_payload)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in {"Yes", "No"}
    assert body["risk_tier"] in {"low", "medium", "high"}


def test_predict_rejects_invalid_categorical_value(
    client: TestClient, valid_customer_payload: dict
) -> None:
    """Um valor fora do domínio esperado deve ser rejeitado com 422."""
    invalid_payload = dict(valid_customer_payload)
    invalid_payload["gender"] = "Unknown"

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_predict_rejects_missing_required_field(
    client: TestClient, valid_customer_payload: dict
) -> None:
    """Um payload incompleto deve ser rejeitado com 422."""
    incomplete_payload = dict(valid_customer_payload)
    del incomplete_payload["tenure"]

    response = client.post("/predict", json=incomplete_payload)

    assert response.status_code == 422


def test_predict_rejects_negative_monthly_charges(
    client: TestClient, valid_customer_payload: dict
) -> None:
    """MonthlyCharges negativo não faz sentido de negócio e deve ser rejeitado."""
    invalid_payload = dict(valid_customer_payload)
    invalid_payload["MonthlyCharges"] = -10.0

    response = client.post("/predict", json=invalid_payload)

    assert response.status_code == 422


def test_predict_high_risk_profile_scores_above_low_risk_profile(
    client: TestClient, valid_customer_payload: dict
) -> None:
    """Validação de sanidade do modelo: um perfil classicamente de alto risco
    (contrato mensal, cliente novo) deve pontuar mais alto que um perfil de
    baixo risco (contrato de 2 anos, cliente antigo) — sem assumir nenhum
    valor numérico exato, apenas a direção esperada do efeito.
    """
    high_risk = dict(valid_customer_payload)
    high_risk.update({"Contract": "Month-to-month", "tenure": 1})

    low_risk = dict(valid_customer_payload)
    low_risk.update(
        {
            "Contract": "Two year",
            "tenure": 60,
            "TotalCharges": 60 * valid_customer_payload["MonthlyCharges"],
        }
    )

    high_risk_proba = client.post("/predict", json=high_risk).json()[
        "churn_probability"
    ]
    low_risk_proba = client.post("/predict", json=low_risk).json()[
        "churn_probability"
    ]

    assert high_risk_proba > low_risk_proba
