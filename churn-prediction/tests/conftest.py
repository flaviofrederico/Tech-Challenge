"""Fixtures compartilhadas entre os módulos de teste."""

from __future__ import annotations

import pandas as pd
import pytest

from churn_prediction.config import RAW_DATA_PATH


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Carrega o CSV bruto uma única vez por sessão de testes."""
    return pd.read_csv(RAW_DATA_PATH)


@pytest.fixture
def valid_customer_payload() -> dict:
    """Payload válido de cliente para testes da API, espelhando um caso real."""
    return {
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
