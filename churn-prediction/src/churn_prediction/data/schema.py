"""Schema de validação dos dados, usando Pandera.

Garante que o dataset (bruto ou em produção via API) respeita o contrato
esperado pelo pipeline antes de seguir para treino/inferência. Isso é o que
o teste de "schema" (exigido pelo desafio) valida.
"""

from __future__ import annotations

from pandera.pandas import Check, Column, DataFrameSchema

# Schema para os dados BRUTOS (antes da limpeza em loader.clean_data).
# TotalCharges ainda é string nesse ponto (formato original do CSV).
RAW_SCHEMA = DataFrameSchema(
    {
        "customerID": Column(str, unique=True),
        "gender": Column(str, Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(int, Check.isin([0, 1])),
        "Partner": Column(str, Check.isin(["Yes", "No"])),
        "Dependents": Column(str, Check.isin(["Yes", "No"])),
        "tenure": Column(int, Check.in_range(0, 100)),
        "PhoneService": Column(str, Check.isin(["Yes", "No"])),
        "MultipleLines": Column(
            str, Check.isin(["Yes", "No", "No phone service"])
        ),
        "InternetService": Column(
            str, Check.isin(["DSL", "Fiber optic", "No"])
        ),
        "OnlineSecurity": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "OnlineBackup": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "DeviceProtection": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "TechSupport": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "StreamingTV": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "StreamingMovies": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "Contract": Column(
            str, Check.isin(["Month-to-month", "One year", "Two year"])
        ),
        "PaperlessBilling": Column(str, Check.isin(["Yes", "No"])),
        "PaymentMethod": Column(
            str,
            Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
        ),
        "MonthlyCharges": Column(float, Check.greater_than_or_equal_to(0)),
        # TotalCharges é string no CSV bruto (pode ter blanks) - validado
        # mais a fundo depois da limpeza.
        "TotalCharges": Column(str),
        "Churn": Column(str, Check.isin(["Yes", "No"])),
    },
    strict=False,
    coerce=False,
)


# Schema para o payload de entrada da API (uma única amostra, sem ID nem target)
API_INPUT_SCHEMA = DataFrameSchema(
    {
        "gender": Column(str, Check.isin(["Male", "Female"])),
        "SeniorCitizen": Column(int, Check.isin([0, 1])),
        "Partner": Column(str, Check.isin(["Yes", "No"])),
        "Dependents": Column(str, Check.isin(["Yes", "No"])),
        "tenure": Column(int, Check.in_range(0, 100)),
        "PhoneService": Column(str, Check.isin(["Yes", "No"])),
        "MultipleLines": Column(
            str, Check.isin(["Yes", "No", "No phone service"])
        ),
        "InternetService": Column(
            str, Check.isin(["DSL", "Fiber optic", "No"])
        ),
        "OnlineSecurity": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "OnlineBackup": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "DeviceProtection": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "TechSupport": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "StreamingTV": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "StreamingMovies": Column(
            str, Check.isin(["Yes", "No", "No internet service"])
        ),
        "Contract": Column(
            str, Check.isin(["Month-to-month", "One year", "Two year"])
        ),
        "PaperlessBilling": Column(str, Check.isin(["Yes", "No"])),
        "PaymentMethod": Column(
            str,
            Check.isin(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ]
            ),
        ),
        "MonthlyCharges": Column(float, Check.greater_than_or_equal_to(0)),
        "TotalCharges": Column(float, Check.greater_than_or_equal_to(0)),
    },
    strict=False,
    coerce=True,
)
