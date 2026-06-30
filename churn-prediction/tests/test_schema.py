"""Testes de schema: garantem que o contrato de dados é respeitado.

Cobre tanto o dataset bruto (entrada do pipeline de treino) quanto a
limpeza aplicada por `clean_data`.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError

from churn_prediction.data.loader import clean_data
from churn_prediction.data.schema import RAW_SCHEMA


def test_raw_data_matches_schema(raw_df: pd.DataFrame) -> None:
    """O CSV bruto deve respeitar o schema esperado (tipos e domínios)."""
    validated = RAW_SCHEMA.validate(raw_df)
    assert len(validated) == len(raw_df)


def test_raw_schema_rejects_invalid_gender(raw_df: pd.DataFrame) -> None:
    """O schema deve rejeitar valores fora do domínio esperado."""
    corrupted = raw_df.copy()
    corrupted.loc[0, "gender"] = "Other"  # valor fora do domínio Male/Female

    with pytest.raises(SchemaError):
        RAW_SCHEMA.validate(corrupted)


def test_clean_data_has_no_missing_values(raw_df: pd.DataFrame) -> None:
    """Após a limpeza, não deve haver valores nulos em nenhuma coluna."""
    cleaned = clean_data(raw_df)
    assert cleaned.isnull().sum().sum() == 0


def test_clean_data_total_charges_is_numeric(raw_df: pd.DataFrame) -> None:
    """TotalCharges deve ser convertido de string para float na limpeza."""
    cleaned = clean_data(raw_df)
    assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])


def test_clean_data_churn_is_binary(raw_df: pd.DataFrame) -> None:
    """A coluna Churn derivada deve conter apenas 0 e 1."""
    cleaned = clean_data(raw_df)
    assert set(cleaned["Churn"].unique()) <= {0, 1}


def test_clean_data_preserves_row_count(raw_df: pd.DataFrame) -> None:
    """A limpeza não deve descartar clientes legítimos (sem duplicados)."""
    cleaned = clean_data(raw_df)
    assert len(cleaned) == len(raw_df)
