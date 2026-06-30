"""Carregamento e limpeza do dataset Telco Customer Churn.

Conhecimentos de domínio aplicados aqui (documentados via EDA, ver
notebooks/01_eda.ipynb):

1. `TotalCharges` vem como string e contém 11 valores em branco,
   todos correspondentes a clientes com `tenure == 0` (clientes novos
   que ainda não foram cobrados). Tratamos como 0.0, não como ausência
   de informação.
2. `SeniorCitizen` vem como int (0/1) mas é semanticamente uma flag binária.
3. `customerID` é apenas identificador, não é uma feature preditiva.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn_prediction.config import ID_COLUMN, RAW_DATA_PATH, TARGET_COLUMN
from churn_prediction.logging_config import get_logger

logger = get_logger(__name__)


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Lê o CSV bruto sem nenhuma transformação."""
    logger.info("loading_raw_data", extra={"path": str(path)})
    df = pd.read_csv(path)
    logger.info(
        "raw_data_loaded",
        extra={"n_rows": len(df), "n_cols": df.shape[1]},
    )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpeza determinística e documentada ao dataframe bruto.

    Não faz nenhuma transformação que dependa do split (isso é feito no
    pipeline do sklearn, em `features/pipeline.py`, para evitar data leakage).
    """
    df = df.copy()

    n_before = len(df)

    # TotalCharges: string -> float, com blanks tratados como 0.0
    # (validado na EDA: todos os blanks têm tenure == 0)
    df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()
    n_blank = (df["TotalCharges"] == "").sum()
    if n_blank:
        logger.info(
            "filling_blank_total_charges",
            extra={"n_blank": int(n_blank)},
        )
    df["TotalCharges"] = df["TotalCharges"].replace("", "0.0").astype(float)

    # Normaliza target para binário 0/1 explícito (mantém Churn p/ leitura humana)
    df["Churn"] = df[TARGET_COLUMN].map({"No": 0, "Yes": 1}).astype(int)

    # Remove duplicados exatos de customerID, se existirem (defensivo)
    n_dup = df[ID_COLUMN].duplicated().sum()
    if n_dup:
        logger.info("dropping_duplicate_customers", extra={"n_dup": int(n_dup)})
        df = df.drop_duplicates(subset=ID_COLUMN, keep="first")

    logger.info(
        "data_cleaned",
        extra={"n_rows_before": n_before, "n_rows_after": len(df)},
    )
    return df


def load_clean_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Atalho: carrega e já limpa os dados."""
    return clean_data(load_raw_data(path))
