"""Data loading utilities for the raw Online Shoppers Purchasing Intention dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET_COLUMN = "Revenue"


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """Load the raw dataset from disk and validate its basic schema.

    Args:
        path: Path to the raw CSV file.

    Returns:
        The loaded dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If the required target column is missing.
    """
    dataframe = pd.read_csv(path)
    _validate_schema(dataframe)
    return dataframe


def _validate_schema(dataframe: pd.DataFrame) -> None:
    """Ensure the dataset contains the expected target column."""
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Dataset is missing required target column '{TARGET_COLUMN}'."
        )