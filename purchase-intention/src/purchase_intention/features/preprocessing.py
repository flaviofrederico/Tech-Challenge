"""Feature preprocessing pipeline for the purchase intention dataset."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Revenue"

NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]


def split_features_and_target(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a raw dataframe into feature matrix and binary target vector.

    Args:
        dataframe: Raw dataset containing both features and the target column.

    Returns:
        A tuple of (features, target) ready to feed into a model pipeline.
    """
    features = dataframe[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    target = dataframe[TARGET_COLUMN].astype(int)
    return features, target


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Create the column transformer used to preprocess raw features.

    Numeric features are standardized; categorical features (including the
    boolean Weekend flag and integer-coded categories) are one-hot encoded.

    Returns:
        A scikit-learn ColumnTransformer ready to be used inside a Pipeline.
    """
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )