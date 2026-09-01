"""Model training and evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from purchase_intention.config import ModelConfig
from purchase_intention.features.preprocessing import build_preprocessing_pipeline


@dataclass(frozen=True)
class TrainingResult:
    """Container for a trained pipeline and its evaluation metrics."""

    pipeline: Pipeline
    metrics: dict[str, float]


def build_model_pipeline(model_config: ModelConfig) -> Pipeline:
    """Assemble the full preprocessing + classifier pipeline.

    Args:
        model_config: Hyperparameters for the RandomForest classifier.

    Returns:
        An unfitted scikit-learn Pipeline combining preprocessing and model.
    """
    classifier = RandomForestClassifier(
        n_estimators=model_config.n_estimators,
        max_depth=model_config.max_depth,
        random_state=model_config.random_state,
        class_weight="balanced",
    )
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            ("classifier", classifier),
        ]
    )


def train_and_evaluate(
    features: pd.DataFrame,
    target: pd.Series,
    model_config: ModelConfig,
    test_size: float,
    random_state: int,
) -> TrainingResult:
    """Split data, train the pipeline and compute evaluation metrics.

    Args:
        features: Feature matrix.
        target: Binary target vector.
        model_config: Hyperparameters for the classifier.
        test_size: Fraction of data reserved for the held-out test set.
        random_state: Seed controlling the train/test split.

    Returns:
        A TrainingResult with the fitted pipeline and test-set metrics.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    pipeline = build_model_pipeline(model_config)
    pipeline.fit(x_train, y_train)
    metrics = _evaluate(pipeline, x_test, y_test)
    return TrainingResult(pipeline=pipeline, metrics=metrics)


def _evaluate(
    pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Compute standard binary classification metrics on the held-out test set."""
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }