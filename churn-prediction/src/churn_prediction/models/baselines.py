"""Modelos baseline (Scikit-Learn) usados para comparação com o MLP.

Inclui:
- DummyClassifier (estratégia 'most_frequent'): piso mínimo de performance.
- Regressão Logística: baseline linear interpretável.
- Random Forest: baseline não-linear baseado em árvores, para checar se a
  rede neural realmente agrega valor sobre um método mais simples e robusto.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn_prediction.config import RANDOM_SEED
from churn_prediction.features.pipeline import build_preprocessing_pipeline


def build_dummy_baseline() -> Pipeline:
    """Baseline trivial: sempre prevê a classe majoritária."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessing_pipeline()),
            (
                "classifier",
                DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED),
            ),
        ]
    )


def build_logistic_regression_baseline() -> Pipeline:
    """Baseline linear: Regressão Logística com balanceamento de classes."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessing_pipeline()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def build_random_forest_baseline() -> Pipeline:
    """Baseline de árvores: Random Forest com balanceamento de classes."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessing_pipeline()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )


BASELINE_BUILDERS = {
    "dummy_most_frequent": build_dummy_baseline,
    "logistic_regression": build_logistic_regression_baseline,
    "random_forest": build_random_forest_baseline,
}
