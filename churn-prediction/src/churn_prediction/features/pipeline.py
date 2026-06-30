"""Pipeline de pré-processamento reutilizável (sklearn).

Usado tanto pelos baselines (Scikit-Learn) quanto para gerar os tensores
de entrada do MLP (PyTorch), garantindo que treino e inferência (API)
aplicam exatamente a mesma transformação — esse é o ponto crítico para
evitar skew entre treino e produção.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_prediction.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


class YesNoEncoder(BaseEstimator, TransformerMixin):
    """Transformador customizado: mapeia colunas binárias Yes/No (e
    variantes como gender Male/Female) para 0/1 de forma explícita.

    Implementado como transformador custom (em vez de usar OrdinalEncoder
    genérico) para deixar a regra de negócio legível e testável:
    "Yes"/"Male" -> 1, "No"/"Female" -> 0.
    """

    _POSITIVE_VALUES = {"Yes", "Male", "1", 1}

    def fit(self, X: pd.DataFrame, y=None) -> YesNoEncoder:
        # Sem estado a aprender; mantido por compatibilidade com a API sklearn.
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else None
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X)
        out = X.apply(
            lambda col: col.apply(lambda v: 1 if v in self._POSITIVE_VALUES else 0)
        )
        return out.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Constrói o ColumnTransformer aplicado a todas as features de entrada.

    - Numéricas: imputação por mediana + padronização (StandardScaler).
    - Binárias (Yes/No, gender, SeniorCitizen): YesNoEncoder customizado.
    - Categóricas nominais (>2 categorias): One-Hot Encoding.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            ("encoder", YesNoEncoder()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("binary", binary_pipeline, BINARY_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor
