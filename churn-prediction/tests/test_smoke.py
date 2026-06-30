"""Smoke tests: garantem que os componentes centrais executam sem erro
e produzem saídas com a forma e o tipo esperados.

Não validam qualidade do modelo (isso é coberto por thresholds de negócio
documentados no Model Card) — apenas que o "encanamento" funciona.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn_prediction.config import ALL_FEATURES, MLPConfig
from churn_prediction.data.loader import clean_data
from churn_prediction.features.pipeline import build_preprocessing_pipeline
from churn_prediction.models.baselines import build_logistic_regression_baseline
from churn_prediction.models.metrics import evaluate_predictions, find_optimal_threshold
from churn_prediction.models.mlp import ChurnMLP
from churn_prediction.models.training import predict_proba, set_seed, train_mlp


def test_preprocessing_pipeline_smoke(raw_df: pd.DataFrame) -> None:
    """O pipeline de pré-processamento deve transformar os dados sem NaNs."""
    df = clean_data(raw_df)
    X = df[ALL_FEATURES]

    pipeline = build_preprocessing_pipeline()
    X_transformed = pipeline.fit_transform(X)

    assert X_transformed.shape[0] == len(df)
    assert not np.isnan(X_transformed).any()


def test_baseline_pipeline_smoke(raw_df: pd.DataFrame) -> None:
    """Um baseline sklearn deve treinar e prever sem erros."""
    df = clean_data(raw_df).sample(n=200, random_state=42)
    X = df[ALL_FEATURES]
    y = df["Churn"].values

    pipe = build_logistic_regression_baseline()
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]

    assert proba.shape == (len(df),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_mlp_training_smoke(raw_df: pd.DataFrame) -> None:
    """O loop de treino da MLP deve rodar algumas épocas e gerar predições válidas."""
    set_seed(42)
    df = clean_data(raw_df).sample(n=300, random_state=42)
    X = df[ALL_FEATURES]
    y = df["Churn"].values

    pipeline = build_preprocessing_pipeline()
    X_transformed = pipeline.fit_transform(X)

    split = int(len(X_transformed) * 0.8)
    X_train, X_val = X_transformed[:split], X_transformed[split:]
    y_train, y_val = y[:split], y[split:]

    config = MLPConfig(max_epochs=3, patience=10)
    model, history = train_mlp(X_train, y_train, X_val, y_val, config=config)

    assert isinstance(model, ChurnMLP)
    assert history.n_epochs_run == 3  # sem early stopping em só 3 épocas
    assert len(history.train_loss) == 3
    assert len(history.val_loss) == 3

    proba = predict_proba(model, X_val)
    assert proba.shape == (len(X_val),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_mlp_reproducibility_with_fixed_seed(raw_df: pd.DataFrame) -> None:
    """Mesma seed deve produzir os mesmos pesos iniciais e, no limite, a
    mesma trajetória de treino determinística (CPU, sem paralelismo
    não-determinístico)."""
    df = clean_data(raw_df).sample(n=200, random_state=42)
    X = df[ALL_FEATURES]
    y = df["Churn"].values

    pipeline = build_preprocessing_pipeline()
    X_transformed = pipeline.fit_transform(X)
    split = int(len(X_transformed) * 0.8)
    X_train, X_val = X_transformed[:split], X_transformed[split:]
    y_train, y_val = y[:split], y[split:]

    config = MLPConfig(max_epochs=2, patience=10)

    set_seed(123)
    model_a, _ = train_mlp(X_train, y_train, X_val, y_val, config=config)
    proba_a = predict_proba(model_a, X_val)

    set_seed(123)
    model_b, _ = train_mlp(X_train, y_train, X_val, y_val, config=config)
    proba_b = predict_proba(model_b, X_val)

    np.testing.assert_allclose(proba_a, proba_b, rtol=1e-5)


def test_persistence_round_trip_preserves_predictions(raw_df: pd.DataFrame, tmp_path) -> None:
    """Salvar e recarregar o pipeline+modelo deve preservar as predições
    exatamente — essencial para garantir que a API serve o mesmo modelo
    que foi treinado e avaliado offline."""
    from churn_prediction.models.persistence import load_artifacts, save_artifacts

    set_seed(42)
    df = clean_data(raw_df).sample(n=200, random_state=42)
    X = df[ALL_FEATURES]
    y = df["Churn"].values

    pipeline = build_preprocessing_pipeline()
    X_transformed = pipeline.fit_transform(X)
    split = int(len(X_transformed) * 0.8)
    X_train, X_val = X_transformed[:split], X_transformed[split:]
    y_train, y_val = y[:split], y[split:]

    config = MLPConfig(max_epochs=3, patience=10)
    model, _ = train_mlp(X_train, y_train, X_val, y_val, config=config)
    proba_before = predict_proba(model, X_val)

    save_artifacts(
        pipeline, model, config, metadata={"model_type": "mlp_pytorch"},
        output_dir=tmp_path,
    )
    loaded_pipeline, loaded_model, metadata = load_artifacts(input_dir=tmp_path)
    proba_after = predict_proba(loaded_model, X_val)

    np.testing.assert_allclose(proba_before, proba_after, rtol=1e-6)
    assert metadata["model_type"] == "mlp_pytorch"


def test_evaluate_predictions_returns_all_required_metrics() -> None:
    """O relatório de avaliação deve conter >= 4 métricas técnicas distintas."""
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    y_proba = np.array([0.1, 0.8, 0.6, 0.3, 0.9, 0.4, 0.2, 0.7])

    result = evaluate_predictions(y_true, y_proba)
    metrics = result.to_dict()

    required = {"roc_auc", "pr_auc", "f1", "precision", "recall"}
    assert required <= set(metrics.keys())
    for key in required:
        assert 0.0 <= metrics[key] <= 1.0


def test_find_optimal_threshold_reduces_or_maintains_cost() -> None:
    """O threshold ótimo nunca deve ter custo pior do que o threshold 0.5."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=500)
    y_proba = np.clip(y_true * 0.6 + rng.normal(0, 0.3, size=500), 0, 1)

    from churn_prediction.models.metrics import compute_business_cost

    cost_at_default = compute_business_cost(y_true, (y_proba >= 0.5).astype(int))
    best_threshold, best_cost = find_optimal_threshold(y_true, y_proba)

    assert best_cost <= cost_at_default
    assert 0.0 < best_threshold < 1.0
