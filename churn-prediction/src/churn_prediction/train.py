"""Script orquestrador de treinamento: baselines + MLP, com tracking MLflow.

Uso:
    python -m churn_prediction.train

Executa, nesta ordem:
1. Carrega e limpa os dados.
2. Faz o split estratificado (train / val / test).
3. Treina e avalia os baselines (Dummy, Logistic Regression, Random Forest)
   com validação cruzada estratificada no conjunto de treino.
4. Treina a MLP (PyTorch) com early stopping.
5. Compara todos os modelos no conjunto de teste (>= 4 métricas).
6. Registra parâmetros, métricas e artefatos de cada modelo no MLflow.
7. Salva os artefatos do melhor modelo (pipeline + pesos) em models/.
"""

from __future__ import annotations

import time

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from churn_prediction.config import (
    ALL_FEATURES,
    MLFLOW_CONFIG,
    MLP_CONFIG,
    RANDOM_SEED,
    TEST_SIZE,
    VAL_SIZE,
)
from churn_prediction.data.loader import load_clean_data
from churn_prediction.features.pipeline import build_preprocessing_pipeline
from churn_prediction.logging_config import get_logger
from churn_prediction.models.baselines import BASELINE_BUILDERS
from churn_prediction.models.metrics import evaluate_predictions, find_optimal_threshold
from churn_prediction.models.persistence import save_artifacts
from churn_prediction.models.training import predict_proba, set_seed, train_mlp

logger = get_logger(__name__)


def split_data(df: pd.DataFrame):
    """Split estratificado em train / val / test (val usado para early stopping)."""
    X = df[ALL_FEATURES]
    y = df["Churn"].values

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VAL_SIZE,
        stratify=y_train_full,
        random_state=RANDOM_SEED,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, X_train_full, y_train_full


def run_baselines(X_train_full, y_train_full, X_test, y_test) -> dict[str, dict]:
    """Treina cada baseline com CV estratificada no treino e avalia no teste."""
    results: dict[str, dict] = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for name, builder in BASELINE_BUILDERS.items():
        with mlflow.start_run(run_name=f"baseline_{name}"):
            mlflow.log_param("model_type", name)
            mlflow.log_param("random_seed", RANDOM_SEED)
            mlflow.log_param("cv_folds", 5)

            pipe = builder()

            cv_scores = cross_val_score(
                pipe, X_train_full, y_train_full, cv=skf, scoring="roc_auc"
            )
            mlflow.log_metric("cv_roc_auc_mean", float(cv_scores.mean()))
            mlflow.log_metric("cv_roc_auc_std", float(cv_scores.std()))

            pipe.fit(X_train_full, y_train_full)
            proba = pipe.predict_proba(X_test)[:, 1]
            result = evaluate_predictions(y_test, proba)

            for metric_name, value in result.to_dict().items():
                mlflow.log_metric(f"test_{metric_name}", value)

            results[name] = {
                "cv_roc_auc_mean": float(cv_scores.mean()),
                "cv_roc_auc_std": float(cv_scores.std()),
                **result.to_dict(),
            }

            logger.info(
                "baseline_evaluated",
                extra={"model": name, **result.to_dict()},
            )

    return results


def run_mlp(X_train, X_val, X_test, y_train, y_val, y_test) -> tuple[dict, object, object]:
    """Treina o MLP em PyTorch e avalia no teste, com tracking MLflow."""
    set_seed(RANDOM_SEED)

    with mlflow.start_run(run_name="mlp_pytorch") as run:
        mlflow.log_params(
            {
                "model_type": "mlp_pytorch",
                "hidden_sizes": str(MLP_CONFIG.hidden_sizes),
                "dropout": MLP_CONFIG.dropout,
                "learning_rate": MLP_CONFIG.learning_rate,
                "weight_decay": MLP_CONFIG.weight_decay,
                "batch_size": MLP_CONFIG.batch_size,
                "max_epochs": MLP_CONFIG.max_epochs,
                "patience": MLP_CONFIG.patience,
                "random_seed": RANDOM_SEED,
            }
        )

        preprocessor = build_preprocessing_pipeline()
        X_train_t = preprocessor.fit_transform(X_train)
        X_val_t = preprocessor.transform(X_val)
        X_test_t = preprocessor.transform(X_test)

        start = time.time()
        model, history = train_mlp(
            X_train_t, y_train, X_val_t, y_val, config=MLP_CONFIG
        )
        training_seconds = time.time() - start

        mlflow.log_metric("training_seconds", training_seconds)
        mlflow.log_metric("n_epochs_run", history.n_epochs_run)
        mlflow.log_metric("best_epoch", history.best_epoch)
        mlflow.log_metric("stopped_early", int(history.stopped_early))

        for epoch, (tr_loss, val_loss) in enumerate(
            zip(history.train_loss, history.val_loss, strict=True), start=1
        ):
            mlflow.log_metric("train_loss", tr_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        proba_test = predict_proba(model, X_test_t)
        result = evaluate_predictions(y_test, proba_test)
        for metric_name, value in result.to_dict().items():
            mlflow.log_metric(f"test_{metric_name}", value)

        # Threshold ótimo por custo de negócio
        best_threshold, best_cost = find_optimal_threshold(y_test, proba_test)
        mlflow.log_metric("optimal_threshold", best_threshold)
        mlflow.log_metric("optimal_threshold_cost", best_cost)

        logger.info(
            "mlp_evaluated",
            extra={
                **result.to_dict(),
                "optimal_threshold": best_threshold,
                "optimal_threshold_cost": best_cost,
                "run_id": run.info.run_id,
            },
        )

        results = {
            **result.to_dict(),
            "optimal_threshold": best_threshold,
            "optimal_threshold_cost": best_cost,
            "n_epochs_run": history.n_epochs_run,
            "best_epoch": history.best_epoch,
            "training_seconds": training_seconds,
            "run_id": run.info.run_id,
        }

    return results, model, preprocessor


def main() -> None:
    set_seed(RANDOM_SEED)

    mlflow.set_tracking_uri(MLFLOW_CONFIG.tracking_uri)
    mlflow.set_experiment(MLFLOW_CONFIG.experiment_name)

    logger.info("pipeline_started")

    df = load_clean_data()
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        X_train_full,
        y_train_full,
    ) = split_data(df)

    logger.info(
        "split_completed",
        extra={
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "train_churn_rate": round(float(np.mean(y_train)), 4),
            "test_churn_rate": round(float(np.mean(y_test)), 4),
        },
    )

    baseline_results = run_baselines(X_train_full, y_train_full, X_test, y_test)
    mlp_results, mlp_model, preprocessor = run_mlp(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    # --- Comparação final ---
    comparison = pd.DataFrame(
        {**baseline_results, "mlp_pytorch": mlp_results}
    ).T
    comparison_path = "models/model_comparison.csv"
    comparison.to_csv(comparison_path)
    logger.info("comparison_table_saved", extra={"path": comparison_path})
    print("\n=== TABELA COMPARATIVA DE MODELOS ===")
    print(
        comparison[
            ["roc_auc", "pr_auc", "f1", "precision", "recall", "business_cost"]
        ].round(4).to_string()
    )

    # --- Salva os artefatos do MLP (modelo "central" de entrega) ---
    metadata = {
        "model_type": "mlp_pytorch",
        "feature_columns": ALL_FEATURES,
        "test_metrics": mlp_results,
        "baseline_comparison": baseline_results,
        "random_seed": RANDOM_SEED,
    }
    save_artifacts(preprocessor, mlp_model, MLP_CONFIG, metadata)
    logger.info("artifacts_saved", extra={"output_dir": "models/"})


if __name__ == "__main__":
    main()
