"""DVC pipeline stage: train the classifier and track/register it via MLflow."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from purchase_intention.config import PipelineConfig
from purchase_intention.features.preprocessing import split_features_and_target
from purchase_intention.models.train import train_and_evaluate

MODEL_REGISTRY_NAME = "purchase_intention_classifier"


def set_global_seed(seed: int) -> None:
    """Fix random seeds across libraries for reproducibility.

    Args:
        seed: Seed value applied to Python's random module and NumPy.
    """
    random.seed(seed)
    np.random.seed(seed)


def run_train(config_path: str) -> None:
    """Train the model, log the run to MLflow and register the trained model.

    Args:
        config_path: Path to the pipeline YAML configuration file.
    """
    config = PipelineConfig.from_yaml(config_path)
    set_global_seed(config.model.random_state)

    dataset_path = config.data.processed_path / "dataset.csv"
    dataframe = pd.read_csv(dataset_path)
    features, target = split_features_and_target(dataframe)

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run():
        result = train_and_evaluate(
            features=features,
            target=target,
            model_config=config.model,
            test_size=config.data.test_size,
            random_state=config.model.random_state,
        )

        mlflow.log_params(
            {
                "n_estimators": config.model.n_estimators,
                "max_depth": config.model.max_depth,
                "random_state": config.model.random_state,
                "test_size": config.data.test_size,
            }
        )
        mlflow.log_metrics(result.metrics)

        model_info = mlflow.sklearn.log_model(
            sk_model=result.pipeline,
            name="model",
            registered_model_name=MODEL_REGISTRY_NAME,
        )

        _write_metrics_file(result.metrics)
        print(f"Model registered as '{MODEL_REGISTRY_NAME}': {model_info.model_uri}")


def _write_metrics_file(metrics: dict[str, float]) -> None:
    """Persist metrics to a JSON file so DVC can track them as an output.

    Args:
        metrics: Mapping of metric name to value.
    """
    metrics_path = Path("metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for this pipeline stage."""
    parser = argparse.ArgumentParser(description="Train the purchase intention model.")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_train(args.config)