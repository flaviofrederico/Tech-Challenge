"""Configuration management for the purchase intention pipeline.

Configuration is expressed as immutable dataclasses loaded from a single
YAML file, keeping all tunable parameters (paths, hyperparameters, MLflow
settings) outside of the source code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    """Paths and split parameters for dataset handling."""

    raw_path: Path
    processed_path: Path
    test_size: float
    random_state: int


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the classification model."""

    name: str
    random_state: int
    n_estimators: int
    max_depth: int


@dataclass(frozen=True)
class MLflowConfig:
    """MLflow tracking server and experiment settings."""

    experiment_name: str
    tracking_uri: str


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration for the full ML pipeline."""

    data: DataConfig
    model: ModelConfig
    mlflow: MLflowConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load and validate pipeline configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A fully populated, immutable PipelineConfig instance.
        """
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            data=DataConfig(
                raw_path=Path(raw["data"]["raw_path"]),
                processed_path=Path(raw["data"]["processed_path"]),
                test_size=float(raw["data"]["test_size"]),
                random_state=int(raw["data"]["random_state"]),
            ),
            model=ModelConfig(**raw["model"]),
            mlflow=MLflowConfig(**raw["mlflow"]),
        )