"""Persistência de artefatos do modelo treinado.

Salva e carrega o par (pipeline de pré-processamento, modelo) necessário
para a API servir predições de forma consistente com o treino.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import torch

from churn_prediction.config import MODELS_DIR, MLPConfig
from churn_prediction.models.mlp import ChurnMLP

PIPELINE_FILENAME = "preprocessing_pipeline.joblib"
MODEL_FILENAME = "churn_mlp.pt"
METADATA_FILENAME = "model_metadata.json"


def save_artifacts(
    pipeline: Any,
    model: ChurnMLP,
    mlp_config: MLPConfig,
    metadata: dict,
    output_dir: str | Path = MODELS_DIR,
) -> None:
    """Salva o pipeline sklearn, os pesos do MLP e metadados associados."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, output_dir / PIPELINE_FILENAME)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": model.network[0].in_features,
            "config": mlp_config.__dict__,
        },
        output_dir / MODEL_FILENAME,
    )

    with open(output_dir / METADATA_FILENAME, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)


def load_artifacts(
    input_dir: str | Path = MODELS_DIR,
) -> tuple[Any, ChurnMLP, dict]:
    """Carrega o pipeline sklearn, o modelo MLP e os metadados salvos."""
    input_dir = Path(input_dir)

    pipeline = joblib.load(input_dir / PIPELINE_FILENAME)

    checkpoint = torch.load(input_dir / MODEL_FILENAME, map_location="cpu")
    config = MLPConfig(**checkpoint["config"])
    model = ChurnMLP(input_dim=checkpoint["input_dim"], config=config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with open(input_dir / METADATA_FILENAME, encoding="utf-8") as f:
        metadata = json.load(f)

    return pipeline, model, metadata
