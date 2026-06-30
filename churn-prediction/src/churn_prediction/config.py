"""Configurações centrais do projeto.

Mantém em um único lugar: seeds, paths, nomes de colunas e
hiperparâmetros default, evitando "magic numbers" espalhados pelo código.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
RAW_DATA_PATH = DATA_RAW_DIR / "Telco-Customer-Churn.csv"

# --------------------------------------------------------------------------- #
# Reprodutibilidade
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Colunas do dataset
# --------------------------------------------------------------------------- #
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

BINARY_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

CATEGORICAL_FEATURES = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # fração do conjunto de treino reservada para validação (early stopping)

# --------------------------------------------------------------------------- #
# Hiperparâmetros default do MLP
# --------------------------------------------------------------------------- #


@dataclass
class MLPConfig:
    hidden_sizes: tuple[int, ...] = (64, 32)
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 15  # early stopping
    class_weight_pos: float | None = None  # calculado em runtime se None


@dataclass
class MLflowConfig:
    experiment_name: str = "churn-prediction"
    # Backend SQLite (recomendado pelo MLflow >= 3.x; o filestore puro em
    # disco está em modo de manutenção e não recebe mais atualizações).
    tracking_uri: str = field(
        default_factory=lambda: f"sqlite:///{ROOT_DIR / 'mlruns.db'}"
    )


MLP_CONFIG = MLPConfig()
MLFLOW_CONFIG = MLflowConfig()

# Threshold de decisão default (pode ser otimizado por custo de negócio)
DEFAULT_DECISION_THRESHOLD = 0.5

# --------------------------------------------------------------------------- #
# Custo de negócio (assunções documentadas no Model Card)
# --------------------------------------------------------------------------- #
# Custo de uma ação de retenção (desconto, contato proativo) por cliente
COST_RETENTION_ACTION = 30.0
# Valor médio perdido quando um cliente realmente cancela (estimado a partir
# de ~12 meses de MonthlyCharges médio, ver docs/model_card.md)
COST_CHURN_LOSS = 800.0
