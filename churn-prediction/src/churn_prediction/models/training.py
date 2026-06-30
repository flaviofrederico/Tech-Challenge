"""Loop de treinamento da MLP com early stopping e mini-batching.

Responsabilidades:
- Treinar em mini-batches via DataLoader (batching).
- Monitorar a loss de validação e interromper o treino quando ela parar
  de melhorar por `patience` épocas consecutivas (early stopping), restaurando
  os pesos da melhor época encontrada.
- Lidar com desbalanceamento de classes via peso positivo na loss
  (pos_weight do BCEWithLogitsLoss), em vez de oversampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from churn_prediction.config import RANDOM_SEED, MLPConfig
from churn_prediction.logging_config import get_logger
from churn_prediction.models.mlp import ChurnMLP

logger = get_logger(__name__)


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False
    n_epochs_run: int = 0


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fixa todas as seeds relevantes para reprodutibilidade."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: MLPConfig | None = None,
    device: str = "cpu",
) -> tuple[ChurnMLP, TrainingHistory]:
    """Treina a MLP com early stopping baseado na loss de validação.

    Retorna o modelo com os pesos da melhor época (menor val_loss) e o
    histórico de treino completo.
    """
    config = config or MLPConfig()
    set_seed(RANDOM_SEED)

    model = ChurnMLP(input_dim=X_train.shape[1], config=config).to(device)

    train_loader = _make_loader(X_train, y_train, config.batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_val, config.batch_size, shuffle=False)

    # Lidar com desbalanceamento: peso maior para a classe positiva (churn).
    if config.class_weight_pos is None:
        n_pos = max(int(y_train.sum()), 1)
        n_neg = len(y_train) - n_pos
        pos_weight_value = n_neg / n_pos
    else:
        pos_weight_value = config.class_weight_pos

    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    history = TrainingHistory()
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    logger.info(
        "training_started",
        extra={
            "max_epochs": config.max_epochs,
            "patience": config.patience,
            "batch_size": config.batch_size,
            "pos_weight": pos_weight_value,
        },
    )

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_running_loss += loss.item() * X_batch.size(0)
        val_loss = val_running_loss / len(val_loader.dataset)

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.n_epochs_run = epoch

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            history.best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "epoch_completed",
                extra={
                    "epoch": epoch,
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                },
            )

        if epochs_without_improvement >= config.patience:
            history.stopped_early = True
            logger.info(
                "early_stopping_triggered",
                extra={"epoch": epoch, "best_epoch": history.best_epoch},
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    logger.info(
        "training_finished",
        extra={
            "n_epochs_run": history.n_epochs_run,
            "best_epoch": history.best_epoch,
            "best_val_loss": round(best_val_loss, 4),
            "stopped_early": history.stopped_early,
        },
    )

    return model, history


def predict_proba(model: ChurnMLP, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Retorna probabilidades (sigmoid dos logits) para um array numpy."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(X_t)
        proba = torch.sigmoid(logits).cpu().numpy().ravel()
    return proba
