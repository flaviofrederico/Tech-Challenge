"""Arquitetura da rede neural MLP (PyTorch) para classificação de churn.

Arquitetura: feed-forward simples (multilayer perceptron) com:
- Camadas lineares intercaladas com ReLU (ativação) e Dropout (regularização).
- Saída de um único logit (BCEWithLogitsLoss aplica a sigmoid internamente,
  o que é numericamente mais estável do que Sigmoid + BCELoss separados).
"""

from __future__ import annotations

import torch
from torch import nn

from churn_prediction.config import MLPConfig


class ChurnMLP(nn.Module):
    """MLP totalmente conectada para classificação binária de churn.

    Parameters
    ----------
    input_dim:
        Número de features de entrada (após o pré-processamento sklearn).
    config:
        Hiperparâmetros de arquitetura (tamanhos das camadas ocultas, dropout).
    """

    def __init__(self, input_dim: int, config: MLPConfig | None = None) -> None:
        super().__init__()
        config = config or MLPConfig()

        layers: list[nn.Module] = []
        in_features = input_dim
        for hidden_size in config.hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            in_features = hidden_size

        # Camada final: 1 logit (sem ativação -> usar BCEWithLogitsLoss)
        layers.append(nn.Linear(in_features, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna logits (não probabilidades) de shape (batch_size, 1)."""
        return self.network(x)
