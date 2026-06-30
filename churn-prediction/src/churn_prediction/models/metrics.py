"""Métricas de avaliação compartilhadas entre baselines e MLP.

Centraliza o cálculo das métricas técnicas (AUC-ROC, PR-AUC, F1, etc.) e
da métrica de negócio (custo esperado), garantindo que todos os modelos
comparados na Etapa 2 usem exatamente as mesmas fórmulas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn_prediction.config import COST_CHURN_LOSS, COST_RETENTION_ACTION


@dataclass
class EvaluationResult:
    """Conjunto padronizado de métricas para um modelo binário de churn."""

    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    business_cost: float
    n_false_positives: int
    n_false_negatives: int
    n_true_positives: int
    n_true_negatives: int
    threshold: float

    def to_dict(self) -> dict[str, float]:
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "business_cost": self.business_cost,
            "n_false_positives": self.n_false_positives,
            "n_false_negatives": self.n_false_negatives,
            "n_true_positives": self.n_true_positives,
            "n_true_negatives": self.n_true_negatives,
            "threshold": self.threshold,
        }


def compute_business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fp: float = COST_RETENTION_ACTION,
    cost_fn: float = COST_CHURN_LOSS,
) -> float:
    """Custo esperado de negócio dado um conjunto de predições.

    - Falso Positivo (FP): a empresa age numa retenção desnecessária ->
      custo de retenção (`cost_fp`), tipicamente baixo (desconto, ligação).
    - Falso Negativo (FN): a empresa não age e o cliente cancela de fato ->
      custo de perda do cliente (`cost_fn`), tipicamente alto.
    - Verdadeiro Positivo (TP) e Verdadeiro Negativo (TN): sem custo extra
      modelado aqui (a ação de retenção em TP é considerada custo-efetiva,
      pois evita a perda; ver Model Card para discussão completa).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp * cost_fp + fn * cost_fn)


def evaluate_predictions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> EvaluationResult:
    """Calcula o conjunto completo de métricas a partir das probabilidades.

    Usa >= 4 métricas técnicas (ROC-AUC, PR-AUC, F1, Precision, Recall,
    Accuracy) e a métrica de custo de negócio, conforme exigido.
    """
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return EvaluationResult(
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        business_cost=compute_business_cost(y_true, y_pred),
        n_false_positives=int(fp),
        n_false_negatives=int(fn),
        n_true_positives=int(tp),
        n_true_negatives=int(tn),
        threshold=threshold,
    )


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fp: float = COST_RETENTION_ACTION,
    cost_fn: float = COST_CHURN_LOSS,
) -> tuple[float, float]:
    """Busca o threshold de decisão que minimiza o custo de negócio.

    Faz uma busca em grade simples (granularidade 0.01) sobre as
    probabilidades preditas. Retorna (melhor_threshold, custo_mínimo).
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = [
        compute_business_cost(
            y_true, (y_proba >= t).astype(int), cost_fp=cost_fp, cost_fn=cost_fn
        )
        for t in thresholds
    ]
    best_idx = int(np.argmin(costs))
    return float(thresholds[best_idx]), float(costs[best_idx])
