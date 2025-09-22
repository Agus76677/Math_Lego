"""Evaluation metrics for classification and unsupervised checks."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def classification_metrics(y_true, y_pred, prob=None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }
    if prob is not None and prob.shape[1] > 1:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, prob, multi_class="ovr"))
        except ValueError:
            metrics["auc"] = float("nan")
    else:
        metrics["auc"] = float("nan")
    return metrics


def unsupervised_alignment_score(before: float, after: float) -> float:
    if before <= 0:
        return 0.0
    return (before - after) / before
