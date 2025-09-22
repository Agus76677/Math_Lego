"""Simple transfer learning strategies (CORAL, MMD, pseudo labels)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_kernels

from .feature_select import compute_coral, compute_mmd
from .models_source import train_model


@dataclass
class TransferResult:
    method: str
    predictions: np.ndarray
    probabilities: np.ndarray
    weights: np.ndarray


def coral_align(Xs: np.ndarray, Xt: np.ndarray) -> np.ndarray:
    cov_s = np.cov(Xs, rowvar=False) + np.eye(Xs.shape[1]) * 1e-3
    cov_t = np.cov(Xt, rowvar=False) + np.eye(Xt.shape[1]) * 1e-3
    U_s, S_s, _ = np.linalg.svd(cov_s)
    U_t, S_t, _ = np.linalg.svd(cov_t)
    whiten = U_s @ np.diag(1.0 / np.sqrt(S_s)) @ U_s.T
    color = U_t @ np.diag(np.sqrt(S_t)) @ U_t.T
    return (Xs - Xs.mean(axis=0)) @ whiten @ color + Xt.mean(axis=0)


def mmd_reweight(Xs: np.ndarray, Xt: np.ndarray, gamma: float = None) -> np.ndarray:
    if gamma is None:
        gamma = 1.0 / (Xs.shape[1] * np.var(np.vstack([Xs, Xt])) + 1e-12)
    Ks = pairwise_kernels(Xs, metric="rbf", gamma=gamma)
    Kt = pairwise_kernels(Xs, Xt, metric="rbf", gamma=gamma)
    weights = np.linalg.solve(Ks + np.eye(len(Xs)) * 1e-6, Kt.mean(axis=1))
    weights = np.maximum(weights, 0)
    weights /= weights.max() + 1e-12
    return weights


def pseudo_label(logits: np.ndarray, thresholds: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    pseudo_labels = -np.ones(len(logits), dtype=int)
    confidences = logits.max(axis=1)
    for thr in thresholds:
        mask = confidences >= thr
        pseudo_labels[mask] = logits[mask].argmax(axis=1)
    return pseudo_labels, confidences


def transfer_predict(
    base_model: str,
    Xs: np.ndarray,
    ys: np.ndarray,
    Xt: np.ndarray,
    align: str = "CORAL",
    sample_weight: np.ndarray | None = None,
    pseudo: bool = False,
    thresholds: Tuple[float, float] = (0.95, 0.9),
) -> TransferResult:
    Xs_aligned = Xs.copy()
    if align.upper() == "CORAL":
        Xs_aligned = coral_align(Xs, Xt)
    elif align.upper() == "MMD":
        weights = mmd_reweight(Xs, Xt)
        if sample_weight is None:
            sample_weight = weights
        else:
            sample_weight = sample_weight * weights
    model = train_model(base_model, Xs_aligned, ys, sample_weight=sample_weight)
    prob = model.predict_proba(Xt)
    preds = prob.argmax(axis=1)
    weights = np.ones(len(ys)) if sample_weight is None else sample_weight
    if pseudo:
        pseudo_y, conf = pseudo_label(prob, thresholds)
        mask = pseudo_y >= 0
        if mask.any():
            X_aug = np.vstack([Xs_aligned, Xt[mask]])
            y_aug = np.concatenate([ys, pseudo_y[mask]])
            w_aug = np.concatenate([weights, np.full(mask.sum(), conf[mask].mean())])
            model = train_model(base_model, X_aug, y_aug, sample_weight=w_aug)
            prob = model.predict_proba(Xt)
            preds = prob.argmax(axis=1)
    return TransferResult(method=align, predictions=preds, probabilities=prob, weights=weights)
