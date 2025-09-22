"""Domain discrepancy quantification and source subset selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@dataclass
class DomainStats:
    mmd: float
    coral: float
    domain_acc: float


def _rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    x_norm = np.sum(x**2, axis=1)[:, None]
    y_norm = np.sum(y**2, axis=1)[None, :]
    return np.exp(-gamma * (x_norm + y_norm - 2 * x @ y.T))


def compute_mmd(source: np.ndarray, target: np.ndarray, gamma: float = None) -> float:
    if source.size == 0 or target.size == 0:
        return float("nan")
    if gamma is None:
        combined = np.vstack([source, target])
        median_sq = np.median(np.sum((combined[:, None, :] - combined[None, :, :]) ** 2, axis=2))
        gamma = 1.0 / (median_sq + 1e-6)
    k_ss = _rbf_kernel(source, source, gamma)
    k_tt = _rbf_kernel(target, target, gamma)
    k_st = _rbf_kernel(source, target, gamma)
    m = source.shape[0]
    n = target.shape[0]
    term_ss = (np.sum(k_ss) - np.trace(k_ss)) / (m * (m - 1) + 1e-12)
    term_tt = (np.sum(k_tt) - np.trace(k_tt)) / (n * (n - 1) + 1e-12)
    term_st = (2 * np.sum(k_st)) / (m * n + 1e-12)
    return float(term_ss + term_tt - term_st)


def compute_coral(source: np.ndarray, target: np.ndarray) -> float:
    if source.size == 0 or target.size == 0:
        return float("nan")
    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    source_centered = source - source_mean
    target_centered = target - target_mean
    cov_source = source_centered.T @ source_centered / (source.shape[0] - 1 + 1e-12)
    cov_target = target_centered.T @ target_centered / (target.shape[0] - 1 + 1e-12)
    diff = cov_source - cov_target
    mean_diff = np.linalg.norm(source_mean - target_mean) ** 2
    return float(mean_diff + np.linalg.norm(diff, ord="fro") ** 2)


def train_domain_classifier(source: np.ndarray, target: np.ndarray, seed: int = 2025) -> float:
    if source.size == 0 or target.size == 0:
        return float("nan")
    X = np.vstack([source, target])
    y = np.concatenate([np.zeros(len(source)), np.ones(len(target))])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return float(accuracy_score(y_test, y_pred))


def compute_domain_stats(source_df: pd.DataFrame, target_df: pd.DataFrame, feature_cols: list[str]) -> DomainStats:
    source = source_df[feature_cols].to_numpy(dtype=float)
    target = target_df[feature_cols].to_numpy(dtype=float)
    return DomainStats(
        mmd=compute_mmd(source, target),
        coral=compute_coral(source, target),
        domain_acc=train_domain_classifier(source, target),
    )


def score_source_similarity(source_df: pd.DataFrame, target_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    target_mean = target_df[feature_cols].mean().to_numpy(dtype=float)
    source_features = source_df[feature_cols].to_numpy(dtype=float)
    distances = np.linalg.norm(source_features - target_mean[None, :], axis=1)
    max_dist = np.nanmax(distances) or 1.0
    scores = 1 - distances / (max_dist + 1e-12)
    result = source_df.copy()
    result["similarity_score"] = scores
    return result
