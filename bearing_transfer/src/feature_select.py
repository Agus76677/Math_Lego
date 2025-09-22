"""Domain distance metrics and subset selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

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


def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float = None) -> float:
    if gamma is None:
        gamma = 1.0 / (X.shape[1] * np.var(np.vstack([X, Y])) + 1e-12)
    XX = np.exp(-gamma * ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    YY = np.exp(-gamma * ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    XY = np.exp(-gamma * ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return float(mmd)


def compute_coral(X: np.ndarray, Y: np.ndarray) -> float:
    cx = X - X.mean(axis=0, keepdims=True)
    cy = Y - Y.mean(axis=0, keepdims=True)
    cov_x = (cx.T @ cx) / (len(X) - 1)
    cov_y = (cy.T @ cy) / (len(Y) - 1)
    diff = cov_x - cov_y
    return float(np.sqrt((diff ** 2).sum()))


def domain_classifier_score(X: np.ndarray, y: np.ndarray) -> float:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)
    return float(accuracy_score(y_val, pred))


def evaluate_domain_gap(source_df: pd.DataFrame, target_df: pd.DataFrame, feature_cols: List[str]) -> DomainStats:
    Xs = source_df[feature_cols].to_numpy()
    Xt = target_df[feature_cols].to_numpy()
    mmd = compute_mmd(Xs, Xt)
    coral = compute_coral(Xs, Xt)
    X = np.vstack([Xs, Xt])
    y = np.array([0] * len(Xs) + [1] * len(Xt))
    domain_acc = domain_classifier_score(X, y)
    return DomainStats(mmd=mmd, coral=coral, domain_acc=domain_acc)


def score_source_samples(source_df: pd.DataFrame, target_df: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    target_mean = target_df[feature_cols].mean()
    distances = np.linalg.norm(source_df[feature_cols] - target_mean, axis=1)
    max_dist = distances.max() + 1e-12
    scores = 1 - distances / max_dist
    return pd.Series(scores, index=source_df.index)


def select_core_subset(source_df: pd.DataFrame, target_df: pd.DataFrame, feature_cols: List[str], keep_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.Series]:
    scores = score_source_samples(source_df, target_df, feature_cols)
    threshold = scores.quantile(1 - keep_ratio)
    mask = scores >= threshold
    core = source_df.loc[mask].copy()
    weights = scores.clip(lower=0.0, upper=1.0)
    return core, weights
