"""Source domain models and training utilities."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    lgb = None

try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None


@dataclass
class FoldResult:
    accuracy: float
    macro_f1: float
    balanced_acc: float
    auc: float


@dataclass
class ModelResult:
    name: str
    metrics: Dict[str, float]
    per_fold: List[FoldResult]
    model: object


METRIC_KEYS = ["accuracy", "macro_f1", "balanced_acc", "auc"]


def _tree_model(name: str):
    if name.lower() == "lightgbm" and lgb is not None:
        return lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=-1)
    if name.lower() == "xgboost" and xgb is not None:
        return xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, eval_metric="mlogloss")
    return HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05)


def _mlp_model(input_dim: int, classes: int) -> MLPClassifier:
    hidden = max(32, min(128, input_dim // 2))
    return MLPClassifier(hidden_layer_sizes=(hidden, hidden // 2), activation="relu", max_iter=200)


def train_model(name: str, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> object:
    model_name = name.lower()
    if model_name in {"lightgbm", "xgboost", "svm"}:
        model = _tree_model(name)
        model.fit(X, y, sample_weight=sample_weight)
        return model
    if model_name == "mlp":
        clf = _mlp_model(X.shape[1], len(np.unique(y)))
        clf.fit(X, y)
        return clf
    if model_name == "cnn1d":  # fallback to mlp for simplicity when torch unavailable
        clf = _mlp_model(X.shape[1], len(np.unique(y)))
        clf.fit(X, y)
        return clf
    raise ValueError(f"Unknown model {name}")


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> FoldResult:
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
    pred = model.predict(X)
    accuracy = accuracy_score(y, pred)
    macro_f1 = f1_score(y, pred, average="macro")
    balanced_acc = balanced_accuracy_score(y, pred)
    if prob is not None and prob.shape[1] > 1:
        try:
            auc = roc_auc_score(y, prob, multi_class="ovr")
        except ValueError:
            auc = float("nan")
    else:
        auc = float("nan")
    return FoldResult(accuracy=accuracy, macro_f1=macro_f1, balanced_acc=balanced_acc, auc=auc)


def cross_validate_model(name: str, X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 2025) -> ModelResult:
    unique_labels, counts = np.unique(y, return_counts=True)
    min_per_class = counts.min()
    if len(y) < n_splits or min_per_class < 2:
        model = train_model(name, X, y)
        result = evaluate_model(model, X, y)
        metrics = {key: getattr(result, key) for key in METRIC_KEYS}
        return ModelResult(name=name, metrics=metrics, per_fold=[result], model=model)
    n_splits = max(2, min(n_splits, len(unique_labels), len(y)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: List[FoldResult] = []
    for train_idx, test_idx in skf.split(X, y):
        model = train_model(name, X[train_idx], y[train_idx])
        result = evaluate_model(model, X[test_idx], y[test_idx])
        folds.append(result)
    metrics = {key: float(np.nanmean([getattr(f, key) for f in folds])) for key in METRIC_KEYS}
    return ModelResult(name=name, metrics=metrics, per_fold=folds, model=None)


def aggregate_results(results: List[ModelResult]) -> Dict[str, Dict[str, float]]:
    return {res.name: res.metrics for res in results}
