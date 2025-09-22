"""Interpretability utilities for tree and neural models."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance

from .utils import ensure_dir

try:  # pragma: no cover - optional dependency
    import shap
except Exception:  # pragma: no cover
    shap = None  # type: ignore


def shap_feature_importance(model, X: pd.DataFrame, out_path: Path, top_k: int = 20) -> Dict[str, float]:
    ensure_dir(out_path.parent)
    if shap is None:
        return {}
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
    except Exception:
        return {}
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]
    return {X.columns[i]: float(mean_abs[i]) for i in order}


def permutation_importance_plot(model, X: pd.DataFrame, y: np.ndarray, out_path: Path, top_k: int = 20) -> Dict[str, float]:
    ensure_dir(out_path.parent)
    result = permutation_importance(model, X, y, n_repeats=10, random_state=0)
    importances = result.importances_mean
    order = np.argsort(importances)[::-1][:top_k]
    plt.figure(figsize=(8, 4))
    plt.barh(range(len(order)), importances[order][::-1])
    plt.yticks(range(len(order)), [X.columns[i] for i in order][::-1])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return {X.columns[i]: float(importances[i]) for i in order}
