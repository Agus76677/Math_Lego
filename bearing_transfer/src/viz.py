"""Visualization utilities with automatic caption generation."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .utils import ensure_dir

try:  # pragma: no cover - optional
    import umap
except Exception:  # pragma: no cover
    umap = None  # type: ignore


COLORS = {
    "source": "tab:blue",
    "target": "tab:orange",
}


def _save_fig(fig: Figure, base_path: Path) -> None:
    ensure_dir(base_path.parent)
    fig.savefig(base_path.with_suffix(".png"), dpi=200)
    fig.savefig(base_path.with_suffix(".svg"))
    plt.close(fig)


def embedding_plot(features: np.ndarray, labels: Iterable[str], domains: Iterable[str], method: str, out_path: Path) -> Path:
    if method.lower() == "umap" and umap is not None:
        reducer = umap.UMAP(random_state=0)
        emb = reducer.fit_transform(features)
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=0, init="pca", learning_rate="auto")
        emb = reducer.fit_transform(features)
    else:
        reducer = PCA(n_components=2)
        emb = reducer.fit_transform(features)
    fig, ax = plt.subplots(figsize=(6, 5))
    domains = list(domains)
    labels = list(labels)
    for domain in sorted(set(domains)):
        mask = [d == domain for d in domains]
        ax.scatter(emb[mask, 0], emb[mask, 1], label=domain, alpha=0.6)
    ax.set_title(f"{method.upper()} embedding")
    ax.legend()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out_path / f"embedding_{method}_{timestamp}"
    _save_fig(fig, base)
    return base.with_suffix(".png")


def plot_confusion_matrix(cm: np.ndarray, classes: Iterable[str], out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.colorbar(im, ax=ax)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out_path / f"confusion_{timestamp}"
    _save_fig(fig, base)
    return base.with_suffix(".png")


def plot_spectrum(freq: np.ndarray, spectrum: np.ndarray, markers: Optional[dict], out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(freq, spectrum)
    if markers:
        for label, f0 in markers.items():
            ax.axvline(f0, color="r", linestyle="--", alpha=0.7)
            ax.text(f0, ax.get_ylim()[1] * 0.9, label, rotation=90, color="r", va="top")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = out_path / f"spectrum_{timestamp}"
    _save_fig(fig, base)
    return base.with_suffix(".png")


def caption_and_insight(fig_path: Path, context: str) -> str:
    fig_name = fig_path.stem.replace("_", " ")
    insights = [f"图 {fig_name} 展示了{context}"]
    if "embedding" in fig_path.name:
        insights.append("源域与目标域在降维空间的重叠度反映了域间差异程度。")
        insights.append("对齐后若点云混合，则迁移风险降低。")
    elif "confusion" in fig_path.name:
        insights.append("混淆矩阵对比了各类预测命中率，便于识别易混淆故障。")
    elif "spectrum" in fig_path.name:
        insights.append("包络谱在关键特征频附近的峰值揭示了故障类型。")
    else:
        insights.append("图像揭示了时频特征的变化趋势。")
    return "\n".join(insights)
