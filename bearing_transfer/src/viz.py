"""Visualization utilities for the bearing transfer project."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .config import FIGURE_DIR, REPORT_PATH
from .utils import ensure_dir, ensure_report

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _choose_embedding(data: np.ndarray, random_state: int = 2025) -> np.ndarray:
    if data.shape[1] > 10:
        pca = PCA(n_components=10, random_state=random_state)
        data = pca.fit_transform(data)
    if data.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto")
        return tsne.fit_transform(data)
    if data.shape[1] == 1:
        return np.column_stack([data[:, 0], np.zeros_like(data[:, 0])])
    return data[:, :2]


def plot_embedding(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    hue: str,
    title: str,
    filename: str,
    random_state: int = 2025,
) -> Path:
    ensure_dir(FIGURE_DIR)
    data = df[list(feature_cols)].to_numpy(dtype=float)
    embedding = _choose_embedding(data, random_state=random_state)
    labels = df[hue].astype(str).to_numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    for idx, label in enumerate(np.unique(labels)):
        mask = labels == label
        color = COLORS[idx % len(COLORS)]
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=12, color=color, label=label, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", fontsize=8, markerscale=2)
    fig.tight_layout()
    path = Path(FIGURE_DIR) / filename
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    return path


def caption_and_insight(fig_path: Path, context: Dict[str, str]) -> str:
    report_path = ensure_report(REPORT_PATH)
    caption = context.get("caption", "图示")
    insight = context.get(
        "insight",
        "嵌入图显示源域与目标域特征分布的差异，域对齐前二者簇中心存在偏移。",
    )
    text = f"![{caption}]({fig_path})\n\n{insight}\n\n"
    with report_path.open("a", encoding="utf-8") as f:
        f.write(text)
    return text
