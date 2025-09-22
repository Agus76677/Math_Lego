"""Question 1: data preparation and common feature extraction."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import config
from src.feature_common import construct_common_feature_table
from src.feature_select import compute_domain_stats, score_source_similarity
from src.io_loader import load_directory, save_segments_metadata
from src.utils import StructuredLogger, ensure_dir, set_seed
from src.viz import caption_and_insight, plot_embedding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare bearing features")
    parser.add_argument("--data_root", type=str, default="bearing_transfer/data")
    parser.add_argument("--win", type=float, default=config.DEFAULT_WINDOW.duration_s)
    parser.add_argument("--overlap", type=float, default=config.DEFAULT_WINDOW.overlap)
    parser.add_argument("--band", type=str, default=config.DEFAULT_BAND.strategy)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed()
    artifact_dir = Path(config.ARTIFACT_DIR)
    ensure_dir(artifact_dir)
    figure_dir = Path(config.FIGURE_DIR)
    ensure_dir(figure_dir)
    logger = StructuredLogger(Path(config.LOG_DIR), name="prepare_features")
    logger.info("start_question1", data_root=args.data_root)

    source_dir = Path(args.data_root) / "source_mat"
    target_dir = Path(args.data_root) / "target_mat"

    source_dataset = load_directory(source_dir, window_duration=args.win, overlap=args.overlap)
    target_dataset = load_directory(target_dir, window_duration=args.win, overlap=args.overlap)

    save_segments_metadata(source_dataset, artifact_dir / "source_segments_meta.csv")
    save_segments_metadata(target_dataset, artifact_dir / "target_segments_meta.csv")

    if not source_dataset.segments:
        logger.warning("no_source_segments")
        return
    if not target_dataset.segments:
        logger.warning("no_target_segments")

    logger.info("extract_features_source", segments=len(source_dataset.segments))
    source_features = construct_common_feature_table(source_dataset.segments)
    source_path = artifact_dir / "source_features.csv"
    source_features.to_csv(source_path, index=False)

    logger.info("extract_features_target", segments=len(target_dataset.segments))
    target_features = construct_common_feature_table(target_dataset.segments)
    target_path = artifact_dir / "target_features.csv"
    target_features.to_csv(target_path, index=False)

    numeric_cols = [
        col
        for col in source_features.columns
        if source_features[col].dtype.kind in {"f", "i"}
        and col not in {"segment_index"}
    ]
    numeric_cols = [c for c in numeric_cols if c in target_features.columns]
    source_numeric = source_features[numeric_cols].fillna(source_features[numeric_cols].median())
    target_numeric = target_features[numeric_cols].fillna(target_features[numeric_cols].median())

    stats = compute_domain_stats(source_numeric, target_numeric, numeric_cols)
    stats_df = pd.DataFrame([
        {"metric": "mmd", "value": stats.mmd},
        {"metric": "coral", "value": stats.coral},
        {"metric": "domain_acc", "value": stats.domain_acc},
    ])
    stats_df.to_csv(artifact_dir / "domain_gap_metrics.csv", index=False)
    logger.info("domain_gap", mmd=stats.mmd, coral=stats.coral, domain_acc=stats.domain_acc)

    similarity = score_source_similarity(source_numeric, target_numeric, numeric_cols)
    similarity.to_csv(artifact_dir / "source_similarity_scores.csv", index=False)

    combined = pd.concat([
        source_features.assign(domain="source"),
        target_features.assign(domain="target"),
    ], ignore_index=True)
    embed_cols = numeric_cols[: min(len(numeric_cols), 50)]
    if embed_cols:
        fig_path = plot_embedding(
            combined,
            feature_cols=embed_cols,
            hue="domain",
            title="域差嵌入对比",
            filename="q1_domain_embedding.png",
        )
        caption_and_insight(
            fig_path,
            {
                "caption": "源域与目标域共性特征嵌入",
                "insight": (
                    "UMAP 嵌入揭示源域与目标域之间的统计差异。MMD={:.3f}, CORAL={:.3f}, 域分类准确率={:.2f}."
                ).format(stats.mmd, stats.coral, stats.domain_acc),
            },
        )

    logger.info("question1_complete", source_segments=len(source_dataset.segments), target_segments=len(target_dataset.segments))


if __name__ == "__main__":
    main()
