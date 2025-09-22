"""Prepare features for source and target domains."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import dsp_features, feature_common, feature_select, io_loader, viz
from src.config import DEFAULT_FEATURE_CONFIG
from src.utils import ensure_dir, save_json


def append_report(section: str, content: str, report_path: Path) -> None:
    ensure_dir(report_path.parent)
    with report_path.open("a", encoding="utf-8") as f:
        f.write(f"\n\n## {section}\n\n{content}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--source_sub", type=str, default="Y_datasets")
    parser.add_argument("--target_sub", type=str, default="MBY_datasets")
    parser.add_argument("--win", type=float, default=DEFAULT_FEATURE_CONFIG.window_sec)
    parser.add_argument("--overlap", type=float, default=DEFAULT_FEATURE_CONFIG.overlap)
    parser.add_argument("--band", type=str, default=DEFAULT_FEATURE_CONFIG.band_strategy)
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()

    out_root = Path(args.output)
    feature_dir = out_root / "artifacts"
    figure_dir = out_root / "figures"
    report_path = Path("reports/report.md")

    ensure_dir(feature_dir)
    ensure_dir(figure_dir)

    source_loader = io_loader.BearingDataLoader(Path(args.data_root) / args.source_sub, window_sec=args.win, overlap=args.overlap)
    target_loader = io_loader.BearingDataLoader(Path(args.data_root) / args.target_sub, window_sec=args.win, overlap=args.overlap)

    source_records = source_loader.load_all()
    target_records = target_loader.load_all()

    io_loader.save_segments_metadata(source_records, feature_dir / "source_segments_meta.csv")
    io_loader.save_segments_metadata(target_records, feature_dir / "target_segments_meta.csv")

    freq_lookup = {}
    for rec in source_records + target_records:
        freq_lookup[(rec.position, rec.fault_type)] = feature_common.bearing_characteristic_frequencies(rec.rpm, rec.geometry)

    source_df = dsp_features.features_dataframe(source_records, freq_lookup, delta=DEFAULT_FEATURE_CONFIG.order_delta)
    target_df = dsp_features.features_dataframe(target_records, freq_lookup, delta=DEFAULT_FEATURE_CONFIG.order_delta)

    for col in source_df.columns:
        if col not in target_df.columns:
            target_df[col] = 0.0
    common_cols = [c for c in source_df.columns if c in target_df.columns]
    target_df = target_df[common_cols]

    source_df.to_csv(feature_dir / "source_features.csv", index=False)
    target_df.to_csv(feature_dir / "target_features.csv", index=False)

    feature_cols = [c for c in source_df.columns if c not in {"file", "segment_id", "position", "fault_type"}]
    stats = feature_select.evaluate_domain_gap(source_df, target_df, feature_cols)
    save_json(stats.__dict__, feature_dir / "domain_gap.json")

    feats = np.vstack([
        np.c_[source_df[feature_cols].to_numpy(), np.zeros(len(source_df))],
        np.c_[target_df[feature_cols].to_numpy(), np.ones(len(target_df))],
    ])
    labels = list(source_df["fault_type"]) + ["?"] * len(target_df)
    domains = ["source"] * len(source_df) + ["target"] * len(target_df)
    emb_path = viz.embedding_plot(feats[:, :-1], labels, domains, method="pca", out_path=figure_dir)
    caption = viz.caption_and_insight(Path(emb_path), "源域与目标域特征分布")
    append_report("题一：特征分布", caption, report_path)

    summary = {
        "source": feature_common.feature_summary(source_df),
        "target": feature_common.feature_summary(target_df),
    }
    save_json(summary, feature_dir / "feature_summary.json")
    append_report("题一：数据概览", json.dumps(summary, ensure_ascii=False, indent=2), report_path)


if __name__ == "__main__":
    main()
