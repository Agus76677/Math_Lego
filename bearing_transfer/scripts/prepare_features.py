"""Prepare features for source and target domains (with progress logs)."""
from __future__ import annotations

import argparse
import json
import sys
import time
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


def _stamp() -> float:
    return time.perf_counter()


def _elapsed(s: float) -> str:
    return f"{time.perf_counter() - s:.2f}s"


def main() -> None:
    t0 = _stamp()
    print("=== Bearing Transfer: Prepare Features ===", flush=True)

    # -------------------------
    # 1) Parse arguments
    # -------------------------
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True,
                   help="Root folder containing source/target subfolders (e.g., Y_datasets, MBY_datasets)")
    p.add_argument("--source_sub", type=str, default="Y_datasets")
    p.add_argument("--target_sub", type=str, default="MBY_datasets")
    p.add_argument("--win", type=float, default=DEFAULT_FEATURE_CONFIG.window_sec)
    p.add_argument("--overlap", type=float, default=DEFAULT_FEATURE_CONFIG.overlap)
    p.add_argument("--band", type=str, default=DEFAULT_FEATURE_CONFIG.band_strategy)
    p.add_argument("--output", type=str, default="outputs",
                   help="Output root for artifacts/figures; report goes to reports/report.md")
    args = p.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"[FATAL] data_root does not exist: {data_root}", flush=True)
        sys.exit(1)

    out_root = Path(args.output)
    feature_dir = out_root / "artifacts"
    figure_dir = out_root / "figures"
    report_path = Path("reports/report.md")

    print(f"[CFG] data_root = {data_root}")
    print(f"[CFG] source_sub = {args.source_sub}")
    print(f"[CFG] target_sub = {args.target_sub}")
    print(f"[CFG] win={args.win}, overlap={args.overlap}, band={args.band}")
    print(f"[CFG] output = {out_root.resolve()}")
    print(f"[CFG] report = {report_path.resolve()}", flush=True)

    # -------------------------
    # 2) Prepare output dirs
    # -------------------------
    t = _stamp()
    ensure_dir(feature_dir)
    ensure_dir(figure_dir)
    print(f"[OK ] Created output dirs -> artifacts: {feature_dir.resolve()}, figures: {figure_dir.resolve()} ({_elapsed(t)})", flush=True)

    # -------------------------
    # 3) Load datasets
    # -------------------------
    t = _stamp()
    print("[.. ] Loading source/target segments ...", flush=True)
    source_loader = io_loader.BearingDataLoader(data_root / args.source_sub, window_sec=args.win, overlap=args.overlap)
    target_loader = io_loader.BearingDataLoader(data_root / args.target_sub, window_sec=args.win, overlap=args.overlap)

    source_records = source_loader.load_all()
    target_records = target_loader.load_all()

    print(f"[OK ] Loaded segments -> source: {len(source_records)}, target: {len(target_records)} ({_elapsed(t)})", flush=True)

    # Save segment metadata
    t = _stamp()
    io_loader.save_segments_metadata(source_records, feature_dir / "source_segments_meta.csv")
    io_loader.save_segments_metadata(target_records, feature_dir / "target_segments_meta.csv")
    print(f"[OK ] Saved segments metadata CSVs ({_elapsed(t)})", flush=True)

    # -------------------------
    # 4) Characteristic frequencies & feature extraction
    # -------------------------
    t = _stamp()
    print("[.. ] Computing characteristic frequencies lookup ...", flush=True)
    freq_lookup = {}
    for rec in source_records + target_records:
        freq_lookup[(rec.position, rec.fault_type)] = feature_common.bearing_characteristic_frequencies(
            rec.rpm, rec.geometry
        )
    print(f"[OK ] Built freq_lookup for {len(freq_lookup)} (position, fault_type) pairs ({_elapsed(t)})", flush=True)

    t = _stamp()
    print("[.. ] Extracting DSP features -> source ...", flush=True)
    source_df = dsp_features.features_dataframe(
        source_records, freq_lookup, delta=DEFAULT_FEATURE_CONFIG.order_delta
    )
    print(f"[OK ] source_df shape = {source_df.shape} ({_elapsed(t)})", flush=True)

    t = _stamp()
    print("[.. ] Extracting DSP features -> target ...", flush=True)
    target_df = dsp_features.features_dataframe(
        target_records, freq_lookup, delta=DEFAULT_FEATURE_CONFIG.order_delta
    )
    print(f"[OK ] target_df shape = {target_df.shape} ({_elapsed(t)})", flush=True)

    # -------------------------
    # 5) Align columns and save
    # -------------------------
    t = _stamp()
    print("[.. ] Aligning feature columns between source/target ...", flush=True)
    for col in source_df.columns:
        if col not in target_df.columns:
            target_df[col] = 0.0
    common_cols = [c for c in source_df.columns if c in target_df.columns]
    target_df = target_df[common_cols]
    print(f"[OK ] Aligned columns. common_cols={len(common_cols)} ({_elapsed(t)})", flush=True)

    t = _stamp()
    source_csv = feature_dir / "source_features.csv"
    target_csv = feature_dir / "target_features.csv"
    source_df.to_csv(source_csv, index=False)
    target_df.to_csv(target_csv, index=False)
    print(f"[OK ] Saved feature CSVs ->\n      {source_csv.resolve()}\n      {target_csv.resolve()} ({_elapsed(t)})", flush=True)

    # -------------------------
    # 6) Evaluate domain gap
    # -------------------------
    t = _stamp()
    feature_cols = [c for c in source_df.columns if c not in {"file", "segment_id", "position", "fault_type"}]
    print(f"[.. ] Evaluating domain gap on {len(feature_cols)} features ...", flush=True)
    stats = feature_select.evaluate_domain_gap(source_df, target_df, feature_cols)
    gap_json = feature_dir / "domain_gap.json"
    save_json(stats.__dict__, gap_json)
    print(f"[OK ] Saved domain_gap -> {gap_json.resolve()} ({_elapsed(t)})", flush=True)

    # -------------------------
    # 7) Embedding visualization
    # -------------------------
    t = _stamp()
    print("[.. ] Building PCA embedding for visualization ...", flush=True)
    feats = np.vstack([
        np.c_[source_df[feature_cols].to_numpy(), np.zeros(len(source_df))],
        np.c_[target_df[feature_cols].to_numpy(), np.ones(len(target_df))],
    ])
    labels = list(source_df["fault_type"]) + ["?"] * len(target_df)
    domains = ["source"] * len(source_df) + ["target"] * len(target_df)
    emb_path = viz.embedding_plot(feats[:, :-1], labels, domains, method="pca", out_path=figure_dir)
    print(f"[OK ] Saved embedding figure -> {Path(emb_path).resolve()} ({_elapsed(t)})", flush=True)

    # -------------------------
    # 8) Report: caption + summaries
    # -------------------------
    t = _stamp()
    print("[.. ] Writing report sections ...", flush=True)
    caption = viz.caption_and_insight(Path(emb_path), "源域与目标域特征分布")
    append_report("题一：特征分布", caption, report_path)

    summary = {
        "source": feature_common.feature_summary(source_df),
        "target": feature_common.feature_summary(target_df),
    }
    summary_json = feature_dir / "feature_summary.json"
    save_json(summary, summary_json)
    append_report("题一：数据概览", json.dumps(summary, ensure_ascii=False, indent=2), report_path)
    print(f"[OK ] Saved feature_summary -> {summary_json.resolve()} and updated report -> {report_path.resolve()} ({_elapsed(t)})", flush=True)

    # -------------------------
    # 9) Done
    # -------------------------
    print(f"=== Done in {_elapsed(t0)} ===", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] User aborted.", flush=True)
        sys.exit(130)
    except Exception as e:
        # 让错误更易读，同时保留默认 traceback（由 Python 打印）
        print(f"[ERROR] {e}", flush=True)
        raise
