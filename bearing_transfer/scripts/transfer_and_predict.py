"""Run transfer learning from source to target domain."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import feature_select, transfer, viz
from src.config import DEFAULT_TRANSFER_CONFIG
from src.utils import ensure_dir, save_json

REPORT_PATH = Path("reports/report.md")


def append_report(section: str, content: str) -> None:
    ensure_dir(REPORT_PATH.parent)
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write(f"\n\n## {section}\n\n{content}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_features", type=str, default="outputs/artifacts/source_features.csv")
    parser.add_argument("--target_features", type=str, default="outputs/artifacts/target_features.csv")
    parser.add_argument("--model_artifact", type=str, default="outputs/artifacts/best_source_model.pkl")
    parser.add_argument("--align", type=str, default=",".join(DEFAULT_TRANSFER_CONFIG.align_methods))
    parser.add_argument("--sample_weight", type=str, default="yes")
    parser.add_argument("--pseudolabel", type=str, default="yes")
    parser.add_argument("--conf_th", type=str, default="0.95,0.90")
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()

    ensure_dir(Path(args.output) / "artifacts")
    ensure_dir(Path(args.output) / "figures")

    source_df = pd.read_csv(args.source_features)
    target_df = pd.read_csv(args.target_features)
    feature_cols = [c for c in source_df.columns if c not in {"file", "segment_id", "position", "fault_type"}]

    model_artifact = joblib.load(args.model_artifact)
    base_model = model_artifact["best_model"]
    Xs = source_df[feature_cols].to_numpy()
    ys = source_df["fault_type"].astype("category").cat.codes.to_numpy()
    label_map = dict(enumerate(source_df["fault_type"].astype("category").cat.categories))
    Xt = target_df[feature_cols].to_numpy()

    align_methods = [m.strip() for m in args.align.split(",") if m.strip()]
    thresholds = tuple(float(x) for x in args.conf_th.split(","))
    sample_weights = None
    if args.sample_weight.lower() == "yes":
        _, weights = feature_select.select_core_subset(source_df, target_df, feature_cols)
        sample_weights = weights.to_numpy()

    outputs = []
    for method in align_methods:
        res = transfer.transfer_predict(
            base_model=base_model,
            Xs=Xs,
            ys=ys,
            Xt=Xt,
            align=method,
            sample_weight=sample_weights,
            pseudo=args.pseudolabel.lower() == "yes",
            thresholds=thresholds,  # type: ignore
        )
        preds = [label_map[idx] for idx in res.predictions]
        confidences = res.probabilities.max(axis=1)
        outputs.append({"method": method, "preds": preds, "conf": confidences})
        df_pred = pd.DataFrame({"file": target_df["file"], "prediction": preds, "confidence": confidences})
        df_pred.to_csv(Path(args.output) / "artifacts" / f"target_predictions_{method}.csv", index=False)

        before = feature_select.evaluate_domain_gap(source_df, target_df, feature_cols)
        aligned_source = transfer.coral_align(Xs, Xt) if method.upper() == "CORAL" else Xs
        after = feature_select.evaluate_domain_gap(pd.DataFrame(aligned_source, columns=feature_cols), target_df, feature_cols)
        improvement = {
            "method": method,
            "mmd_reduction": before.mmd - after.mmd,
            "coral_reduction": before.coral - after.coral,
        }
        save_json(improvement, Path(args.output) / "artifacts" / f"transfer_improvement_{method}.json")

        feats = np.vstack([aligned_source, Xt])
        labels = list(source_df["fault_type"]) + ["?"] * len(target_df)
        domains = ["source_aligned"] * len(source_df) + ["target"] * len(target_df)
        emb_path = viz.embedding_plot(feats, labels, domains, method="pca", out_path=Path(args.output) / "figures")
        caption = viz.caption_and_insight(Path(emb_path), f"{method} 对齐后的嵌入分布")
        append_report("题三：迁移对齐", caption)

    best = max(outputs, key=lambda o: np.mean(o["conf"]))
    summary = {
        "best_method": best["method"],
        "mean_confidence": float(np.mean(best["conf"])),
    }
    append_report("题三：最优方案", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
