"""Generate additional interpretability artefacts."""
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

from src import interpret, viz
from src.utils import ensure_dir

REPORT_PATH = Path("reports/report.md")


def append_report(section: str, content: str) -> None:
    ensure_dir(REPORT_PATH.parent)
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write(f"\n\n## {section}\n\n{content}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_csv", type=str, default="outputs/artifacts/source_features.csv")
    parser.add_argument("--model_artifact", type=str, default="outputs/artifacts/best_source_model.pkl")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()

    ensure_dir(Path(args.output) / "figures")

    df = pd.read_csv(args.feature_csv)
    model_art = joblib.load(args.model_artifact)
    model = model_art["model"]
    feature_cols = model_art["features"]

    shap_path = Path(args.output) / "figures" / "explain_shap.png"
    shap_info = interpret.shap_feature_importance(model, df[feature_cols], shap_path, top_k=args.top_k)
    if not shap_info:
        perm_path = Path(args.output) / "figures" / "explain_perm.png"
        shap_info = interpret.permutation_importance_plot(model, df[feature_cols], df["fault_type"], perm_path, top_k=args.top_k)
        caption = viz.caption_and_insight(perm_path, "Permutation 特征重要性")
    else:
        caption = viz.caption_and_insight(shap_path, "SHAP 特征重要性")
    append_report("题四：事后解释", json.dumps(shap_info, ensure_ascii=False, indent=2) + "\n" + caption)


if __name__ == "__main__":
    main()
