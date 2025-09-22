"""Train source domain models and log results."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import eval as eval_utils
from src import interpret, viz
from src.config import DEFAULT_MODEL_CONFIG
from src.models_source import METRIC_KEYS, aggregate_results, cross_validate_model, train_model
from src.utils import ensure_dir

REPORT_PATH = Path("reports/report.md")


def append_report(section: str, content: str) -> None:
    ensure_dir(REPORT_PATH.parent)
    with REPORT_PATH.open("a", encoding="utf-8") as f:
        f.write(f"\n\n## {section}\n\n{content}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="outputs/artifacts/source_features.csv")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODEL_CONFIG.models))
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()

    features_path = Path(args.features)
    df = pd.read_csv(features_path)
    feature_cols = [c for c in df.columns if c not in {"file", "segment_id", "position", "fault_type"}]
    X = df[feature_cols].to_numpy()
    y = df["fault_type"].astype(str).to_numpy()

    results = []
    for model in args.models.split(","):
        res = cross_validate_model(model, X, y)
        results.append(res)
    metrics = aggregate_results(results)
    ensure_dir(Path(args.output) / "artifacts")
    ensure_dir(Path(args.output) / "figures")
    with (Path(args.output) / "artifacts" / "source_model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    best_model_name = max(metrics.items(), key=lambda item: item[1]["macro_f1"])[0]

    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2025, stratify=y)
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2025, stratify=None)
    model = train_model(best_model_name, X_train, y_train)
    prob = model.predict_proba(X_val)
    classes = model.classes_ if hasattr(model, "classes_") else np.unique(y_train)
    pred_idx = prob.argmax(axis=1)
    pred = np.array([classes[i] for i in pred_idx])
    metrics_val = eval_utils.classification_metrics(y_val, pred, prob)
    with (Path(args.output) / "artifacts" / "source_val_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_val, f, ensure_ascii=False, indent=2)

    cm = confusion_matrix(y_val, pred, labels=sorted(np.unique(y)))
    fig_path = viz.plot_confusion_matrix(cm, sorted(np.unique(y)), Path(args.output) / "figures")
    caption = viz.caption_and_insight(Path(fig_path), "源域验证集混淆矩阵")
    append_report("题二：源域模型表现", caption)

    shap_path = Path(args.output) / "figures" / "shap_source.png"
    shap_info = interpret.shap_feature_importance(model, pd.DataFrame(X_train, columns=feature_cols), shap_path)
    if not shap_info:
        perm_path = Path(args.output) / "figures" / "perm_importance.png"
        shap_info = interpret.permutation_importance_plot(model, pd.DataFrame(X_val, columns=feature_cols), y_val, perm_path)
        caption = viz.caption_and_insight(Path(perm_path), "源域重要特征排名")
    else:
        caption = viz.caption_and_insight(shap_path, "源域SHAP重要性")
    append_report("题二：特征解释", json.dumps(shap_info, ensure_ascii=False, indent=2) + "\n" + caption)

    model_path = Path(args.output) / "artifacts" / "best_source_model.pkl"
    joblib.dump({"model": model, "features": feature_cols, "best_model": best_model_name}, model_path)


if __name__ == "__main__":
    main()
