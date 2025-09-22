"""Physical frequency helpers and feature harmonisation."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .io_loader import DEFAULT_GEOMETRY


def bearing_characteristic_frequencies(rpm: float, geometry: Dict[str, float] | None = None) -> Dict[str, float]:
    geom = DEFAULT_GEOMETRY.copy()
    if geometry:
        geom.update({k: v for k, v in geometry.items() if v is not None})
    fr = rpm / 60.0
    Z = geom.get("Z", 9)
    bd = geom.get("bd", 0.2858)
    pd = geom.get("pd", 1.245)
    theta = math.radians(geom.get("contact_angle_deg", 0.0))
    cos_theta = math.cos(theta)
    ratio = bd / pd if pd else 0.0
    ftf = 0.5 * fr * (1 - ratio * cos_theta)
    bpfo = 0.5 * Z * fr * (1 - ratio * cos_theta)
    bpfi = 0.5 * Z * fr * (1 + ratio * cos_theta)
    bsf = fr * (pd / (2 * bd)) * (1 - (ratio * cos_theta) ** 2)
    return {
        "FTF": ftf,
        "BPFO": bpfo,
        "BPFI": bpfi,
        "BSF": bsf,
    }


def combine_hz_order_features(df: pd.DataFrame, rpm_column: str = "RPM") -> pd.DataFrame:
    df = df.copy()
    rpm = df[rpm_column].to_numpy()
    fr = rpm / 60.0
    fr[fr == 0] = np.nan
    phys_cols = [c for c in df.columns if c.startswith("freq_peak_amp")]
    for col in phys_cols:
        base_freq = float(col.split("_")[-1])
        order_col = f"order_{col.split('_')[-1]}"
        df[order_col] = base_freq / fr
    df.fillna(0.0, inplace=True)
    return df


def standardize_features(train_df: pd.DataFrame, test_df: pd.DataFrame | None = None) -> Tuple[pd.DataFrame, pd.DataFrame | None, Dict[str, Tuple[float, float]]]:
    stats: Dict[str, Tuple[float, float]] = {}
    scaled_train = train_df.copy()
    scaled_test = test_df.copy() if test_df is not None else None
    for col in train_df.columns:
        if col in {"file", "segment_id", "position", "fault_type"}:
            continue
        mean = float(train_df[col].mean())
        std = float(train_df[col].std() + 1e-12)
        stats[col] = (mean, std)
        scaled_train[col] = (train_df[col] - mean) / std
        if scaled_test is not None:
            scaled_test[col] = (test_df[col] - mean) / std
    return scaled_train, scaled_test, stats


def feature_summary(df: pd.DataFrame) -> Dict[str, float]:
    summary = {
        "n_samples": len(df),
        "n_features": df.select_dtypes(include=[float, int]).shape[1],
        "mean_rpm": float(df["RPM"].mean()) if "RPM" in df else 0.0,
        "std_rpm": float(df["RPM"].std()) if "RPM" in df else 0.0,
    }
    return summary
