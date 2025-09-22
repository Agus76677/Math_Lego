"""Construction of common physical features for source/target domains."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .config import DEFAULT_FEATURE
from .dsp_features import (
    aggregate_band_energy,
    bandpass_filter,
    envelope_spectrum,
    spectral_kurtosis,
    time_domain_stats,
    tuned_band_from_sk,
)
from .utils import load_geometry


def bearing_characteristic_freqs(fr: float | None, geometry: Dict[str, float]) -> Dict[str, float | None]:
    if fr is None:
        return {key: None for key in ("FTF", "BPFO", "BPFI", "BSF")}
    z = geometry.get("Z", 8)
    bd = geometry.get("bd", 0.3126)
    pd = geometry.get("pd", 1.537)
    theta = math.radians(geometry.get("contact_angle_deg", 0.0))
    cos_theta = math.cos(theta)
    ftf = 0.5 * fr * (1 - (bd / pd) * cos_theta)
    bpfo = 0.5 * z * fr * (1 - (bd / pd) * cos_theta)
    bpfi = 0.5 * z * fr * (1 + (bd / pd) * cos_theta)
    bsf = (pd / (2 * bd)) * fr * (1 - ((bd / pd) * cos_theta) ** 2)
    return {"FTF": ftf, "BPFO": bpfo, "BPFI": bpfi, "BSF": bsf}


def _physical_band(center: float | None, delta: float) -> tuple[float, float] | None:
    if center is None or np.isnan(center):
        return None
    return center * (1 - delta), center * (1 + delta)


def _peak_features(freqs: np.ndarray, power: np.ndarray, center: float | None, delta: float, prefix: str) -> Dict[str, float]:
    features: Dict[str, float] = {
        f"{prefix}_amp": float("nan"),
        f"{prefix}_snr": float("nan"),
        f"{prefix}_freq": float("nan"),
        f"{prefix}_freq_dev": float("nan"),
    }
    if center is None or np.isnan(center) or freqs.size == 0:
        return features
    band = _physical_band(center, delta)
    if band is None:
        return features
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return features
    local_power = power[mask]
    local_freqs = freqs[mask]
    idx = int(np.argmax(local_power))
    amp = float(local_power[idx])
    freq = float(local_freqs[idx])
    noise_floor = float(np.median(power[~mask])) if np.any(~mask) else 1e-12
    snr = amp / (noise_floor + 1e-12)
    features[f"{prefix}_amp"] = amp
    features[f"{prefix}_snr"] = snr
    features[f"{prefix}_freq"] = freq
    features[f"{prefix}_freq_dev"] = freq - center
    return features


def _sideband_energy(freqs: np.ndarray, power: np.ndarray, center: float | None, fr: float | None, k: int, delta: float, prefix: str) -> Dict[str, float]:
    features: Dict[str, float] = {}
    if center is None or fr is None or freqs.size == 0:
        features[f"{prefix}_sb{k}_energy"] = float("nan")
        return features
    left = _physical_band(center - k * fr, delta)
    right = _physical_band(center + k * fr, delta)
    if left is None or right is None:
        features[f"{prefix}_sb{k}_energy"] = float("nan")
        return features
    mask_left = (freqs >= left[0]) & (freqs <= left[1])
    mask_right = (freqs >= right[0]) & (freqs <= right[1])
    energy = float(np.trapz(power[mask_left], freqs[mask_left]) + np.trapz(power[mask_right], freqs[mask_right]))
    features[f"{prefix}_sb{k}_energy"] = energy
    return features


def _harmonic_energy(freqs: np.ndarray, power: np.ndarray, center: float | None, harmonic: int, delta: float, prefix: str) -> Dict[str, float]:
    features: Dict[str, float] = {}
    if center is None or np.isnan(center):
        features[f"{prefix}_harm{harmonic}_energy"] = float("nan")
        return features
    band = _physical_band(center * harmonic, delta)
    if band is None:
        features[f"{prefix}_harm{harmonic}_energy"] = float("nan")
        return features
    mask = (freqs >= band[0]) & (freqs <= band[1])
    energy = float(np.trapz(power[mask], freqs[mask])) if np.any(mask) else float("nan")
    features[f"{prefix}_harm{harmonic}_energy"] = energy
    return features


DEFAULT_BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]


def extract_features_for_segment(segment, feature_config=DEFAULT_FEATURE) -> Dict[str, float]:
    signal = segment.data
    fs = segment.fs
    freqs_sk, sk = spectral_kurtosis(signal, fs, window=1024)
    low, high = tuned_band_from_sk(freqs_sk, sk, min_bandwidth=500.0)
    filtered = bandpass_filter(signal, fs, low, high)
    stats = time_domain_stats(filtered)
    env_freqs, env_power = envelope_spectrum(filtered, fs)
    features: Dict[str, float] = {f"time_{k}": v for k, v in stats.items()}
    features.update({"sk_max": float(np.nanmax(sk)), "sk_band_low": low, "sk_band_high": high})

    geom = load_geometry({})
    fr = segment.fr
    char_freqs = bearing_characteristic_freqs(fr, geom)
    delta = feature_config.physical_delta

    for name, center in char_freqs.items():
        prefix = f"{name.lower()}"
        features.update(_peak_features(env_freqs, env_power, center, delta, prefix))
        for harmonic in (2, 3):
            features.update(_harmonic_energy(env_freqs, env_power, center, harmonic, delta, prefix))
        for k in (1, 2):
            features.update(_sideband_energy(env_freqs, env_power, center, fr, k, delta, prefix))
        if feature_config.include_orders and fr:
            order = center / fr if center else float("nan")
            features[f"{prefix}_order"] = order
            freq_val = features.get(f"{prefix}_freq", float("nan"))
            features[f"{prefix}_freq_order"] = freq_val / (fr + 1e-12)
            freq_dev = features.get(f"{prefix}_freq_dev", float("nan"))
            features[f"{prefix}_order_dev"] = freq_dev / (fr + 1e-12)

    bands = DEFAULT_BANDS
    features.update(aggregate_band_energy(env_freqs, env_power, bands))
    features["fs"] = fs
    features["rpm"] = segment.rpm if segment.rpm is not None else float("nan")
    features["fr"] = fr if fr is not None else float("nan")
    if feature_config.include_fs_tag:
        features["fs_tag"] = segment.fs_tag
    features["channel"] = segment.channel
    features["fault_type"] = segment.fault_type
    features["fault_size"] = segment.fault_size or "unknown"
    features["file"] = segment.file_path.name
    features["segment_index"] = segment.segment_index
    return features


def construct_common_feature_table(segments: Iterable, feature_config=DEFAULT_FEATURE) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for segment in segments:
        rows.append(extract_features_for_segment(segment, feature_config))
    df = pd.DataFrame(rows)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace({np.inf: np.nan, -np.inf: np.nan})
    return df
