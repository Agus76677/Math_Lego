#!/usr/bin/env python3
"""auto_classify.py

Usage:

```bash
# 放置 A.mat … P.mat 到 data_target/
python auto_classify.py > final_predictions.csv
# 文件即为最终 16 行结果（末行是注释行）
```

The script performs iterative self-refining classification of the 16 input
samples into four classes (OR/IR/B/N) until convergence metrics are met or a
high iteration ceiling is reached.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import io as sio
from scipy import signal
from scipy.fft import rfft, rfftfreq
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


DATA_DIR = os.path.join(os.path.dirname(__file__), "0.data", "MBY_datasets")
DEFAULT_FS = 32000.0
FAULT_CLASSES = ["OR", "IR", "B", "N"]
HISTORY_LENGTH = 50
RECENT_WINDOW = 20
MAX_ITERS = 2000


@dataclass
class SampleFeature:
    name: str
    features: Dict[str, float]
    physics_probs: np.ndarray
    physics_label: str
    physics_margin: float


def load_mat_file(path: str) -> Dict[str, np.ndarray]:
    mat = sio.loadmat(path)
    channels: Dict[str, np.ndarray] = {}
    rpm_value = None
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        arr = np.array(value)
        if arr.size == 0:
            continue
        if arr.dtype.names:  # Structured array
            for field in arr.dtype.names:
                data = np.array(arr[field]).squeeze()
                if data.ndim == 1:
                    channels[field] = data.astype(float)
        else:
            data = arr.squeeze()
            if data.ndim == 1:
                channels[key] = data.astype(float)
            elif data.ndim == 2:
                rows, cols = data.shape
                if rows == 1 or cols == 1:
                    channels[key] = data.reshape(-1).astype(float)
                else:
                    for idx in range(cols):
                        channels[f"{key}_{idx}"] = data[:, idx].astype(float)
    if "RPM" in mat:
        rpm_arr = np.array(mat["RPM"]).squeeze()
        if rpm_arr.size > 0:
            rpm_value = float(rpm_arr.flat[0])
    if rpm_value is not None:
        channels["__RPM__"] = np.array([rpm_value], dtype=float)
    return channels


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    if len(values) == 0:
        return 0.0
    sorted_idx = np.argsort(values)
    values_arr = np.array(values)[sorted_idx]
    weights_arr = np.array(weights)[sorted_idx]
    cumulative = np.cumsum(weights_arr)
    cutoff = 0.5 * np.sum(weights_arr)
    median_index = np.searchsorted(cumulative, cutoff)
    median_index = min(median_index, len(values_arr) - 1)
    return float(values_arr[median_index])


def channel_weight(name: str) -> float:
    upper = name.upper()
    if "DE" in upper:
        return 1.0
    if "FE" in upper:
        return 0.95
    if "BA" in upper:
        return 0.8
    return 0.9


def detrend_signal(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)


def compute_envelope(x: np.ndarray) -> np.ndarray:
    analytic = signal.hilbert(x)
    return np.abs(analytic)


def estimate_fr(envelope: np.ndarray, fs: float) -> float:
    spectrum = np.abs(rfft(envelope * signal.windows.hann(len(envelope))))
    freqs = rfftfreq(len(envelope), d=1.0 / fs)
    mask = (freqs > 0.5) & (freqs < 50.0)
    if not np.any(mask):
        return 10.0
    local_spec = spectrum[mask]
    local_freqs = freqs[mask]
    idx = int(np.argmax(local_spec))
    fr = float(local_freqs[idx])
    return max(fr, 1.0)


def compute_order_spectrum(envelope: np.ndarray, fs: float, fr: float) -> Tuple[np.ndarray, np.ndarray]:
    windowed = envelope * signal.windows.hann(len(envelope))
    spec = np.abs(rfft(windowed))
    freqs = rfftfreq(len(envelope), d=1.0 / fs)
    if fr <= 0:
        fr = 1.0
    orders = freqs / fr
    return orders, spec


def integrate_band(order_axis: np.ndarray, spectrum: np.ndarray, center: float, width: float) -> float:
    lower = max(center - width, 0.0)
    upper = center + width
    mask = (order_axis >= lower) & (order_axis <= upper)
    if not np.any(mask):
        return 0.0
    band_power = np.sum((spectrum[mask]) ** 2)
    return float(band_power)


def fault_energy_metrics(
    order_axis: np.ndarray,
    spectrum: np.ndarray,
    base_range: Tuple[float, float],
    fr: float,
    harmonics: int = 4,
    peak_half_width: float = 0.15,
    sidebands: int = 3,
    sideband_half_width: float = 0.12,
) -> Tuple[float, float, float]:
    base_energy = 0.0
    sideband_symmetry_scores: List[float] = []
    dominant_order = 0.0
    for harmonic in range(1, harmonics + 1):
        start = base_range[0] * harmonic
        end = base_range[1] * harmonic
        mask = (order_axis >= start) & (order_axis <= end)
        if not np.any(mask):
            continue
        local_spec = spectrum[mask]
        local_orders = order_axis[mask]
        peak_index = int(np.argmax(local_spec))
        center_order = float(local_orders[peak_index])
        dominant_order = max(dominant_order, center_order)
        energy = integrate_band(order_axis, spectrum, center_order, peak_half_width)
        base_energy += energy / harmonic
        for sb in range(1, sidebands + 1):
            plus_center = center_order + sb
            minus_center = max(center_order - sb, 0.0)
            e_plus = integrate_band(order_axis, spectrum, plus_center, sideband_half_width)
            e_minus = integrate_band(order_axis, spectrum, minus_center, sideband_half_width)
            total = e_plus + e_minus
            if total > 0:
                symmetry = 1.0 - abs(e_plus - e_minus) / total
                sideband_symmetry_scores.append(symmetry)
    symmetry_value = float(np.mean(sideband_symmetry_scores)) if sideband_symmetry_scores else 0.0
    return base_energy, symmetry_value, dominant_order


def spectral_flatness(power: np.ndarray) -> float:
    power = np.maximum(power, 1e-12)
    geom = np.exp(np.mean(np.log(power)))
    arith = np.mean(power)
    return float(geom / arith) if arith > 0 else 0.0


def spectral_kurtosis_peak(power: np.ndarray) -> float:
    centered = power - np.mean(power)
    std = np.std(centered) + 1e-12
    normalized = centered / std
    kurtosis_values = normalized ** 4
    return float(np.max(kurtosis_values))


def physics_scoring(feature_map: Dict[str, float]) -> Tuple[np.ndarray, str, float]:
    energy_bpfo = max(feature_map.get("energy_bpfo", 0.0), 0.0)
    energy_bpfi = max(feature_map.get("energy_bpfi", 0.0), 0.0)
    energy_bsf = max(feature_map.get("energy_bsf", 0.0), 0.0)
    side_bpfo = max(feature_map.get("sideband_bpfo", 0.0), 0.0)
    side_bpfi = max(feature_map.get("sideband_bpfi", 0.0), 0.0)
    side_bsf = max(feature_map.get("sideband_bsf", 0.0), 0.0)
    rms = feature_map.get("rms", 0.0)
    kurt = feature_map.get("kurtosis", 3.0)
    flatness = feature_map.get("spectral_flatness", 0.0)
    sk_peak = feature_map.get("spectral_kurtosis_peak", 0.0)

    fault_strength = np.array(
        [
            energy_bpfo * (0.6 + 0.4 * side_bpfo),
            energy_bpfi * (0.6 + 0.4 * side_bpfi),
            energy_bsf * (0.6 + 0.4 * side_bsf),
        ]
    )
    norm_fault_strength = fault_strength / (np.max(fault_strength) + 1e-9)
    healthy_indicator = 1.0 / (1.0 + np.sum(norm_fault_strength))
    healthy_indicator *= (0.8 + 0.2 * flatness)
    healthy_indicator *= 1.0 / (1.0 + max(kurt - 3.0, 0.0))
    healthy_indicator *= 1.0 / (1.0 + max(sk_peak - 3.0, 0.0))

    raw_scores = np.concatenate([norm_fault_strength, [healthy_indicator + max(0.0, 0.5 - rms)]])
    raw_scores = np.maximum(raw_scores, 1e-9)
    probs = raw_scores / np.sum(raw_scores)
    top_two = np.sort(probs)[-2:]
    margin = float(top_two[-1] - top_two[-2]) if len(top_two) >= 2 else float(top_two[-1])
    label = FAULT_CLASSES[int(np.argmax(probs))]
    return probs, label, margin


def aggregate_sample_features(name: str, channels: Dict[str, np.ndarray], fs: float = DEFAULT_FS) -> SampleFeature:
    rpm_value = None
    if "__RPM__" in channels:
        rpm_value = float(channels.pop("__RPM__")[0])
    channel_features: Dict[str, Dict[str, float]] = {}
    channel_weights: Dict[str, float] = {}
    physics_map: Dict[str, np.ndarray] = {}
    for channel_name, signal_data in channels.items():
        if not isinstance(signal_data, np.ndarray):
            continue
        x = detrend_signal(signal_data.astype(float))
        if x.size < 8:
            continue
        envelope = compute_envelope(x)
        fr = rpm_value / 60.0 if rpm_value else estimate_fr(envelope, fs)
        orders, spectrum = compute_order_spectrum(envelope, fs, fr)
        power = (spectrum ** 2)
        bpfo_energy, bpfo_side, bpfo_dom = fault_energy_metrics(orders, spectrum, (3.0, 4.5), fr)
        bpfi_energy, bpfi_side, bpfi_dom = fault_energy_metrics(orders, spectrum, (4.5, 6.5), fr)
        bsf_energy, bsf_side, bsf_dom = fault_energy_metrics(orders, spectrum, (2.0, 3.5), fr)

        rms = float(np.sqrt(np.mean(x ** 2)))
        centered = x - np.mean(x)
        m4 = np.mean(centered ** 4)
        m2 = np.mean(centered ** 2) + 1e-12
        kurt = float(m4 / (m2 ** 2))
        flat = spectral_flatness(power)
        sk_peak = spectral_kurtosis_peak(power)

        feature_map = {
            "energy_bpfo": bpfo_energy,
            "energy_bpfi": bpfi_energy,
            "energy_bsf": bsf_energy,
            "sideband_bpfo": bpfo_side,
            "sideband_bpfi": bpfi_side,
            "sideband_bsf": bsf_side,
            "dominant_bpfo": bpfo_dom,
            "dominant_bpfi": bpfi_dom,
            "dominant_bsf": bsf_dom,
            "rms": rms,
            "kurtosis": kurt,
            "spectral_flatness": flat,
            "spectral_kurtosis_peak": sk_peak,
            "fr": fr,
        }
        probs, label, margin = physics_scoring(feature_map)
        channel_features[channel_name] = feature_map
        channel_weights[channel_name] = channel_weight(channel_name)
        physics_map[channel_name] = np.concatenate([probs, [margin]])

    if not channel_features:
        raise ValueError(f"No valid channels found in {name}")

    aggregated: Dict[str, float] = {}
    for key in next(iter(channel_features.values())).keys():
        values = [feat[key] for feat in channel_features.values()]
        weights = [channel_weights[ch] for ch in channel_features.keys()]
        aggregated[key] = weighted_median(values, weights)
    aggregated["fr"] = float(np.median([feat["fr"] for feat in channel_features.values()]))

    combined_probs = np.zeros(len(FAULT_CLASSES))
    margins = []
    for ch, packed in physics_map.items():
        weight = channel_weights[ch]
        combined_probs += packed[:-1] * weight
        margins.append(packed[-1])
    combined_probs = np.maximum(combined_probs, 1e-9)
    combined_probs /= np.sum(combined_probs)
    median_margin = float(np.median(margins)) if margins else float(combined_probs.max())
    physics_label = FAULT_CLASSES[int(np.argmax(combined_probs))]
    aggregated_probs = combined_probs
    return SampleFeature(name=name, features=aggregated, physics_probs=aggregated_probs, physics_label=physics_label, physics_margin=median_margin)


def prepare_samples(data_dir: str) -> List[SampleFeature]:
    files = sorted(f for f in os.listdir(data_dir) if f.lower().endswith(".mat"))
    samples: List[SampleFeature] = []
    for file_name in files:
        path = os.path.join(data_dir, file_name)
        try:
            channels = load_mat_file(path)
            sample = aggregate_sample_features(os.path.splitext(file_name)[0], channels)
            samples.append(sample)
        except Exception:
            continue
    return samples


def build_feature_matrix(samples: Sequence[SampleFeature]) -> Tuple[np.ndarray, List[str]]:
    keys = sorted({key for sample in samples for key in sample.features.keys()})
    matrix = np.array([[sample.features.get(key, 0.0) for key in keys] for sample in samples], dtype=float)
    return matrix, keys


def jaccard_similarity(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0
    same = sum(1 for la, lb in zip(labels_a, labels_b) if la == lb)
    union = len(labels_a) * 2 - same
    if union == 0:
        return 1.0
    return same / union


def enforce_all_classes(labels: List[str], margins: Sequence[float]) -> List[str]:
    labels = list(labels)
    counts = Counter(labels)
    missing = [cls for cls in FAULT_CLASSES if counts.get(cls, 0) == 0]
    if not missing:
        return labels
    margin_indices = np.argsort(margins)
    used_indices = set()
    for cls in missing:
        for idx in margin_indices:
            if idx in used_indices:
                continue
            current_label = labels[idx]
            if counts[current_label] <= 1:
                continue
            labels[idx] = cls
            counts[current_label] -= 1
            counts[cls] += 1
            used_indices.add(idx)
            break
    return labels


def assign_clusters_to_faults(cluster_labels: np.ndarray, physics_probs: np.ndarray) -> Dict[int, int]:
    unique_clusters = sorted(set(cluster_labels))
    num_clusters = len(unique_clusters)
    score_matrix = np.zeros((num_clusters, len(FAULT_CLASSES)))
    for idx, cluster in enumerate(unique_clusters):
        indices = np.where(cluster_labels == cluster)[0]
        if indices.size == 0:
            continue
        avg_probs = np.mean(physics_probs[indices], axis=0)
        score_matrix[idx] = avg_probs
    # cost matrix is negative score for maximization
    cost = -score_matrix
    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {unique_clusters[row]: int(col) for row, col in zip(row_ind, col_ind)}
    # Assign remaining clusters greedily if any
    unused = set(range(len(FAULT_CLASSES))) - set(mapping.values())
    for cluster in unique_clusters:
        if cluster in mapping:
            continue
        best_class = max(unused, key=lambda cls: score_matrix[unique_clusters.index(cluster)][cls]) if unused else int(np.argmax(score_matrix[unique_clusters.index(cluster)]))
        mapping[cluster] = best_class
        if best_class in unused:
            unused.remove(best_class)
    return mapping


def compute_metrics(
    feature_subset: np.ndarray,
    labels: Sequence[str],
    physics_labels: Sequence[str],
) -> Tuple[float, float, bool, float]:
    encoded = np.array([FAULT_CLASSES.index(lbl) for lbl in labels])
    cluster_metric_value = -1.0
    db_value = math.inf
    separation_ok = False
    unique, counts = np.unique(encoded, return_counts=True)
    if np.all(counts > 1):
        try:
            cluster_metric_value = float(silhouette_score(feature_subset, encoded, metric="euclidean"))
            if cluster_metric_value >= 0.35:
                separation_ok = True
        except Exception:
            cluster_metric_value = -1.0
    if not separation_ok:
        try:
            db_value = float(davies_bouldin_score(feature_subset, encoded))
            if db_value <= 1.2:
                separation_ok = True
        except Exception:
            db_value = math.inf
    physics_match = np.mean([1.0 if a == b else 0.0 for a, b in zip(labels, physics_labels)])
    return cluster_metric_value, db_value, separation_ok, float(physics_match)


def iterative_classification(samples: Sequence[SampleFeature], rng: np.random.Generator) -> Tuple[List[str], Dict[str, float], int, bool]:
    feature_matrix, _ = build_feature_matrix(samples)
    physics_probs = np.vstack([sample.physics_probs for sample in samples])
    physics_labels = [sample.physics_label for sample in samples]
    history: List[List[str]] = []
    best_record = None
    best_stability = -1.0
    converged = False
    last_iteration = 0

    for iteration in range(1, MAX_ITERS + 1):
        last_iteration = iteration
        feature_indices = rng.choice(
            feature_matrix.shape[1],
            size=int(max(4, rng.integers(low=4, high=feature_matrix.shape[1] + 1))),
            replace=False,
        )
        subset = feature_matrix[:, feature_indices]
        if rng.random() < 0.7:
            scaler = StandardScaler()
            subset_scaled = scaler.fit_transform(subset)
        else:
            min_vals = subset.min(axis=0)
            max_vals = subset.max(axis=0)
            denom = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
            subset_scaled = (subset - min_vals) / denom

        cluster_method = rng.choice(["kmeans", "spectral"])
        if cluster_method == "kmeans":
            random_state = int(rng.integers(0, 10_000_000))
            n_init = int(rng.integers(10, 16))
            kmeans = KMeans(n_clusters=4, random_state=random_state, n_init=n_init)
            cluster_labels = kmeans.fit_predict(subset_scaled)
        else:
            random_state = int(rng.integers(0, 10_000_000))
            gamma = float(rng.uniform(0.1, 1.5))
            spectral = SpectralClustering(
                n_clusters=4,
                affinity="rbf",
                gamma=gamma,
                random_state=random_state,
                assign_labels="kmeans",
            )
            cluster_labels = spectral.fit_predict(subset_scaled)

        cluster_to_fault = assign_clusters_to_faults(cluster_labels, physics_probs)
        combined_scores = physics_probs.copy()
        cluster_boost = float(0.2 + 0.3 * rng.random())
        for idx, cluster in enumerate(cluster_labels):
            class_idx = cluster_to_fault.get(cluster, 0)
            combined_scores[idx, class_idx] += cluster_boost
        combined_scores = np.maximum(combined_scores, 1e-9)
        combined_scores /= np.sum(combined_scores, axis=1, keepdims=True)
        margins = np.partition(combined_scores, -2, axis=1)
        top = combined_scores.max(axis=1)
        second = margins[:, -2]
        decision_margins = top - second
        labels = [FAULT_CLASSES[int(np.argmax(row))] for row in combined_scores]
        labels = enforce_all_classes(labels, decision_margins)

        subset_for_metrics = subset_scaled
        silhouette_value, db_value, separation_ok, physics_match = compute_metrics(
            subset_for_metrics, labels, physics_labels
        )

        history.append(labels)
        if len(history) > HISTORY_LENGTH:
            history.pop(0)

        recent_window = min(len(history) - 1, RECENT_WINDOW)
        if recent_window <= 0:
            stability_recent = 1.0
        else:
            current = history[-1]
            comparisons = history[-recent_window - 1 : -1]
            jaccs = [jaccard_similarity(current, prev) for prev in comparisons]
            stability_recent = float(np.mean(jaccs)) if jaccs else 1.0

        if stability_recent > best_stability:
            best_stability = stability_recent
            best_record = {
                "labels": list(labels),
                "silhouette": silhouette_value,
                "db": db_value,
                "physics_match": physics_match,
                "iterations": last_iteration,
                "stability": stability_recent,
            }

        if (
            stability_recent >= 0.95
            and len(history) - 1 >= RECENT_WINDOW
            and separation_ok
            and physics_match >= 0.75
        ):
            converged = True
            best_record = {
                "labels": list(labels),
                "silhouette": silhouette_value,
                "db": db_value,
                "physics_match": physics_match,
                "iterations": last_iteration,
                "stability": stability_recent,
            }
            break

    if best_record is None:
        raise RuntimeError("Failed to obtain any classification result")

    total_iterations = last_iteration if last_iteration else len(history)
    return best_record["labels"], best_record, total_iterations, converged


def main() -> None:
    samples = prepare_samples(DATA_DIR)
    if len(samples) != 16:
        print("# warning: expected 16 samples, got {}".format(len(samples)), file=sys.stderr)
    seed = int(time.time() * 1000) % (2 ** 32 - 1)
    rng = np.random.default_rng(seed)
    labels, info, iterations, converged = iterative_classification(samples, rng)
    name_to_label = {sample.name: label for sample, label in zip(samples, labels)}
    warning_line = None
    if not converged:
        warning_line = "# warning: reached max iterations without full convergence"
    for sample in sorted(samples, key=lambda s: s.name):
        print(f"{sample.name},{name_to_label.get(sample.name, 'N')}")
    silhouette_value = info.get("silhouette", -1.0)
    if warning_line:
        print(warning_line)
    print(
        "# seed={} iters={} stability={:.4f} silhouette={:.4f} db={:.4f} physics_consistency={:.4f}".format(
            seed,
            iterations,
            info.get("stability", 0.0),
            silhouette_value,
            info.get("db", math.inf),
            info.get("physics_match", 0.0),
        )
    )


if __name__ == "__main__":
    main()
