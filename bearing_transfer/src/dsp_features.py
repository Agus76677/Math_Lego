"""Signal processing feature extraction utilities."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert, stft, welch
from scipy.stats import kurtosis

from .utils import numpy_seed


@dataclass
class BandSelection:
    low: float
    high: float
    score: float


DEFAULT_BANDS = [
    (200, 1200),
    (500, 2500),
    (1000, 4000),
    (2000, 6000),
    (3000, 9000),
]


def _butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999)
    if high <= low:
        high = min(low + 0.01, 0.999)
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(signal: np.ndarray, fs: float, band: Tuple[float, float]) -> np.ndarray:
    b, a = _butter_bandpass(band[0], band[1], fs)
    return filtfilt(b, a, signal)


def compute_spectral_kurtosis(signal: np.ndarray, fs: float, nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg)
    magnitude = np.abs(Zxx) ** 2
    sk = kurtosis(magnitude, axis=1, fisher=False, bias=False)
    return f, sk


def select_band_via_sk(signal: np.ndarray, fs: float, candidate_bands: Iterable[Tuple[float, float]] = DEFAULT_BANDS) -> BandSelection:
    best = BandSelection(low=0.0, high=fs / 2, score=-np.inf)
    for low, high in candidate_bands:
        filtered = bandpass_filter(signal, fs, (low, high))
        f, sk = compute_spectral_kurtosis(filtered, fs)
        score = float(np.nanmax(sk))
        if score > best.score:
            best = BandSelection(low=low, high=high, score=score)
    return best


def hilbert_envelope(signal: np.ndarray) -> np.ndarray:
    analytic = hilbert(signal)
    return np.abs(analytic)


def compute_time_features(signal: np.ndarray) -> Dict[str, float]:
    feats = {
        "rms": float(np.sqrt(np.mean(signal ** 2))),
        "kurtosis": float(kurtosis(signal, fisher=False, bias=False)),
        "skew": float(np.mean(((signal - np.mean(signal)) / (np.std(signal) + 1e-12)) ** 3)),
        "crest_factor": float(np.max(np.abs(signal)) / (np.sqrt(np.mean(signal ** 2)) + 1e-12)),
        "impulse_factor": float(np.max(np.abs(signal)) / (np.mean(np.abs(signal)) + 1e-12)),
    }
    return feats


def compute_band_energy(psd_freq: np.ndarray, psd_power: np.ndarray, bands: Iterable[Tuple[float, float]]) -> Dict[str, float]:
    features = {}
    for idx, (low, high) in enumerate(bands):
        mask = (psd_freq >= low) & (psd_freq < high)
        energy = float(np.trapz(psd_power[mask], psd_freq[mask])) if mask.any() else 0.0
        features[f"band_energy_{idx}_{int(low)}_{int(high)}"] = energy
    return features


def spectral_peaks(freq: np.ndarray, spectrum: np.ndarray, peaks: Iterable[float], delta: float) -> Dict[str, float]:
    features = {}
    for pk in peaks:
        if pk <= 0:
            continue
        band = (pk * (1 - delta), pk * (1 + delta))
        mask = (freq >= band[0]) & (freq <= band[1])
        if mask.any():
            idx = np.argmax(spectrum[mask])
            sub_freq = freq[mask]
            sub_spec = spectrum[mask]
            peak_freq = float(sub_freq[idx])
            peak_amp = float(sub_spec[idx])
            snr = peak_amp / (float(np.mean(sub_spec) + 1e-6))
            features[f"peak_amp_{pk:.1f}"] = peak_amp
            features[f"peak_snr_{pk:.1f}"] = snr
            features[f"peak_dev_{pk:.1f}"] = peak_freq - pk
        else:
            features[f"peak_amp_{pk:.1f}"] = 0.0
            features[f"peak_snr_{pk:.1f}"] = 0.0
            features[f"peak_dev_{pk:.1f}"] = 0.0
    return features


def compute_frequency_features(signal: np.ndarray, fs: float, rpm: float, phys_freqs: Dict[str, float], delta: float = 0.1) -> Dict[str, float]:
    f, pxx = welch(signal, fs=fs, nperseg=min(1024, len(signal)))
    features = compute_band_energy(
        f,
        pxx,
        bands=[(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, fs / 2)],
    )
    peaks = [v for v in phys_freqs.values() if np.isfinite(v)]
    features.update(spectral_peaks(f, pxx, peaks, delta))
    if rpm > 0:
        order_freq = rpm / 60.0
        orders = [pk / order_freq if order_freq else 0 for pk in peaks]
        order_freqs = {f"order_{name}": val for name, val in zip(phys_freqs.keys(), orders)}
        features.update(order_freqs)
    return features


def build_feature_vector(
    signal: np.ndarray,
    fs: float,
    rpm: float,
    phys_freqs_hz: Dict[str, float],
    delta: float = 0.1,
    apply_envelope: bool = True,
    band: Tuple[float, float] | None = None,
) -> Dict[str, float]:
    base_signal = signal - np.mean(signal)
    if band is not None:
        base_signal = bandpass_filter(base_signal, fs, band)
    envelope_signal = hilbert_envelope(base_signal) if apply_envelope else base_signal
    time_feats = {f"time_{k}": v for k, v in compute_time_features(base_signal).items()}
    env_feats = {f"env_{k}": v for k, v in compute_time_features(envelope_signal).items()}
    freq_feats = {f"freq_{k}": v for k, v in compute_frequency_features(envelope_signal, fs, rpm, phys_freqs_hz, delta).items()}
    features = {}
    features.update(time_feats)
    features.update(env_feats)
    features.update(freq_feats)
    return features


def features_dataframe(records, phys_freqs_lookup: Dict[Tuple[str, str], Dict[str, float]], delta: float = 0.1, apply_envelope: bool = True) -> pd.DataFrame:
    rows = []
    for rec in records:
        key = (rec.position, rec.fault_type)
        phys = phys_freqs_lookup.get(key, phys_freqs_lookup.get((rec.position, "generic"), {}))
        feats = build_feature_vector(
            signal=rec.signal,
            fs=rec.fs,
            rpm=rec.rpm,
            phys_freqs_hz=phys,
            delta=delta,
            apply_envelope=apply_envelope,
        )
        feats.update({
            "file": rec.file,
            "segment_id": rec.segment_id,
            "RPM": rec.rpm,
            "fs": rec.fs,
            "position": rec.position,
            "fault_type": rec.fault_type,
        })
        rows.append(feats)
    return pd.DataFrame(rows)
