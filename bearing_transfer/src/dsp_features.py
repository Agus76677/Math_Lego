"""Signal processing feature extraction."""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import butter, filtfilt, hilbert, welch


def _butter_bandpass(low: float, high: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    low = low / nyq if low else 0.0
    high = high / nyq if high else 1.0
    b, a = butter(order, [max(low, 1e-5), min(high, 0.9999)], btype="band")
    return b, a


def bandpass_filter(signal: ArrayLike, fs: float, low: float | None, high: float | None) -> np.ndarray:
    data = np.asarray(signal, dtype=float)
    if low is None and high is None:
        return data
    b, a = _butter_bandpass(low or 1.0, high or (0.45 * fs), fs)
    return filtfilt(b, a, data)


def spectral_kurtosis(signal: ArrayLike, fs: float, window: int = 1024, step: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(signal, dtype=float)
    if step is None:
        step = window // 2
    num = max(len(data) - window, 1)
    kurtosis_values = []
    freqs = np.fft.rfftfreq(window, d=1 / fs)
    for start in range(0, len(data) - window + 1, step):
        segment = data[start : start + window]
        spectrum = np.fft.rfft(segment * np.hanning(window))
        power = np.abs(spectrum) ** 2
        kurtosis_values.append(power)
    power_stack = np.vstack(kurtosis_values) if kurtosis_values else np.zeros((1, len(freqs)))
    sk = np.mean(((power_stack - power_stack.mean(axis=0)) ** 4), axis=0)
    sk /= (np.var(power_stack, axis=0) ** 2 + 1e-12)
    return freqs, sk


def tuned_band_from_sk(freqs: np.ndarray, sk: np.ndarray, min_bandwidth: float = 200.0) -> tuple[float, float]:
    if freqs.size == 0:
        return 0.0, min_bandwidth
    idx = np.argmax(sk)
    center = freqs[idx]
    low = max(center - min_bandwidth / 2, freqs[0])
    high = min(center + min_bandwidth / 2, freqs[-1])
    if high <= low:
        high = min(freqs[-1], low + min_bandwidth)
    return low, high


def compute_envelope(signal: ArrayLike) -> np.ndarray:
    analytic = hilbert(np.asarray(signal, dtype=float))
    return np.abs(analytic)


def compute_welch(signal: ArrayLike, fs: float, nperseg: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    freq, power = welch(signal, fs=fs, nperseg=min(nperseg, len(signal)))
    return freq, power


def time_domain_stats(signal: ArrayLike) -> dict[str, float]:
    x = np.asarray(signal, dtype=float)
    x = x - np.nanmean(x)
    rms = np.sqrt(np.nanmean(x**2))
    kurt = np.nanmean((x / (np.nanstd(x) + 1e-12)) ** 4)
    skew = np.nanmean((x / (np.nanstd(x) + 1e-12)) ** 3)
    peak = np.nanmax(np.abs(x))
    impulse = peak / (np.nanmean(np.abs(x)) + 1e-12)
    clearance = peak / (np.sqrt(np.abs(np.nanmean(np.sqrt(np.abs(x))))) + 1e-12)
    return {
        "rms": float(rms),
        "kurtosis": float(kurt),
        "skewness": float(skew),
        "peak": float(peak),
        "impulse_factor": float(impulse),
        "clearance_factor": float(clearance),
    }


def envelope_spectrum(signal: ArrayLike, fs: float, nperseg: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    env = compute_envelope(signal)
    freq, power = compute_welch(env, fs, nperseg)
    return freq, power


def aggregate_band_energy(freqs: np.ndarray, power: np.ndarray, bands: list[tuple[float, float]]) -> dict[str, float]:
    features: dict[str, float] = {}
    for low, high in bands:
        mask = (freqs >= low) & (freqs <= high)
        features[f"band_{low:.0f}_{high:.0f}"] = float(np.trapz(power[mask], freqs[mask])) if np.any(mask) else 0.0
    return features
