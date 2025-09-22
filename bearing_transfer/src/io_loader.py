"""Loading utilities for bearing vibration .mat files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .config import DEFAULT_WINDOW
from .utils import detect_fs_tag, list_mat_files, rpm_to_fr


CHANNEL_MAP = {
    "DE_time": "DE",
    "FE_time": "FE",
    "BA_time": "BA",
    "XDE_time": "DE",
    "XFE_time": "FE",
    "XBA_time": "BA",
}


@dataclass
class Segment:
    data: np.ndarray
    fs: float
    rpm: float | None
    channel: str
    fault_type: str
    fault_size: str | None
    file_path: Path
    segment_index: int

    @property
    def fr(self) -> float | None:
        return rpm_to_fr(self.rpm)

    @property
    def fs_tag(self) -> str:
        return detect_fs_tag(self.fs)


@dataclass
class LoadedDataset:
    segments: List[Segment]
    metadata: pd.DataFrame


def _normalize_key(key: str) -> str:
    key = key.strip()
    return CHANNEL_MAP.get(key, key)


def _extract_channel_data(mat: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    channels: Dict[str, np.ndarray] = {}
    for key, value in mat.items():
        if not isinstance(value, np.ndarray):
            continue
        norm_key = _normalize_key(key)
        if norm_key in {"DE", "FE", "BA"}:
            channels[norm_key] = np.asarray(value).squeeze()
    return channels


def _load_rpm(mat: Dict[str, np.ndarray]) -> float | None:
    for key in ("RPM", "rpm", "Motor_Speed", "rps"):
        if key in mat:
            arr = np.asarray(mat[key]).squeeze()
            if arr.size == 0:
                continue
            return float(arr.mean())
    return None


def segment_signal(signal: np.ndarray, fs: float, duration: float, overlap: float) -> Iterable[np.ndarray]:
    length = signal.shape[0]
    step = int(fs * duration * (1 - overlap))
    window = int(fs * duration)
    if step <= 0:
        step = window
    if window > length:
        yield signal
        return
    for start in range(0, length - window + 1, step):
        yield signal[start : start + window]


def load_mat_file(path: Path, window_duration: float, overlap: float) -> List[Segment]:
    mat = loadmat(path)
    fs_candidates = [key for key in mat.keys() if key.lower().startswith("fs")]
    fs = None
    if fs_candidates:
        fs_value = np.asarray(mat[fs_candidates[0]]).squeeze()
        if fs_value.size:
            fs = float(fs_value.mean())
    if fs is None:
        if "12k" in path.name.lower():
            fs = 12000.0
        elif "48k" in path.name.lower():
            fs = 48000.0
        elif "32k" in path.name.lower():
            fs = 32000.0
        else:
            fs = 12000.0
    rpm = _load_rpm(mat)
    channels = _extract_channel_data(mat)
    segments: List[Segment] = []
    for channel, data in channels.items():
        data = np.asarray(data, dtype=float)
        data = data - np.nanmean(data)
        for idx, seg in enumerate(segment_signal(data, fs, window_duration, overlap)):
            segments.append(
                Segment(
                    data=np.asarray(seg, dtype=float),
                    fs=fs,
                    rpm=rpm,
                    channel=channel,
                    fault_type=_infer_fault_type(path.name),
                    fault_size=_infer_fault_size(path.name),
                    file_path=path,
                    segment_index=idx,
                )
            )
    return segments


def _infer_fault_type(name: str) -> str:
    lower = name.lower()
    if "or" in lower:
        return "OR"
    if "ir" in lower:
        return "IR"
    if "b" in lower and "ball" in lower:
        return "B"
    return "N"


def _infer_fault_size(name: str) -> Optional[str]:
    tokens = [t for t in name.replace("-", "_").split("_") if t]
    for token in tokens:
        if token.endswith("mil"):
            return token
    return None


def load_directory(root: Path | str, window_duration: float | None = None, overlap: float | None = None) -> LoadedDataset:
    root = Path(root)
    window_duration = window_duration or DEFAULT_WINDOW.duration_s
    overlap = overlap if overlap is not None else DEFAULT_WINDOW.overlap
    all_segments: List[Segment] = []
    metadata_rows: List[Dict[str, object]] = []
    for path in list_mat_files(root):
        segments = load_mat_file(path, window_duration, overlap)
        for seg in segments:
            all_segments.append(seg)
            metadata_rows.append(
                {
                    "file": path.name,
                    "path": str(path),
                    "channel": seg.channel,
                    "fault_type": seg.fault_type,
                    "fault_size": seg.fault_size,
                    "fs": seg.fs,
                    "rpm": seg.rpm,
                    "segment_index": seg.segment_index,
                    "fs_tag": seg.fs_tag,
                }
            )
    metadata = pd.DataFrame(metadata_rows)
    if not metadata.empty:
        metadata["fr"] = metadata["rpm"].apply(rpm_to_fr)
    return LoadedDataset(all_segments, metadata)


def save_segments_metadata(dataset: LoadedDataset, path: Path | str) -> None:
    metadata = dataset.metadata.copy()
    if metadata.empty:
        metadata.to_csv(path, index=False)
        return
    metadata["num_segments"] = metadata.groupby(["file", "channel"])["segment_index"].transform("count")
    metadata.drop_duplicates(subset=["file", "channel"], inplace=True)
    metadata.to_csv(path, index=False)
