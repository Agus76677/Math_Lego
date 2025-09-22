"""MAT data loading and segmentation utilities."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly

from .utils import SEED, ensure_dir, set_seed

MAT_KEY_MAPPING = {
    "DE_time": "DE",
    "FE_time": "FE",
    "BA_time": "BA",
    "RPM": "RPM",
}

DEFAULT_GEOMETRY = {
    "Z": 9,
    "bd": 0.2858,
    "pd": 1.245,
    "contact_angle_deg": 0.0,
}


@dataclass
class SegmentRecord:
    signal: np.ndarray
    fs: int
    rpm: float
    position: str
    fault_type: str
    file: str
    segment_id: int
    geometry: Dict[str, float]
    missing_mask: Dict[str, bool]

    def to_metadata(self) -> Dict[str, object]:
        data = {
            "file": self.file,
            "segment_id": self.segment_id,
            "fs": self.fs,
            "RPM": self.rpm,
            "position": self.position,
            "fault_type": self.fault_type,
        }
        data.update({f"missing_{k}": v for k, v in self.missing_mask.items()})
        for k, v in self.geometry.items():
            data[f"geom_{k}"] = v
        return data


class BearingDataLoader:
    """Loader for Case Western Reserve University like bearing datasets."""

    def __init__(
        self,
        source_dir: os.PathLike[str] | str,
        window_sec: float = 1.0,
        overlap: float = 0.5,
        seed: int = SEED,
        resample_to: Optional[int] = None,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.window_sec = window_sec
        self.overlap = overlap
        self.seed = seed
        self.resample_to = resample_to
        set_seed(seed)

    def discover_mat_files(self) -> List[Path]:
        mats = sorted(self.source_dir.rglob("*.mat"))
        if not mats:
            raise FileNotFoundError(f"No .mat files found under {self.source_dir}")
        return mats

    @staticmethod
    def _normalize_keys(mat_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        normalized: Dict[str, np.ndarray] = {}
        for key, value in mat_dict.items():
            if key.startswith("__"):
                continue
            base = key
            for raw, mapped in MAT_KEY_MAPPING.items():
                if raw in key:
                    base = mapped
                    break
            arr = np.asarray(value).squeeze()
            normalized[base] = arr
        return normalized

    def _segment_signal(self, signal: np.ndarray, fs: int) -> Iterable[np.ndarray]:
        win_length = int(self.window_sec * fs)
        hop = int(win_length * (1 - self.overlap))
        if hop <= 0:
            raise ValueError("overlap too high resulting in non-positive hop")
        total = len(signal)
        if total < win_length:
            yield signal
            return
        for start in range(0, total - win_length + 1, hop):
            yield signal[start : start + win_length]

    @staticmethod
    def _infer_fault_type(file_name: str) -> str:
        lower = file_name.lower()
        if "ball" in lower or "bs" in lower:
            return "B"
        if "inner" in lower or "ir" in lower:
            return "IR"
        if "outer" in lower or "or" in lower:
            return "OR"
        if "normal" in lower or "n" in lower:
            return "N"
        return "UNK"

    def load_file(self, path: os.PathLike[str] | str, position_hint: Optional[str] = None) -> List[SegmentRecord]:
        path = Path(path)
        mat_raw = loadmat(path)
        normalized = self._normalize_keys(mat_raw)
        fs = int(normalized.get("fs", mat_raw.get("fs", 12000)))
        rpm = float(normalized.get("RPM", mat_raw.get("RPM", 1797)))

        records: List[SegmentRecord] = []
        missing_mask = {k: False for k in ["DE", "FE", "BA"]}
        for position in ["DE", "FE", "BA"]:
            if position not in normalized:
                missing_mask[position] = True
                continue
            signal = normalized[position].astype(float)
            if signal.ndim != 1:
                signal = signal.reshape(-1)
            if self.resample_to and fs != self.resample_to:
                up = self.resample_to
                signal = resample_poly(signal, up, fs)
                fs = self.resample_to
            for seg_id, segment in enumerate(self._segment_signal(signal, fs)):
                records.append(
                    SegmentRecord(
                        signal=segment,
                        fs=fs,
                        rpm=rpm,
                        position=position if position_hint is None else position_hint,
                        fault_type=self._infer_fault_type(path.name),
                        file=path.name,
                        segment_id=seg_id,
                        geometry=DEFAULT_GEOMETRY.copy(),
                        missing_mask=missing_mask.copy(),
                    )
                )
        return records

    def load_all(self) -> List[SegmentRecord]:
        all_records: List[SegmentRecord] = []
        for mat_path in self.discover_mat_files():
            all_records.extend(self.load_file(mat_path))
        return all_records


def records_to_dataframe(records: List[SegmentRecord]) -> pd.DataFrame:
    return pd.DataFrame([r.to_metadata() for r in records])


def save_segments_metadata(records: List[SegmentRecord], path: os.PathLike[str] | str) -> pd.DataFrame:
    df = records_to_dataframe(records)
    ensure_dir(Path(path).parent)
    df.to_csv(path, index=False)
    return df
