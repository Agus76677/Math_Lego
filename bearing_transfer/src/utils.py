"""Utility helpers for the bearing transfer project."""
from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
try:  # optional dependency
    import torch
except ImportError:  # pragma: no cover - optional fallback
    class _CudaStub:
        def is_available(self) -> bool:
            return False

        def manual_seed_all(self, seed: int) -> None:
            pass

    class _TorchStub:
        cuda = _CudaStub()

        @staticmethod
        def manual_seed(seed: int) -> None:
            pass

        @staticmethod
        def manual_seed_all(seed: int) -> None:
            pass

        @staticmethod
        def device(name: str) -> str:
            return "cpu"

    torch = _TorchStub()  # type: ignore


DEFAULT_SEED = 2025


@dataclass
class TimerRecord:
    name: str
    elapsed: float


class StructuredLogger:
    """Simple JSON logger that also prints formatted messages."""

    def __init__(self, log_dir: Path, name: str = "bearing_transfer") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_path = self.log_dir / f"{name}_{timestamp}.jsonl"
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    def info(self, message: str, **kwargs: Any) -> None:
        payload = {"level": "info", "message": message, **kwargs, "time": time.time()}
        logging.info(message + (" " + json.dumps(kwargs, ensure_ascii=False) if kwargs else ""))
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def warning(self, message: str, **kwargs: Any) -> None:
        payload = {"level": "warning", "message": message, **kwargs, "time": time.time()}
        logging.warning(message + (" " + json.dumps(kwargs, ensure_ascii=False) if kwargs else ""))
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
        return torch.device("cuda") if hasattr(torch, "device") else "cuda"  # type: ignore[return-value]
    return torch.device("cpu") if hasattr(torch, "device") else "cpu"  # type: ignore[return-value]


@contextmanager
def time_block(name: str, logger: Optional[StructuredLogger] = None):
    start = time.time()
    yield
    elapsed = time.time() - start
    if logger is not None:
        logger.info("timer", name=name, elapsed=elapsed)


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_numpy(arr: np.ndarray, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def list_mat_files(root: Path | str) -> Iterable[Path]:
    root = Path(root)
    for ext in ("*.mat", "*.MAT"):
        yield from root.glob(ext)


def write_dataframe(df, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def safe_divide(a: float, b: float) -> float:
    if b == 0:
        return float("nan")
    return a / b


def describe_array(arr: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }


def detect_fs_tag(fs: float) -> str:
    if fs <= 13000:
        return "12k"
    if fs <= 20000:
        return "20k"
    if fs <= 33000:
        return "32k"
    if fs <= 50000:
        return "48k"
    return "high"


def default_geometry() -> Dict[str, float]:
    # CWRU-style defaults (inch -> mm conversions omitted as ratios suffice)
    return {
        "Z": 8,
        "bd": 0.3126,
        "pd": 1.537,
        "contact_angle_deg": 0.0,
    }


def load_geometry(metadata: Dict[str, Any]) -> Dict[str, float]:
    geom = default_geometry().copy()
    for key in ("Z", "bd", "pd", "contact_angle_deg"):
        if key in metadata and metadata[key] not in (None, "", float("nan")):
            geom[key] = float(metadata[key])
    return geom


def rpm_to_fr(rpm: float | None) -> float | None:
    if rpm is None or np.isnan(rpm):
        return None
    return rpm / 60.0


def order_from_freq(freq: np.ndarray, fr: float | None) -> np.ndarray:
    if fr is None or fr == 0:
        return np.full_like(freq, np.nan, dtype=float)
    return freq / fr


def ensure_report(path: Path | str) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# 轴承迁移诊断报告\n\n", encoding="utf-8")
    return path
