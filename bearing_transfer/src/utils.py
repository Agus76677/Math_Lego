"""Utility helpers for the bearing transfer project."""
from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for CPU only envs
    torch = None  # type: ignore


SEED = 2025


@dataclass
class Timer:
    """Context manager for measuring execution time."""

    name: str
    logger: Optional[logging.Logger] = None

    def __enter__(self) -> "Timer":
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        elapsed = time.time() - self.start
        msg = f"{self.name} took {elapsed:.2f} s"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)


def set_seed(seed: int = SEED) -> None:
    """Set random seed for numpy, random and torch (if available)."""

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no branch - depends on HW
            torch.cuda.manual_seed_all(seed)


def detect_device() -> str:
    """Return preferred computation device."""

    if torch is not None and torch.cuda.is_available():  # pragma: no cover - GPU specific
        return "cuda"
    return "cpu"


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    """Create directory if missing and return Path."""

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict[str, Any], path: os.PathLike[str] | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: os.PathLike[str] | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def configure_logging(log_path: os.PathLike[str] | str) -> logging.Logger:
    ensure_dir(Path(log_path).parent)
    logger = logging.getLogger("bearing_transfer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


@contextlib.contextmanager
def numpy_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def chunk_iterable(data: Iterable[Any], chunk_size: int):
    chunk = []
    for item in data:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def format_insight(lines: Iterable[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)
