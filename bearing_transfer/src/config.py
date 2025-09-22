"""Global configuration and method pool definitions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .utils import SEED


@dataclass
class FeatureConfig:
    window_sec: float = 1.0
    overlap: float = 0.5
    band_strategy: str = "auto_sk_tk"
    envelope: bool = True
    order_delta: float = 0.1
    include_order_features: bool = True
    include_hz_features: bool = True
    add_conditionals: Tuple[str, ...] = ("RPM", "fs_tag")


@dataclass
class ModelConfig:
    models: Tuple[str, ...] = ("LightGBM", "XGBoost", "MLP", "cnn1d")
    metric: str = "macro_f1"
    n_folds: int = 5
    repeats: int = 1


@dataclass
class TransferConfig:
    align_methods: Tuple[str, ...] = ("CORAL", "MMD")
    sample_weight: bool = True
    pseudo_label: bool = True
    pseudo_thresholds: Tuple[float, float] = (0.95, 0.9)
    consistency_reg: bool = True
    lambda_candidates: Tuple[float, ...] = (0.1, 0.3, 1.0)


@dataclass
class SearchSpace:
    preprocessing: Dict[str, List] = field(
        default_factory=lambda: {
            "window_sec": [1.0],
            "overlap": [0.5, 0.75],
            "band_strategy": ["auto_sk_tk", "fixed"],
            "envelope": [True, False],
        }
    )
    features: Dict[str, List] = field(
        default_factory=lambda: {
            "order_delta": [0.05, 0.1, 0.15],
            "include_order_features": [True, False],
            "include_hz_features": [True],
        }
    )
    models: Dict[str, List] = field(
        default_factory=lambda: {
            "model": ["LightGBM", "XGBoost", "MLP"],
        }
    )
    transfer: Dict[str, List] = field(
        default_factory=lambda: {
            "align": ["None", "CORAL", "MMD"],
            "sample_weight": [True, False],
            "pseudo_label": [True, False],
        }
    )


DEFAULT_FEATURE_CONFIG = FeatureConfig()
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRANSFER_CONFIG = TransferConfig()
DEFAULT_SEARCH_SPACE = SearchSpace()
DEFAULT_SEED = SEED
