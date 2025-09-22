"""Configuration and search spaces for the bearing transfer project."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class WindowConfig:
    duration_s: float = 1.0
    overlap: float = 0.5


@dataclass
class BandConfig:
    strategy: str = "auto_sk_tk"  # or "fixed"
    low: float | None = None
    high: float | None = None


@dataclass
class FeatureConfig:
    envelope: bool = True
    freq_resolution: float = 1.0
    physical_delta: float = 0.1
    include_orders: bool = True
    include_rpm: bool = True
    include_fs_tag: bool = True


@dataclass
class MethodPool:
    window_options: Sequence[WindowConfig] = (
        WindowConfig(1.0, 0.5),
        WindowConfig(1.0, 0.75),
    )
    band_options: Sequence[BandConfig] = (
        BandConfig("auto_sk_tk"),
        BandConfig("fixed", 500, 8000),
    )
    feature_options: Sequence[FeatureConfig] = (FeatureConfig(),)


DEFAULT_WINDOW = WindowConfig()
DEFAULT_BAND = BandConfig()
DEFAULT_FEATURE = FeatureConfig()


REPORT_PATH = "reports/report.md"
FIGURE_DIR = "outputs/figures"
ARTIFACT_DIR = "outputs/artifacts"
LOG_DIR = "outputs/logs"
