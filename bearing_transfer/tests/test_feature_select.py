import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import feature_select


def test_mmd_monotonicity():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(100, 3))
    Y = rng.normal(0.5, 1, size=(100, 3))
    Z = rng.normal(1.5, 1, size=(100, 3))
    mmd_xy = feature_select.compute_mmd(X, Y)
    mmd_xz = feature_select.compute_mmd(X, Z)
    assert mmd_xz > mmd_xy


def test_coral_zero_when_equal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4))
    Y = X + rng.normal(scale=1e-6, size=X.shape)
    coral = feature_select.compute_coral(X, Y)
    assert coral < 1e-2


def test_domain_gap_stats():
    rng = np.random.default_rng(2)
    df_source = pd.DataFrame(rng.normal(size=(40, 4)), columns=[f"f{i}" for i in range(4)])
    df_target = pd.DataFrame(rng.normal(loc=0.5, size=(40, 4)), columns=[f"f{i}" for i in range(4)])
    stats = feature_select.evaluate_domain_gap(df_source, df_target, list(df_source.columns))
    assert stats.mmd > 0
    assert 0.5 <= stats.domain_acc <= 1.0
