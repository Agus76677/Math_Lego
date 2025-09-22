import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import models_source


def test_cross_validate_model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = np.array([0, 1, 2] * 20)[:60]
    result = models_source.cross_validate_model("LightGBM", X, y, n_splits=3)
    assert result.metrics["macro_f1"] <= 1.0
    assert len(result.per_fold) == 3
