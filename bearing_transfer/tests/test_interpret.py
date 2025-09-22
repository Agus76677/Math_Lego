import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import interpret


def test_permutation_importance(tmp_path):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(100, 3)), columns=list("abc"))
    y = (X["a"] + X["b"] > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    out_path = tmp_path / "perm.png"
    info = interpret.permutation_importance_plot(model, X, y, out_path)
    assert out_path.exists()
    assert len(info) > 0
