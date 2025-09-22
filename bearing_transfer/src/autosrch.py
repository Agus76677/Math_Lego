"""Automatic exploration of the method pool."""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .config import DEFAULT_SEARCH_SPACE
from .utils import ensure_dir, numpy_seed


class MethodExplorer:
    def __init__(self, search_space=DEFAULT_SEARCH_SPACE, max_trials: int = 5) -> None:
        self.space = search_space
        self.max_trials = max_trials
        self.trials: List[Dict] = []

    def sample_configs(self) -> List[Dict[str, object]]:
        keys = []
        values = []
        for section in [self.space.preprocessing, self.space.features, self.space.models]:
            for k, v in section.items():
                keys.append(k)
                values.append(v)
        combos = list(itertools.product(*values))
        configs = []
        with numpy_seed(2025):
            chosen = np.random.choice(len(combos), size=min(self.max_trials, len(combos)), replace=False)
        for idx in chosen:
            config = dict(zip(keys, combos[idx]))
            configs.append(config)
        return configs

    def record_trial(self, config: Dict[str, object], metrics: Dict[str, float], out_path: Path) -> None:
        entry = {"config": config, "metrics": metrics}
        self.trials.append(entry)
        ensure_dir(out_path.parent)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(self.trials, f, ensure_ascii=False, indent=2)

    def best_trial(self, metric: str = "macro_f1") -> Dict[str, object]:
        if not self.trials:
            return {}
        best = max(self.trials, key=lambda t: t["metrics"].get(metric, 0))
        return best
