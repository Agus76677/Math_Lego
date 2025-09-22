import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import dsp_features
from src.io_loader import DEFAULT_GEOMETRY, SegmentRecord


def test_feature_vector_shapes():
    fs = 12000
    t = np.linspace(0, 1, fs, endpoint=False)
    signal = np.sin(2 * np.pi * 200 * t) + 0.3 * np.random.randn(fs)
    record = SegmentRecord(signal=signal, fs=fs, rpm=1800, position="DE", fault_type="IR", file="mock.mat", segment_id=0, geometry=DEFAULT_GEOMETRY, missing_mask={"DE": False, "FE": True, "BA": True})
    phys = {"BPFI": 100, "BPFO": 150, "BSF": 200, "FTF": 50}
    feats = dsp_features.build_feature_vector(record.signal, record.fs, record.rpm, phys)
    assert len(feats) > 10
    assert not np.isnan(list(feats.values())).any()


def test_band_selection():
    fs = 12000
    t = np.linspace(0, 1, fs, endpoint=False)
    signal = np.sin(2 * np.pi * 3000 * t)
    band = dsp_features.select_band_via_sk(signal, fs)
    assert band.high > band.low
    assert band.score > 0
