import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def create_mock_dataset(root: Path) -> None:
    source_dir = root / "Y_datasets"
    target_dir = root / "MBY_datasets"
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    fs = 12000
    t = np.linspace(0, 2, 2 * fs, endpoint=False)
    sig1 = np.sin(2 * np.pi * 200 * t)
    sig2 = np.sin(2 * np.pi * 400 * t)
    savemat(source_dir / "inner_fault.mat", {"DE_time": sig1, "RPM": np.array([[1797]])})
    savemat(source_dir / "outer_fault.mat", {"DE_time": sig2, "RPM": np.array([[1797]])})
    savemat(target_dir / "A.mat", {"DE_time": sig1, "RPM": np.array([[600]])})
    savemat(target_dir / "B.mat", {"DE_time": sig2, "RPM": np.array([[600]])})


def run_script(script: str, args: list[str], tmpdir: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    cmd = [sys.executable, str(SCRIPTS / script)] + args
    subprocess.run(cmd, check=True, env=env, cwd=ROOT)


def test_full_pipeline(tmp_path):
    data_root = tmp_path / "data"
    create_mock_dataset(data_root)
    out_dir = ROOT / "outputs"
    if out_dir.exists():
        for item in out_dir.glob("**/*"):
            if item.is_file():
                item.unlink()
    run_script("prepare_features.py", ["--data_root", str(data_root)], tmp_path)
    assert (ROOT / "outputs" / "artifacts" / "source_features.csv").exists()
    run_script("train_source.py", [], tmp_path)
    assert (ROOT / "outputs" / "artifacts" / "best_source_model.pkl").exists()
    run_script("transfer_and_predict.py", [], tmp_path)
    run_script("explainability.py", [], tmp_path)
    run_script("make_report.py", [], tmp_path)
    assert (ROOT / "reports" / "report.md").exists()
