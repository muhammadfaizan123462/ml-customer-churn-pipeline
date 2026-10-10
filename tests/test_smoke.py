"""Smoke tests: verify the pipeline trains and the API serves (fast, CPU-only)."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def run(cmd: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def test_data_generation(tmp_path):
    out = tmp_path / "churn.csv"
    r = run([sys.executable, str(SRC / "data.py"), "--n-samples", "500",
             "--out", str(out)])
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "train.csv").exists() and (tmp_path / "test.csv").exists()


def test_training_produces_artifacts(tmp_path, monkeypatch):
    # Generate small data, then train with few search iterations.
    r = run([sys.executable, str(SRC / "data.py"), "--n-samples", "600",
             "--out", str(tmp_path / "churn.csv")])
    assert r.returncode == 0, r.stderr
    # Point train.py at the tmp data by moving it into data/.
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    (tmp_path / "train.csv").replace(data_dir / "train.csv")
    (tmp_path / "test.csv").replace(data_dir / "test.csv")
    r = run([sys.executable, str(SRC / "train.py"), "--n-iter", "2"])
    assert r.returncode == 0, r.stderr
    assert (ROOT / "artifacts" / "model.joblib").exists()
    metrics = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
    assert metrics["roc_auc"] > 0.7, metrics
