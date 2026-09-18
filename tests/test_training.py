"""Training CLI outputs and protection of completed runs."""

import csv
import hashlib
import math
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.ppo, pytest.mark.integration]


def test_training_writes_usable_outputs_and_refuses_to_overwrite(tmp_path):
    torch = pytest.importorskip("torch")
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "scripts/utilities/train_periodic_ppo.py"),
        "--num-episodes",
        "1",
        "--eval-interval",
        "1",
        "--eval-episodes",
        "1",
        "--update-epochs",
        "1",
        "--output-dir",
        str(tmp_path),
        "--run-name",
        "trial",
    ]
    result = subprocess.run(
        command, cwd=root, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    run = tmp_path / "trial"
    with (run / "metrics.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["phase"] for row in rows] == ["train", "evaluation_sampled"]
    assert float(rows[0]["optimizer_updates"]) > 0
    assert float(rows[1]["optimizer_updates"]) == 0
    for row in rows:
        assert math.isfinite(float(row["reward"]))
        assert math.isfinite(float(row["policy_loss"]))
        assert float(row["invalid_action_count"]) == 0
    for name in ("checkpoint_best.pt", "checkpoint_final.pt"):
        checkpoint = torch.load(run / name, map_location="cpu", weights_only=True)
        assert checkpoint["total_updates"] > 0
        assert all(
            torch.isfinite(value).all() for value in checkpoint["network"].values()
        )
    outputs = {
        name: hashlib.sha256((run / name).read_bytes()).hexdigest()
        for name in (
            "config.json",
            "metrics.csv",
            "checkpoint_best.pt",
            "checkpoint_final.pt",
        )
    }
    result = subprocess.run(
        command, cwd=root, capture_output=True, text=True, timeout=120
    )
    assert result.returncode != 0
    assert "fresh training directory" in result.stderr
    assert not (run / ".training.lock").exists()
    for name, digest in outputs.items():
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == digest
