import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.train_with_tracking import train_with_tracking


def test_training_with_tracking_returns_payload():
    payload = train_with_tracking()
    assert "metrics" in payload
    assert payload["metrics"]["accuracy"] > 0.7


def test_training_with_tracking_writes_artifacts(tmp_path):
    payload = train_with_tracking(artifact_dir=str(tmp_path))
    assert "metrics" in payload
    assert (tmp_path / "metrics_summary.json").exists()
    assert (tmp_path / "metrics_plot.png").exists()
    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "roc_curve.png").exists()
