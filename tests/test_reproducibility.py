import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.train_with_tracking import train_with_tracking


def test_training_with_tracking_returns_payload():
    payload = train_with_tracking()
    assert "metrics" in payload
    assert payload["metrics"]["accuracy"] > 0.7
