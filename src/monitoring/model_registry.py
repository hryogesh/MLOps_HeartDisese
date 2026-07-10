from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict

import joblib


def register_model(model_payload: Dict[str, Any], artifact_path: str | None = None) -> str:
    """Create a lightweight local model registry entry for the trained model."""
    artifact_path = Path(artifact_path or "artifacts")
    artifact_path.mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())[:8]
    artifact_file = artifact_path / f"model_{run_id}.joblib"
    joblib.dump(model_payload, artifact_file)
    return run_id
