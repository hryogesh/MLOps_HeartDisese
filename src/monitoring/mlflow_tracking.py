from __future__ import annotations

import os
from pathlib import Path

import mlflow
from mlflow.sklearn import log_model

from src.logger import get_logger

logger = get_logger(__name__)


def configure_mlflow(tracking_uri: str | None = None) -> None:
    """Configure MLflow for local experiment tracking."""
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("heart-disease-mlops")


def log_training_run(params: dict, metrics: dict, model, artifact_dir: str | None = None) -> None:
    """Log model parameters, metrics, and artifact to MLflow."""
    artifact_dir = Path(artifact_dir or "artifacts/mlflow")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    scalar_metrics = {
        key: value for key, value in metrics.items() if isinstance(value, (int, float))
    }

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(scalar_metrics)
        log_model(model, artifact_path="model")
        logger.info("MLflow run logged: %s", run.info.run_id)
