from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from mlflow.sklearn import log_model
from sklearn.metrics import confusion_matrix, roc_curve

from src.logger import get_logger

logger = get_logger(__name__)


def configure_mlflow(tracking_uri: str | None = None) -> None:
    """Configure MLflow for local experiment tracking."""
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("heart-disease-mlops")


def log_training_run(params: dict, metrics: dict, model, artifact_dir: str | None = None, y_true=None, y_pred=None, y_prob=None) -> None:
    """Log model parameters, metrics, and artifact to MLflow."""
    artifact_dir = Path(artifact_dir or "artifacts/mlflow")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    scalar_metrics = {
        key: value for key, value in metrics.items() if isinstance(value, (int, float))
    }

    summary_path = artifact_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(scalar_metrics, handle, indent=2)

    plot_path = artifact_dir / "metrics_plot.png"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(scalar_metrics.keys()), list(scalar_metrics.values()))
    ax.set_title("Training Metrics")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    confusion_path = artifact_dir / "confusion_matrix.png"
    if y_true is not None and y_pred is not None:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        fig.savefig(confusion_path)
        plt.close(fig)
    else:
        confusion_path.touch(exist_ok=True)

    roc_path = artifact_dir / "roc_curve.png"
    if y_true is not None and y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label="ROC curve")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        fig.tight_layout()
        fig.savefig(roc_path)
        plt.close(fig)
    else:
        roc_path.touch(exist_ok=True)

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(scalar_metrics)
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(plot_path))
        mlflow.log_artifact(str(confusion_path))
        mlflow.log_artifact(str(roc_path))
        log_model(model, artifact_path="model")
        logger.info("MLflow run logged: %s", run.info.run_id)
