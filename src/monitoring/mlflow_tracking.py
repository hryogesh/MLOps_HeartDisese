from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.sklearn import log_model
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.pipeline import Pipeline

from src.logger import get_logger

logger = get_logger(__name__)


def configure_mlflow(tracking_uri: str | None = None) -> None:
    """Configure MLflow for local experiment tracking."""
    project_root = Path(__file__).resolve().parents[2]
    default_uri = f"file:{project_root / 'mlruns'}"
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", default_uri)

    if tracking_uri.startswith("file:"):
        uri_path = Path(tracking_uri[5:])
        if not uri_path.is_absolute():
            uri_path = (project_root / uri_path).resolve()
        tracking_uri = f"file:{uri_path}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("heart-disease-mlops")


def log_training_run(
    params: dict,
    metrics: dict,
    model,
    artifact_dir: str | None = None,
    y_true=None,
    y_pred=None,
    y_prob=None,
    preprocessor=None,
    feature_columns=None,
) -> None:
    """Log model parameters, metrics, and artifact to MLflow."""
    project_root = Path(__file__).resolve().parents[2]
    artifact_dir = Path(artifact_dir or "artifacts/mlflow")
    if not artifact_dir.is_absolute():
        artifact_dir = (project_root / artifact_dir).resolve()
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

    default_feature_columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal"
    ]
    columns = feature_columns or default_feature_columns

    if preprocessor is not None:
        logged_model = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
    else:
        logged_model = model

    input_example = pd.DataFrame([ [0.0] * len(columns) ], columns=columns)
    try:
        prediction = logged_model.predict(input_example)
        signature = infer_signature(input_example, prediction)
    except Exception:
        signature = None

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(scalar_metrics)
        mlflow.log_artifacts(str(artifact_dir))
        log_model(
            logged_model,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
        )
        logger.info("MLflow run logged: %s", run.info.run_id)
