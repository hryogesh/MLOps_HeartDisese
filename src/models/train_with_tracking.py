from __future__ import annotations

from src.models.train_model import ModelTrainer
from src.monitoring.mlflow_tracking import configure_mlflow, log_training_run


def train_with_tracking(artifact_dir: str | None = None) -> dict:
    configure_mlflow()
    trainer = ModelTrainer()
    payload = trainer.train()
    log_training_run(
        params={"model": "logistic_regression"},
        metrics=payload["metrics"],
        model=payload["model"],
        artifact_dir=artifact_dir or "artifacts/mlflow",
        y_true=payload["evaluation"]["y_true"],
        y_pred=payload["evaluation"]["y_pred"],
        y_prob=payload["evaluation"]["y_prob"],
    )
    return payload


if __name__ == "__main__":
    train_with_tracking()
