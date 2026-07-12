from src.exception import CustomException
from src.logger import get_logger
from src.models.train_model import ModelTrainer
from src.monitoring.mlflow_tracking import configure_mlflow, log_training_run

logger = get_logger(__name__)


def main():
    try:
        configure_mlflow()
        trainer = ModelTrainer()
        payload = trainer.train()
        log_training_run(
            params={"model": "logistic_regression"},
            metrics=payload["metrics"],
            model=payload["model"],
            y_true=payload["evaluation"]["y_true"],
            y_pred=payload["evaluation"]["y_pred"],
            y_prob=payload["evaluation"]["y_prob"],
            preprocessor=payload["preprocessor"],
            feature_columns=payload["feature_columns"],
        )
        logger.info("Training completed successfully")
        return payload
    except Exception as exc:
        logger.exception("Training failed")
        raise CustomException(str(exc)) from exc


if __name__ == "__main__":
    main()
