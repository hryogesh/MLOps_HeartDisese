from src.models.train_model import ModelTrainer


if __name__ == "__main__":
    trainer = ModelTrainer()
    payload = trainer.train()
    print(payload["metrics"])
