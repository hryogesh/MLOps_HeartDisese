from __future__ import annotations

import joblib
from pathlib import Path
from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from src.data.load_data import load_dataset
from src.features.preprocess import prepare_features


class ModelTrainer:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path or "models/model.joblib")
        self.model = LogisticRegression(max_iter=1000)

    def train(self):
        df = load_dataset(path="data/heart_disease.csv")
        prepared = prepare_features(df)
        self.model.fit(prepared["X_train"], prepared["y_train"])

        predictions = self.model.predict(prepared["X_test"])
        accuracy = accuracy_score(prepared["y_test"], predictions)
        probabilities = self.model.predict_proba(prepared["X_test"])[:, 1]
        report = classification_report(prepared["y_test"], predictions, output_dict=True)
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]
        f1_score = report["weighted avg"]["f1-score"]

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "preprocessor": prepared["preprocessor"],
            "feature_columns": [
                "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
                "exang", "oldpeak", "slope", "ca", "thal"
            ],
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "classification_report": report,
            },
            "evaluation": {
                "y_true": prepared["y_test"],
                "y_pred": predictions,
                "y_prob": probabilities,
            },
        }
        joblib.dump(payload, self.model_path)
        return payload


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()
