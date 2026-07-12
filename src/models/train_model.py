from __future__ import annotations

import joblib
from pathlib import Path
from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from src.data.load_data import load_dataset
from src.features.preprocess import prepare_features


class ModelTrainer:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path or "models/model.joblib")
        self.model = LogisticRegression(max_iter=1000)

    def train(self):
        df = load_dataset(path="data/heart_disease.csv")
        prepared = prepare_features(df)
        full_pipeline = Pipeline(
            steps=[
                ("preprocessor", prepared["preprocessor"]),
                ("classifier", self.model),
            ]
        )
        full_pipeline.fit(prepared["X_train_raw"], prepared["y_train"])

        predictions = full_pipeline.predict(prepared["X_test_raw"])
        accuracy = accuracy_score(prepared["y_test"], predictions)
        probabilities = full_pipeline.predict_proba(prepared["X_test_raw"])[:, 1]
        report = classification_report(prepared["y_test"], predictions, output_dict=True)
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]
        f1_score = report["weighted avg"]["f1-score"]

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "pipeline": full_pipeline,
            "preprocessor": prepared["preprocessor"],
            "feature_columns": prepared["feature_columns"],
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
