from __future__ import annotations

import joblib
from pathlib import Path
from typing import Any, List

import pandas as pd

from src.models.train_model import ModelTrainer


def predict(input_features: List[float], model_path: str | None = None) -> dict[str, Any]:
    model_path = Path(model_path or "models/model.joblib")
    if not model_path.exists():
        ModelTrainer(model_path=str(model_path)).train()

    payload = joblib.load(model_path)
    if "model" not in payload or "preprocessor" not in payload:
        ModelTrainer(model_path=str(model_path)).train()
        payload = joblib.load(model_path)

    model = payload["model"]
    preprocessor = payload["preprocessor"]
    feature_columns = payload.get("feature_columns", [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal"
    ])

    features = pd.DataFrame([input_features], columns=feature_columns)
    features_scaled = preprocessor.transform(features)
    prediction = int(model.predict(features_scaled)[0])
    probability = float(model.predict_proba(features_scaled)[0].max())

    return {"prediction": prediction, "probability": probability}
