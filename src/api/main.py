from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.predict import predict

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")


class PredictionRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(request: PredictionRequest):
    if len(request.features) != 13:
        raise HTTPException(status_code=400, detail="Expected exactly 13 feature values")
    result = predict(request.features)
    return result
