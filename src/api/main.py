from fastapi import FastAPI
from pydantic import BaseModel
from src.models.predict import predict

app = FastAPI(title="MLOps Demo API", version="1.0.0")


class PredictionRequest(BaseModel):
    features: list


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(request: PredictionRequest):
    if len(request.features) != 4:
        return {"error": "Expected exactly 4 feature values"}
    result = predict(request.features)
    return result
