from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from pydantic import BaseModel

from src.models.predict import predict

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["endpoint", "method"])


class PredictionRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health_check():
    REQUEST_COUNT.labels(endpoint="/health", method="GET").inc()
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    REQUEST_COUNT.labels(endpoint="/metrics", method="GET").inc()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict_endpoint(request: PredictionRequest):
    REQUEST_COUNT.labels(endpoint="/predict", method="POST").inc()
    if len(request.features) != 13:
        raise HTTPException(status_code=400, detail="Expected exactly 13 feature values")
    result = predict(request.features)
    return result
