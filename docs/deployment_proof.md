# Deployment Proof

## Local API Test
Run the API locally:

```bash
uvicorn src.api.main:app --reload
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"features": [63,1,1,145,233,1,2,150,0,2.3,3,0,6]}'
```

Expected response:

```json
{"prediction": 1, "probability": 0.8}
```

## Docker Build Proof
```bash
docker build -f docker/Dockerfile -t heart-disease-api .
```

## Kubernetes Deployment Proof
```bash
kubectl apply -f deployment.yaml
kubectl get pods
kubectl get svc
```
