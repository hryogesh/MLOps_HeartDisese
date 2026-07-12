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

## Monitoring Proof
```bash
docker compose -f monitoring/docker-compose.monitoring.yml up -d
```

Then open:
- http://127.0.0.1:9090 for Prometheus
- http://127.0.0.1:3000 for Grafana
- http://127.0.0.1:3000/d/heart-disease-api/heart-disease-api-monitoring for the dashboard
- http://127.0.0.1:8001/metrics for API metrics

Prometheus last-2-hour query example:
```bash
start=$(date -u -v -2H +%s)
end=$(date -u +%s)

curl -G 'http://127.0.0.1:9090/api/v1/query_range' \
  --data-urlencode 'query=increase(http_requests_total[1m])' \
  --data-urlencode "start=$start" \
  --data-urlencode "end=$end" \
  --data-urlencode 'step=60'
```
