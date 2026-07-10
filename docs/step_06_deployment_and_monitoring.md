# Step 6 — Deployment and Monitoring

## Deployment
1. Build the Docker image:
   ```bash
   docker build -f docker/Dockerfile -t heart-disease-api .
   ```
2. Run the container locally:
   ```bash
   docker run -p 8000:8000 heart-disease-api
   ```
3. Test the endpoint:
   ```bash
   curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"features": [63,1,1,145,233,1,2,150,0,2.3,3,0,6]}'
   ```

## Monitoring
1. Start Prometheus and Grafana:
   ```bash
   docker compose -f monitoring/docker-compose.monitoring.yml up -d
   ```
2. Open Grafana at http://localhost:3000.
3. Add Prometheus as a data source.
