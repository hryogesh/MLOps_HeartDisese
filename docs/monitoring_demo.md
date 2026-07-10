# Monitoring Demo

## Prometheus and Grafana
Use the monitoring stack to observe the system:

```bash
docker compose -f monitoring/docker-compose.monitoring.yml up -d
```

Then open:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## What to observe
- API health endpoint
- Basic metrics exposure
- Container and service availability
