# Architecture Diagram

```text
User / Client
    |
    v
Streamlit UI
    |
    v
FastAPI Service
    |
    v
Model Artifact (joblib)
    |
    +--> MLflow Tracking
    +--> Logging
    +--> Monitoring (Prometheus / Grafana)
```
