# Step 7 — Submission Checklist

## Must-have items
- Dataset download helper
- EDA plots and summary
- Model comparison and training
- MLflow tracking
- Dockerized API
- CI workflow
- Kubernetes manifest
- Report document
- Screenshots folder

## Suggested final proof
- Run tests: `pytest -q`
- Generate EDA plots: `python scripts/eda_report.py`
- Build Docker image: `docker build -f docker/Dockerfile -t heart-disease-api .`
- Run API locally: `uvicorn src.api.main:app --reload`
