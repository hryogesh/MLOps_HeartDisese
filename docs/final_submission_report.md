# Final Submission Report

## 1. Project Overview
This repository presents an end-to-end MLOps workflow for heart disease prediction using the Cleveland heart disease dataset. The project covers data ingestion, preprocessing, model training, experiment tracking, testing, API deployment, Dockerization, and documentation.

## 2. Dataset and EDA
The dataset is stored in data/heart_disease.csv and the repository includes a download helper at scripts/download_dataset.py. EDA artifacts are available under artifacts/eda and screenshots under screenshots/.

## 3. Model Development
The training pipeline includes preprocessing, feature selection, model comparison, and serialization. The main training entrypoint is src/train.py and the model artifact is stored in models/model.joblib.

## 4. Experiment Tracking
MLflow is used for experiment tracking. Training runs log parameters, metrics, and artifacts into mlruns/.

## 5. CI/CD and Testing
CI is implemented using GitHub Actions in .github/workflows/ci.yml. Tests are available in tests/ and are executed automatically on push and pull request.

## 6. Containerization and Deployment
The FastAPI service is containerized with Docker using docker/Dockerfile and docker/docker-compose.yml. A deployment manifest is available in deployment.yaml.

## 7. Local Access Instructions
- API docs: http://127.0.0.1:8000/docs
- Streamlit UI: http://127.0.0.1:8501
- Run tests: pytest -q
- Run API: uvicorn src.api.main:app --host 0.0.0.0 --port 8000

## 8. Submission Assets
- Code and scripts: repository root
- Tests: tests/
- Screenshots: screenshots/
- Notebooks: notebooks/
- Final report: docs/final_submission_report.md
