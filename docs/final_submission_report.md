# Final Submission Report

## 1. Project Overview
This repository presents an end-to-end MLOps workflow for heart disease prediction using the Cleveland heart disease dataset. The project covers data ingestion, preprocessing, model training, experiment tracking, testing, API deployment, Dockerization, and monitoring.

## 2. Task-by-Task Assignment Summary
### Task 1 — Data acquisition, validation, cleaning, and EDA
- Loaded the Cleveland heart disease dataset into `data/heart_disease.csv`.
- Implemented data validation and cleaning in `src/data/load_data.py`.
- Generated exploratory analysis artifacts and summary reports in `artifacts/eda`.

### Task 2 — Feature engineering, model training, evaluation, and comparison
- Built preprocessing and feature engineering logic in `src/features/preprocess.py`.
- Trained candidate models and compared results in `src/models/model_comparison.py`.
- Selected and serialized the best model pipeline to `models/model.joblib`.

### Task 3 — Experiment tracking and MLflow logging
- Configured MLflow tracking in `src/train.py` and `src/monitoring/mlflow_tracking.py`.
- Logged parameters, metrics, and artifacts for reproducible experiment tracking.
- Added MLflow model signature and input example for clean model registry support.

### Task 4 — Model packaging, serialization, reproducibility, and inference serving
- Packaged the preprocessing pipeline and classifier into a single joblib artifact.
- Implemented inference logic in `src/models/predict.py` to support reproducible predictions.
- Built a FastAPI service for the model in `src/api/main.py`.

### Task 5 — CI/CD readiness and automated testing
- Included tests in `tests/` covering model behavior and pipeline reproducibility.
- Prepared CI automation via GitHub Actions in `.github/workflows/ci.yml`.
- Enabled linting, unit tests, and artifact validation as part of the workflow.

### Task 6 — Containerization with Docker
- Containerized the FastAPI service with `docker/Dockerfile`.
- Added Docker Compose support in `docker/docker-compose.yml` for local API deployment.
- Verified service exposure on host port `8001`.

### Task 7 — Local Kubernetes deployment support
- Provided `k8s-deployment.yaml` for API deployment in a local Kubernetes cluster.
- Included helper commands for local port-forwarding and service verification.
- Documented the deployment manifest and service access points.

### Task 8 — Monitoring and logging with Prometheus and Grafana
- Added Prometheus configuration in `monitoring/prometheus.yml`.
- Configured Grafana provisioning under `monitoring/grafana/provisioning`.
- Included a pre-provisioned dashboard for request rate and target health.
- Ensured the dashboard defaults to the last 2 hours and exposed metrics at `/metrics`.

## 3. Dataset and EDA
The dataset is stored in data/heart_disease.csv and the repository includes a download helper at scripts/download_dataset.py. EDA artifacts are available under artifacts/eda and screenshots under screenshots/.

## 4. Model Development
The training pipeline includes preprocessing, feature selection, model comparison, and serialization. The main training entrypoint is src/train.py and the model artifact is stored in models/model.joblib.

## 4. Experiment Tracking
MLflow is used for experiment tracking. Training runs log parameters, metrics, and artifacts into mlruns/.

## 5. CI/CD and Testing
CI is implemented using GitHub Actions in .github/workflows/ci.yml. Tests are available in tests/ and are executed automatically on push and pull request.

## 6. Containerization and Deployment
The FastAPI service is containerized with Docker using docker/Dockerfile and docker/docker-compose.yml. A deployment manifest is available in deployment.yaml.

## 7. Monitoring
The monitoring stack includes Prometheus and Grafana. Prometheus scrapes the FastAPI service metrics from `http://127.0.0.1:8001/metrics` and Grafana is preconfigured with the `Heart Disease API Monitoring` dashboard.

Access:
- http://127.0.0.1:9090 for Prometheus
- http://127.0.0.1:3000 for Grafana
- http://127.0.0.1:3000/d/heart-disease-api/heart-disease-api-monitoring for the pre-provisioned dashboard

The dashboard defaults to the last 2 hours.

## 8. Local Access Instructions
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
