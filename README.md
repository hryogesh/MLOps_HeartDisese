# End-to-End MLOps Project for Heart Disease Prediction

This repository implements a production-style MLOps workflow for predicting heart disease using the Cleveland heart disease dataset from the attached archive.

## Features
- Data ingestion from the attached dataset archive
- Data validation and cleaning
- Feature preprocessing and train/test split
- Logistic regression model training and evaluation
- Model serialization and prediction
- FastAPI inference service
- Streamlit demo app
- MLflow experiment tracking
- Logging and exception handling
- Pytest-based testing
- Docker support

## Project Structure
- src/data - data loading and validation
- src/features - preprocessing
- src/models - training and prediction
- src/api - FastAPI app
- app - Streamlit app
- tests - automated tests
- config - YAML configuration

## Assignment task coverage
This repository implements the full assignment workflow with end-to-end support for data ingestion, model development, deployment, and monitoring.

- [x] Task 1 — Data acquisition, validation, cleaning, and exploratory analysis
- [x] Task 2 — Feature engineering, model training, evaluation, and comparison
- [x] Task 3 — Experiment tracking and MLflow logging
- [x] Task 4 — Model packaging, serialization, reproducibility, and inference serving via FastAPI
- [x] Task 5 — CI/CD readiness with automated linting and pytest test coverage
- [x] Task 6 — Containerization for both API and Streamlit demo apps using Docker
- [x] Task 7 — Local Kubernetes deployment support for API and Streamlit
- [x] Task 8 — Monitoring and logging with Prometheus and Grafana

## End-to-end run guide for the assignment tasks (1 to 8)

### Task 1 — Setup environment and dataset acquisition
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dataset: Title: Heart Disease UCI Dataset
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
If `data/heart_disease.csv` is not present, run the download helper with the archive located at `../Downloads/heart+disease.zip` or :
```bash
python scripts/download_dataset.py
```
This creates the cleaned file at `data/heart_disease.csv`.

```bash
python - <<'PY'
from src.data.load_data import load_dataset

df = load_dataset(path='data/heart_disease.csv')
print(df.head())
print(df.shape)
print(df.isnull().sum())
PY
```

Optional EDA report:
```bash
python scripts/eda_report.py
```

### Task 2 — Feature engineering and model development
```bash
python - <<'PY'
from src.data.load_data import load_dataset
from src.features.preprocess import prepare_features
from src.models.model_comparison import compare_models

df = load_dataset(path='data/heart_disease.csv')
prepared = prepare_features(df)
print(prepared['X_train'].shape, prepared['X_test'].shape)
comparison = compare_models(df)
print(comparison['best_model'])
print(comparison['results'])
PY
```

### Task 3 — Experiment tracking with MLflow
```bash
python -m src.train
mlflow ui --host 127.0.0.1 --port 5000
```

Then open:
- http://127.0.0.1:5000

### Task 4 — Model packaging and reproducibility
Train the model once to create a reusable artifact at [models/model.joblib](models/model.joblib). The saved joblib bundle contains the full scikit-learn pipeline, including the preprocessing transformer (imputer + scaler) and the classifier, so predictions remain reproducible without manually reapplying feature transforms.

```bash
python -m src.train
```

Then run inference with the packaged model:
```bash
python - <<'PY'
from src.models.predict import predict

result = predict([63.0, 1.0, 1.0, 145.0, 233.0, 1.0, 2.0, 150.0, 0.0, 2.3, 3.0, 0.0, 6.0])
print(result)
PY
```

The model payload is stored as `models/model.joblib` and includes:
- `pipeline`: full preprocess + classifier pipeline
- `preprocessor`: transformer stack for imputing and scaling
- `feature_columns`: ordered feature names for reproducible input

MLflow also logs the same pipeline as an MLflow model artifact under `mlruns/`.

### Task 5 — CI/CD and automated testing
Run the local test suite with:
```bash
pytest -q
```

This repository also includes a GitHub Actions workflow at [.github/workflows/ci.yml](.github/workflows/ci.yml) that automates:
- dependency installation
- linting with Ruff
- unit tests with Pytest
- model training
- artifact upload for the trained model and MLflow outputs

Each workflow run publishes:
- [models/model.joblib](models/model.joblib) as the trained model artifact
- MLflow run data under [mlruns](mlruns)
- auxiliary artifacts under [artifacts/mlflow](artifacts/mlflow)

### Task 6 — Model containerization
Build and run the model-serving API directly with Docker Compose:
```bash
docker compose -f docker/docker-compose.yml up --build
```

The containerized FastAPI service exposes:
- http://127.0.0.1:8001/docs for the Swagger UI
- http://127.0.0.1:8001/health for the health check endpoint
- http://127.0.0.1:8001/predict for JSON inference requests

Sample prediction request:
```bash
curl -X POST http://127.0.0.1:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{"features":[63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0]}'
```

Example response:
```json
{"prediction":0,"probability":0.6578291651317967}
```

### Task 7 — Production deployment
Deploy the containerized API directly to a local Kubernetes cluster using the manifest in [k8s-deployment.yaml](k8s-deployment.yaml):
```bash
kubectl apply -f k8s-deployment.yaml
```

If you are using Docker Desktop, build the required images first:
```bash
docker build -f docker/Dockerfile -t heart-disease-api:latest .
docker build -f docker/Dockerfile.streamlit -t heart-disease-streamlit:latest .
```

If the LoadBalancer IP stays pending, verify the deployment locally with port-forwarding:
```bash
kubectl get deployments,services
kubectl port-forward svc/heart-disease-api 8000:80
kubectl port-forward svc/heart-disease-streamlit 8501:8501
```

Then open:
- API: http://127.0.0.1:8000/docs
- Streamlit: http://127.0.0.1:8501

If you only need the API manifest, see [deployment.yaml](deployment.yaml).

### Task 8 — Monitoring and logging
Run the monitoring stack directly:
```bash
docker compose -f monitoring/docker-compose.monitoring.yml up --build
```

Then open:
- http://127.0.0.1:9090 for Prometheus
- http://127.0.0.1:3000 for Grafana
- http://127.0.0.1:3000/d/heart-disease-api/heart-disease-api-monitoring for the pre-provisioned dashboard

The Grafana dashboard is configured to show the last 2 hours by default. If you want a different range, use the Grafana time-picker in the top-right and select a relative range such as "Last 2 hours" or "Last 1 hour".

Prometheus query examples:
- In the Prometheus UI, run: `increase(http_requests_total[1m])`
- To check target health, run: `up{job="heart-disease-api"}`

API query_range example for the last two hours:
```bash
start=$(date -u -v -2H +%s)
end=$(date -u +%s)

curl -G 'http://127.0.0.1:9090/api/v1/query_range' \
  --data-urlencode 'query=increase(http_requests_total[1m])' \
  --data-urlencode "start=$start" \
  --data-urlencode "end=$end" \
  --data-urlencode 'step=60'
```

Your API metrics are now scraped at `/metrics` from:
- http://127.0.0.1:8001/metrics

> Note: the Docker Compose API service exposes the container on host port `8001`.

### 10. Optional local UI demo
```bash
streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Then open:
- http://127.0.0.1:8501

## Test
```bash
pytest -q
```
