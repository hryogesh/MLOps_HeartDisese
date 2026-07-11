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

If `data/heart_disease.csv` is not present, run the download helper with the archive located at `../Downloads/heart+disease.zip`:
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
```bash
python - <<'PY'
from src.models.predict import predict

result = predict([63.0, 1.0, 1.0, 145.0, 233.0, 1.0, 2.0, 150.0, 0.0, 2.3, 3.0, 0.0, 6.0])
print(result)
PY
```

### Task 5 — CI/CD and automated testing
```bash
pytest -q
```

### Task 6 — Containerization with Docker
```bash
docker compose -f docker/docker-compose.yml up --build
```

Then open:
- http://127.0.0.1:8000/docs for the Swagger UI
- http://127.0.0.1:8000/health for the health check endpoint

### Task 7 — Local end-to-end deployment
Use the helper script to run dataset download, linting, tests, training, and either Docker Compose or Kubernetes in one command.

Docker Compose deployment:
```bash
./scripts/run_end_to_end.sh --detach --streamlit
```

Kubernetes deployment:
```bash
./scripts/run_end_to_end.sh --k8s --streamlit
```

This will:
- run lint and tests
- train the model
- build the API and Streamlit images
- deploy via Docker Compose or render and apply `k8s-deployment.yaml`

To skip lint or tests:
```bash
./scripts/run_end_to_end.sh --k8s --streamlit --no-lint --no-test
```

Then open:
- API: http://127.0.0.1:8000/docs
- Streamlit: http://127.0.0.1:8501

This workflow is also available via the local Kubernetes helper script:
```bash
./scripts/deploy_local_k8s.sh
```
If you only need the API manifest, see [deployment.yaml](deployment.yaml).

### Task 8 — Monitoring and logging
Run the monitoring stack:
```bash
docker compose -f monitoring/docker-compose.monitoring.yml up --build
```

Then open:
- http://127.0.0.1:9090 for Prometheus
- http://127.0.0.1:3000 for Grafana

Your API metrics are now scraped at `/metrics` from:
- http://127.0.0.1:8000/metrics

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
