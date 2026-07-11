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

## End-to-end run guide for tasks 1 to 8

### 1. Setup environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Task 1 — Data acquisition and EDA
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

### 3. Task 2 — Feature engineering and model development
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

### 4. Task 3 — Experiment tracking with MLflow
```bash
python -m src.train
mlflow ui --host 127.0.0.1 --port 5000
```

Then open:
- http://127.0.0.1:5000

### 5. Task 4 — Model packaging and reproducibility
```bash
python - <<'PY'
from src.models.predict import predict

result = predict([63.0, 1.0, 1.0, 145.0, 233.0, 1.0, 2.0, 150.0, 0.0, 2.3, 3.0, 0.0, 6.0])
print(result)
PY
```

### 6. Task 5 — CI/CD and automated testing
```bash
pytest -q
```

### 7. Task 6 — Containerization with Docker
```bash
docker compose -f docker/docker-compose.yml up --build
```

Then open:
- http://127.0.0.1:8000/docs for the Swagger UI
- http://127.0.0.1:8000/health for the health check endpoint

### 8. Task 7 — Production deployment
The Kubernetes deployment manifest is available in [deployment.yaml](deployment.yaml).

### 9. Task 8 — Monitoring and logging
```bash
curl http://127.0.0.1:8000/metrics
```

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
