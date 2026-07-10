# Project Summary

## Project Title
End-to-End MLOps Project for Heart Disease Prediction

## Problem Statement
Build a machine learning classifier to predict the risk of heart disease based on patient health data and deploy the solution as a cloud-ready, monitored API.

## Assignment Task Coverage

### 1. Data Acquisition and EDA
- The Cleveland heart disease dataset is loaded from the attached archive using [src/data/load_data.py](src/data/load_data.py).
- Data validation and cleaning are implemented in [src/data/validate.py](src/data/validate.py).
- An EDA summary is documented in [eda_report.md](eda_report.md).

### 2. Feature Engineering and Model Development
- Feature preprocessing is implemented in [src/features/preprocess.py](src/features/preprocess.py).
- Feature selection is included in [src/features/feature_selection.py](src/features/feature_selection.py).
- Model development and comparison are implemented in [src/models/model_comparison.py](src/models/model_comparison.py).
- Logistic Regression and Random Forest are compared as part of the workflow.

### 3. Experiment Tracking
- MLflow experiment tracking is configured in [src/monitoring/mlflow_tracking.py](src/monitoring/mlflow_tracking.py).

### 4. Model Packaging and Reproducibility
- The trained model is saved as a reusable joblib artifact in [src/models/train_model.py](src/models/train_model.py).
- A dataset download helper is available in [scripts/download_dataset.py](scripts/download_dataset.py).
- Requirements are listed in [requirements.txt](requirements.txt).

### 5. CI/CD Pipeline and Automated Testing
- Unit tests are implemented in [tests/test_pipeline.py](tests/test_pipeline.py) and [tests/test_model_comparison.py](tests/test_model_comparison.py).
- GitHub Actions workflow is configured in [.github/workflows/ci.yml](.github/workflows/ci.yml).

### 6. Model Containerization
- Docker support is available in [docker/Dockerfile](docker/Dockerfile).
- The API is served using FastAPI in [src/api/main.py](src/api/main.py).

### 7. Production Deployment
- Kubernetes deployment manifests are available in [deployment.yaml](deployment.yaml).

### 8. Monitoring and Logging
- Logging is implemented in [src/logger.py](src/logger.py).
- Monitoring configuration is provided in [monitoring/prometheus.yml](monitoring/prometheus.yml).

### 9. Documentation and Reporting
- Project documentation is available in [README.md](README.md) and [docs/final_report.md](docs/final_report.md).
- A notebook for EDA and training is available at [notebooks/eda_training.ipynb](notebooks/eda_training.ipynb).

## Tech Stack
- Python
- scikit-learn
- FastAPI
- Streamlit
- MLflow
- DVC
- Docker
- Kubernetes
- GitHub Actions
- Pytest

## Project Workflow
1. Load the heart disease dataset
2. Perform data validation and cleaning
3. Apply preprocessing and feature selection
4. Train and compare models
5. Track experiments using MLflow
6. Serve predictions through FastAPI
7. Test the pipeline using Pytest
8. Containerize and deploy the solution

## Current Status
The project is implemented and verified. The latest test run completed successfully with 10 passed tests.

