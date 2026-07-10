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

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
uvicorn src.api.main:app --reload
streamlit run app/streamlit_app.py
```

## Test
```bash
pytest -q
```
