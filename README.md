# MLOps_HeartDisese

This repository implements an end-to-end MLOps workflow for a public classification dataset and is structured for academic submission and GitHub portfolio use.

## Project Goals
- Build an end-to-end ML pipeline: data loading, preprocessing, training, evaluation, model packaging, and serving.
- Demonstrate MLOps practices: versioned code, automated testing, CI/CD, containerization, and API deployment readiness.
- Provide a clean, documented project structure suitable for evaluation and discussion.

## Dataset
- Source: scikit-learn Iris dataset
- Task: multi-class classification
- Samples: 150
- Features: 4 numeric measurements
- Target: 3-class label

## Tech Stack
- Python
- scikit-learn
- FastAPI
- MLflow
- pytest
- Docker
- GitHub Actions

## Project Structure
- data/ – local copy of the dataset
- src/ – reusable training, inference, and API code
- tests/ – unit tests for the pipeline
- docker/ – Docker assets
- .github/workflows/ – CI/CD pipeline definitions
- eda_report.md – exploratory data analysis summary

## Quickstart
1. Create and activate a virtual environment.
2. Install dependencies: pip install -r requirements.txt
3. Run training: python -m src.train
4. Run API locally: uvicorn src.api.main:app --reload
5. Run tests: pytest -q

## Deliverables Included
- Dataset ingestion and persistence
- Feature preprocessing and train/test split
- Model training and evaluation
- Prediction service with API endpoint
- Automated tests
- Docker and CI workflow setup
