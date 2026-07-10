# Final Submission Writeup

## 1. Project Title
End-to-End MLOps Project for Heart Disease Prediction

## 2. Problem Statement
The project aims to build a machine learning classifier that predicts the risk of heart disease using patient health data and deploys the solution as a cloud-ready, monitored API.

## 3. Objectives
- Acquire and clean the dataset
- Perform exploratory data analysis
- Engineer features and train machine learning models
- Track experiments with MLflow
- Package the model and make it reproducible
- Test the system with automated unit tests
- Containerize and deploy the API
- Monitor the deployment and document the workflow

## 4. Dataset
The Cleveland heart disease dataset is used for this project. The dataset is loaded from the attached heart disease archive and stored locally as a CSV file for reproducibility.

## 5. Methodology
The workflow includes data loading, validation, preprocessing, feature selection, model training, comparison of multiple classifiers, and API serving. Logistic Regression and Random Forest are used for model comparison, while preprocessing steps include imputation and scaling.

## 6. Experiment Tracking
MLflow is used to track experiments, parameters, metrics, and model artifacts. The configuration is implemented in the project source code.

## 7. Machine Learning Models
Two models are included in the workflow:
- Logistic Regression
- Random Forest

The models are compared using cross-validation and evaluation metrics such as accuracy.

## 8. Deployment and Containerization
The trained model is served using FastAPI. Docker is used to containerize the service, and Kubernetes deployment manifests are included for orchestration.

## 9. CI/CD and Testing
Pytest is used for unit testing, and GitHub Actions is configured to run linting, tests, and model training automatically.

## 10. Monitoring and Logging
The project includes logging and monitoring configuration for the service, with Prometheus and Grafana support included.

## 11. Results
The project is implemented and verified successfully. The latest test run reports 9 passing tests.

## 12. Conclusion
This project demonstrates an end-to-end MLOps workflow for a real-world classification problem and is suitable for academic submission, GitHub portfolio presentation, and technical discussion.
