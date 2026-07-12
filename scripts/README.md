# Scripts Documentation

This directory contains utility scripts for the MLOps heart disease prediction project.

## Available Scripts

### 1. **data_pipeline.py** - Complete Data Processing Pipeline
Orchestrates the entire data workflow from download through EDA to feature engineering.

**Usage:**
```bash
# Run entire pipeline
python scripts/data_pipeline.py

# Run only specific step
python scripts/data_pipeline.py --step data      # Download & validate dataset
python scripts/eda_report.py --step eda       # Run EDA only
python scripts/feature_engineering.py --step features  # Run feature engineering only
```

**Workflow:**
1. Download & validate dataset
2. Generate EDA reports and plots
3. Run feature selection and preprocessing

**Output:**
- EDA plots → `artifacts/eda/`
- Feature analysis → `artifacts/features/`

---

### 2. **download_dataset.py** - Dataset Acquisition
Downloads and prepares the Cleveland heart disease dataset.

**Usage:**
```bash
python scripts/download_dataset.py
```

**Output:**
- Cleaned CSV file → `data/heart_disease.csv`
- Dataset size: 303 rows × 14 columns

---

### 3. **eda_report.py** - Exploratory Data Analysis
Generates comprehensive EDA plots and summary statistics.

**Usage:**
```bash
python scripts/eda_report.py
```

**Generates:**
1. **Class Distribution Plot** → Shows target variable balance
2. **Feature Distributions** → Histograms for all numeric features
3. **Correlation Heatmap** → Feature correlation analysis
4. **Feature-Target Relationship** → Boxplot analysis
5. **Model Comparison Results** → Performance metrics for 4 models
6. **Summary Report** → Text file with all statistics

**Output Directory:** `artifacts/eda/`

**Output Files:**
- `01_class_distribution.png` - Class balance visualization
- `02_hist_*.png` - Individual feature distributions
- `03_correlation_heatmap.png` - Feature correlations
- `04_target_vs_thalach.png` - Feature-target relationship
- `summary.txt` - Comprehensive statistics report

---

### 4. **feature_engineering.py** - Feature Processing Pipeline
Handles feature selection and preprocessing in a standalone script.

**Usage:**
```bash
# Run all feature engineering steps
python scripts/feature_engineering.py --mode all

# Run only feature selection
python scripts/feature_engineering.py --mode selection

# Run only preprocessing
python scripts/feature_engineering.py --mode preprocessing

# Specify custom output directory
python scripts/feature_engineering.py --mode all --output-dir custom/path
```

**Features:**

**Selection Mode:**
- Selects top 8 features using ANOVA F-statistic
- Shows feature importance scores
- Compares selected vs non-selected features

**Preprocessing Mode:**
- Imputes missing values (median strategy)
- Scales features (StandardScaler)
- Splits data (80% train / 20% test, stratified)
- Reports class distribution

**Output Directory:** `artifacts/features/` (default)

**Output Files:**
- `feature_selection_results.txt` - Selected features and scores
- `preprocessing_statistics.txt` - Train/test split and class distribution

---

### 5. **run_end_to_end.sh** - End-to-End Pipeline Automation
Bash script that automates the complete workflow including linting, testing, training, and deployment.

**Usage:**
```bash
# Run with Docker Compose
./scripts/run_end_to_end.sh --detach --streamlit

# Run with Kubernetes
./scripts/run_end_to_end.sh --k8s --streamlit

# Skip lint and tests
./scripts/run_end_to_end.sh --k8s --streamlit --no-lint --no-test

# Show help
./scripts/run_end_to_end.sh --help
```

**Workflow Steps:**
1. Data validation
2. Linting (ruff check)
3. Unit tests (pytest)
4. Model training
5. Docker image build
6. Service deployment (Docker Compose or K8s)
7. Monitoring stack startup

---

### 6. **deploy_local_k8s.sh** - Kubernetes Deployment Helper
Deploys the API and Streamlit services to a local Kubernetes cluster.

**Usage:**
```bash
./scripts/deploy_local_k8s.sh
```

**Prerequisites:**
- Docker installed and running
- Kubernetes cluster accessible (minikube, Docker Desktop K8s, etc.)

**Deploys:**
- FastAPI inference service
- Streamlit demo application
- Monitoring stack (optional)

---

### 7. **export_submission_docs.py** - Documentation Export
Exports project documentation and artifacts for submission.

**Usage:**
```bash
python scripts/export_submission_docs.py
```

**Exports:**
- Architecture diagrams
- EDA reports
- Model metrics
- Deployment proof
- Submission HTML document

---

## Quick Start Workflow

### Complete Data & Feature Pipeline
```bash
# Run entire pipeline (data → EDA → features)
python scripts/data_pipeline.py

# Or run individually
python scripts/download_dataset.py
python scripts/eda_report.py
python scripts/feature_engineering.py
```

### Train Model & Deploy
```bash
# Train model with experiment tracking
python -m src.train

# View MLflow experiments
mlflow ui --host 127.0.0.1 --port 5000

# Deploy with Docker
./scripts/run_end_to_end.sh --detach --streamlit

# Or deploy to Kubernetes
./scripts/run_end_to_end.sh --k8s --streamlit
```

### Feature Engineering Only
```bash
# Select best features
python scripts/feature_engineering.py --mode selection

# Preprocess features
python scripts/feature_engineering.py --mode preprocessing

# Both
python scripts/feature_engineering.py --mode all
```

---

## Output Artifacts

### Directory Structure
```
artifacts/
├── eda/                           # EDA outputs
│   ├── 01_class_distribution.png
│   ├── 02_hist_*.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_target_vs_thalach.png
│   └── summary.txt
├── features/                      # Feature engineering outputs
│   ├── feature_selection_results.txt
│   └── preprocessing_statistics.txt
└── mlflow/                        # Model training artifacts
    └── metrics_summary.json
```

---

## Troubleshooting

### Dataset Not Found
```bash
python scripts/download_dataset.py
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Permission Denied for Shell Scripts
```bash
chmod +x scripts/*.sh
```

### Python Path Issues
Scripts automatically handle path resolution. Run from project root:
```bash
cd /path/to/mlops-end-to-end-project
python scripts/script_name.py
```

---

## Integration with CI/CD

All scripts support integration with GitHub Actions:
- Linting: `ruff check .`
- Testing: `pytest tests -q`
- Training: `python -m src.train`
- Data Pipeline: `python scripts/data_pipeline.py`

See `.github/workflows/ci.yml` for full CI configuration.
