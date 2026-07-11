#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DOWNLOAD_DATA=false
TRAIN_MODEL=true
START_API=true
START_MONITORING=true
START_STREAMLIT=false
RUN_LINT=true
RUN_TESTS=true
DETACH=false
DEPLOY_K8S=false

usage() {
  cat <<'EOF'
Usage: ./scripts/run_end_to_end.sh [options]

Options:
  --download-data    download dataset from ../Downloads/heart+disease.zip
  --no-train         skip model training
  --no-monitoring    skip Prometheus / Grafana startup
  --no-lint          skip lint checks
  --no-test          skip running tests
  --streamlit        start the Streamlit demo app
  --k8s              deploy locally to Kubernetes instead of Docker Compose
  --detach           start services in detached mode
  --help             show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --download-data)
      DOWNLOAD_DATA=true
      shift
      ;;
    --no-train)
      TRAIN_MODEL=false
      shift
      ;;
    --no-monitoring)
      START_MONITORING=false
      shift
      ;;
    --no-lint)
      RUN_LINT=false
      shift
      ;;
    --no-test)
      RUN_TESTS=false
      shift
      ;;
    --streamlit)
      START_STREAMLIT=true
      shift
      ;;
    --k8s)
      DEPLOY_K8S=true
      shift
      ;;
    --detach)
      DETACH=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "$DOWNLOAD_DATA" == true ]]; then
  echo "Downloading dataset..."
  python3 scripts/download_dataset.py
elif [[ ! -f "data/heart_disease.csv" ]]; then
  echo "Dataset not found at data/heart_disease.csv"
  echo "Run with --download-data or place the archive at ../Downloads/heart+disease.zip"
  exit 1
fi

if [[ "$RUN_LINT" == true || "$RUN_TESTS" == true ]]; then
  if [[ -f "requirements-dev.txt" ]]; then
    echo "Installing development dependencies for lint and tests..."
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements-dev.txt
  else
    echo "Warning: requirements-dev.txt not found; lint and tests may fail if dev dependencies are missing."
  fi
fi

if [[ "$RUN_LINT" == true ]]; then
  echo "Running lint checks..."
  python3 -m pip install ruff
  ruff check .
fi

if [[ "$RUN_TESTS" == true ]]; then
  echo "Running tests..."
  python3 -m pytest tests -q
fi

if [[ "$TRAIN_MODEL" == true ]]; then
  echo "Running model training..."
  python3 -m src.train
fi

if [[ "$START_MONITORING" == true && "$START_API" == false ]]; then
  echo "Monitoring requires the API service to be available. Remove --no-monitoring or enable API startup."
  exit 1
fi

if [[ "$DEPLOY_K8S" == true ]]; then
  if [[ "$START_MONITORING" == true ]]; then
    echo "Kubernetes mode currently deploys only API and optional Streamlit. Monitoring is disabled for this run."
    START_MONITORING=false
  fi
  if [[ "$DETACH" == true ]]; then
    echo "Note: --detach is ignored in Kubernetes mode."
  fi

  if [[ "$START_STREAMLIT" == true ]]; then
    echo "Deploying API and Streamlit to Kubernetes..."
  else
    echo "Deploying API to Kubernetes..."
  fi

  ./scripts/deploy_local_k8s.sh
  exit 0
fi

if [[ "$START_API" == true ]]; then
  echo "Building API Docker image..."
  docker build -f docker/Dockerfile -t heart-disease-api .
fi

if [[ "$START_STREAMLIT" == true ]]; then
  echo "Building Streamlit Docker image..."
  docker build -f docker/Dockerfile.streamlit -t heart-disease-streamlit .
fi

COMPOSE_ARGS=()
COMPOSE_ARGS+=("-f" "docker/docker-compose.yml")

if [[ "$START_MONITORING" == true ]]; then
  COMPOSE_ARGS+=("-f" "monitoring/docker-compose.monitoring.yml")
fi

if [[ "$START_STREAMLIT" == true ]]; then
  COMPOSE_ARGS+=("-f" "docker/docker-compose.streamlit.yml")
fi

COMPOSE_ARGS+=(up --build)
if [[ "$DETACH" == true ]]; then
  COMPOSE_ARGS+=("-d")
fi

echo "Starting end-to-end stack..."
docker compose "${COMPOSE_ARGS[@]}"

echo
if [[ "$START_API" == true ]]; then
  echo "API available at http://127.0.0.1:8000"
  echo "Swagger UI: http://127.0.0.1:8000/docs"
fi
if [[ "$START_MONITORING" == true ]]; then
  echo "Prometheus available at http://127.0.0.1:9090"
  echo "Grafana available at http://127.0.0.1:3000"
  echo "Grafana login: admin / admin"
fi
if [[ "$START_STREAMLIT" == true ]]; then
  echo "Streamlit demo available at http://127.0.0.1:8501"
fi
