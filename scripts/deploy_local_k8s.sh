#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v kubectl >/dev/null 2>&1; then
  echo "Error: kubectl is required but not installed."
  exit 1
fi

if ! kubectl version --client >/dev/null 2>&1; then
  echo "Error: kubectl cannot connect to a cluster. Ensure your kubeconfig is valid."
  exit 1
fi

IMAGE_API="heart-disease-api:latest"
IMAGE_STREAMLIT="heart-disease-streamlit:latest"

echo "Building local Docker images..."
docker build -f docker/Dockerfile -t "$IMAGE_API" .
docker build -f docker/Dockerfile.streamlit -t "$IMAGE_STREAMLIT" .

echo "Rendering Kubernetes manifest..."
K8S_MANIFEST="$(mktemp -t k8s-deployment)"
python3 - "$ROOT_DIR" "$IMAGE_API" "$IMAGE_STREAMLIT" > "$K8S_MANIFEST" <<'PY'
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
image_api = sys.argv[2]
image_streamlit = sys.argv[3]
text = (root / 'k8s-deployment.yaml').read_text()
text = text.replace('${IMAGE_API}', image_api)
text = text.replace('${IMAGE_STREAMLIT}', image_streamlit)
print(text)
PY

echo "Applying Kubernetes manifest..."
kubectl apply -f "$K8S_MANIFEST"

echo "Waiting for deployments to become ready..."
kubectl rollout status deployment/heart-disease-api --timeout=120s
kubectl rollout status deployment/heart-disease-streamlit --timeout=120s

echo
kubectl get deployments,services heart-disease-api heart-disease-streamlit

echo
cat <<'EOF'
Local Kubernetes deployment complete.
If the LoadBalancer IP is pending, use port-forwarding:
  kubectl port-forward svc/heart-disease-api 8000:80
  kubectl port-forward svc/heart-disease-streamlit 8501:8501
Then open:
  API: http://127.0.0.1:8000/docs
  Streamlit: http://127.0.0.1:8501
EOF
