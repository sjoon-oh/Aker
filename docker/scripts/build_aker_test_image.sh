#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE_TAG="${AKER_TEST_IMAGE_TAG:-aker_test:latest}"
DOCKERFILE_PATH="${PROJECT_ROOT}/docker/images/Dockerfile.aker_test"

FAISS_REPO_URL="${FAISS_REPO_URL:-https://github.com/facebookresearch/faiss.git}"
FAISS_REF="${FAISS_REF:-}"
AKER_MODE="${AKER_MODE:-standard}"

docker build \
    -f "${DOCKERFILE_PATH}" \
    -t "${IMAGE_TAG}" \
    --build-arg "FAISS_REPO_URL=${FAISS_REPO_URL}" \
    --build-arg "FAISS_REF=${FAISS_REF}" \
    --build-arg "AKER_MODE=${AKER_MODE}" \
    "${PROJECT_ROOT}"

echo "Built image: ${IMAGE_TAG}"
echo "Run: docker run --rm ${IMAGE_TAG} --help"
