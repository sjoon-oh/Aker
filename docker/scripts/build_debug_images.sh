#!/bin/bash
set -euo pipefail

# Build debug container images.
#
# These images are meant for quick local iteration:
# - Start from the *vanilla* pgvector image.
# - Add build toolchains and a baseline Aker source tree.
# - Rebuild + install Aker and pgvector inside a temporary container on each run.
#
# This script intentionally does NOT modify the existing build_images.sh pipeline.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Persist build output for post-mortem debugging.
LOG_DIR="${PROJECT_ROOT}/temp"
mkdir -p "${LOG_DIR}"
LOG_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/docker_build_debug_images_${LOG_TIMESTAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

IMAGE_PREFIX="${IMAGE_PREFIX:-aker_pgvector}"
BASE_IMAGE="${BASE_IMAGE:-${IMAGE_PREFIX}_vanilla:latest}"

log() {
    printf "[build_debug_images] %s\n" "$*"
}

build_debug_variant() {
    local aker_mode="$1"
    local tag="${IMAGE_PREFIX}_${aker_mode}_debug:latest"

    log "Building debug image (${aker_mode}): ${tag}"
    docker build \
        -f "${PROJECT_ROOT}/docker/images/debug/Dockerfile.pgvector_debug" \
        --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
        -t "${tag}" \
        "${PROJECT_ROOT}"
}

main() {
    log "Logging to: ${LOG_FILE}"
    log "Base image: ${BASE_IMAGE}"

    build_debug_variant standard
    build_debug_variant potluck
    build_debug_variant proximity

    log "Done. Built images:"
    log "  ${IMAGE_PREFIX}_standard_debug:latest"
    log "  ${IMAGE_PREFIX}_potluck_debug:latest"
    log "  ${IMAGE_PREFIX}_proximity_debug:latest"
}

main "$@"
