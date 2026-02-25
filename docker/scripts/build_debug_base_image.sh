#!/bin/bash
set -euo pipefail

# Build the debug base image (Option B) on top of the already-built vanilla image.
#
# Expected workflow:
#   1) ./docker/scripts/build_images.sh
#   2) ./docker/scripts/build_debug_base_image.sh
#   3) ./pgvector-bench/ak_bench_run_search_debug.sh ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LOG_DIR="${PROJECT_ROOT}/temp"
mkdir -p "${LOG_DIR}"
LOG_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/docker_build_debug_base_${LOG_TIMESTAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

IMAGE_PREFIX="${IMAGE_PREFIX:-aker_pgvector}"
VANILLA_IMAGE="${IMAGE_PREFIX}_vanilla:latest"
DEBUG_BASE_IMAGE="${IMAGE_PREFIX}_debug_base:latest"

log() {
    printf "[build_debug_base] %s\n" "$*"
}

require_cmd() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        log "ERROR: required command not found: ${cmd}"
        exit 1
    fi
}

image_exists() {
    local image_tag="$1"
    docker image inspect "${image_tag}" >/dev/null 2>&1
}

main() {
    require_cmd docker

    log "Logging to: ${LOG_FILE}"

    if ! image_exists "${VANILLA_IMAGE}"; then
        log "ERROR: Vanilla image not found: ${VANILLA_IMAGE}"
        log "       Please run: ${PROJECT_ROOT}/docker/scripts/build_images.sh"
        exit 1
    fi

    log "Building debug base image: ${DEBUG_BASE_IMAGE} (base=${VANILLA_IMAGE})"
    docker build \
        -f "${PROJECT_ROOT}/docker/images/debug/Dockerfile.pgvector_debug_base" \
        --build-arg "BASE_IMAGE=${VANILLA_IMAGE}" \
        -t "${DEBUG_BASE_IMAGE}" \
        "${PROJECT_ROOT}"

    log "Done. Built image: ${DEBUG_BASE_IMAGE}"
}

main "$@"
