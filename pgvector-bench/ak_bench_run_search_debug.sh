#!/bin/bash

set -euo pipefail

#
# Run Search-workload benchmark in a *temporary* debug container.
#
# Goal:
# - Avoid rebuilding Docker images for each code edit.
# - Rebuild and install Aker + pgvector inside a fresh container each run.
# - Remove the container at the end so the debug image remains untouched.
#
# Workflow:
#   1) Launch a debug container based on vanilla PostgreSQL + pgvector.
#   2) Copy host-side modified sources into /tmp inside the container:
#        - <project>/inc/
#        - <project>/src/
#        - <project>/test/                 (optional)
#        - <project>/apps/pgvector/pgvector
#   3) Rebuild + install Aker (selected mode) and rebuild + install pgvector.
#   4) Run the same benchmark pipeline as ak_bench_run_search_workload.sh.
#   5) Stop + remove the container.
#
# Assumption: user launches this script from the pgvector-bench working directory.
#

BENCH_ROOT="$(pwd)"

# shellcheck disable=SC1091
source "${BENCH_ROOT}/ak_bench_env_activate.sh"
# shellcheck disable=SC1091
source "${BENCH_ROOT}/ak_bench_pg_safeguard.sh"

CONFIG_PATH=""
OUTPUT_ROOT="output"
FORCE_GENERATE=false
AKER_CONFIG_OVERRIDE=""

# Debug mode selection.
AKER_MODE="${AK_BENCH_DEBUG_AKER_MODE:-standard}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --force)
            FORCE_GENERATE=true
            shift 1
            ;;
        --aker-config)
            AKER_CONFIG_OVERRIDE="$2"
            shift 2
            ;;
        --mode)
            AKER_MODE="$2"
            shift 2
            ;;
        *)
            printf "Unknown argument: %s\n" "$1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${CONFIG_PATH}" ]]; then
    printf "--config is required\n" >&2
    exit 1
fi

case "${AKER_MODE}" in
    standard|potluck|proximity)
        ;;
    *)
        printf "--mode must be one of: standard|potluck|proximity (got: %s)\n" "${AKER_MODE}" >&2
        exit 1
        ;;
esac

CONFIG_PATH="$(resolve_config_path "${CONFIG_PATH}")"

mkdir -p "${OUTPUT_ROOT}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/runs/search_debug_${AKER_MODE}_${RUN_ID}"
MERGED_TMP_DIR="${OUTPUT_ROOT}/merged_tmp_traces"
mkdir -p "${RUN_DIR}" "${MERGED_TMP_DIR}"

TMP_BEFORE_LIST="${RUN_DIR}/tmp_trace_before.txt"
CLEANUP_DONE=0

cleanup() {
    local exit_code=$?

    if [[ "${CLEANUP_DONE}" == "1" ]]; then
        exit "${exit_code}"
    fi
    CLEANUP_DONE=1

    set +e

    if [[ "${exit_code}" != "0" && -n "${AK_BENCH_DOCKER_CONTAINER_INTERNAL:-}" ]]; then
        if command -v docker >/dev/null 2>&1; then
            docker logs "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" > "${RUN_DIR}/docker_container.log" 2>&1 || true
        fi
    fi

    # Stop PostgreSQL.
    # - On success, stop gracefully and propagate failure.
    # - On error, stop best-effort only.
    if [[ "${exit_code}" == "0" ]]; then
        maybe_stop_postgres_graceful "${CONFIG_PATH}"
        local stop_rc=$?
        if [[ "${stop_rc}" != "0" ]]; then
            log_err "PostgreSQL graceful stop failed during cleanup"
            exit_code=1
        fi
    else
        maybe_stop_postgres_best_effort "${CONFIG_PATH}" || true
    fi

    # Avoid long waits on error by default.
    if [[ "${exit_code}" == "0" ]]; then
        maybe_wait_for_aker_trace_export
    else
        local wait_sec="${AK_BENCH_TRACE_EXPORT_WAIT_ON_ERROR_SEC:-0}"
        if [[ -n "${wait_sec}" && "${wait_sec}" != "0" ]]; then
            log_info "Waiting ${wait_sec} seconds for trace export after error"
            sleep "${wait_sec}"
        fi
    fi

    if [[ -f "${TMP_BEFORE_LIST}" ]]; then
        collect_new_tmp_traces "${TMP_BEFORE_LIST}" "${RUN_DIR}/tmp_traces" "${MERGED_TMP_DIR}" || true
    fi

    collect_postgres_logs "${CONFIG_PATH}" "${RUN_DIR}" || true

    docker_cleanup_tmp_traces || true

    maybe_shutdown_docker_container "${CONFIG_PATH}" || true

    exit "${exit_code}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

require_cmd docker

docker_image_exists() {
    local image_tag="$1"
    docker image inspect "${image_tag}" >/dev/null 2>&1
}

select_debug_image() {
    # Allow users to override the image manually.
    if [[ -n "${AK_BENCH_DOCKER_IMAGE:-}" ]]; then
        printf "%s" "${AK_BENCH_DOCKER_IMAGE}"
        return 0
    fi

    local prefix="${AK_BENCH_DEBUG_IMAGE_PREFIX:-aker_pgvector}"
    printf "%s_%s_debug:latest" "${prefix}" "${AKER_MODE}"
}

debug_rebuild_and_install_in_container() {
    local image_tag="$1"
    local project_root
    project_root="$(cd "${BENCH_ROOT}/.." && pwd)"

    local host_nproc
    host_nproc="$(nproc)"

    local work_dir
    work_dir="/tmp/aker_debug_${RUN_ID}"

    local mode_flags=""
    case "${AKER_MODE}" in
        standard)
            mode_flags="-DAKER_ENABLE_PROXIMITY_MODE=OFF -DAKER_ENABLE_POTLUCK_MODE=OFF"
            ;;
        potluck)
            mode_flags="-DAKER_ENABLE_PROXIMITY_MODE=OFF -DAKER_ENABLE_POTLUCK_MODE=ON"
            ;;
        proximity)
            mode_flags="-DAKER_ENABLE_PROXIMITY_MODE=ON -DAKER_ENABLE_POTLUCK_MODE=OFF"
            ;;
        *)
            log_err "Unexpected AKER_MODE=${AKER_MODE}"
            exit 1
            ;;
    esac

    local local_inc="${project_root}/inc"
    local local_src="${project_root}/src"
    local local_test="${project_root}/test"
    local local_pgvector_repo="${project_root}/apps/pgvector/pgvector"

    if [[ ! -d "${local_inc}" || ! -d "${local_src}" ]]; then
        log_err "Expected Aker sources not found under project root: ${project_root}"
        log_err "Missing required directories: inc/ and/or src/"
        exit 1
    fi

    if [[ ! -d "${local_pgvector_repo}" ]]; then
        log_err "Local pgvector working tree not found: ${local_pgvector_repo}"
        log_err "Create it by cloning pgvector v0.8.0 under apps/pgvector/pgvector and apply your edits there."
        exit 1
    fi

    local build_log="${RUN_DIR}/container_rebuild_install.log"
    log_info "Rebuilding Aker (${AKER_MODE}) + pgvector inside container (log: ${build_log})"

    # Build inside a throwaway workspace under /tmp so the image stays pristine.
    # NOTE: we run as root because installation writes into /usr/local.
    set +e
    docker_exec_root "\
        set -euo pipefail; \
        if [[ ! -d /opt/src/aker_base ]]; then \
            echo 'ERROR: baseline Aker source tree not found: /opt/src/aker_base' >&2; \
            exit 1; \
        fi; \
        rm -rf '${work_dir}'; \
        mkdir -p '${work_dir}'; \
        cp -a /opt/src/aker_base '${work_dir}/aker'; \
        rm -rf '${work_dir}/aker/inc'; cp -a '${local_inc}' '${work_dir}/aker/inc'; \
        rm -rf '${work_dir}/aker/src'; cp -a '${local_src}' '${work_dir}/aker/src'; \
        if [[ -d '${local_test}' ]]; then \
            rm -rf '${work_dir}/aker/test'; cp -a '${local_test}' '${work_dir}/aker/test'; \
        fi; \
        cmake -S '${work_dir}/aker' -B '${work_dir}/aker/build' -G Ninja \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DCMAKE_INSTALL_PREFIX=/usr/local \
            ${mode_flags}; \
        cmake --build '${work_dir}/aker/build' -j${host_nproc}; \
        cmake --install '${work_dir}/aker/build'; \
        ldconfig; \
        rm -rf '${work_dir}/pgvector'; \
        cp -a '${local_pgvector_repo}' '${work_dir}/pgvector'; \
        cd '${work_dir}/pgvector'; \
        make -j${host_nproc} PG_CONFIG=/usr/local/pgsql/bin/pg_config; \
        make PG_CONFIG=/usr/local/pgsql/bin/pg_config install; \
        echo '[debug] rebuild/install completed'\
    " >"${build_log}" 2>&1
    local rc=$?
    set -e

    if [[ "${rc}" != "0" ]]; then
        log_err "Container rebuild/install failed (see: ${build_log})"
        exit "${rc}"
    fi

    log_info "Container rebuild/install succeeded: ${image_tag}"
}

# Use a dedicated container name per run to avoid reusing mutable state.
export AK_BENCH_DOCKER_REMOVE_CONTAINER_ON_EXIT=1
export AK_BENCH_DOCKER_FORCE_RECREATE=1
export AK_BENCH_DOCKER_CONTAINER_NAME="akerbench_debug_${AKER_MODE}_${RUN_ID}_5432"

AK_BENCH_DOCKER_IMAGE="$(select_debug_image)"
export AK_BENCH_DOCKER_IMAGE

if ! docker_image_exists "${AK_BENCH_DOCKER_IMAGE}"; then
    log_err "Debug Docker image not found: ${AK_BENCH_DOCKER_IMAGE}"
    log_err "Build it from the project root: ./docker/scripts/build_debug_images.sh"
    exit 1
fi

# Start container first (required for rebuild/install).
prepare_docker_environment "${CONFIG_PATH}"

# Capture /tmp trace candidates BEFORE rebuild, so any early trace exports are also collected.
capture_tmp_trace_list "${TMP_BEFORE_LIST}"

# Rebuild/install Aker + pgvector inside the container.
debug_rebuild_and_install_in_container "${AK_BENCH_DOCKER_IMAGE}"

# Restore snapshot and start postgres (same as the normal benchmark runner).
restore_clean_snapshot_and_start "${CONFIG_PATH}" "search" "${AKER_CONFIG_OVERRIDE}"

if [[ "${FORCE_GENERATE}" == "true" ]]; then
    run_bench_cli generate-search-workload --config "${CONFIG_PATH}" --force
else
    run_bench_cli generate-search-workload --config "${CONFIG_PATH}"
fi

run_bench_cli_numactl run-search-workload --config "${CONFIG_PATH}" --output-dir "${RUN_DIR}"

printf "[OK] Search-workload finished (debug): %s\n" "${RUN_DIR}"
