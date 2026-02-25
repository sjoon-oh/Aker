#!/bin/bash

set -euo pipefail

#
# Debug variant of Search-workload benchmark.
#
# Differences vs ak_bench_run_search_workload.sh:
# - Uses a *debug base* Docker image (derived from pgvector vanilla) that contains build deps + FAISS.
# - Before starting PostgreSQL, it rebuilds:
#   1) Aker from a baseline checkout (image) + local overrides (inc/src/test)
#   2) pgvector from a locally cloned repo (apps/pgvector/pgvector)
# - The container is removed after the run (docker rm), keeping the base image untouched.
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

# Debug build options.
AKER_DEBUG_MODE="${AKER_DEBUG_MODE:-standard}"
AKER_DEBUG_BUILD_TYPE="${AKER_DEBUG_BUILD_TYPE:-Release}"
DEBUG_IMAGE_DEFAULT="${AK_BENCH_DEBUG_DOCKER_IMAGE:-aker_pgvector_debug_base:latest}"

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
        --aker-mode)
            AKER_DEBUG_MODE="$2"
            shift 2
            ;;
        --aker-build-type)
            AKER_DEBUG_BUILD_TYPE="$2"
            shift 2
            ;;
        --docker-image)
            DEBUG_IMAGE_DEFAULT="$2"
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

CONFIG_PATH="$(resolve_config_path "${CONFIG_PATH}")"

# Force docker mode.
export AK_BENCH_DOCKER_IMAGE="${DEBUG_IMAGE_DEFAULT}"

# Ensure the debug container is ephemeral.
export AK_BENCH_DOCKER_REMOVE_CONTAINER_ON_EXIT=1
export AK_BENCH_DOCKER_FORCE_RECREATE=0

# Local pgvector source must exist.
LOCAL_PGVECTOR_DIR="${PROJECT_ROOT}/apps/pgvector/pgvector"
if [[ ! -d "${LOCAL_PGVECTOR_DIR}" ]]; then
    log_err "Local pgvector repo is missing: ${LOCAL_PGVECTOR_DIR}"
    log_err "Clone pgvector v0.8.0 into that path and apply your local modifications."
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/runs/search_debug_${RUN_ID}"
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

capture_tmp_trace_list "${TMP_BEFORE_LIST}"

prepare_docker_environment "${CONFIG_PATH}"

rebuild_debug_stack_in_container() {
    local mode="$1"
    local build_type="$2"

    case "${mode}" in
        standard|potluck|proximity) ;;
        *)
            log_err "Unknown --aker-mode: ${mode} (expected: standard|potluck|proximity)"
            exit 1
            ;;
    esac

    local mode_flags=""
    case "${mode}" in
        standard)
            mode_flags="-DAKER_ENABLE_PROXIMITY_MODE=OFF -DAKER_ENABLE_POTLUCK_MODE=OFF"
            ;;
        potluck)
            mode_flags="-DAKER_ENABLE_PROXIMITY_MODE=OFF -DAKER_ENABLE_POTLUCK_MODE=ON"
            ;;
        proximity)
            mode_flags="-DAKER_ENABLE_PROXIMITY_MODE=ON -DAKER_ENABLE_POTLUCK_MODE=OFF"
            ;;
    esac

    local aker_src_dir="/opt/src/aker"
    local pg_config="/usr/local/pgsql/bin/pg_config"
    local pgvector_tmp="/tmp/pgvector_debug_src"

    if [[ -z "${AK_BENCH_DOCKER_CONTAINER_INTERNAL:-}" ]]; then
        log_err "Docker container is not initialized (AK_BENCH_DOCKER_CONTAINER_INTERNAL is empty)"
        exit 1
    fi

    local container_name="${AK_BENCH_DOCKER_CONTAINER_INTERNAL}"

    log_info "[debug] Rebuilding Aker (mode=${mode}, build_type=${build_type}) and pgvector from local sources"

    log_info "[debug] Overwriting container sources with host working tree"
    docker_exec_root "rm -rf '${aker_src_dir}/inc' '${aker_src_dir}/src' '${aker_src_dir}/test' '${pgvector_tmp}' || true"
    docker_exec_root "mkdir -p '${aker_src_dir}' '${pgvector_tmp}'"

    docker cp "${PROJECT_ROOT}/inc" "${container_name}:${aker_src_dir}/"
    docker cp "${PROJECT_ROOT}/src" "${container_name}:${aker_src_dir}/"
    if [[ -d "${PROJECT_ROOT}/test" ]]; then
        docker cp "${PROJECT_ROOT}/test" "${container_name}:${aker_src_dir}/"
    fi

    # Copy pgvector sources from the host working tree.
    # Use '/.' to copy contents, and rely on the pre-cleaned destination to avoid stale files.
    docker cp "${LOCAL_PGVECTOR_DIR}/." "${container_name}:${pgvector_tmp}/"

    docker_exec_root "\
        set -euo pipefail; \
        echo '[debug] Building and installing Aker'; \
        rm -rf '${aker_src_dir}/build' || true; \
        cmake -S '${aker_src_dir}' -B '${aker_src_dir}/build' -G Ninja \
            -DCMAKE_BUILD_TYPE='${build_type}' \
            -DCMAKE_INSTALL_PREFIX=/usr/local \
            ${mode_flags}; \
        cmake --build '${aker_src_dir}/build' -j\$(nproc); \
        cmake --install '${aker_src_dir}/build'; \
        ldconfig; \
        echo '[debug] Building and installing pgvector from debug source tree'; \
        cd '${pgvector_tmp}'; \
        make -j\$(nproc) PG_CONFIG='${pg_config}'; \
        make PG_CONFIG='${pg_config}' install; \
        ldconfig; \
        echo '[debug] Build/install done.'\
    "
}

verify_debug_installation_or_abort() {
    #
    # Verify that:
    #  1) libaker.so exists in the default install path.
    #  2) The installed vector.so depends on libaker.so.
    #
    # Abort early (with full cleanup) if any check fails.
    #
    local pg_config="/usr/local/pgsql/bin/pg_config"
    local pkglibdir
    pkglibdir="$(docker_exec_root "${pg_config} --pkglibdir")"

    local libaker_path="/usr/local/lib/libaker.so"
    local vector_so="${pkglibdir}/vector.so"

    local libaker_exists="NO"
    if docker_exec_root "test -f '${libaker_path}'"; then
        libaker_exists="YES"
    fi

    local vector_depends="NO"
    if docker_exec_root "test -f '${vector_so}' && ldd '${vector_so}' | grep -q 'libaker\.so'"; then
        vector_depends="YES"
    fi

    log_info "[debug] Install verification:"
    log_info "[debug]   libaker.so exists at ${libaker_path}: ${libaker_exists}"
    log_info "[debug]   vector.so depends on libaker.so: ${vector_depends} (vector_so=${vector_so})"

    {
        echo "libaker_path=${libaker_path}"
        echo "libaker_exists=${libaker_exists}"
        echo "pkglibdir=${pkglibdir}"
        echo "vector_so=${vector_so}"
        echo "vector_depends_on_libaker=${vector_depends}"
        echo ""
        echo "ldd_vector_so="
        docker_exec_root "ldd '${vector_so}' || true"
    } > "${RUN_DIR}/debug_install_check.txt"

    if [[ "${libaker_exists}" != "YES" || "${vector_depends}" != "YES" ]]; then
        log_err "[debug] Installation verification failed; aborting run early."
        log_err "[debug] See ${RUN_DIR}/debug_install_check.txt for details."
        exit 1
    fi
}

rebuild_debug_stack_in_container "${AKER_DEBUG_MODE}" "${AKER_DEBUG_BUILD_TYPE}"
verify_debug_installation_or_abort

restore_clean_snapshot_and_start "${CONFIG_PATH}" "search" "${AKER_CONFIG_OVERRIDE}"

if [[ "${FORCE_GENERATE}" == "true" ]]; then
    run_bench_cli generate-search-workload --config "${CONFIG_PATH}" --force
else
    run_bench_cli generate-search-workload --config "${CONFIG_PATH}"
fi

run_bench_cli_numactl run-search-workload --config "${CONFIG_PATH}" --output-dir "${RUN_DIR}"

printf "[OK] Search-debug-workload finished: %s\n" "${RUN_DIR}"
