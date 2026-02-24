#!/bin/bash

set -euo pipefail

#
# End-to-end pgvector benchmark driver (legacy-style).
#
# This script demonstrates a complete pipeline:
#   1) Prepare storage (initdb + create table + COPY + build index + clean snapshot)
#   2) For each run:
#        - restore clean snapshot
#        - ALWAYS drop OS page cache
#        - export [env] variables so postgres can read getenv (e.g., AKER_CONFIG_PATH)
#        - start postgres
#        - (re)generate workload pkl if missing (optional force)
#        - run benchmark (NUMA pinned via numactl)
#        - collect /tmp traces (e.g., /tmp/aker_trace_*)
#        - stop postgres
#
# NOTE:
# - Workload formats (pkl) and output formats are kept compatible with legacy scripts.
# - Assumption: user launches this script from the pgvector-bench working directory.
#

BENCH_ROOT="$(pwd)"

CONFIG_PATH=""
OUTPUT_ROOT="output"
WORKLOAD_KIND="search"  # search | stress
RUNS=1
INVALIDATE_FRACTION="0.10"
FORCE_GENERATE=false
FORCE_REBUILD_STORAGE=false
AKER_CONFIG_OVERRIDE=""

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
        --workload)
            WORKLOAD_KIND="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --invalidate)
            INVALIDATE_FRACTION="$2"
            shift 2
            ;;
        --force)
            FORCE_GENERATE=true
            shift 1
            ;;
        --force-rebuild-storage)
            FORCE_REBUILD_STORAGE=true
            shift 1
            ;;
        --aker-config)
            AKER_CONFIG_OVERRIDE="$2"
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

# Normalize config path under configs/ if needed.
# shellcheck disable=SC1091
source "${BENCH_ROOT}/ak_bench_env_activate.sh"
# shellcheck disable=SC1091
source "${BENCH_ROOT}/ak_bench_pg_safeguard.sh"
CONFIG_PATH="$(resolve_config_path "${CONFIG_PATH}")"

mkdir -p "${OUTPUT_ROOT}"

#
# Prepare storage (create clean snapshot) if missing or forced.
#
DATASTORE_VALUE="$(ini_get_value "${CONFIG_PATH}" "postgres" "datastore")"
DATASTORE_PATH="$(resolve_datastore_path "${DATASTORE_VALUE}")"
DATASTORE_CLEAN_VALUE="$(ini_get_value "${CONFIG_PATH}" "postgres" "datastore_clean")"
if [[ -z "${DATASTORE_CLEAN_VALUE}" ]]; then
    DATASTORE_CLEAN_PATH="${DATASTORE_PATH}-clean"
else
    DATASTORE_CLEAN_PATH="$(resolve_datastore_path "${DATASTORE_CLEAN_VALUE}")"
fi

if [[ "${FORCE_REBUILD_STORAGE}" == "true" || ! -d "${DATASTORE_CLEAN_PATH}" ]]; then
    if [[ "${FORCE_REBUILD_STORAGE}" == "true" ]]; then
        printf "[INFO] Forcing storage rebuild (--force-rebuild-storage)\n"
        "${BENCH_ROOT}/ak_bench_prepare_storage_pgvector.sh" --config "${CONFIG_PATH}" --force
    else
        printf "[INFO] Clean snapshot missing; preparing storage first: %s\n" "${DATASTORE_CLEAN_PATH}"
        "${BENCH_ROOT}/ak_bench_prepare_storage_pgvector.sh" --config "${CONFIG_PATH}"
    fi
fi

#
# Run benchmark 반복.
#
for ((i=1; i<=RUNS; i++)); do
    printf "[INFO] Run %d/%d (%s)\n" "${i}" "${RUNS}" "${WORKLOAD_KIND}"

    case "${WORKLOAD_KIND}" in
        search)
            args=(--config "${CONFIG_PATH}" --output-dir "${OUTPUT_ROOT}")
            if [[ "${FORCE_GENERATE}" == "true" ]]; then
                args+=(--force)
            fi
            if [[ -n "${AKER_CONFIG_OVERRIDE}" ]]; then
                args+=(--aker-config "${AKER_CONFIG_OVERRIDE}")
            fi
            "${BENCH_ROOT}/ak_bench_run_search_workload.sh" "${args[@]}"
            ;;
        stress)
            args=(--config "${CONFIG_PATH}" --output-dir "${OUTPUT_ROOT}" --invalidate "${INVALIDATE_FRACTION}")
            if [[ "${FORCE_GENERATE}" == "true" ]]; then
                args+=(--force)
            fi
            if [[ -n "${AKER_CONFIG_OVERRIDE}" ]]; then
                args+=(--aker-config "${AKER_CONFIG_OVERRIDE}")
            fi
            "${BENCH_ROOT}/ak_bench_run_stress_workload.sh" "${args[@]}"
            ;;
        *)
            printf "[ERROR] Unknown --workload: %s (expected search|stress)\n" "${WORKLOAD_KIND}" >&2
            exit 1
            ;;
    esac
 done

printf "[OK] End-to-end completed. Output: %s\n" "${OUTPUT_ROOT}"
