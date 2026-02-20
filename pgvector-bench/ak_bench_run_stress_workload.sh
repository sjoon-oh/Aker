#!/bin/bash

set -euo pipefail

#
# Run Stress-workload benchmark.
# Assumption: user launches this script from the pgvector-bench working directory.
#

BENCH_ROOT="$(pwd)"
# shellcheck disable=SC1091
source "${BENCH_ROOT}/ak_bench_env_activate.sh"
# shellcheck disable=SC1091
source "${BENCH_ROOT}/ak_bench_pg_safeguard.sh"

CONFIG_PATH=""
OUTPUT_ROOT="output"
INVALIDATE_FRACTION="0.10"
FORCE_GENERATE=false
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
        --invalidate)
            INVALIDATE_FRACTION="$2"
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

mkdir -p "${OUTPUT_ROOT}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/runs/stress_${RUN_ID}"
MERGED_TMP_DIR="${OUTPUT_ROOT}/merged_tmp_traces"
mkdir -p "${RUN_DIR}" "${MERGED_TMP_DIR}"

#
# Phase A: Generate stress workload (may INSERT into DB as part of legacy semantics).
# We restore from clean snapshot but DO NOT drop OS cache because this phase is not the measured benchmark.
#
restore_clean_snapshot_and_start_no_cache_drop "${CONFIG_PATH}" "stress-generate" "${AKER_CONFIG_OVERRIDE}"

if [[ "${FORCE_GENERATE}" == "true" ]]; then
    run_bench_cli generate-stress-workload --config "${CONFIG_PATH}" --force
else
    run_bench_cli generate-stress-workload --config "${CONFIG_PATH}"
fi

run_bench_cli fix-trace --config "${CONFIG_PATH}"

maybe_stop_postgres "${CONFIG_PATH}"

#
# Phase B: Timed benchmark run.
# Legacy safeguard:
# - restore clean snapshot (<datastore_clean> -> <datastore>)
# - ALWAYS drop OS page cache
# - export [env] variables for postgres getenv
#
TMP_BEFORE_LIST="${RUN_DIR}/tmp_trace_before.txt"
capture_tmp_trace_list "${TMP_BEFORE_LIST}"

restore_clean_snapshot_and_start "${CONFIG_PATH}" "stress-run" "${AKER_CONFIG_OVERRIDE}"

run_bench_cli_numactl run-stress-workload \
    --config "${CONFIG_PATH}" \
    --output-dir "${RUN_DIR}" \
    --invalidate "${INVALIDATE_FRACTION}"

collect_new_tmp_traces "${TMP_BEFORE_LIST}" "${RUN_DIR}/tmp_traces" "${MERGED_TMP_DIR}"

maybe_stop_postgres "${CONFIG_PATH}"

printf "[OK] Stress-workload finished: %s\n" "${RUN_DIR}"
