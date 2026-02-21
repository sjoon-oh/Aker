#!/bin/bash

set -euo pipefail

#
# Run Search-workload benchmark.
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

# Current refactor requires Docker mode by default.
prepare_docker_environment "${CONFIG_PATH}"

mkdir -p "${OUTPUT_ROOT}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_ROOT}/runs/search_${RUN_ID}"
MERGED_TMP_DIR="${OUTPUT_ROOT}/merged_tmp_traces"
mkdir -p "${RUN_DIR}" "${MERGED_TMP_DIR}"

#
# Capture /tmp trace candidates BEFORE recovery/start, so traces created during postgres startup are also collected.
#
TMP_BEFORE_LIST="${RUN_DIR}/tmp_trace_before.txt"
capture_tmp_trace_list "${TMP_BEFORE_LIST}"

#
# Legacy safeguard:
# - restore clean snapshot (<datastore_clean> -> <datastore>) if configured
# - ALWAYS drop OS page cache
# - export [env] variables so postgres can read getenv (e.g., AKER_CONFIG_PATH)
# - start postgres using pg_ctl (if datastore is configured)
#
restore_clean_snapshot_and_start "${CONFIG_PATH}" "search" "${AKER_CONFIG_OVERRIDE}"

#
# Workload generation (pkl) is idempotent unless --force is provided.
#
if [[ "${FORCE_GENERATE}" == "true" ]]; then
    run_bench_cli generate-search-workload --config "${CONFIG_PATH}" --force
else
    run_bench_cli generate-search-workload --config "${CONFIG_PATH}"
fi

#
# Run benchmark under fixed NUMA binding.
#
run_bench_cli_numactl run-search-workload --config "${CONFIG_PATH}" --output-dir "${RUN_DIR}"

# Wait before shutdown to allow Aker to export traces (default: 600s when AKER_CONFIG_PATH is set).
maybe_wait_for_aker_trace_export

#
# Collect newly created /tmp trace directories.
#
maybe_stop_postgres "${CONFIG_PATH}"

collect_new_tmp_traces "${TMP_BEFORE_LIST}" "${RUN_DIR}/tmp_traces" "${MERGED_TMP_DIR}"

docker_cleanup_tmp_traces

maybe_shutdown_docker_container "${CONFIG_PATH}"

printf "[OK] Search-workload finished: %s\n" "${RUN_DIR}"
