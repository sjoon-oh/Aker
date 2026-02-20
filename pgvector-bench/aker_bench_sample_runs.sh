#!/bin/bash

set -euo pipefail

#
# Convenience script to run a small sample end-to-end flow.
# Assumption: user launches this script from the pgvector-bench working directory.
#

BENCH_ROOT="$(pwd)"

# shellcheck disable=SC1091
source "${BENCH_ROOT}/ak_bench_env_activate.sh"

CONFIG_DIR="${BENCH_ROOT}/configs"
OUT_ROOT="${BENCH_ROOT}/output"
TS="$(date +"%Y%m%d_%H%M%S")"

#
# Optional GT backend selection.
# - postgres: legacy exact scan (slow, but matches legacy semantics)
# - numpy: SIMD + threaded exact GT (requires dataset.base)
#
export GT_BACKEND=${GT_BACKEND:-postgres}
export GT_NUMPY_WORKERS=${GT_NUMPY_WORKERS:-8}
export GT_NUMPY_BASE_CHUNK_ROWS=${GT_NUMPY_BASE_CHUNK_ROWS:-100000}
export GT_NUMPY_QUERY_BATCH_SIZE=${GT_NUMPY_QUERY_BATCH_SIZE:-16}

#
# Optional: auto-create clean snapshot if missing.
# WARNING: expensive on large PGDATA.
#
export AK_BENCH_CREATE_CLEAN_SNAPSHOT_IF_MISSING=${AK_BENCH_CREATE_CLEAN_SNAPSHOT_IF_MISSING:-0}

RUN_OUT="${OUT_ROOT}/sample_${TS}"
mkdir -p "${RUN_OUT}"

# -------- Search-workload (HNSW) --------
SEARCH_CFG="${CONFIG_DIR}/search_workload_hnsw.ini"
"${BENCH_ROOT}/ak_bench_run_search_workload.sh" --config "${SEARCH_CFG}" --output-dir "${RUN_OUT}"

# -------- Stress-workload (HNSW) --------
# NOTE: This requires the DB to provide topkache_invalidate_random(fraction).
STRESS_CFG="${CONFIG_DIR}/stress_workload_hnsw.ini"
INVALIDATE_FRACTION="0.10"
"${BENCH_ROOT}/ak_bench_run_stress_workload.sh" \
    --config "${STRESS_CFG}" \
    --invalidate "${INVALIDATE_FRACTION}" \
    --output-dir "${RUN_OUT}"

printf "\nDone. Output root:\n  %s\n" "${RUN_OUT}" >&2
