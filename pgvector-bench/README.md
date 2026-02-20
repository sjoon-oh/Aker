# pgvector-bench (Search-workload / Stress-workload)

This directory is a refactoring of the legacy `script/pgvector-*/2-runs/*.py` pipeline.

## Goals
- Keep **workload pickle formats** compatible with the legacy scripts.
- Keep **report output formats** compatible (`report.csv`, `search-results.pkl`, `trace-extract-info.csv`).
- Keep compatibility with the revised Aker/TopKache runtime by **not changing** the SQL protocol used by the runners.
- Current project scope (hard-coded):
  - Distance operator: **L2** (`<->`)
  - Ground truth size: **Top-100**
  - Table name: **items**

## Directory layout (assume you run from this directory)

```text
pgvector-bench/
├─ ak_bench/                 # python -m ak_bench ...
├─ configs/                  # INI files
├─ dataset/                  # Place datasets here (dataset/<DATASET>/...)
│  └─ SPACEV/                # example dataset name
├─ datastore/                # PostgreSQL data directories (PGDATA) live here
├─ output/                   # Results, merged /tmp traces, generated workloads
└─ *.sh                      # Bash entrypoints (run from here)
```

## What is supported
- **Search-workload** (legacy `workloada` trace format: `list[dict]`)
- **Stress-workload** (legacy `trace-stress.py` format: `dict{search:list[dict], insert:list[dict]}`)

## Key behaviors kept from legacy
- **GT generation (default)** uses PostgreSQL exact scan:
  - `SET LOCAL enable_indexscan = off`
  - `SELECT id, embedding <-> '[...]' AS _score FROM items ORDER BY _score LIMIT 100;`
- Optional GT backend: **NumPy SIMD + threaded exact GT** (`GT_BACKEND=numpy`)
  - `gt_scores` are stored as **Euclidean distance** (sqrt), matching PostgreSQL `<->` semantics.
- **Benchmark SQL** is identical to legacy runners:
  - `SELECT id, embedding <-> %s AS _score FROM items ORDER BY _score LIMIT %s;`
  - `SET hnsw.ef_search = ...` / `SET ivfflat.probes = ...`
  - Stress invalidation: `SELECT topkache_invalidate_random(fraction);`
- **Benchmark client is always NUMA pinned** (required):
  - `numactl --cpunodebind=0 --membind=0 ...`
- If `postgres.datastore` is configured in the INI, the runner reproduces legacy safeguards:
  - restore `<datastore_clean> -> <datastore>`
  - **ALWAYS** drop OS page cache (sudo password prompt is OK)
  - export `[env]` variables so PostgreSQL can read `getenv()` (e.g., `AKER_CONFIG_PATH`)
  - start postgres using `pg_ctl -D <datastore>`
- After each benchmark run, newly created **/tmp trace directories** are collected:
  - per-run: `output/.../runs/<run_id>/tmp_traces/`
  - merged: `output/.../merged_tmp_traces/`

## Pipeline overview

```text
(A) Prepare storage once (base pgvector)
    - initdb
    - create table
    - COPY vectors (contiguous IDs)
    - build index
    - create PGDATA-clean snapshot

(B) Generate workload (when storage exists)
    - restore clean snapshot (NO cache drop)
    - start postgres
    - generate *.pkl (legacy format)
    - fix-trace (legacy hole fixer)
    - stop postgres

(C) Run benchmark (repeatable)
    - restore clean snapshot
    - ALWAYS drop OS page cache
    - export [env] (and optional --aker-config override)
    - start postgres
    - run benchmark under numactl
    - collect /tmp aker traces
    - stop postgres
```

## Sample configs
Example INI files under `configs/`:
- `configs/search_workload_hnsw.ini`
- `configs/stress_workload_hnsw.ini`
- `configs/search_workload_ivfflat.ini`
- `configs/stress_workload_ivfflat.ini`

You must edit these values to match your environment:
- `dataset.base`, `dataset.search` (place the `.npy` files under `dataset/<DATASET>/`)
- `dataset.gt_trace` (recommended under `output/workloads/...`)
- `postgres.host/port/user/password/database`
- (recommended for reproducible repeated runs)
  - `postgres.datastore` / `postgres.datastore_clean`

### PostgreSQL configuration file (`postgresql.conf`)

The legacy scripts used a custom `postgresql.conf`. This refactor ships the same file at:

- `configs/postgresql.conf`

By default, the bash scripts pass this file to PostgreSQL via `pg_ctl -o "--config-file=..."`.
You can override it per INI using:

- `postgres.psql_config = <relative or absolute path>`

If `postgres.psql_config` is empty, the scripts fall back to `configs/postgresql.conf` when present.

### About `postgres.datastore`
- If `postgres.datastore` is a **name** (no '/'), it maps to: `datastore/<name>`.
- If it contains '/', it is treated as a **relative path under pgvector-bench/**.

## How to run

### 1) Prepare storage only (COPY + index build + clean snapshot)

```bash
./ak_bench_prepare_storage_pgvector.sh --config search_workload_hnsw.ini
```

(If you pass just the ini filename, the script looks under `configs/` automatically.)

### 2) Generate workload only (when storage exists)

```bash
./ak_bench_generate_workload_from_storage.sh --config search_workload_hnsw.ini
```

### 3) Run Search-workload benchmark

```bash
./ak_bench_run_search_workload.sh --config search_workload_hnsw.ini --output-dir output
```

### 4) Run Stress-workload benchmark

```bash
./ak_bench_run_stress_workload.sh --config stress_workload_hnsw.ini --invalidate 0.10 --output-dir output
```

### 5) End-to-end driver

```bash
./ak_bench_end_to_end_pgvector.sh --config search_workload_hnsw.ini --workload search --runs 3 --output-dir output
```

### Passing the Aker bootstrap/config file (getenv)

If your Aker-integrated pgvector reads `AKER_CONFIG_PATH` via `getenv()`, you can provide it at runtime:

```bash
./ak_bench_run_search_workload.sh --config search_workload_hnsw.ini --aker-config configs/aker_bootstrap.ini
```

This overrides any `AKER_CONFIG_PATH` set in `[env]`.

### Optional: enable the NumPy GT backend

```bash
export GT_BACKEND=numpy
./ak_bench_generate_workload_from_storage.sh --config search_workload_hnsw.ini
```

