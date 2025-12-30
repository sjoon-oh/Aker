#!/usr/bin/env python3
import configparser
import os
import numpy as np
import pickle
import argparse
import random
import logging
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pgvector.psycopg
import psycopg

import dscheck


# ---------------------------
# Connectivity & config utils
# ---------------------------
def connect_to_postgres(config: configparser.ConfigParser) -> psycopg.Connection:
    """
    Create a fresh PostgreSQL connection (one per thread).
    """
    db_params = {
        'host': config.get('postgres', 'host'),
        'dbname': config.get('postgres', 'database'),
        'user': config.get('postgres', 'user'),
        'password': config.get('postgres', 'password'),
        "autocommit": True,
        "port": config.getint('postgres', 'port', fallback=5432)
    }
    conn = psycopg.connect(**db_params)
    # Safe if already installed; cheap no-op
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    pgvector.psycopg.register_vector(conn)
    return conn


def load_configuration(config_path: str) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def get_distance_operator(config: configparser.ConfigParser) -> str:
    distance = config.get('pgvector', 'distance', fallback=None)
    if distance is None:
        logging.error("No distance type specified in [pgvector].distance")
        raise ValueError("Distance type must be specified in the configuration under 'pgvector' section.")
    if distance == 'vector_l2_ops':
        return '<->'
    elif distance == 'vector_cosine_ops':
        return '<=>'
    elif distance == 'vector_ip_ops':
        return '<#>'
    else:
        raise ValueError(
            f"Unsupported distance type: {distance}. "
            f"Supported types: 'vector_l2_ops', 'vector_cosine_ops', 'vector_ip_ops'."
        )


def split_contiguous(lst: List[Any], n: int) -> List[List[Any]]:
    """Split list into n contiguous chunks (last may be smaller)."""
    n = max(1, n)
    L = len(lst)
    if L == 0:
        return [[]]
    base = L // n
    rem = L % n
    chunks = []
    start = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        end = start + size
        if size > 0:
            chunks.append(lst[start:end])
        start = end
    return chunks


# ---------------------------
# Thread-safe helpers
# ---------------------------
class IDAllocator:
    """Thread-safe monotonically increasing ID allocator."""
    def __init__(self, start: int):
        self._next = start
        self._lock = threading.Lock()

    def alloc(self) -> int:
        with self._lock:
            v = self._next
            self._next += 1
            return v


class ProgressTracker:
    """Global progress across threads; logs % and ETA like the original."""
    def __init__(self, total: int, start_time: float):
        self.total = max(1, total)
        self.start_time = start_time
        self.completed = 0
        self.next_pct = 1
        self._lock = threading.Lock()

    def tick(self, end_time: float):
        with self._lock:
            self.completed += 1
            percentage = self.completed / self.total * 100.0
            if percentage >= self.next_pct:
                elapsed = end_time - self.start_time
                qps = self.completed / elapsed if elapsed > 0 else 0.0
                est_remaining = (elapsed / percentage) * (100 - percentage) if percentage > 0 else 0.0
                logging.info(f"Progress: {percentage:.2f}% completed.")
                logging.info(
                    f"Estimated remaining time: {est_remaining:.2f} seconds. "
                    f"(QPS: {qps:.2f})"
                )
                self.next_pct += 1


# ---------------------------
# Worker
# ---------------------------
def set_session_for_index(cursor, config, index_type: str):
    if index_type == 'hnsw':
        ef_search = config.getint('pgvector', 'ef_search', fallback=None)
        if ef_search is None:
            raise ValueError("Missing [pgvector].ef_search for HNSW")
        cursor.execute(f"SET hnsw.ef_search = {ef_search};")
    elif index_type == 'ivfflat':
        nprobe = config.getint('pgvector', 'nprobe', fallback=None)
        if nprobe is None:
            raise ValueError("Missing [pgvector].nprobe for IVFFlat")
        cursor.execute(f"SET ivfflat.probes = {nprobe};")
    else:
        raise ValueError(f"Unsupported index type: {index_type} (expected 'ivfflat' or 'hnsw')")


def worker_run(
    thread_id: int,
    requests_slice: List[Dict[str, Any]],
    config: configparser.ConfigParser,
    search_sql: str,
    insert_sql: str,
    delete_sql: str,
    index_type: str,
    limit: int,
    id_alloc: IDAllocator,
    progress: ProgressTracker,
) -> List[Dict[str, Any]]:
    conn = connect_to_postgres(config)
    cur = conn.cursor()
    set_session_for_index(cur, config, index_type)

    results: List[Dict[str, Any]] = []

    for req in requests_slice:
        idx = req['__idx__']
        operation = req['operation']
        req_start = time.perf_counter()

        if operation == 'search':
            search_vector = req['payload']
            if not isinstance(search_vector, np.ndarray):
                raise TypeError(f"[idx={idx}] Expected numpy.ndarray for search payload, got {type(search_vector)}")

            cur.execute(search_sql, (search_vector, limit), binary=True, prepare=True)
            rows = cur.fetchall()
            req_end = time.perf_counter()

            result_ids = [row[0] for row in rows]
            result_scores = [row[1] for row in rows]

            results.append({
                'operation': operation,
                'search_vector': search_vector,
                'result_ids': result_ids,
                'result_scores': result_scores,
                'gt_ids': req['gt_ids'],
                'gt_scores': req['gt_scores'],
                'latency': req_end - req_start,
                # will fill after merge:
                'latency_accumulated': None,
                'qps_moment': None,
                # merge helpers:
                '__idx__': idx,
                '__end_time__': req_end,
            })
            progress.tick(req_end)

        elif operation == 'insert':
            vector = req['payload']
            new_id = id_alloc.alloc()
            cur.execute(insert_sql, (new_id, vector), binary=True, prepare=True)
            req_end = time.perf_counter()

            results.append({
                'operation': operation,
                'search_vector': None,
                'result_ids': [],
                'result_scores': [],
                'gt_ids': [],
                'gt_scores': [],
                'latency': req_end - req_start,
                'latency_accumulated': None,
                'qps_moment': None,
                '__idx__': idx,
                '__end_time__': req_end,
            })
            progress.tick(req_end)

        elif operation == 'delete':
            del_id = req['payload']
            cur.execute(delete_sql, (del_id,), binary=True, prepare=True)
            req_end = time.perf_counter()

            results.append({
                'operation': operation,
                'search_vector': None,
                'result_ids': [],
                'result_scores': [],
                'gt_ids': [],
                'gt_scores': [],
                'latency': req_end - req_start,
                'latency_accumulated': None,
                'qps_moment': None,
                '__idx__': idx,
                '__end_time__': req_end,
            })
            progress.tick(req_end)

        else:
            raise ValueError(f"[idx={idx}] Unsupported operation: {operation}")

    try:
        cur.close()
    finally:
        conn.close()
    return results


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Multithreaded pgvector benchmark runner.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--threads', type=int, default=None, help='Number of worker threads (overrides config).')
    args = parser.parse_args()

    # Load configuration
    config = load_configuration(args.config)

    # Connect once to verify index and read globals
    conn0 = connect_to_postgres(config)
    cur0 = conn0.cursor()

    # Check index
    is_ready = dscheck.check_index_exists(conn0, config)
    if is_ready:
        logging.info("Index check completed successfully.")
    else:
        logging.error("Index check failed. Please ensure the index exists in the database.")
        raise SystemExit(2)

    # Check trace path
    trace_path = config.get('dataset', 'gt_trace', fallback='trace.pkl')
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Trace file {trace_path} does not exist.")

    workload_type = config.get('workload', 'wtype', fallback=None)
    if workload_type is None:
        raise ValueError("Workload type must be specified in the configuration under 'workload' section.")

    # Count rows
    cur0.execute("SELECT COUNT(*) FROM items;")
    row_count = cur0.fetchone()[0]
    logging.info(f"Rows in items: {row_count}")

    # Max vector id (for inserts)
    cur0.execute("SELECT MAX(id) FROM items;")
    max_vector_id = cur0.fetchone()[0] or 0

    # Prepare SQL strings
    limit = config.getint('workload', 'limit', fallback=10)
    distance_operator = get_distance_operator(config)
    search_sql = "SELECT id, embedding " + distance_operator + " %s AS _score FROM items ORDER BY _score LIMIT %s;"
    insert_sql = "INSERT INTO items (id, embedding) VALUES (%s, %s);"
    delete_sql = "DELETE FROM items WHERE id = %s;"

    # Index type & per-session params will be set per-thread
    index_type = config.get('pgvector', 'type', fallback=None)
    if index_type not in ['ivfflat', 'hnsw']:
        raise ValueError(f"Unsupported index type: {index_type}. Supported types: 'ivfflat' and 'hnsw'.")

    # Load trace
    with open(trace_path, 'rb') as f:
        trace = pickle.load(f)

    # Sanity: ensure search requests have GT ids
    # for i, req in enumerate(trace):
    #     if req['operation'] == 'search':
    #         gt_ids = req.get('gt_ids', None)
    #         if not gt_ids:
    #             raise ValueError(f"[idx={i}] Ground truth IDs are missing for a search request.")

    logging.info(f"Loaded trace with {len(trace)} requests.")

    # Annotate with stable original indices (for deterministic merge/output)
    annotated_trace = []
    for i, req in enumerate(trace):
        d = dict(req)
        d['__idx__'] = i
        annotated_trace.append(d)

    # Determine threads
    threads_cfg = config.getint('workload', 'threads', fallback=None)
    num_threads = args.threads if args.threads is not None else (threads_cfg if threads_cfg is not None else 1)
    num_threads = max(1, int(num_threads))
    if num_threads == 1:
        logging.info("Running single-threaded; results will match original behavior precisely.")
    else:
        logging.info(f"Running multi-threaded with {num_threads} threads (contiguous slicing).")

    # Close initial connection before spawning workers
    cur0.close()
    conn0.close()

    # Split trace and run
    slices = split_contiguous(annotated_trace, num_threads)
    id_alloc = IDAllocator(start=(max_vector_id + 1))
    workload_start = time.perf_counter()
    progress = ProgressTracker(total=len(annotated_trace), start_time=workload_start)

    all_results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        futs = [
            ex.submit(
                worker_run,
                t_id,
                chunk,
                config,
                search_sql,
                insert_sql,
                delete_sql,
                index_type,
                limit,
                id_alloc,
                progress,
            )
            for t_id, chunk in enumerate(slices)
        ]

        for fut in as_completed(futs):
            res = fut.result()
            all_results.extend(res)

    workload_end = time.perf_counter()
    total_latency = workload_end - workload_start
    qps = (len(annotated_trace) / total_latency) if total_latency > 0 else 0.0

    # Normalize timing-derived fields to mimic original semantics:
    # - latency_accumulated: completion_time - workload_start
    # - qps_moment: (#completed by this completion_time) / elapsed
    completed_order = sorted(all_results, key=lambda r: r['__end_time__'])
    for rank, r in enumerate(completed_order, start=1):
        elapsed = r['__end_time__'] - workload_start
        r['latency_accumulated'] = elapsed
        r['qps_moment'] = (rank / elapsed) if elapsed > 0 else 0.0

    # For output files, order by original idx (stable wrt single-threaded script’s notion of “i”)
    search_result = sorted(all_results, key=lambda r: r['__idx__'])

    logging.info(f"Workload type: {workload_type} completed. Total requests: {len(search_result)}.")
    logging.info(f"Total latency: {total_latency:.4f} seconds, QPS: {qps:.2f}")

    # Aggregate stats
    search_latencies = [r['latency'] for r in search_result if r['operation'] == 'search']
    delete_latencies = [r['latency'] for r in search_result if r['operation'] == 'delete']
    insert_latencies = [r['latency'] for r in search_result if r['operation'] == 'insert']

    def safe_stats(vals: List[float]):
        if not vals:
            return 0.0, 0.0, 0.0
        return float(np.mean(vals)), float(np.percentile(vals, 50)), float(np.percentile(vals, 99))

    average_search_latency, p50_search_latency, p99_search_latency = safe_stats(search_latencies)
    average_delete_latency, p50_delete_latency, p99_delete_latency = safe_stats(delete_latencies)
    average_insert_latency, p50_insert_latency, p99_insert_latency = safe_stats(insert_latencies)

    logging.info(f"Average search latency: {average_search_latency:.4f} s, P50: {p50_search_latency:.4f}, P99: {p99_search_latency:.4f}")
    logging.info(f"Average delete latency: {average_delete_latency:.4f} s, P50: {p50_delete_latency:.4f}, P99: {p99_delete_latency:.4f}")
    logging.info(f"Average insert latency: {average_insert_latency:.4f} s, P50: {p50_insert_latency:.4f}, P99: {p99_insert_latency:.4f}")

    # Recall
    recalls = []
    for r in search_result:
        if r['operation'] != 'search':
            continue
        gt_ids = set(r['gt_ids'][:limit])
        result_ids = set(r['result_ids'])
        recalls.append(len(gt_ids.intersection(result_ids)) / limit if limit > 0 else 0.0)
    avg_recall = float(np.mean(recalls)) if recalls else 0.0
    logging.info(f"Average recall: {avg_recall:.4f}, total records: {len(recalls)}")

    # report.csv
    report_file = "report.csv"
    index_type_cfg = config.get('pgvector', 'type', fallback='unknown')
    if index_type_cfg == 'ivfflat':
        search_params = f"nprobe={config.getint('pgvector', 'nprobe', fallback=-1)}"
    elif index_type_cfg == 'hnsw':
        search_params = f"ef_search={config.getint('pgvector', 'ef_search', fallback=-1)}"
    else:
        search_params = "unknown"

    workload_name = config.get('workload', 'name', fallback='unknown')

    with open(report_file, 'w') as f:
        f.write("Name\tWorkload Type\tSearch Params\tQPS\t")
        f.write("Avg Search Latency (s)\t50%ile Search Latency (s)\t99%ile Search Latency (s)\t")
        f.write("Avg Delete Latency (s)\t50%ile Delete Latency (s)\t99%ile Delete Latency (s)\t")
        f.write("Avg Insert Latency (s)\t50%ile Insert Latency (s)\t99%ile Insert Latency (s)\t")
        f.write("Avg Recall\n")
        f.write(f"{workload_name}\t{workload_type}\t{search_params}\t{qps:.2f}\t")
        f.write(f"{average_search_latency:.4f}\t{p50_search_latency:.4f}\t{p99_search_latency:.4f}\t")
        f.write(f"{average_delete_latency:.4f}\t{p50_delete_latency:.4f}\t{p99_delete_latency:.4f}\t")
        f.write(f"{average_insert_latency:.4f}\t{p50_insert_latency:.4f}\t{p99_insert_latency:.4f}\t")
        f.write(f"{avg_recall:.4f}\n")

    # search-results.pkl
    with open("search-results.pkl", 'wb') as f:
        pickle.dump(search_result, f)

    logging.info("Search results written to search-results.pkl.")

    # trace-extract-info.csv (use original idx for stable row numbering)
    recall_by_idx: Dict[int, float] = {}
    rec_it = iter(recalls)
    for r in search_result:
        if r['operation'] == 'search':
            recall_by_idx[r['__idx__']] = next(rec_it, 0.0)
    with open("trace-extract-info.csv", 'w') as f:
        for r in search_result:
            i = r['__idx__']
            op = r['operation'][0].upper() if r['operation'] else 'U'
            lat = r['latency']
            qps_moment = r['qps_moment'] if r['qps_moment'] is not None else 0.0
            if op == 'S':
                rec = recall_by_idx.get(i, 0.0)
                f.write(f"{i}\t{op}\t{rec:.4f}\t{lat:.6f}\t{qps_moment:.6f}\n")
            else:
                f.write(f"{i}\t{op}\t0.0000\t{lat:.6f}\t{qps_moment:.6f}\n")
