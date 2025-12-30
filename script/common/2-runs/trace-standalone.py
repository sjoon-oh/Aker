#!/usr/bin/env python3
import os
import pickle
import argparse
import logging
import random

import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1) Batched in-memory ground-truth search (IDs + Scores)
# -----------------------------------------------------------------------------

def process_query_batch(
    thread_id: int,
    snapshot: np.ndarray,
    base_ids: np.ndarray,
    query_vecs: np.ndarray,
    start_idx: int,
    end_idx: int,
    results_ids: list,
    results_scores: list,
    batch_size: int = 64
):
    vid_arr   = np.asarray(base_ids)
    live_mask = vid_arr != -1
    live_vecs = snapshot[live_mask]   # (N_live, dim)
    live_vids = vid_arr[live_mask]    # (N_live,)

    base_sq = np.sum(live_vecs * live_vecs, axis=1)  # ||b_j||²

    total_q = end_idx - start_idx
    pbar = tqdm(total=total_q,
                desc=f"Thread {thread_id}",
                position=thread_id,
                unit="q")

    for bstart in range(start_idx, end_idx, batch_size):
        bend    = min(bstart + batch_size, end_idx)
        q_batch = query_vecs[bstart:bend]               # (B, dim)
        q_sq    = np.sum(q_batch * q_batch, axis=1)     # ||q_i||²

        cross = -2.0 * (q_batch @ live_vecs.T)           # (B, N_live)
        D2    = cross + base_sq[None, :] + q_sq[:, None] # (B, N_live)

        cand_idx   = np.argpartition(D2, 100, axis=1)[:, :100]  
        cand_dists = D2[np.arange(bend-bstart)[:,None], cand_idx]
        order      = np.argsort(cand_dists, axis=1)
        top_pos    = cand_idx[np.arange(bend-bstart)[:,None], order]  # (B,100)
        top_dists  = D2[np.arange(bend-bstart)[:,None], top_pos]

        for i in range(bend - bstart):
            results_ids   [bstart + i] = live_vids[top_pos[i]].tolist()
            results_scores[bstart + i] = top_dists[i].tolist()
            pbar.update(1)

    pbar.close()

def find_gt_snapshot(
    snapshot: np.ndarray,
    base_ids: np.ndarray,
    query_vecs: np.ndarray,
    batch_size: int = 128,
    num_threads: int = None
) -> tuple[list[list[int]], list[list[float]]]:
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()

    num_threads = 4

    num_q          = len(query_vecs)
    results_ids    = [None] * num_q
    results_scores = [None] * num_q

    ranges = []
    for t in range(num_threads):
        s =   t     * num_q // num_threads
        e = (t + 1) * num_q // num_threads if t < num_threads-1 else num_q
        ranges.append((s, e))

    with ThreadPoolExecutor(max_workers=num_threads) as exe:
        futures = []
        for tid, (s, e) in enumerate(ranges):
            futures.append(
                exe.submit(
                    process_query_batch,
                    tid,
                    snapshot,
                    base_ids,
                    query_vecs,
                    s, e,
                    results_ids,
                    results_scores,
                    batch_size
                )
            )
        for f in futures:
            f.result()

    return results_ids, results_scores

# -----------------------------------------------------------------------------
# 2) Trace-building helpers
# -----------------------------------------------------------------------------

def find_search_segments(op_list: list[dict]) -> list[tuple[int,int]]:
    segments, start = [], None
    for i, op in enumerate(op_list):
        if op['operation'] == 'search':
            if start is None:
                start = i
        else:
            if start is not None:
                segments.append((start, i-1))
                start = None
    if start is not None:
        segments.append((start, len(op_list)-1))
    return segments

# -----------------------------------------------------------------------------
# 3) Workload-A trace generation
# -----------------------------------------------------------------------------

def make_workloada_trace(
    trace_path: str,
    base_path: str,
    search_path: str,
    batch_size: int
):
    if os.path.exists(trace_path):
        logging.info(f"WorkloadA trace already exists at {trace_path}; skipping.")
        return

    # Load search vectors
    query_vectors = np.load(search_path)

    # Initialize trace
    trace = [{
        'operation': 'search',
        'payload':   vec,
        'gt_ids':    None,
        'gt_scores': None
    } for vec in query_vectors]

    # Load base snapshot + IDs
    base_vectors = np.load(base_path)
    base_ids     = np.arange(len(base_vectors), dtype=int)

    # The shape must match
    if base_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError("Base vectors and search vectors must have the same dimension.")

    # Compute GT per search-segment
    for (s, e) in find_search_segments(trace):
        logging.info(f"WorkloadA: computing GT for segment [{s},{e}]")
        qs = np.stack([trace[i]['payload'] for i in range(s, e+1)], axis=0)
        id_lists, score_lists = find_gt_snapshot(
            base_vectors, base_ids, qs, batch_size=batch_size
        )
        for idx, (ids, scores) in enumerate(zip(id_lists, score_lists)):
            trace[s+idx]['gt_ids']    = np.array(ids,    dtype=np.int32)
            trace[s+idx]['gt_scores'] = np.array(scores, dtype=np.float32)

    # Save
    with open(trace_path, 'wb') as f:
        pickle.dump(trace, f)
    logging.info(f"WorkloadA trace saved to {trace_path}")

# -----------------------------------------------------------------------------
# 4) Workload-B trace generation
# -----------------------------------------------------------------------------

def make_workloadb_trace(
    trace_path:   str,
    base_pattern: str,
    search_path:  str,
    split_num:    int,
    insert_ratio: float,
    batch_size:   int
):
    if os.path.exists(trace_path):
        logging.info(f"WorkloadB trace already exists at {trace_path}; skipping.")
        return

    # Load/concatenate base vectors
    if '*' in base_pattern:
        import glob
        files = sorted(glob.glob(base_pattern))
        loaded = 0
        for p in files:
            vecs = np.load(p)
            base_vectors = vecs if loaded == 0 else np.vstack([base_vectors, vecs])
            loaded += len(vecs)
            if loaded >= split_num * 2:
                break
    else:
        base_vectors = np.load(base_pattern)
    base_ids = np.arange(len(base_vectors), dtype=int)

    # Load search vectors and prepare inserts/deletes
    search_vectors = np.load(search_path)
    first_split    = base_vectors[:split_num]
    second_split   = base_vectors[split_num:split_num*2]

    insert_num     = int(len(first_split) * insert_ratio)
    insert_vectors = second_split[np.random.choice(len(second_split),
                                                  insert_num, replace=False)]
    delete_num     = insert_num
    delete_ids     = np.random.choice(base_ids, size=delete_num, replace=False)

    # Build interleaved trace
    total_ops   = len(search_vectors) + insert_num + delete_num
    ops         = ['search', 'insert', 'delete']
    p_search    = len(search_vectors) / total_ops
    p_insert    = p_delete = (1 - p_search) / 2
    probs       = [p_search, p_insert, p_delete]
    idx_s = idx_i = idx_d = 0
    left_s = len(search_vectors)
    left_i = insert_num
    left_d = delete_num

    trace = []
    for _ in range(total_ops):
        op = random.choices(ops, weights=probs, k=1)[0]
        if op == 'search' and left_s > 0:
            trace.append({'operation':'search','payload':search_vectors[idx_s],
                          'gt_ids':None,'gt_scores':None})
            idx_s += 1; left_s -= 1
        elif op == 'insert' and left_i > 0:
            trace.append({'operation':'insert','payload':insert_vectors[idx_i],
                          'gt_ids':None,'gt_scores':None})
            idx_i += 1; left_i -= 1
        elif op == 'delete' and left_d > 0:
            trace.append({'operation':'delete','payload':delete_ids[idx_d],
                          'gt_ids':None,'gt_scores':None})
            idx_d += 1; left_d -= 1
        else:
            # fallback if one category is exhausted
            if left_s > 0:
                trace.append({'operation':'search','payload':search_vectors[idx_s],
                              'gt_ids':None,'gt_scores':None})
                idx_s += 1; left_s -= 1
            elif left_i > 0:
                trace.append({'operation':'insert','payload':insert_vectors[idx_i],
                              'gt_ids':None,'gt_scores':None})
                idx_i += 1; left_i -= 1
            else:
                trace.append({'operation':'delete','payload':delete_ids[idx_d],
                              'gt_ids':None,'gt_scores':None})
                idx_d += 1; left_d -= 1

        total_left = left_s + left_i + left_d
        if total_left == 0:
            break
        p_search = round(left_s / total_left, 4)
        p_insert = round(left_i / total_left, 4)
        p_delete = round(left_d / total_left, 4)
        probs    = [p_search, p_insert, p_delete]

    # Compute GT for each search segment
    for (s, e) in find_search_segments(trace):
        logging.info(f"WorkloadB: computing GT for segment [{s},{e}]")
        qs = np.stack([trace[i]['payload'] for i in range(s, e+1)], axis=0)
        id_lists, score_lists = find_gt_snapshot(
            base_vectors, base_ids, qs, batch_size=batch_size
        )
        for idx, (ids, scores) in enumerate(zip(id_lists, score_lists)):
            trace[s+idx]['gt_ids']    = np.array(ids,    dtype=np.int32)
            trace[s+idx]['gt_scores'] = np.array(scores, dtype=np.float32)

    # Save
    with open(trace_path, 'wb') as f:
        pickle.dump(trace, f)
    logging.info(f"WorkloadB trace saved to {trace_path}")

# -----------------------------------------------------------------------------
# 5) Main entry
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate pgvector workload trace with in-memory GT (IDs+Scores)"
    )
    parser.add_argument(
        "--workload-type",
        choices=["workloada", "workloadb"],
        required=True,
        help="Which workload to generate"
    )
    parser.add_argument(
        "--trace-path",
        required=True,
        help="Output path for pickled trace"
    )
    parser.add_argument(
        "--base-vectors",
        required=True,
        help="Path or glob for base-vector .npy files"
    )
    parser.add_argument(
        "--search-vectors",
        required=True,
        help="Path to search/query-vector .npy file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size for ground-truth search"
    )
    parser.add_argument(
        "--split-num",
        type=int,
        default=None,
        help="(workloadb only) split index for base vectors"
    )
    parser.add_argument(
        "--insert-ratio",
        type=float,
        default=0.001,
        help="(workloadb only) fraction of second split to insert"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.workload_type == "workloada":
        make_workloada_trace(
            trace_path   = args.trace_path,
            base_path    = args.base_vectors,
            search_path  = args.search_vectors,
            batch_size   = args.batch_size
        )

    else:  # workloadb
        if args.split_num is None:
            parser.error("--split-num is required for workloadb")
        make_workloadb_trace(
            trace_path   = args.trace_path,
            base_pattern = args.base_vectors,
            search_path  = args.search_vectors,
            split_num    = args.split_num,
            insert_ratio = args.insert_ratio,
            batch_size   = args.batch_size
        )