import configparser
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
    """
    Worker: for queries[start_idx:end_idx),
    compute top-100 neighbor IDs + squared-L2 scores.
    """
    vid_arr   = np.asarray(base_ids)
    live_mask = vid_arr != -1
    live_vecs = snapshot[live_mask]   # (N_live, dim)
    live_vids = vid_arr[live_mask]    # (N_live,)

    base_sq = np.sum(live_vecs * live_vecs, axis=1)  # ||b_j||^2

    total_q = end_idx - start_idx
    pbar = tqdm(total=total_q,
                desc=f"Thread {thread_id}",
                position=thread_id,
                unit="queries")

    for bstart in range(start_idx, end_idx, batch_size):
        bend    = min(bstart + batch_size, end_idx)
        q_batch = query_vecs[bstart:bend]               # (B, dim)
        q_sq    = np.sum(q_batch * q_batch, axis=1)     # ||q_i||^2

        # squared distances: ||q−b||² = ||q||² + ||b||² − 2 q·b
        cross = -2.0 * (q_batch @ live_vecs.T)           # (B, N_live)
        D2    = cross + base_sq[None, :] + q_sq[:, None] # (B, N_live)

        # 1) partial sort → 100 smallest indices per row
        cand_idx   = np.argpartition(D2, 100, axis=1)[:, :100]  
        # 2) gather & sort those distances
        cand_dists = D2[np.arange(bend - bstart)[:, None], cand_idx]  # (B,100)
        order      = np.argsort(cand_dists, axis=1)                  # (B,100)
        top_pos    = cand_idx[np.arange(bend - bstart)[:, None], order]  # (B,100)

        # 3) extract IDs + scores
        top_dists  = D2[np.arange(bend - bstart)[:, None], top_pos]     # (B,100)
        for i in range(bend - bstart):
            results_ids   [bstart + i] = live_vids[top_pos[i]].tolist()
            results_scores[bstart + i] = top_dists[i].tolist()
            pbar.update(1)

    pbar.close()


def find_gt_snapshot(
    snapshot: np.ndarray,
    base_ids: np.ndarray,
    query_vecs: np.ndarray,
    batch_size: int = 64,
    num_threads: int = None
) -> tuple[list[list[int]], list[list[float]]]:
    """
    Threaded batched L2² ground-truth search.
    Returns (id_lists, score_lists), each a list of length len(query_vecs).
    """
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()

    num_q         = len(query_vecs)
    results_ids   = [None] * num_q
    results_scores= [None] * num_q

    # split into chunks per thread
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
# 2) Trace‐building helpers
# -----------------------------------------------------------------------------

def load_configuration(path: str):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg

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

def make_workloada_trace(config):
    trace_path = config.get('dataset', 'gt_trace', fallback=None)
    if not trace_path:
        raise ValueError("Set 'dataset.gt_trace' in config")
    if os.path.exists(trace_path):
        logging.info("WorkloadA trace exists; skipping.")
        return

    # load query vectors
    search_path   = config.get('dataset', 'search', fallback=None)
    if not search_path:
        raise ValueError("Set 'dataset.search' in config")
    query_vectors = np.load(search_path)

    # initialize trace entries
    trace = [{
        'operation':  'search',
        'payload':    vec,
        'gt_ids':     None,
        'gt_scores':  None
    } for vec in query_vectors]

    # load base snapshot + IDs
    base_path    = config.get('dataset', 'base', fallback=None)
    if not base_path:
        raise ValueError("Set 'dataset.base' in config")
    base_vectors = np.load(base_path)
    base_ids     = np.arange(len(base_vectors), dtype=int)

    # compute GT for each search-segment
    batch_size = config.getint('workload', 'batch_size', fallback=64)
    for (s, e) in find_search_segments(trace):
        logging.info(f"WorkloadA: computing GT for segment [{s},{e}]")
        qs        = np.stack([trace[i]['payload'] for i in range(s, e+1)], axis=0)
        id_lists, score_lists = find_gt_snapshot(
            base_vectors, base_ids, qs, batch_size=batch_size
        )
        for idx, (ids, scores) in enumerate(zip(id_lists, score_lists)):
            trace[s + idx]['gt_ids']    = np.array(ids,    dtype=np.int32)
            trace[s + idx]['gt_scores'] = np.array(scores, dtype=np.float32)

    # save trace
    with open(trace_path, 'wb') as f:
        pickle.dump(trace, f)
    logging.info(f"WorkloadA trace saved to {trace_path}")


# -----------------------------------------------------------------------------
# 4) Workload-B trace generation
# -----------------------------------------------------------------------------

def make_workloadb_trace(config):
    trace_path = config.get('dataset', 'gt_trace', fallback=None)
    if not trace_path:
        raise ValueError("Set 'dataset.gt_trace' in config")
    if os.path.exists(trace_path):
        logging.info("WorkloadB trace exists; skipping.")
        return

    # load/concatenate base vectors
    base_pattern = config.get('dataset', 'base', fallback=None)
    if not base_pattern:
        raise ValueError("Set 'dataset.base' in config")
    if '*' in base_pattern:
        import glob
        files = sorted(glob.glob(base_pattern))
        loaded = 0
        for p in files:
            vecs = np.load(p)
            base_vectors = vecs if loaded == 0 else np.vstack([base_vectors, vecs])
            loaded += len(vecs)
            if loaded >= config.getint('dataset', 'split_num') * 2:
                break
    else:
        base_vectors = np.load(base_pattern)

    base_ids = np.arange(len(base_vectors), dtype=int)

    # load search vectors + split for inserts/deletes
    search_path  = config.get('dataset', 'search', fallback=None)
    split_num    = config.getint('dataset', 'split_num', fallback=None)
    insert_ratio = config.getfloat('workload', 'insert_ratio', fallback=0.001)
    if not search_path or split_num is None:
        raise ValueError("Set 'dataset.search' and 'dataset.split_num' in config")

    search_vectors = np.load(search_path)
    first_split    = base_vectors[:split_num]
    second_split   = base_vectors[split_num:split_num*2]

    insert_num     = int(len(first_split) * insert_ratio)
    insert_vectors = second_split[np.random.choice(len(second_split),
                                                  insert_num, replace=False)]
    delete_num     = insert_num
    delete_ids     = np.random.choice(base_ids, size=delete_num, replace=False)

    # build interleaved trace
    total_ops   = len(search_vectors) + insert_num + delete_num
    ops         = ['search', 'insert', 'delete']
    p_search    = len(search_vectors) / total_ops
    p_insert    = p_delete = (1 - p_search) / 2
    probs       = [p_search, p_insert, p_delete]
    idx_search = idx_insert = idx_delete = 0
    left_search = len(search_vectors)
    left_insert = insert_num
    left_delete = delete_num

    trace = []
    for _ in range(total_ops):
        op = random.choices(ops, weights=probs, k=1)[0]
        if op == 'search' and left_search > 0:
            trace.append({'operation':'search','payload':search_vectors[idx_search],
                          'gt_ids':None,'gt_scores':None})
            idx_search += 1; left_search -= 1
        elif op == 'insert' and left_insert > 0:
            trace.append({'operation':'insert','payload':insert_vectors[idx_insert],
                          'gt_ids':None,'gt_scores':None})
            idx_insert += 1; left_insert -= 1
        elif op == 'delete' and left_delete > 0:
            trace.append({'operation':'delete','payload':delete_ids[idx_delete],
                          'gt_ids':None,'gt_scores':None})
            idx_delete += 1; left_delete -= 1
        else:
            # fallback if chosen op exhausted
            if left_search > 0:
                trace.append({'operation':'search','payload':search_vectors[idx_search],
                              'gt_ids':None,'gt_scores':None})
                idx_search += 1; left_search -= 1
            elif left_insert > 0:
                trace.append({'operation':'insert','payload':insert_vectors[idx_insert],
                              'gt_ids':None,'gt_scores':None})
                idx_insert += 1; left_insert -= 1
            else:
                trace.append({'operation':'delete','payload':delete_ids[idx_delete],
                              'gt_ids':None,'gt_scores':None})
                idx_delete += 1; left_delete -= 1

        # update probabilities
        total_left = left_search + left_insert + left_delete
        if total_left == 0:
            break
        p_search = round(left_search / total_left, 4)
        p_insert = round(left_insert / total_left, 4)
        p_delete = round(left_delete / total_left, 4)
        probs    = [p_search, p_insert, p_delete]

    # compute GT for each search segment
    batch_size = config.getint('workload', 'batch_size', fallback=64)
    for (s, e) in find_search_segments(trace):
        logging.info(f"WorkloadB: computing GT for segment [{s},{e}]")
        qs        = np.stack([trace[i]['payload'] for i in range(s, e+1)], axis=0)
        id_lists, score_lists = find_gt_snapshot(
            base_vectors, base_ids, qs, batch_size=batch_size
        )
        for idx, (ids, scores) in enumerate(zip(id_lists, score_lists)):
            trace[s + idx]['gt_ids']    = np.array(ids,    dtype=np.int32)
            trace[s + idx]['gt_scores'] = np.array(scores, dtype=np.float32)

    # save
    with open(trace_path, 'wb') as f:
        pickle.dump(trace, f)
    logging.info(f"WorkloadB trace saved to {trace_path}")


# -----------------------------------------------------------------------------
# 5) Main entry
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate pgvector workload trace with in-memory GT (IDs+Scores)"
    )
    p.add_argument('--config', required=True, help="Path to configuration file")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg   = load_configuration(args.config)
    wtype = cfg.get('workload', 'wtype', fallback=None)

    if   wtype == 'workloada':
        make_workloada_trace(cfg)
    elif wtype == 'workloadb':
        make_workloadb_trace(cfg)
    else:
        logging.error("Set workload.wtype to 'workloada' or 'workloadb'")
        exit(1)
