import numpy as np
import argparse
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def process_query_batch(
    thread_id,
    snapshot,
    base_ids,
    query_vecs,
    start_idx,
    end_idx,
    results,
    batch_size=64
):
    """
    Internal: for queries [start_idx:end_idx), compute top-100 neighbors by L2^2
    over only the live vectors (base_ids != -1), in mini-batches.
    """
    # mask out deleted slots
    vid_arr     = np.asarray(base_ids)
    live_mask   = vid_arr != -1
    live_vecs   = snapshot[live_mask]   # (N_live, dim)
    live_vids   = vid_arr[live_mask]    # (N_live,)

    # precompute ||b_j||^2
    base_sq     = np.sum(live_vecs * live_vecs, axis=1)  # (N_live,)

    total_q = end_idx - start_idx
    pbar = tqdm(total=total_q,
                desc=f"Thread {thread_id}",
                position=thread_id,
                unit="queries")

    for bstart in range(start_idx, end_idx, batch_size):
        bend    = min(bstart + batch_size, end_idx)
        q_batch = query_vecs[bstart:bend]                       # (B, dim)
        q_sq    = np.sum(q_batch * q_batch, axis=1)             # (B,)

        # compute squared distances: D2[i,j] = ||q_i||^2 + ||b_j||^2 - 2 q_i·b_j
        cross   = -2.0 * (q_batch @ live_vecs.T)               # (B, N_live)
        D2      = cross + base_sq[None, :] + q_sq[:, None]     # (B, N_live)

        # get the 100 smallest per row
        cand_idx   = np.argpartition(D2, 100, axis=1)[:, :100]  # (B,100)
        cand_dists = D2[np.arange(bend - bstart)[:, None], cand_idx]
        order      = np.argsort(cand_dists, axis=1)            # (B,100)
        top_pos    = cand_idx[np.arange(bend - bstart)[:, None], order]  # (B,100)

        # map back to original vector IDs
        for i in range(bend - bstart):
            results[bstart + i] = live_vids[top_pos[i]].tolist()
            pbar.update(1)

    pbar.close()

def find_gt_snapshot(
    snapshot: np.ndarray,
    base_ids: np.ndarray,
    query_vecs: np.ndarray,
    batch_size: int = 64,
    num_threads: int = None
) -> list[list[int]]:
    """
    Compute ground-truth top-100 neighbors for `query_vecs` against a given
    base‐vector snapshot + ID array. Returns a list of neighbor‐ID lists.
    """
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()

    num_queries = len(query_vecs)
    results     = [None] * num_queries

    # split query indices into roughly equal chunks
    thread_ranges = []
    for t in range(num_threads):
        start =   t     * num_queries // num_threads
        end   = (t + 1) * num_queries // num_threads if t < num_threads - 1 else num_queries
        thread_ranges.append((start, end))

    # launch threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for tid, (s, e) in enumerate(thread_ranges):
            futures.append(
                executor.submit(
                    process_query_batch,
                    tid,
                    snapshot,
                    base_ids,
                    query_vecs,
                    s,
                    e,
                    results,
                    batch_size
                )
            )
        # wait
        for f in futures:
            f.result()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch brute-force ground-truth search (vector-ID snapshots)."
    )
    parser.add_argument("--base-vectors",  dest="base_path",  required=True,
                        help="Path to base vectors (.npy)")
    parser.add_argument("--query-vectors", dest="query_path", required=True,
                        help="Path to query vectors (.npy)")
    parser.add_argument("--batch-size",    dest="batch_size", type=int, default=64,
                        help="Queries per mini-batch")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Loading base vectors from %s", args.base_path)
    base_vecs = np.load(args.base_path)

    # Example snapshot of IDs; replace/modify as your application requires:
    base_ids = np.arange(len(base_vecs), dtype=int)
    # e.g. base_ids[k] = -1 marks that slot as deleted.

    logging.info("Loading query vectors from %s", args.query_path)
    query_vecs = np.load(args.query_path)

    first_query = query_vecs[:10000]  # First 10k queries for ground truth
    logging.info("Running ground-truth search for the first 10k queries…")

    # run and retrieve the neighbor-ID lists
    logging.info("Starting ground-truth search over snapshot…")
    neighbors = find_gt_snapshot(
        snapshot   = base_vecs,
        base_ids   = base_ids,
        query_vecs = first_query,
        batch_size = args.batch_size
    )

    dim = base_vecs.shape[1]
    max_id = np.max(base_ids) if len(base_ids) > 0 else -1

    # Test randomly to add, mark delete vectors
    import random
    for i in range(50):
        delete_idx = random.randint(0, len(base_ids) - 1)
        base_ids[delete_idx] = -1

    for i in range(100):
        insert_vec = np.random.rand(dim).astype(np.float32)
        base_vecs = np.vstack([base_vecs, insert_vec])
        base_ids = np.append(base_ids, max_id + 1)
        max_id += 1

    second_query = query_vecs[10000:20000]  # Next 10k queries
    logging.info("Running ground-truth search for the next 10k queries…")
    neighbors += find_gt_snapshot(
        snapshot   = base_vecs,
        base_ids   = base_ids,
        query_vecs = second_query,
        batch_size = args.batch_size
    )

    # save out
    out_file = "ground_truth.npy"
    logging.info("Search complete. Saving to %s", out_file)
    np.save(out_file, neighbors)
