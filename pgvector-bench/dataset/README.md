# Dataset

Please download the dataset files from the Hugging Face dataset page:

- https://huggingface.co/datasets/sjoon-oh/aker/

If you use a different dataset name or directory layout, update the corresponding `configs/*.ini` paths.

---

## File formats

### `.npy` — NumPy vector arrays

The benchmark expects `.npy` files for vector data.
These are standard NumPy arrays loaded with `np.load(...)`.

Typical uses:

- **Base vectors**: a 2-D array with shape `(N, dim)`. Examples:
  - `spacev-10m.npy`
  - `sphere-10m.npy`

- **Query vectors**: a 2-D array with shape `(Q, dim)`. Examples:
  - `spacev-sim-100k-0.3.npy`
  - `triviaqa-sim-100k-0.3.npy`

### `.pkl` — Pre-generated workload traces

`.pkl` files for pre-generated workloads.
These are Python pickle files for benchmark, not raw vector datasets.
  - `spacev-sim-100k-0.3.pkl`
  - `triviaqa-sim-100k-0.3.pkl`

## Notes

- The repository keeps only small-scale test data.