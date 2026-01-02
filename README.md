![aker-logo](aker-logo-1.png)

# Aker
**Aker**: Density-Aware Approximate Caching for Vector Search

## Requirements

- C++14 or later
- GCC/G++ 9 or later
- Boost 1.80 or later
- FAISS (Meta/Facebook)

### Boost

```sh
./bootstrap
sudo ./b2 install --prefix=/opt/boost_1_86 --with=all

# If the dynamic linker cannot find Boost at runtime:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/boost_1_86/lib
```

### FAISS

```sh
cmake -B build . \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DFAISS_ENABLE_CUVS=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DFAISS_ENABLE_C_API=ON \
    -DFAISS_ENABLE_MKL=OFF

make -C build -j faiss
make -C build -j faiss_avx2
make -C build -j faiss_avx512

make -C build install

# If the dynamic linker cannot find FAISS at runtime:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```

For more environment/setup details, see `script/activate.sh`.

## Build

From the repository root directory (`Aker/`), run:

```sh
cmake .
make -j
```

## Run

Run scripts are located under:
- `script/pgvector-base`
- `script/pgvector-aker`

Run each script from its corresponding parent directory (e.g., from `script/pgvector-aker/` when using scripts in that folder). Each script expects the following relative directories to exist:

- `0-conf/` : Configuration `.ini` files for each PostgreSQL datastore
- `0-data/` : Datastore directory where data will be placed
- `1-init/` : Initialization files used when launching PostgreSQL (e.g., `postgresql.conf`)

Example configurations are provided in `script/configs/`. A configuration file looks like:

```text
[workload]
wtype           = workloada
name            = spacev-10m-hnsw-workloada-m32-efc64-0.3-top10
datastore       = 0-data/spacev-10m-hnsw-workloada-m16-efc128
insert_ratio    = 0.01
limit           = 10

[dataset]
base            = dataset/spacev-10m/vectors.bin.10m.npy
search          = dataset/spacev-10m/manual/spacev-sim-100k-0.3.npy
split_num       = 10000000
dim             = 100
dtype           = int8
gt_trace        = dataset/spacev-10m/manual/spacev-sim-100k-0.3-phys.pkl

[postgres]
host            = localhost
port            = 5441
user            = postgres
password        = postgres
database        = postgres
table_name      = items
psql_config     = 1-init/postgresql.conf

[pgvector]
type            = hnsw
m               = 16
ef_construction = 128
ef_search       = 152
index_name      = items_hnsw_idx
distance        = vector_l2_ops
```

Set the dataset path in `[dataset]/base`, and the query vectors in `[dataset]/search` (in `.npy` format). The scripts generate a ground-truth trace and save it to `[dataset]/gt_trace` (as a `.pkl` file).

In the script file, you can adjust the pool size. 

```sh
vector_pool_size=(
    500000
)
```

You can set a fixed fixed threshold. If this value is set to zero, Aker mode is activated. If not, the threshold value is used for approximate hits.

```sh
fixed_threshold=(
    0
)
```

## Applications

### PostgreSQL `pgvector` Extension

In `apps/pgvector/pgvector/`, run:

```sh
git checkout tags/v0.8.0
patch -p1 < ../pgvector.patch
```
