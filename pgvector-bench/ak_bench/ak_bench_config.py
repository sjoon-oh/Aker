"""Configuration loader for the refactor sample.

This module intentionally keeps compatibility with the legacy INI configs.
Only a subset of keys are used for Search-workload and Stress-workload.

In this sample scope, we hard-code:
- L2 distance operator (<->)
- GT Top-100
- Table name: items

We still parse pgvector index build parameters to support an end-to-end pipeline.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from typing import Optional

#
# Hard-coded semantics for the current experiment scope.
#
TABLE_NAME = "items"
DISTANCE_OPERATOR = "<->"  # L2
GT_TOPK = 100


@dataclass(frozen=True)
class PostgresConfig:
    """PostgreSQL connection parameters."""

    host: str
    port: int
    user: str
    password: str
    database: str


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset and trace paths."""

    base_path: Optional[str]
    search_path: str
    gt_trace_path: str
    split_num: Optional[int]
    dim: Optional[int]


@dataclass(frozen=True)
class PgvectorConfig:
    """pgvector index parameters (build + runtime)."""

    index_type: str
    index_name: str
    distance_type: str

    # Build parameters.
    m: int
    ef_construction: int
    nlist: int

    # Runtime parameters.
    ef_search: Optional[int]
    nprobe: Optional[int]


@dataclass(frozen=True)
class WorkloadConfig:
    """Workload settings."""

    wtype: str
    name: str
    limit: int
    insert_ratio: Optional[float]


@dataclass(frozen=True)
class BenchConfig:
    """Unified configuration object."""

    config_path: str
    postgres: PostgresConfig
    dataset: DatasetConfig
    pgvector: PgvectorConfig
    workload: WorkloadConfig

    def getWorkloadKind(self) -> str:
        """Return the normalized workload kind: 'search' or 'stress'."""

        wtype = self.workload.wtype.strip().lower()

        if wtype in ("workloada", "search-workload", "search"):
            return "search"

        if wtype.startswith("stress-") or wtype in ("stress-workload", "stress"):
            return "stress"

        # Keep legacy names observable to the caller.
        raise ValueError(f"Unsupported workload type: {self.workload.wtype}")

    def getStressMode(self) -> str:
        """Return the stress mode string compatible with legacy runners."""

        wtype = self.workload.wtype.strip().lower()

        if wtype in ("stress-workload", "stress"):
            # Default to the most common legacy mode.
            return "stress-insert"

        if wtype.startswith("stress-"):
            return wtype

        raise ValueError(f"Not a stress workload: {self.workload.wtype}")


def loadFromIni(config_path: str) -> BenchConfig:
    """Load legacy INI config into a typed BenchConfig."""

    parser = configparser.ConfigParser()
    parser.read(config_path)

    #
    # PostgreSQL section.
    #
    postgres = PostgresConfig(
        host=parser.get("postgres", "host"),
        port=parser.getint("postgres", "port", fallback=5432),
        user=parser.get("postgres", "user"),
        password=parser.get("postgres", "password"),
        database=parser.get("postgres", "database"),
    )

    #
    # Dataset section.
    #
    dataset = DatasetConfig(
        base_path=parser.get("dataset", "base", fallback=None),
        search_path=parser.get("dataset", "search"),
        gt_trace_path=parser.get("dataset", "gt_trace"),
        split_num=parser.getint("dataset", "split_num", fallback=None),
        dim=parser.getint("dataset", "dim", fallback=None),
    )

    #
    # Workload section.
    #
    workload = WorkloadConfig(
        wtype=parser.get("workload", "wtype"),
        name=parser.get("workload", "name", fallback="unknown"),
        limit=parser.getint("workload", "limit", fallback=10),
        insert_ratio=parser.getfloat("workload", "insert_ratio", fallback=None),
    )

    #
    # pgvector section.
    #
    index_type = parser.get("pgvector", "type")
    index_name = parser.get("pgvector", "index_name", fallback="")
    if index_name.strip() == "":
        # Preserve the legacy fallback behavior (even though it is flawed),
        # but make it explicit.
        index_name = f"index_${index_type}_idx"

    distance_type = parser.get("pgvector", "distance", fallback="vector_l2_ops")

    pgvector = PgvectorConfig(
        index_type=index_type,
        index_name=index_name,
        distance_type=distance_type,
        m=parser.getint("pgvector", "m", fallback=16),
        ef_construction=parser.getint("pgvector", "ef_construction", fallback=64),
        nlist=parser.getint("pgvector", "nlist", fallback=100),
        ef_search=parser.getint("pgvector", "ef_search", fallback=None),
        nprobe=parser.getint("pgvector", "nprobe", fallback=None),
    )

    return BenchConfig(
        config_path=config_path,
        postgres=postgres,
        dataset=dataset,
        pgvector=pgvector,
        workload=workload,
    )
