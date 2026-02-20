"""Index existence checks.

This matches the legacy `dscheck.py` behavior, including its fallback index name.
"""

from __future__ import annotations

import logging

import psycopg

from ak_bench.ak_bench_config import BenchConfig


def checkIndexExists(conn: psycopg.Connection, config: BenchConfig) -> bool:
    """Return True if the configured index exists."""

    cursor = conn.cursor()

    index_name = config.pgvector.index_name

    cursor.execute(
        """
        SELECT indexname FROM pg_indexes
        WHERE schemaname = 'public' AND indexname = %s;
        """,
        (index_name,),
    )
    index_lists = cursor.fetchall()

    flat = [item[0] for item in index_lists]
    logging.info("Checking for index '%s' in the database: %s", index_name, flat)

    if index_name in flat:
        logging.info("Index '%s' exists in the database.", index_name)
        return True

    logging.error("Index '%s' does not exist in the database.", index_name)
    return False
