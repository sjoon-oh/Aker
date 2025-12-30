-- PostgreSQL pgvector index build query script

SELECT
  p.pid,
  t.relname AS table_name,
  i.relname AS index_name,
  p.phase,
  p.tuples_done,
  p.tuples_total
FROM
  pg_stat_progress_create_index p
JOIN
  pg_class t ON p.relid = t.oid
JOIN
  pg_class i ON p.index_relid = i.oid;