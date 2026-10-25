"""三層 parquet テーブル（raw / core / marts）の入出力。design.md §5 のスキーマを実装する。"""

from cascade.io.tables import (
    SCHEMAS,
    SchemaError,
    append_rows,
    duckdb_connect,
    read_table,
    run_sql,
    table_lock,
    table_path,
    write_table,
)

__all__ = [
    "SCHEMAS",
    "SchemaError",
    "append_rows",
    "duckdb_connect",
    "read_table",
    "run_sql",
    "table_lock",
    "table_path",
    "write_table",
]
