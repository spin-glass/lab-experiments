"""cascade.io.tables のスキーマ検証・往復・DuckDB ビュー登録（design.md §5）。ネットワーク不要。"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet  # noqa: F401 — pa.parquet を使う
import pytest

from cascade import paths
from cascade.io import tables
from cascade.io.tables import SCHEMAS, SchemaError

_TS = datetime(2026, 9, 7, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def layers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    d = {k: tmp_path / k for k in ("raw", "core", "marts")}
    monkeypatch.setattr(paths, "LAYERS", d)
    return d


def _scope_rows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "photo_id": ["p1", "p2"],
            "source": ["yelp", "commons"],
            "content": ["food", "non_food"],
            "operational_truth": ["food", "out_of_scope"],
            "subject_type": [None, "doll_character_statue"],
            "boundary_flag": [False, True],
            "quality": ["ok", "ok"],
            "dup_group_id": ["g1", "g2"],
            "role": ["eval", "tune"],
            "sampling_prob": [1.0, 0.5],
            "license_short": [None, "CC BY-SA 4.0"],
            "attribution": [None, "someone"],
            "source_url": [None, "https://example.org/x"],
            "sha256": ["a" * 64, "b" * 64],
            "annotator": ["ann-01", "ann-01"],
            "guideline_version": ["0.1", "0.1"],
            "ts": [_TS, _TS],
        }
    )


def test_schemas_use_bigquery_compatible_types_only() -> None:
    allowed = {pa.string(), pa.int64(), pa.float64(), pa.bool_(), pa.timestamp("us", tz="UTC")}
    for name, schema in SCHEMAS.items():
        for field in schema:
            t = field.type
            if pa.types.is_list(t):
                assert t.value_type == pa.string(), (name, field.name)
            else:
                assert t in allowed, (name, field.name, t)


def test_design_md_tables_present() -> None:
    expected = {
        "labels",
        "gold_annotations",
        "scope_annotations",
        "predictions",
        "runs",
        "eval_sets",
        "ood_sets",
        "ood_scores",
        "gate_decisions",
    }
    assert expected <= set(SCHEMAS)
    assert SCHEMAS["runs"].names == [
        "run_id",
        "git_sha",
        "taxonomy_version",
        "rulebook_version",
        "eval_set_id",
        "model_id",
        "config_json",
        "metrics_json",
        "wandb_url",
        "ts",
    ]


def test_missing_column_raises(layers: dict[str, Path]) -> None:
    df = _scope_rows().drop("sha256")
    with pytest.raises(SchemaError, match="sha256"):
        tables.write_table("scope_annotations", df, "core")


def test_extra_column_raises(layers: dict[str, Path]) -> None:
    df = _scope_rows().with_columns(pl.lit("x").alias("business_id"))
    with pytest.raises(SchemaError, match="business_id"):
        tables.write_table("scope_annotations", df, "core")


def test_type_mismatch_raises(layers: dict[str, Path]) -> None:
    df = _scope_rows().with_columns(pl.lit("not-a-number").alias("sampling_prob"))
    with pytest.raises(SchemaError):
        tables.write_table("scope_annotations", df, "core")


def test_unknown_table_or_layer_raises(layers: dict[str, Path]) -> None:
    with pytest.raises(SchemaError):
        tables.write_table("nope", _scope_rows(), "core")
    with pytest.raises(SchemaError):
        tables.write_table("scope_annotations", _scope_rows(), "gold")


def test_scope_annotations_round_trip(layers: dict[str, Path]) -> None:
    df = _scope_rows()
    path = tables.write_table("scope_annotations", df, "core")
    assert path == layers["core"] / "scope_annotations.parquet"
    back = tables.read_table("scope_annotations", "core")
    assert back.columns == SCHEMAS["scope_annotations"].names
    assert back.schema["ts"] == pl.Datetime("us", "UTC")
    assert back.schema["boundary_flag"] == pl.Boolean
    assert back.to_dicts() == df.to_dicts()


def test_runs_round_trip_and_append(layers: dict[str, Path]) -> None:
    row = {
        "run_id": "r1",
        "git_sha": "abc",
        "taxonomy_version": "0.1.0",
        "rulebook_version": "未導入",
        "eval_set_id": "eval-v1",
        "model_id": "ViT-B-16",
        "config_json": "{}",
        "metrics_json": '{"acc": 0.5}',
        "wandb_url": None,
        "ts": _TS,
    }
    tables.append_rows("runs", [row], "marts")
    tables.append_rows("runs", [{**row, "run_id": "r2"}], "marts")
    back = tables.read_table("runs", "marts")
    assert back["run_id"].to_list() == ["r1", "r2"]
    assert back.schema["ts"] == pl.Datetime("us", "UTC")
    # 未知キーは拒否
    with pytest.raises(SchemaError, match="bogus"):
        tables.append_rows("runs", [{**row, "bogus": 1}], "marts")


def test_read_missing_returns_empty_with_schema(layers: dict[str, Path]) -> None:
    df = tables.read_table("labels", "raw")
    assert df.height == 0
    assert df.columns == SCHEMAS["labels"].names


def test_write_overwrites(layers: dict[str, Path]) -> None:
    tables.write_table("scope_annotations", _scope_rows(), "core")
    tables.write_table("scope_annotations", _scope_rows().head(1), "core")
    assert tables.read_table("scope_annotations", "core").height == 1


def test_duckdb_connect_registers_views_and_runs_neutral_sql(
    layers: dict[str, Path], tmp_path: Path
) -> None:
    tables.write_table("scope_annotations", _scope_rows(), "core")
    tables.append_rows(
        "runs",
        [
            {
                "run_id": "r1",
                "git_sha": "abc",
                "metrics_json": "{}",
                "ts": _TS,
            }
        ],
        "marts",
    )
    # 論理名 → 物理名の写像を config から注入（BigQuery では ds_core.scope_annotations 等になる）
    config = {"tables": {"scope_annotations": "scope_annotations", "runs": "runs_v1"}}
    conn = tables.duckdb_connect(config)
    assert conn.execute("SELECT count(*) FROM scope_annotations").fetchone()[0] == 2
    assert conn.execute("SELECT count(*) FROM runs_v1").fetchone()[0] == 1
    # 層付きの名前も参照できる
    assert conn.execute("SELECT count(*) FROM core.scope_annotations").fetchone()[0] == 2
    assert conn.execute("SELECT count(*) FROM marts.runs").fetchone()[0] == 1

    sql_path = tmp_path / "q.sql"
    sql_path.write_text(
        "SELECT source, count(*) AS n FROM {{scope}} "
        "WHERE role = $role GROUP BY source ORDER BY source",
        encoding="utf-8",
    )
    out = tables.run_sql(conn, sql_path, {"scope": "scope_annotations"}, {"role": "eval"})
    assert out.to_dicts() == [{"source": "yelp", "n": 1}]
    with pytest.raises(KeyError):
        tables.run_sql(conn, sql_path, {}, {"role": "eval"})


def test_duckdb_dotted_physical_name(layers: dict[str, Path]) -> None:
    tables.write_table(
        "labels",
        pl.DataFrame(
            {
                "photo_id": ["p"],
                "label_source": ["yelp"],
                "label": ["food"],
                "provenance": ["photos.json"],
                "ts": [_TS],
            }
        ),
        "raw",
    )
    conn = tables.duckdb_connect({"tables": {"labels": "ds_raw.labels"}})
    assert conn.execute("SELECT label FROM ds_raw.labels").fetchone()[0] == "food"


def test_duckdb_connect_with_no_data_dirs(layers: dict[str, Path]) -> None:
    conn = tables.duckdb_connect({})
    assert conn.execute("SELECT 1").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# レビュー指摘の回帰テスト（並行 append・SQL 注入・tz・naive datetime・型 allowlist・欠損列）
# ---------------------------------------------------------------------------


def _append_worker(worker: int, n_rows: int) -> None:
    """子プロセス（spawn）: CASCADE_DATA_DIR 配下の marts/runs.parquet に n_rows 行を追記する。"""
    from datetime import UTC, datetime

    from cascade.io import tables as t

    for i in range(n_rows):
        t.append_rows(
            "runs",
            [
                {
                    "run_id": f"w{worker}-{i}",
                    "git_sha": "abc",
                    "metrics_json": "{}",
                    "ts": datetime.now(UTC),
                }
            ],
            "marts",
        )


def test_concurrent_append_rows_do_not_lose_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """4 プロセス × 25 行の同時 append で 100 行そろい、parquet が壊れない（flock + 一意 tmp）。"""
    import multiprocessing as mp

    monkeypatch.setenv("CASCADE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(paths, "LAYERS", {k: tmp_path / k for k in ("raw", "core", "marts")})
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_append_worker, args=(w, 25)) for w in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    df = tables.read_table("runs", "marts")
    assert df.height == 100
    assert df["run_id"].n_unique() == 100
    leftovers = [p.name for p in (tmp_path / "marts").iterdir() if p.suffix == ".tmp"]
    assert leftovers == []
    assert (tmp_path / "marts" / "runs.parquet.lock").exists()


def test_write_table_uses_unique_tmp_and_leaves_none(layers: dict[str, Path]) -> None:
    tables.write_table("scope_annotations", _scope_rows(), "core")
    names = sorted(p.name for p in layers["core"].iterdir())
    assert names == ["scope_annotations.parquet", "scope_annotations.parquet.lock"] or names == [
        "scope_annotations.parquet"
    ]
    # 他プロセスの書きかけを装った固定名 tmp は触られない
    foreign = layers["core"] / "scope_annotations.parquet.tmp"
    foreign.write_bytes(b"garbage")
    tables.write_table("scope_annotations", _scope_rows().head(1), "core")
    assert foreign.read_bytes() == b"garbage"
    assert tables.read_table("scope_annotations", "core").height == 1


def test_table_lock_is_reentrant_in_thread_and_exclusive_across_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fcntl

    monkeypatch.setattr(paths, "LAYERS", {k: tmp_path / k for k in ("raw", "core", "marts")})
    with tables.table_lock("runs", "marts") as lp, tables.table_lock("runs", "marts"):
        # 別の fd から非ブロッキングで取れない（排他中）
        fd = os.open(lp, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
    fd = os.open(lp, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # 解放済み
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def test_render_sql_identifiers_only_and_literals_via_parameters(
    layers: dict[str, Path], tmp_path: Path
) -> None:
    tables.write_table("scope_annotations", _scope_rows(), "core")
    conn = tables.duckdb_connect({})
    sql_path = tmp_path / "q.sql"
    sql_path.write_text(
        "SELECT source, count(*) AS n FROM {{scope}} WHERE role = $role "
        "GROUP BY source ORDER BY source",
        encoding="utf-8",
    )
    out = tables.run_sql(conn, sql_path, {"scope": "scope_annotations"}, {"role": "eval"})
    assert out.to_dicts() == [{"source": "yelp", "n": 1}]
    # `'` を含むリテラルもパラメータなら壊れない（注入されない）
    out = tables.run_sql(
        conn, sql_path, {"scope": "scope_annotations"}, {"role": "eval' OR '1'='1"}
    )
    assert out.height == 0
    # プレースホルダは識別子専用: `'` や空白を含む値は拒否
    for bad in ("scope_annotations; DROP TABLE x", "x' OR '1'='1", "a b", "", 1, None):
        with pytest.raises(ValueError):
            tables.render_sql("SELECT * FROM {{t}}", {"t": bad})
    assert tables.render_sql("SELECT * FROM {{t}}", {"t": "ds_core.scope"}) == (
        'SELECT * FROM "ds_core"."scope"'
    )
    assert tables.to_bigquery_params("WHERE role = $role AND n > $n") == (
        "WHERE role = @role AND n > @n"
    )


def test_duckdb_session_timezone_is_utc(layers: dict[str, Path], tmp_path: Path) -> None:
    tables.write_table("scope_annotations", _scope_rows(), "core")
    conn = tables.duckdb_connect({"duckdb": None})  # `duckdb: null` でも落ちない
    assert conn.execute("SELECT current_setting('TimeZone')").fetchone()[0] == "UTC"
    df = conn.execute("SELECT ts FROM core.scope_annotations").pl()
    assert df.schema["ts"] == pl.Datetime("us", "UTC")
    # TIMESTAMPTZ の取り出し（pytz 依存）とホスト tz 非依存の文字列化
    (row,) = conn.execute(
        "SELECT ts, CAST(ts AS VARCHAR) FROM core.scope_annotations LIMIT 1"
    ).fetchall()
    assert row[0].utcoffset().total_seconds() == 0
    assert row[1].startswith("2026-09-07 12:00:00")


def test_naive_datetime_rejected(layers: dict[str, Path]) -> None:
    naive = datetime(2026, 9, 7, 12, 0, 0)
    df = _scope_rows().with_columns(pl.lit(naive).alias("ts"))
    assert df.schema["ts"] == pl.Datetime("us", None)
    with pytest.raises(SchemaError, match="naive"):
        tables.write_table("scope_annotations", df, "core")
    row = {"run_id": "r", "git_sha": "abc", "metrics_json": "{}", "ts": naive}
    with pytest.raises(SchemaError, match="naive"):
        tables.append_rows("runs", [row], "marts")
    # ISO 文字列は SchemaError（ArrowTypeError を漏らさない）
    with pytest.raises(SchemaError):
        tables.append_rows("runs", [{**row, "ts": "2026-09-07T12:00:00+00:00"}], "marts")
    assert not tables.table_path("runs", "marts").exists()


def test_cast_allowlist_rejects_silent_conversions(layers: dict[str, Path]) -> None:
    base = _scope_rows()
    with pytest.raises(SchemaError):  # 数値文字列 → float
        tables.write_table(
            "scope_annotations", base.with_columns(pl.lit("0.25").alias("sampling_prob")), "core"
        )
    with pytest.raises(SchemaError):  # int 0/1 → bool
        tables.write_table(
            "scope_annotations", base.with_columns(pl.lit(1).alias("boundary_flag")), "core"
        )
    # 許容変換: int32 → int64、float32 → float64、null 列 → 任意
    ok = base.with_columns(
        pl.col("sampling_prob").cast(pl.Float32),
        pl.lit(None).alias("attribution"),
    )
    tables.write_table("scope_annotations", ok, "core")
    labels = pl.DataFrame(
        {
            "photo_id": ["p"],
            "label_source": ["yelp"],
            "label": ["food"],
            "provenance": ["x"],
            "ts": [_TS],
        }
    )
    tables.write_table("labels", labels, "raw")
    nodes = pl.DataFrame(
        {"node_id": ["a"], "level": [1], "parent_id": [None], "ancestors": [[]]},
        schema={
            "node_id": pl.Utf8,
            "level": pl.Int32,
            "parent_id": pl.Utf8,
            "ancestors": pl.List(pl.Utf8),
        },
    )
    tables.write_table("taxonomy_nodes", nodes, "core")
    assert tables.read_table("taxonomy_nodes", "core").schema["level"] == pl.Int64


def test_read_table_fills_missing_columns_but_rejects_extra(layers: dict[str, Path]) -> None:
    path = tables.table_path("runs", "marts")
    path.parent.mkdir(parents=True)
    old = pa.table(
        {
            "run_id": ["r1"],
            "git_sha": ["abc"],
            "ts": pa.array([_TS], type=pa.timestamp("us", tz="UTC")),
        }
    )
    pa.parquet.write_table(old, path)
    df = tables.read_table("runs", "marts")
    assert df.columns == SCHEMAS["runs"].names
    assert df["metrics_json"].to_list() == [None]
    assert df.schema["metrics_json"] == pl.Utf8
    # 追記も通る（スキーマへ列を足しても過去の parquet で止まらない）
    tables.append_rows("runs", [{"run_id": "r2", "ts": _TS}], "marts")
    assert tables.read_table("runs", "marts")["run_id"].to_list() == ["r1", "r2"]
    extra = old.append_column("bogus", pa.array([1]))
    pa.parquet.write_table(extra, path)
    with pytest.raises(SchemaError, match="bogus"):
        tables.read_table("runs", "marts")


def test_ancestors_null_policy(layers: dict[str, Path]) -> None:
    schema = {
        "node_id": pl.Utf8,
        "level": pl.Int64,
        "parent_id": pl.Utf8,
        "ancestors": pl.List(pl.Utf8),
    }
    ok = pl.DataFrame(
        {
            "node_id": ["a", "b"],
            "level": [1, 2],
            "parent_id": [None, "a"],
            "ancestors": [[], ["a"]],
        },
        schema=schema,
    )
    tables.write_table("taxonomy_nodes", ok, "core")
    null_array = pl.DataFrame(
        {"node_id": ["a"], "level": [1], "parent_id": [None], "ancestors": [None]}, schema=schema
    )
    with pytest.raises(SchemaError, match="NULL 配列"):
        tables.write_table("taxonomy_nodes", null_array, "core")
    null_elem = pl.DataFrame(
        {"node_id": ["b"], "level": [2], "parent_id": ["a"], "ancestors": [["a", None]]},
        schema=schema,
    )
    with pytest.raises(SchemaError, match="NULL 要素"):
        tables.write_table("taxonomy_nodes", null_elem, "core")
