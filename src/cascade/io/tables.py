"""三層 parquet テーブルのスキーマ定義と入出力（design.md §5、計画 v4「記録スキーマ」）。

- 正本はローカル parquet（``paths.LAYERS[layer]/<name>.parquet``）。BigQuery 互換型のみを使う
  （string / int64 / float64 / bool / timestamp[us, UTC] / list<string>）ので、同じ論理スキーマを
  そのまま BigQuery 三層（ds_raw / ds_core / ds_marts）へ移せる（design.md §3）
- 書き込み時にスキーマ検証を行う。欠損列・余分な列はともに ``SchemaError``。型は許容変換の
  allowlist（int→int64、float→float64、string 系→string、null→任意、tz 付き timestamp→UTC）に限り、
  文字列→数値や int→bool の黙った変換はしない。naive datetime（tz なし）は拒否する
- 読み込み時は欠損列を型付き null で補い（スキーマへ列を足しても過去の parquet が読める）、
  余分な列のみ拒否する
- 並行実行の保護: ``table_lock`` が ``<name>.parquet.lock`` を ``fcntl.flock`` で排他する。
  ``append_rows`` の read→concat→write と、呼び出し側の read-modify-write（adapter の wandb_url
  更新）はこのロック内で行う。一時ファイルは一意名（tempfile）で、rename 前の書きかけを他プロセスが
  拾わない
- 分析 SQL は DuckDB / BigQuery 両対応の方言中立な書き方に限定し、テーブル名は config から
  注入する（CLAUDE.md 開発規約）。``duckdb_connect`` が論理名→物理名の写像でビューを張り、
  ``run_sql`` が ``{{table_name}}`` プレースホルダ（識別子専用）を置換し、リテラルは prepared
  parameter（DuckDB ``$name``、BigQuery では ``@name``）で渡す
"""

from __future__ import annotations

import fcntl
import os
import re
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from cascade import paths

# ---------------------------------------------------------------------------
# スキーマ（design.md §5）
# ---------------------------------------------------------------------------

_STR = pa.string()
_I64 = pa.int64()
_F64 = pa.float64()
_BOOL = pa.bool_()
_TS = pa.timestamp("us", tz="UTC")
_STR_LIST = pa.list_(pa.string())


class SchemaError(ValueError):
    """テーブル名・列・型がスキーマと一致しない。"""


SCHEMAS: dict[str, pa.Schema] = {
    # label は第2階層クラス。gold は gold_annotations.content_label を射影して label_source='gold'
    "labels": pa.schema(
        [
            ("photo_id", _STR),
            ("label_source", _STR),
            ("label", _STR),
            ("provenance", _STR),
            ("ts", _TS),
        ]
    ),
    # 第1階層（food / non-food）は持たない（導出のみ）
    "gold_annotations": pa.schema(
        [
            ("photo_id", _STR),
            ("content_label", _STR),
            ("quality", _STR),
            ("boundary_flag", _BOOL),
            ("food_visible", _BOOL),
            ("secondary", _STR),
            ("annotator", _STR),
            ("guideline_version", _STR),
            ("ts", _TS),
        ]
    ),
    # gold_annotations の拡張（G0）。ライセンス列は外部源のみ（Yelp は null）
    "scope_annotations": pa.schema(
        [
            ("photo_id", _STR),
            ("source", _STR),
            ("content", _STR),
            ("operational_truth", _STR),
            ("subject_type", _STR),
            ("boundary_flag", _BOOL),
            ("quality", _STR),
            ("dup_group_id", _STR),
            ("role", _STR),
            ("sampling_prob", _F64),
            ("license_short", _STR),
            ("attribution", _STR),
            ("source_url", _STR),
            ("sha256", _STR),
            ("annotator", _STR),
            ("guideline_version", _STR),
            ("ts", _TS),
        ]
    ),
    # probs_json は第2階層の確率（葉スコアを含めてよい）、branch_margin = S_food − S_non
    "predictions": pa.schema(
        [
            ("run_id", _STR),
            ("photo_id", _STR),
            ("stage", _STR),
            ("probs_json", _STR),
            ("margin", _F64),
            ("branch_margin", _F64),
            ("primary", _STR),
            ("secondary", _STR),
            ("flag", _STR),
            ("model_id", _STR),
            ("cost_tokens", _I64),
            ("ts", _TS),
        ]
    ),
    "runs": pa.schema(
        [
            ("run_id", _STR),
            ("git_sha", _STR),
            ("taxonomy_version", _STR),
            ("rulebook_version", _STR),
            ("eval_set_id", _STR),
            ("model_id", _STR),
            ("config_json", _STR),
            ("metrics_json", _STR),
            ("wandb_url", _STR),
            ("ts", _TS),
        ]
    ),
    # role ∈ {tune, work, eval}。photo_ids_hash で凍結
    "eval_sets": pa.schema(
        [
            ("eval_set_id", _STR),
            ("role", _STR),
            ("photo_ids_hash", _STR),
            ("definition", _STR),
            ("created_at", _TS),
        ]
    ),
    "ood_sets": pa.schema(
        [
            ("ood_set_id", _STR),
            ("source", _STR),
            ("subject_type", _STR),
            ("role", _STR),
            ("n", _I64),
            ("ids_sha256", _STR),
            ("lock_path", _STR),
            ("created_at", _TS),
        ]
    ),
    # score の向きは大＝対象内。mode ∈ {P1, P2}
    "ood_scores": pa.schema(
        [
            ("run_id", _STR),
            ("photo_id", _STR),
            ("source", _STR),
            ("mode", _STR),
            ("method_id", _STR),
            ("encoder_id", _STR),
            ("score", _F64),
            ("role", _STR),
            ("fold_axis", _STR),
            ("fold_id", _STR),
            ("ts", _TS),
        ]
    ),
    "gate_decisions": pa.schema(
        [
            ("run_id", _STR),
            ("photo_id", _STR),
            ("mode", _STR),
            ("method_id", _STR),
            ("threshold_id", _STR),
            ("decision", _STR),
            ("stage1_pred", _STR),
            ("final_label", _STR),
            ("ts", _TS),
        ]
    ),
}

# 推移閉包用（design.md §5 末尾）。ancestors は list<string>。BigQuery の ARRAY は NULL 配列・NULL 要素を
# 持てない（NULL 配列は空配列として読まれる）ので、書き込み時に NULL を拒否し第1階層は [] を明示する
SCHEMAS["taxonomy_nodes"] = pa.schema(
    [
        ("node_id", _STR),
        ("level", _I64),
        ("parent_id", _STR),
        ("ancestors", _STR_LIST),
    ]
)

# 論理名を解決するときの優先順（marts > core > raw）
_LAYER_PRIORITY = ("marts", "core", "raw")


# ---------------------------------------------------------------------------
# パス
# ---------------------------------------------------------------------------


def _schema_of(name: str) -> pa.Schema:
    try:
        return SCHEMAS[name]
    except KeyError as e:
        raise SchemaError(f"未知のテーブル名: {name!r}（既知: {sorted(SCHEMAS)}）") from e


def _layer_dir(layer: str) -> Path:
    try:
        return paths.LAYERS[layer]
    except KeyError as e:
        raise SchemaError(f"未知の層: {layer!r}（既知: {sorted(paths.LAYERS)}）") from e


def table_path(name: str, layer: str) -> Path:
    """テーブルの parquet パス（存在は問わない）。"""
    _schema_of(name)
    return _layer_dir(layer) / f"{name}.parquet"


def lock_path(name: str, layer: str) -> Path:
    """テーブルのロックファイル（``<name>.parquet.lock``。glob ``*.parquet`` に掛からない）。"""
    p = table_path(name, layer)
    return p.with_name(p.name + ".lock")


# ---------------------------------------------------------------------------
# ロック（プロセス間: fcntl.flock、同一スレッド内は再入可）
# ---------------------------------------------------------------------------

_LOCK_STATE = threading.local()


@contextmanager
def table_lock(name: str, layer: str) -> Iterator[Path]:
    """``<name>.parquet.lock`` を排他ロックする。read-modify-write はこの中で行う。

    同一スレッド内でネストした場合は再取得しない（append_rows → write_table の入れ子で
    デッドロックしない）。別スレッド・別プロセスは flock で直列化される。
    """
    lp = lock_path(name, layer)
    held: dict[Path, int] = getattr(_LOCK_STATE, "held", None) or {}
    _LOCK_STATE.held = held
    if lp in held:
        held[lp] += 1
        try:
            yield lp
        finally:
            held[lp] -= 1
            if held[lp] == 0:
                del held[lp]
        return
    lp.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lp, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        held[lp] = 1
        try:
            yield lp
        finally:
            del held[lp]
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# 検証・変換
# ---------------------------------------------------------------------------


def _is_string_like(t: pa.DataType) -> bool:
    return pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_string_view(t)


def _compatible(actual: pa.DataType, expected: pa.DataType, column: str) -> None:
    """期待型へ黙って cast してよい arrow 型か（allowlist）。不可なら SchemaError。"""
    if actual == expected or pa.types.is_null(actual):
        return
    if expected == _STR and _is_string_like(actual):
        return
    if expected == _I64 and pa.types.is_integer(actual):
        return
    if expected == _F64 and (pa.types.is_floating(actual) or pa.types.is_integer(actual)):
        return
    if expected == _BOOL and pa.types.is_boolean(actual):
        return
    if expected == _TS and pa.types.is_timestamp(actual):
        if actual.tz is None:
            raise SchemaError(
                f"列 {column!r}: naive datetime（tz なし）は受け付けない。tz 付き（UTC）で渡す"
            )
        return
    if (
        expected == _STR_LIST
        and (pa.types.is_list(actual) or pa.types.is_large_list(actual))
        and (_is_string_like(actual.value_type) or pa.types.is_null(actual.value_type))
    ):
        return
    raise SchemaError(f"列 {column!r}: 型 {actual} は期待型 {expected} へ変換できない")


def _check_list_nulls(name: str, table: pa.Table) -> None:
    """list 列の NULL 配列・NULL 要素を拒否する（BigQuery ARRAY と意味が変わるため）。"""
    for field in table.schema:
        if not pa.types.is_list(field.type):
            continue
        col = table.column(field.name)
        if col.null_count:
            raise SchemaError(
                f"table {name!r}: 列 {field.name!r} に NULL 配列がある（[] を明示する）"
            )
        if pc.list_flatten(col).null_count:
            raise SchemaError(f"table {name!r}: 列 {field.name!r} に NULL 要素がある")


def _validate_and_cast(
    name: str, df: pl.DataFrame, *, fill_missing: bool = False, check_lists: bool = True
) -> pa.Table:
    """polars → arrow。列集合の一致を検証し、allowlist の範囲でスキーマへ cast する。

    ``fill_missing=True`` は欠損列を型付き null で補う（読み込み用）。余分な列は常に SchemaError。
    """
    schema = _schema_of(name)
    expected = list(schema.names)
    got = list(df.columns)
    missing = [c for c in expected if c not in got]
    extra = [c for c in got if c not in expected]
    if extra or (missing and not fill_missing):
        raise SchemaError(f"table {name!r}: 欠損列={missing} 余分な列={extra}")
    table = df.select([c for c in expected if c in got]).to_arrow()
    for field in schema:
        if field.name in missing:
            continue
        _compatible(table.schema.field(field.name).type, field.type, field.name)
    try:
        table = table.cast(
            pa.schema([schema.field(c) for c in expected if c not in missing]), safe=True
        )
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as e:
        raise SchemaError(f"table {name!r}: 型変換に失敗: {e}") from e
    for c in missing:
        table = table.append_column(schema.field(c), pa.nulls(table.num_rows, schema.field(c).type))
    table = table.select(expected)
    if check_lists:
        _check_list_nulls(name, table)
    return table


def _empty(name: str) -> pl.DataFrame:
    return pl.from_arrow(_schema_of(name).empty_table())  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------


def _write_atomic(table: pa.Table, path: Path) -> None:
    """一意名の一時ファイルへ書いてから rename（他プロセスの書きかけを拾わない）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_table(name: str, df: pl.DataFrame, layer: str) -> Path:
    """スキーマ検証のうえ parquet へ上書き保存し、パスを返す（rename による原子的な置換）。"""
    table = _validate_and_cast(name, df)
    path = table_path(name, layer)
    _write_atomic(table, path)
    return path


def read_table(name: str, layer: str) -> pl.DataFrame:
    """parquet を読み、スキーマ型で返す。ファイルが無ければ空テーブル。

    スキーマに後から足された列（既存 parquet に無い列）は型付き null で補う。余分な列は SchemaError。
    """
    path = table_path(name, layer)
    if not path.exists():
        return _empty(name)
    table = pq.read_table(path)
    return _validate_and_cast_arrow(name, table)


def _validate_and_cast_arrow(name: str, table: pa.Table) -> pl.DataFrame:
    df = pl.from_arrow(table)
    assert isinstance(df, pl.DataFrame)
    return pl.from_arrow(_validate_and_cast(name, df, fill_missing=True, check_lists=False))  # type: ignore[return-value]


def _check_rows(name: str, rows: list[dict[str, Any]]) -> None:
    schema = _schema_of(name)
    known = set(schema.names)
    for i, row in enumerate(rows):
        unknown = sorted(set(row) - known)
        if unknown:
            raise SchemaError(f"table {name!r}: rows[{i}] に未知のキー {unknown}")
        for k, v in row.items():
            if isinstance(v, datetime) and v.tzinfo is None:
                raise SchemaError(
                    f"table {name!r}: rows[{i}][{k!r}] が naive datetime。tz 付き（UTC）で渡す"
                )


def append_rows(name: str, rows: list[dict[str, Any]], layer: str) -> Path:
    """行（dict）を既存 parquet に追記する（ロック内の read→concat→write）。

    未知のキー・naive datetime・スキーマ型へ入らない値は SchemaError。欠けたキーは null になる
    （追記の利便性のため。列集合の厳密検証は ``write_table`` の責務）。
    """
    schema = _schema_of(name)
    _check_rows(name, rows)
    try:
        new = pa.Table.from_pylist(rows, schema=schema)
    except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError) as e:
        raise SchemaError(f"table {name!r}: 行をスキーマへ変換できない: {e}") from e
    new_df = pl.from_arrow(new)
    assert isinstance(new_df, pl.DataFrame)
    with table_lock(name, layer):
        existing = read_table(name, layer)
        merged = pl.concat([existing, new_df], how="vertical") if existing.height else new_df
        return write_table(name, merged, layer)


# ---------------------------------------------------------------------------
# DuckDB（分析 SQL は方言中立。BigQuery では同じ論理名で dataset.table に写す）
# ---------------------------------------------------------------------------

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_literal(s: str) -> str:
    """SQL 文字列リテラル（CREATE VIEW は prepared parameter を受け付けないため）。"""
    return "'" + s.replace("'", "''") + "'"


def _quote_ident(name: str) -> str:
    """``schema.table`` 形式を許容し、各要素を検証して引用する。"""
    if not isinstance(name, str):
        raise ValueError(f"識別子は文字列でなければならない: {name!r}")
    parts = name.split(".")
    for p in parts:
        if not _IDENT.match(p):
            raise ValueError(f"不正な識別子: {name!r}")
    return ".".join(f'"{p}"' for p in parts)


def duckdb_connect(config: dict[str, Any]) -> duckdb.DuckDBPyConnection:
    """三層下の既存 parquet ごとにビューを張った DuckDB 接続を返す。

    - セッションの TimeZone を UTC に固定する（BigQuery と同じ。TIMESTAMPTZ の演算・文字列化・
      ``.pl()`` の dtype がホストの tz に依存しない）
    - ``<layer>.<name>``（例 ``marts.runs``）を層ごとに登録
    - 論理名（``config["tables"]`` で物理名へ写像、既定は同名）を marts > core > raw の優先で登録
    - ``config["duckdb"]["path"]`` があればそのファイル DB、無ければ in-memory
    """
    duck_cfg = config.get("duckdb") or {}
    db_path = str(duck_cfg.get("path") or ":memory:")
    conn = duckdb.connect(db_path)
    conn.execute("SET TimeZone='UTC'")
    mapping: dict[str, str] = dict(config.get("tables") or {})
    resolved: dict[str, Path] = {}
    for layer in reversed(_LAYER_PRIORITY):  # raw → core → marts の順に上書きし、marts が勝つ
        d = paths.LAYERS[layer]
        if not d.exists():
            continue
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {_quote_ident(layer)}")
        for p in sorted(d.glob("*.parquet")):
            logical = p.stem
            if not _IDENT.match(logical):
                continue
            conn.execute(
                f"CREATE OR REPLACE VIEW {_quote_ident(f'{layer}.{logical}')} "
                f"AS SELECT * FROM read_parquet({_quote_literal(str(p))})"
            )
            resolved[logical] = p
    for logical, p in resolved.items():
        physical = mapping.get(logical, logical)
        if "." in physical:
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {_quote_ident(physical.rsplit('.', 1)[0])}")
        conn.execute(
            f"CREATE OR REPLACE VIEW {_quote_ident(physical)} "
            f"AS SELECT * FROM read_parquet({_quote_literal(str(p))})"
        )
    return conn


_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def render_sql(sql: str, params: dict[str, Any]) -> str:
    """``{{name}}`` プレースホルダを **識別子** として検証・引用して置換する。

    値は ``schema.table`` 形式の識別子のみ（それ以外は ValueError）。リテラル（役割名・閾値等）は
    ここでは埋めず、``run_sql(..., literals=...)`` の prepared parameter（``$name``）で渡す。
    """

    def sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in params:
            raise KeyError(f"SQL プレースホルダ {{{{{key}}}}} に対応する値が無い")
        return _quote_ident(params[key])

    return _PLACEHOLDER.sub(sub, sql)


def to_bigquery_params(sql: str) -> str:
    """DuckDB の ``$name`` 名前付きパラメータを BigQuery の ``@name`` に写す（識別子置換後に使う）。"""
    return re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", r"@\1", sql)


def run_sql(
    conn: duckdb.DuckDBPyConnection,
    sql_path: str | Path,
    params: dict[str, Any] | None = None,
    literals: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """SQL ファイルを読み、``{{table_name}}``（識別子）を置換し、``$name`` にリテラルを束縛して実行する。"""
    sql = Path(sql_path).read_text(encoding="utf-8")
    rendered = render_sql(sql, params or {})
    if literals:
        return conn.execute(rendered, dict(literals)).pl()
    return conn.execute(rendered).pl()
