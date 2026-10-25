"""目視確認ラベル（jsonl）→ ``scope_annotations``（design.md §5 / 計画「記録スキーマ」）への取り込み。

- 2 パス目の行だけを ``scope_annotations`` に写す。1 パス目（keep / drop / boundary）は別 DataFrame で返す
- enum（content / operational_truth / subject_type / quality）を検証し、違反は ValueError
- 同一 item の複数行は最新（ts → ファイル順）が勝つ
- ``guideline_version`` が一致する行だけを取り込む（パイロット版のラベルを本番へ混ぜない）
- 書き出しは ``cascade.io.tables.write_table`` が存在する場合のみ（遅延 import。lane L6 に依存しない）
- 自己一致 κ（計画「目視確認の設計」: 10% を再判定）は ``operational_truth`` 上の Cohen's κ
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl

from cascade.data.review_queue import (
    CONTENT_VALUES,
    OPERATIONAL_TRUTH_VALUES,
    PASS1_DECISIONS,
    QUALITY_VALUES,
    SUBJECT_TYPE_VALUES,
    latest_labels,
    load_item_map,
    read_jsonl,
)

SCOPE_COLUMNS: tuple[str, ...] = (
    "photo_id",
    "source",
    "content",
    "operational_truth",
    "subject_type",
    "boundary_flag",
    "quality",
    "dup_group_id",
    "role",
    "sampling_prob",
    "license_short",
    "attribution",
    "source_url",
    "sha256",
    "annotator",
    "guideline_version",
    "ts",
)

SCOPE_DTYPES: dict[str, pl.DataType] = {
    "photo_id": pl.Utf8,
    "source": pl.Utf8,
    "content": pl.Utf8,
    "operational_truth": pl.Utf8,
    "subject_type": pl.Utf8,
    "boundary_flag": pl.Boolean,
    "quality": pl.Utf8,
    "dup_group_id": pl.Utf8,
    "role": pl.Utf8,
    "sampling_prob": pl.Float64,
    "license_short": pl.Utf8,
    "attribution": pl.Utf8,
    "source_url": pl.Utf8,
    "sha256": pl.Utf8,
    "annotator": pl.Utf8,
    "guideline_version": pl.Utf8,
    "ts": pl.Utf8,
}

PASS1_COLUMNS: tuple[str, ...] = (
    "item_id",
    "photo_id",
    "source",
    "decision",
    "annotator",
    "guideline_version",
    "ts",
)

_ENUMS: dict[str, frozenset[str]] = {
    "content": frozenset(CONTENT_VALUES),
    "operational_truth": frozenset(OPERATIONAL_TRUTH_VALUES),
    "subject_type": frozenset(SUBJECT_TYPE_VALUES),
    "quality": frozenset(QUALITY_VALUES),
}


def validate_pass2_row(row: dict[str, Any]) -> None:
    """2 パス目ラベル 1 行の enum / 型を検証する。違反は ValueError。"""
    for field, allowed in _ENUMS.items():
        v = row.get(field)
        if v not in allowed:
            raise ValueError(
                f"{field}={v!r} は不正（item_id={row.get('item_id')}、許容: {sorted(allowed)}）"
            )
    bf = row.get("boundary_flag", False)
    if bf is not None and not isinstance(bf, bool):
        raise ValueError(f"boundary_flag は bool か null（item_id={row.get('item_id')}）")


def validate_pass1_row(row: dict[str, Any]) -> None:
    if row.get("decision") not in PASS1_DECISIONS:
        raise ValueError(f"decision={row.get('decision')!r} は不正（item_id={row.get('item_id')}）")


def _to_bool(v: Any) -> bool:
    """``boundary_flag`` の正規化: null（未指定）は False。validate_pass2_row と同じ契約。"""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    raise ValueError(f"boundary_flag は bool: {v!r}")


def split_by_guideline(
    rows: Iterable[dict[str, Any]], guideline_version: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(guideline_version が一致する行, それ以外の行)。"""
    rows = list(rows)
    kept = [r for r in rows if r.get("guideline_version") == guideline_version]
    dropped = [r for r in rows if r.get("guideline_version") != guideline_version]
    return kept, dropped


def ingest(
    labels_jsonl: str | Path | Iterable[dict[str, Any]],
    manifest_lookup: dict[str, dict[str, Any]],
    guideline_version: str,
    *,
    write: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """ラベル jsonl を ``(scope_annotations, pass1_decisions)`` の 2 DataFrame に変換する。

    ``manifest_lookup``: item_id → {photo_id, source, dup_group_id?, role?, sampling_prob?, license_short?,
    attribution?, source_url?, sha256?}（``review_queue.load_item_map`` の出力で足りる）。
    未知の item_id は ValueError。``write=True`` なら ``cascade.io.tables`` があれば core 層に書く。
    """
    rows = read_jsonl(labels_jsonl) if isinstance(labels_jsonl, str | Path) else list(labels_jsonl)
    rows, _dropped = split_by_guideline(rows, guideline_version)

    scope_records: list[dict[str, Any]] = []
    for item_id, r in sorted(latest_labels(rows, pass_no=2).items()):
        validate_pass2_row(r)
        m = manifest_lookup.get(item_id)
        if m is None:
            raise ValueError(f"manifest_lookup に item_id={item_id} が無い")
        scope_records.append(
            {
                "photo_id": str(m.get("photo_id") or m.get("ext_id")),
                "source": m.get("source"),
                "content": r["content"],
                "operational_truth": r["operational_truth"],
                "subject_type": r["subject_type"],
                "boundary_flag": _to_bool(r.get("boundary_flag")),
                "quality": r["quality"],
                "dup_group_id": m.get("dup_group_id"),
                "role": m.get("role"),
                "sampling_prob": m.get("sampling_prob"),
                "license_short": m.get("license_short"),
                "attribution": m.get("attribution"),
                "source_url": m.get("source_url"),
                "sha256": m.get("sha256"),
                "annotator": r.get("annotator"),
                "guideline_version": guideline_version,
                "ts": r.get("ts"),
            }
        )

    pass1_records: list[dict[str, Any]] = []
    for item_id, r in sorted(latest_labels(rows, pass_no=1).items()):
        validate_pass1_row(r)
        m = manifest_lookup.get(item_id, {})
        pass1_records.append(
            {
                "item_id": item_id,
                "photo_id": (str(m["photo_id"]) if m.get("photo_id") else None),
                "source": m.get("source"),
                "decision": r["decision"],
                "annotator": r.get("annotator"),
                "guideline_version": guideline_version,
                "ts": r.get("ts"),
            }
        )

    scope_df = pl.DataFrame(scope_records, schema=SCOPE_DTYPES).select(SCOPE_COLUMNS)
    pass1_df = pl.DataFrame(pass1_records, schema=dict.fromkeys(PASS1_COLUMNS, pl.Utf8)).select(
        PASS1_COLUMNS
    )

    if write:
        write_scope_annotations(scope_df)
    return scope_df, pass1_df


def write_scope_annotations(df: pl.DataFrame) -> Path | None:
    """``cascade.io.tables`` が使えれば core 層へ書く。無ければ None（呼び出し側は DataFrame を持っている）。"""
    try:
        from cascade.io.tables import write_table  # noqa: PLC0415
    except ImportError:
        return None
    return write_table("scope_annotations", df, "core")


def _as_map(
    labels: str | Path | Iterable[dict[str, Any]] | dict[str, dict[str, Any]],
    guideline_version: str | None = None,
) -> dict[str, dict[str, Any]]:
    if isinstance(labels, dict):
        if guideline_version is None:
            return labels
        return {k: v for k, v in labels.items() if v.get("guideline_version") == guideline_version}
    return latest_labels(labels, pass_no=2, guideline_version=guideline_version)


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    """名義尺度の Cohen's κ。片方が退化（p_e = 1）なら完全一致で 1.0、そうでなければ 0.0。"""
    n = len(pairs)
    if n == 0:
        raise ValueError("再判定ペアが 0 件")
    po = sum(a == b for a, b in pairs) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def self_agreement_report(
    labels_a: str | Path | Iterable[dict[str, Any]] | dict[str, dict[str, Any]],
    labels_b: str | Path | Iterable[dict[str, Any]] | dict[str, dict[str, Any]],
    field: str = "operational_truth",
    guideline_version: str | None = None,
) -> dict[str, Any]:
    """両方に存在する item_id について κ・一致率・boundary_flag 率を返す。

    ``guideline_version`` を与えると、その版の行だけで比較する（パイロット版の混入を防ぐ）。
    """
    a, b = _as_map(labels_a, guideline_version), _as_map(labels_b, guideline_version)
    common = sorted(set(a) & set(b))
    pairs = [(str(a[i][field]), str(b[i][field])) for i in common]
    kappa = cohen_kappa(pairs)
    n = len(common)
    return {
        "n": n,
        "field": field,
        "guideline_version": guideline_version,
        "kappa": kappa,
        "agreement": sum(x == y for x, y in pairs) / n,
        "boundary_rate_a": sum(bool(a[i].get("boundary_flag")) for i in common) / n,
        "boundary_rate_b": sum(bool(b[i].get("boundary_flag")) for i in common) / n,
    }


def kappa_self_agreement(
    labels_a: str | Path | Iterable[dict[str, Any]] | dict[str, dict[str, Any]],
    labels_b: str | Path | Iterable[dict[str, Any]] | dict[str, dict[str, Any]],
) -> float:
    """再判定した項目（両方に存在する item_id）の ``operational_truth`` 上の Cohen's κ。"""
    return float(self_agreement_report(labels_a, labels_b)["kappa"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="review labels jsonl → scope_annotations")
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument(
        "--map", required=True, type=Path, help="build_queue が書いた <queue>.map.jsonl"
    )
    ap.add_argument("--guideline-version", required=True)
    ap.add_argument("--write", action="store_true", help="cascade.io.tables があれば core 層へ書く")
    ap.add_argument(
        "--rejudge", type=Path, default=None, help="再判定ラベル jsonl（自己一致 κ を出す）"
    )
    args = ap.parse_args(argv)

    all_rows = read_jsonl(args.labels)
    _kept, dropped = split_by_guideline(all_rows, args.guideline_version)
    scope_df, pass1_df = ingest(
        all_rows, load_item_map(args.map), args.guideline_version, write=args.write
    )
    summary: dict[str, Any] = {
        "guideline_version": args.guideline_version,
        "scope_rows": scope_df.height,
        "pass1_rows": pass1_df.height,
        "n_rows_dropped_other_guideline": len(dropped),
        "dropped_guideline_versions": dict(
            Counter(str(r.get("guideline_version")) for r in dropped)
        ),
        "operational_truth": dict(Counter(scope_df["operational_truth"].to_list())),
        "boundary_rate": (float(scope_df["boundary_flag"].mean()) if scope_df.height else None),
    }
    if args.rejudge is not None:
        summary["self_agreement"] = self_agreement_report(
            args.labels, args.rejudge, guideline_version=args.guideline_version
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
