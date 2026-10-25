"""目視確認キューの生成（盲検）と、ラベル jsonl の読み出し。

計画「目視確認の設計」/ design.md §2・§5（scope_annotations）を実装する。

- キュー行は ``item_id`` / ``pass`` の 2 キー **のみ** を持つ。path（``ext/<source>/images/<title>.jpg`` は源と
  候補ラベルを含む）・source・候補ラベル・スコア・split など判定に影響しうる情報は載せない（盲検）。
  表示順はシード付きシャッフルで決定的
- ``item_id`` は ``sha1(seed | source | photo_id)`` の先頭 16 桁の不透明 ID。source を含む元 ID をそのまま
  使うと ID 文字列から源が漏れるため。元情報（path を含む）への写像は別ファイル（map jsonl）に書き、
  レビュー中は開かない。サーバは map を読んで ``/img/<item_id>`` をサーバ側で解決する
- 出力は既定で ``paths.REVIEW`` 配下。``paths.DATA`` の外への書き出しは拒否する（map には Yelp ID 等が入り、
  .gitignore は data/ しか守らない）
- 本モジュールは標準ライブラリのみに依存する（``scripts/review_server.py`` から import するため）
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from cascade import paths

QUEUE_KEYS: tuple[str, ...] = ("item_id", "pass")
"""キュー行に許される唯一のキー集合。これ以外は盲検を破る。"""

PASS1_DECISIONS: frozenset[str] = frozenset({"keep", "drop", "boundary"})
CONTENT_VALUES: tuple[str, ...] = ("food", "drink", "non_food", "mixed", "unjudgeable")
OPERATIONAL_TRUTH_VALUES: tuple[str, ...] = (
    "food",
    "drink",
    "menu",
    "inside",
    "outside",
    "out_of_scope",
)
SUBJECT_TYPE_VALUES: tuple[str, ...] = (
    "doll_character_statue",
    "amusement_tourist",
    "signage_decor_goods",
    "person_animal",
    "vehicle_street",
    "near_food_nonfood",
    "texture_document_screen",
    "food_contrast",
    "other",
)
QUALITY_VALUES: tuple[str, ...] = ("ok", "degraded", "unjudgeable")

_ID_KEYS: tuple[str, ...] = ("item_id", "photo_id", "ext_id")


def make_item_id(seed: str, source: str | None, native_id: str) -> str:
    """不透明な item_id を作る（決定的。source と元 ID から逆算できない）。"""
    h = hashlib.sha1(f"{seed}|{source or ''}|{native_id}".encode()).hexdigest()
    return h[:16]


def relative_to_data(path: str | Path) -> str:
    """``paths.DATA`` からの相対パス（POSIX 表記）に正規化する。

    絶対パスは DATA 配下であることを要求し、相対パスは ``..`` や絶対化を含まないことを要求する
    （``../../etc/passwd`` のような行をキュー生成の時点で止める）。
    """
    p = Path(path)
    if p.is_absolute():
        try:
            p = p.resolve().relative_to(paths.DATA.resolve())
        except ValueError as e:
            raise ValueError(f"path が DATA の外を指している: {path}") from e
    rel = PurePosixPath(p.as_posix())
    if rel.is_absolute() or any(part in ("..", "") for part in rel.parts) or not rel.parts:
        raise ValueError(f"path が不正（.. や空要素）: {path}")
    return rel.as_posix()


def default_queue_path(pass_no: int) -> Path:
    return paths.REVIEW / f"queue_p{pass_no}.jsonl"


def map_path_for(queue_path: str | Path) -> Path:
    return Path(queue_path).with_suffix(".map.jsonl")


def _require_under_data(p: Path, what: str) -> Path:
    root = paths.DATA.resolve()
    if not p.resolve().is_relative_to(root):
        raise ValueError(f"{what} は paths.DATA ({root}) 配下に置く（git 管理外）: {p}")
    return p


def _native_id(item: dict[str, Any]) -> str:
    for k in ("photo_id", "ext_id"):
        v = item.get(k)
        if v:
            return str(v)
    raise ValueError("item には photo_id か ext_id が必要")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def build_queue(
    items: Iterable[dict[str, Any]],
    pass_no: int,
    seed: str,
    out_path: str | Path | None = None,
    map_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """マニフェスト行（``path`` と ``photo_id``/``ext_id``、任意で ``source`` 等）から盲検キューを書く。

    - ``out_path``（既定 ``paths.REVIEW/queue_p<pass>.jsonl``）: キュー jsonl。行は ``{item_id, pass}`` のみ
    - ``map_path``（既定 ``<out_path>.map.jsonl``）: ``item_id`` → 元情報（path / photo_id / source / その他全キー）。
      サーバの画像解決と、レビュー完了後の取り込み（``review_ingest.ingest``）だけが読む
    - どちらも ``paths.DATA`` 配下でなければ ValueError
    - 順序: 元 ID でソート → ``random.Random(seed)`` でシャッフル。入力順に依存しない
    """
    if pass_no not in (1, 2):
        raise ValueError(f"pass_no は 1 か 2: {pass_no}")
    out_path = Path(out_path) if out_path is not None else default_queue_path(pass_no)
    map_path = Path(map_path) if map_path is not None else map_path_for(out_path)
    _require_under_data(out_path, "キュー jsonl")
    _require_under_data(map_path, "map jsonl")

    rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    for item in items:
        if "path" not in item:
            raise ValueError("item には path が必要")
        source = item.get("source")
        native = _native_id(item)
        item_id = item.get("item_id") or make_item_id(seed, source, native)
        if item_id in seen:
            raise ValueError(f"item_id が重複: {item_id}")
        seen.add(item_id)
        q = {"item_id": item_id, "pass": pass_no}
        m = {k: v for k, v in item.items() if k != "item_id"}
        m["item_id"] = item_id
        m["photo_id"] = native
        m["path"] = relative_to_data(item["path"])
        rows.append((f"{source or ''}|{native}", q, m))

    rows.sort(key=lambda r: r[0])
    random.Random(seed).shuffle(rows)
    _write_jsonl(out_path, (q for _, q, _ in rows))
    _write_jsonl(map_path, (m for _, _, m in rows))
    return [q for _, q, _ in rows]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_queue(path: str | Path) -> list[dict[str, Any]]:
    """キュー jsonl を読み、盲検キー制約を検証する。"""
    rows = read_jsonl(path)
    for r in rows:
        extra = set(r) - set(QUEUE_KEYS)
        if extra:
            raise ValueError(f"キュー行に禁止キー {sorted(extra)}（盲検違反）: {r.get('item_id')}")
        missing = set(QUEUE_KEYS) - set(r)
        if missing:
            raise ValueError(f"キュー行にキー不足 {sorted(missing)}")
    return rows


def load_item_map(map_path: str | Path) -> dict[str, dict[str, Any]]:
    """map jsonl を ``item_id`` → 元情報 dict に読む。"""
    return {r["item_id"]: r for r in read_jsonl(map_path)}


def latest_labels(
    labels: str | Path | Iterable[dict[str, Any]],
    pass_no: int | None = None,
    guideline_version: str | None = None,
) -> dict[str, dict[str, Any]]:
    """ラベル行のうち (item_id, pass) ごとの最新（ts 昇順 → ファイル順の最後）を返す。

    ``pass_no`` / ``guideline_version`` を指定するとその行だけを対象にする。戻り値のキーは item_id。
    """
    rows = read_jsonl(labels) if isinstance(labels, str | Path) else list(labels)
    best: dict[str, tuple[tuple[str, int], dict[str, Any]]] = {}
    for i, r in enumerate(rows):
        if pass_no is not None and int(r.get("pass", 0)) != pass_no:
            continue
        if guideline_version is not None and r.get("guideline_version") != guideline_version:
            continue
        key = (str(r.get("ts", "")), i)
        cur = best.get(r["item_id"])
        if cur is None or key > cur[0]:
            best[r["item_id"]] = (key, r)
    return {k: v[1] for k, v in best.items()}


def build_pass2_queue(
    labels_pass1_jsonl: str | Path,
    seed: str,
    out_path: str | Path | None,
    queue_pass1_jsonl: str | Path,
    map_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """1 パス目の決定が keep / boundary の項目だけで 2 パス目キューを作る。

    ``item_id`` は 1 パス目と同じものを引き継ぐ。map は 1 パス目の map（``<queue1>.map.jsonl``）から
    該当項目だけを ``map_path``（既定 ``<out_path>.map.jsonl``）へ書き出す。
    """
    decided = latest_labels(labels_pass1_jsonl, pass_no=1)
    keep_ids = {k for k, r in decided.items() if r.get("decision") in ("keep", "boundary")}
    q1 = load_queue(queue_pass1_jsonl)
    rows = [{"item_id": r["item_id"], "pass": 2} for r in q1 if r["item_id"] in keep_ids]
    rows.sort(key=lambda r: r["item_id"])
    random.Random(seed).shuffle(rows)

    out_path = Path(out_path) if out_path is not None else default_queue_path(2)
    map_path = Path(map_path) if map_path is not None else map_path_for(out_path)
    _require_under_data(out_path, "キュー jsonl")
    _require_under_data(map_path, "map jsonl")
    _write_jsonl(out_path, rows)
    m1 = load_item_map(map_path_for(queue_pass1_jsonl))
    _write_jsonl(map_path, (m1[r["item_id"]] for r in rows if r["item_id"] in m1))
    return rows
