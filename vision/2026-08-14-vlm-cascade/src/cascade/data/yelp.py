"""Yelp Open Dataset photos の検証・統計・分割・eval 凍結（design.md §2、g0_nonfood_plan.md §3.1 / §3.7、計画 Phase B-b）。

- :func:`verify_layout`      手動配置（paths.YELP）のレイアウト確認と DOWNLOAD.txt（規約の版・DL 日）の読取
- :func:`load_photos_json`   photos.json（JSON Lines）→ DataFrame(photo_id, business_id, caption, label)
- :func:`class_counts`       クラス別枚数とラベル欠損率
- :func:`sample_image_stats` 標本の幅・高さ・バイト数・JPEG 品質推定（A2 §11.1 の長辺サイズ・品質分布）
- :func:`build_splits`       business_id ∪ dup 群の連結成分単位で tune_fit / tune_cal / work / eval を決定的に割り当て、
                             役割別・クラス別の最小 n を満たす（満たせなければ :class:`SplitSizeError`）
- :func:`freeze_eval_sets`   役割ごとの photo_id リストの sha256 を eval_sets 行にする
- :func:`m0_report`          reports/m0_setup.md を描画する（集計値のみ。photo_id・画像は載せない）

画像・生データ・photo_id 行はリポジトリに置かない（CLAUDE.md 絶対制約）。リポジトリに残るのは
config・seed・sha256 のみ。
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from cascade import paths
from cascade.data import dedup
from cascade.data.preprocess import measure_image_stats

__all__ = [
    "CLASSES",
    "ROLES",
    "SplitSizeError",
    "balanced_slice",
    "build_splits",
    "class_counts",
    "compute_phash_shards",
    "freeze_eval_sets",
    "ids_sha256",
    "load_photos_json",
    "m0_report",
    "read_download_record",
    "sample_image_stats",
    "split_summary",
    "verify_layout",
]

CLASSES: tuple[str, ...] = ("food", "drink", "menu", "inside", "outside")
ROLES: tuple[str, ...] = ("tune_fit", "tune_cal", "work", "eval")
RESIDUAL_ROLE = "work"

_PHOTOS_SCHEMA = {
    "photo_id": pl.Utf8,
    "business_id": pl.Utf8,
    "caption": pl.Utf8,
    "label": pl.Utf8,
}


class SplitSizeError(ValueError):
    """役割別・クラス別の最小 n を満たす分割が作れない。"""


# ---------------------------------------------------------------------------
# レイアウトと規約の記録
# ---------------------------------------------------------------------------


def _yelp_cfg(config: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict((config or {}).get("yelp", {}) or {})


def resolve_root(config: Mapping[str, Any] | None = None, root: str | Path | None = None) -> Path:
    """Yelp の配置先。引数 > config.yelp.root > paths.YELP の順。"""
    if root is not None:
        return Path(root)
    cfg_root = _yelp_cfg(config).get("root")
    if cfg_root:
        return Path(str(cfg_root))
    return paths.YELP


def _locate(root: Path, name: str) -> Path:
    """<root>/<name> が無ければ <root>/yelp_photos/<name>（tar 展開時の入れ子）を探す。"""
    direct = root / name
    if direct.exists():
        return direct
    nested = root / "yelp_photos" / name
    if nested.exists():
        return nested
    return direct


def read_download_record(path: str | Path) -> dict[str, str]:
    """DOWNLOAD.txt（``key: value`` の行。``#`` 行は無視）を読む。無ければ空 dict。"""
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or ":" not in s:
            continue
        k, v = s.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def verify_layout(
    root: str | Path | None = None, config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """配置を検査し、存在フラグ・パス・DOWNLOAD.txt の内容と sha256 を返す。

    戻り値の ``ok`` は photos.json と photos/ が存在し、DOWNLOAD.txt の必須項目が揃っているとき True。
    """
    cfg = _yelp_cfg(config)
    root = resolve_root(config, root)
    photos_json = _locate(root, str(cfg.get("photos_json", "photos.json")))
    photos_dir = _locate(root, str(cfg.get("photos_dir", "photos")))
    record_path = _locate(root, str(cfg.get("download_record", "DOWNLOAD.txt")))
    pdfs = sorted(p.name for p in root.glob("**/*.pdf")) if root.exists() else []
    record = read_download_record(record_path)
    terms = cfg.get("terms", {}) or {}
    required = [str(k) for k in terms.get("required_fields", ["terms_version", "downloaded_at"])]
    missing_fields = [k for k in required if not record.get(k)]
    n_files = 0
    if photos_dir.is_dir():
        ext = str(cfg.get("image_ext", ".jpg")).lower()
        n_files = sum(1 for p in photos_dir.iterdir() if p.suffix.lower() == ext)
    problems: list[str] = []
    if not root.exists():
        problems.append(f"root が無い: {root}")
    if not photos_json.is_file():
        problems.append(f"photos.json が無い: {photos_json}")
    if not photos_dir.is_dir():
        problems.append(f"photos/ が無い: {photos_dir}")
    if not record_path.is_file():
        problems.append(f"DOWNLOAD.txt が無い（規約の版と DL 日を記録する）: {record_path}")
    elif missing_fields:
        problems.append(f"DOWNLOAD.txt に必須項目が無い: {missing_fields}")
    if not pdfs:
        problems.append(
            "利用規約 PDF が見つからない（design.md §2: 規約確認を M0 の最初の工程とする）"
        )
    return {
        "root": str(root),
        "photos_json": str(photos_json),
        "photos_dir": str(photos_dir),
        "download_record": str(record_path),
        "download_record_sha256": dedup.sha256_file(record_path) if record_path.is_file() else None,
        "download_record_fields": record,
        "terms_pdfs": pdfs,
        "n_image_files": n_files,
        "missing_fields": missing_fields,
        "problems": problems,
        "ok": not problems,
    }


# ---------------------------------------------------------------------------
# photos.json とクラス別統計
# ---------------------------------------------------------------------------


def load_photos_json(path: str | Path) -> pl.DataFrame:
    """photos.json（1 行 1 JSON）を読む。label の欠損・空文字は null に正規化する。

    列: photo_id, business_id, caption, label。photo_id の重複は ValueError。
    """
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: JSON として読めない: {e}") from e
            if "photo_id" not in obj:
                raise ValueError(f"{path}:{lineno}: photo_id が無い")
            label = obj.get("label")
            if isinstance(label, str):
                label = label.strip().lower() or None
            elif label is not None:
                label = str(label)
            rows.append(
                {
                    "photo_id": str(obj["photo_id"]),
                    "business_id": None
                    if obj.get("business_id") in (None, "")
                    else str(obj["business_id"]),
                    "caption": None if obj.get("caption") is None else str(obj["caption"]),
                    "label": label,
                }
            )
    df = pl.DataFrame(rows, schema=_PHOTOS_SCHEMA)
    if df.height and df["photo_id"].n_unique() != df.height:
        raise ValueError("photos.json に重複した photo_id がある")
    return df


def class_counts(df: pl.DataFrame, classes: Sequence[str] = CLASSES) -> pl.DataFrame:
    """クラス別枚数（欠損は label = null の行）。列: label, n, share。"""
    total = df.height
    counts: dict[str | None, int] = {}
    if total:
        for lb, n in df.group_by("label").len().iter_rows():
            counts[lb] = int(n)
    rows = [{"label": c, "n": counts.get(c, 0)} for c in classes]
    known = set(classes)
    for k in sorted(k for k in counts if k is not None and k not in known):
        rows.append({"label": k, "n": counts[k]})
    rows.append({"label": None, "n": counts.get(None, 0)})
    out = pl.DataFrame(rows, schema={"label": pl.Utf8, "n": pl.Int64})
    return out.with_columns((pl.col("n") / max(total, 1)).alias("share"))


def missing_label_rate(df: pl.DataFrame) -> float:
    if not df.height:
        return 0.0
    return float(df["label"].null_count() / df.height)


def photo_path(root: Path, photo_id: str, photos_dir: str = "photos", ext: str = ".jpg") -> Path:
    return _locate(root, photos_dir) / f"{photo_id}{ext}"


def sample_image_stats(
    df: pl.DataFrame,
    root: str | Path,
    n: int = 2000,
    seed: int = 0,
    *,
    photos_dir: str = "photos",
    ext: str = ".jpg",
) -> pl.DataFrame:
    """seed 付き無作為 n 枚の画像統計（cascade.data.preprocess.measure_image_stats）。

    列: photo_id, label, path, width, height, bytes, format, jpeg_quality_estimate, long_side。
    """
    root = Path(root)
    if df.height == 0:
        return pl.DataFrame(
            schema={
                "photo_id": pl.Utf8,
                "label": pl.Utf8,
                "path": pl.Utf8,
                "width": pl.Int64,
                "height": pl.Int64,
                "bytes": pl.Int64,
                "format": pl.Utf8,
                "jpeg_quality_estimate": pl.Int64,
                "long_side": pl.Int64,
            }
        )
    k = min(int(n), df.height)
    sub = df.select("photo_id", "label").sample(n=k, seed=int(seed), shuffle=True)
    files = [photo_path(root, pid, photos_dir, ext) for pid in sub["photo_id"].to_list()]
    stats = measure_image_stats(files)
    return sub.hstack(stats).with_columns(
        pl.max_horizontal(pl.col("width"), pl.col("height")).alias("long_side"),
    )


def summarize_image_stats(stats: pl.DataFrame) -> dict[str, Any]:
    """長辺サイズ・JPEG 品質・バイト数の中央値と範囲（全体とクラス別）。"""

    def _one(d: pl.DataFrame) -> dict[str, Any]:
        ok = d.filter(pl.col("width").is_not_null())
        if not ok.height:
            return {"n": int(d.height), "n_readable": 0}
        return {
            "n": int(d.height),
            "n_readable": int(ok.height),
            "long_side_median": float(ok["long_side"].median()),
            "long_side_min": int(ok["long_side"].min()),
            "long_side_max": int(ok["long_side"].max()),
            "bytes_median": float(ok["bytes"].median()),
            "jpeg_quality_median": (
                None
                if ok["jpeg_quality_estimate"].null_count() == ok.height
                else float(ok["jpeg_quality_estimate"].median())
            ),
            "formats": {
                str(k): int(v) for k, v in ok.group_by("format").len().sort("format").iter_rows()
            },
        }

    out: dict[str, Any] = {"all": _one(stats), "by_label": {}}
    for lb in stats["label"].unique().to_list():
        key = "null" if lb is None else str(lb)
        out["by_label"][key] = _one(
            stats.filter(pl.col("label").is_null() if lb is None else pl.col("label") == lb)
        )
    return out


# ---------------------------------------------------------------------------
# pHash（Yelp 規模・再開可能）
# ---------------------------------------------------------------------------


def compute_phash_shards(
    df: pl.DataFrame,
    root: str | Path,
    shard_dir: str | Path,
    *,
    batch_size: int = 5000,
    photos_dir: str = "photos",
    ext: str = ".jpg",
) -> dict[str, int]:
    """photo_id 順に batch_size ごとの pHash シャード（parquet: photo_id, phash）を書く。既存シャードは飛ばす。

    戻り値は photo_id → pHash（読めなかった画像は含めない）。
    """
    root = Path(root)
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(df["photo_id"].to_list())
    out: dict[str, int] = {}
    for start in range(0, len(ids), int(batch_size)):
        idx = start // int(batch_size)
        shard = shard_dir / f"shard_{idx:05d}.parquet"
        if shard.exists():
            part = pl.read_parquet(shard)
        else:
            pids: list[str] = []
            vals: list[int] = []
            for pid in ids[start : start + int(batch_size)]:
                p = photo_path(root, pid, photos_dir, ext)
                try:
                    h = dedup.phash64(p)
                except OSError:
                    continue
                pids.append(pid)
                vals.append(int(h))
            part = pl.DataFrame(
                {"photo_id": pids, "phash": vals}, schema={"photo_id": pl.Utf8, "phash": pl.UInt64}
            )
            tmp = shard.with_suffix(".parquet.tmp")
            part.write_parquet(tmp)
            tmp.replace(shard)
        out.update(
            zip(part["photo_id"].to_list(), (int(v) for v in part["phash"].to_list()), strict=True)
        )
    return out


def build_dedup_groups(phashes: Mapping[str, int], phash_max: int = 8) -> dict[str, str]:
    """pHash のみで dup 群（photo_id → dup_group_id）を作る。business_id は分割側で連結する。"""
    items = [{"photo_id": pid, "phash": h} for pid, h in sorted(phashes.items())]
    return dedup.build_groups(items, phash_max=phash_max, use_group_key=False)


# ---------------------------------------------------------------------------
# 分割
# ---------------------------------------------------------------------------


def _components(
    df: pl.DataFrame, groups: Mapping[str, str] | None
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """business_id ∪ dup 群の連結成分。戻り値は (photo_id → component_id, component_id → photo_ids)。"""
    uf = dedup.UnionFind()
    first_of_biz: dict[str, str] = {}
    first_of_grp: dict[str, str] = {}
    for pid, biz in zip(df["photo_id"].to_list(), df["business_id"].to_list(), strict=True):
        uf.add(pid)
        if biz not in (None, ""):
            if biz in first_of_biz:
                uf.union(first_of_biz[biz], pid)
            else:
                first_of_biz[biz] = pid
        g = None if groups is None else groups.get(pid)
        if g is not None:
            if g in first_of_grp:
                uf.union(first_of_grp[g], pid)
            else:
                first_of_grp[g] = pid
    comp_of: dict[str, str] = {}
    members_of: dict[str, list[str]] = {}
    for members in uf.components().values():
        cid = dedup._group_id(members)
        members_of[cid] = members
        for pid in members:
            comp_of[pid] = cid
    return comp_of, members_of


def _minimums(sizes: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    mins = sizes.get("minimums", {}) or {}
    out: dict[str, dict[str, int]] = {}
    for role in ROLES:
        out[role] = {str(k): int(v) for k, v in (mins.get(role, {}) or {}).items()}
    return out


def _deficits(
    counts: Mapping[str, Mapping[str, int]], minimums: Mapping[str, Mapping[str, int]]
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for role, need in minimums.items():
        d = {c: n - counts.get(role, {}).get(c, 0) for c, n in need.items()}
        d = {c: v for c, v in d.items() if v > 0}
        if d:
            out[role] = d
    return out


def build_splits(
    df: pl.DataFrame,
    groups: Mapping[str, str] | None,
    sizes: Mapping[str, Any],
    seed: str | int,
) -> pl.DataFrame:
    """tune_fit / tune_cal / work / eval を成分単位で決定的に割り当てる。

    1. 成分 = business_id ∪ dup 群の連結成分（``groups`` は photo_id → dup_group_id。None なら business_id のみ）
    2. 基礎割当 = ``sizes["ratios"]`` に従う :func:`cascade.data.dedup.assign_roles`（成分 ID のハッシュで決定的）
    3. 補填 = ``sizes["minimums"]``（役割 → クラス → 最小 n）に届かない役割へ、不足クラスを含む成分を
       ハッシュ順に移す。提供元は残余役割 ``work`` を優先し、尽きたら自身の最小 n を保てる範囲で他の役割
       からも移す。それでも届かなければ :class:`SplitSizeError`（不足表を含む）
    4. 非跨ぎを :func:`cascade.data.dedup.check_no_crossing` で検証する

    戻り値: DataFrame(photo_id, role, component_id, eval_balanced)。``eval_balanced`` は eval 内の
    クラス別均等スライス（``sizes["eval_balanced_per_class"]``。0 / None なら全て False）。
    """
    if df.height == 0:
        raise SplitSizeError("photos が空")
    raw_ratios = {str(k): float(v) for k, v in (sizes.get("ratios") or {}).items()}
    if set(raw_ratios) != set(ROLES):
        raise ValueError(f"ratios の役割は {ROLES} に一致させる: {sorted(raw_ratios)}")
    ratios = {r: raw_ratios[r] for r in ROLES}  # 区間の順序を固定（YAML のキー順に依存しない）
    if ratios[RESIDUAL_ROLE] <= 0:
        raise ValueError(f"残余役割 {RESIDUAL_ROLE!r} の比率は正にする")
    minimums = _minimums(sizes)
    seed_s = str(seed)
    if not bool(sizes.get("include_unlabeled", True)):
        df = df.filter(pl.col("label").is_not_null())

    comp_of, members_of = _components(df, groups)
    label_of = dict(zip(df["photo_id"].to_list(), df["label"].to_list(), strict=True))
    comp_ids = sorted(members_of)
    role_of_comp = dedup.assign_roles(comp_ids, ratios, seed_s)

    # 成分ごとのクラス別枚数
    comp_counts: dict[str, dict[str, int]] = {}
    for cid, members in members_of.items():
        cc: dict[str, int] = defaultdict(int)
        for pid in members:
            lb = label_of.get(pid)
            if lb is not None:
                cc[lb] += 1
        comp_counts[cid] = dict(cc)

    def _role_counts() -> dict[str, dict[str, int]]:
        rc: dict[str, dict[str, int]] = {r: defaultdict(int) for r in ROLES}
        for cid, r in role_of_comp.items():
            for lb, n in comp_counts[cid].items():
                rc[r][lb] += n
        return {r: dict(v) for r, v in rc.items()}

    # 補填: 不足クラスを含む成分をハッシュ順に移す。提供元は work（残余）を優先し、work が尽きたら
    # 自身の最小 n を保てる範囲で他の役割からも移す
    order = sorted(comp_ids, key=lambda c: dedup._uniform_from_key(seed_s, c))
    counts = _role_counts()
    deficits = _deficits(counts, minimums)

    def _move(cid: str, src: str, dst: str) -> None:
        role_of_comp[cid] = dst
        for lb, n in comp_counts[cid].items():
            counts[dst][lb] = counts[dst].get(lb, 0) + n
            counts[src][lb] = counts[src].get(lb, 0) - n

    def _donor_ok(cid: str, src: str) -> bool:
        need = minimums.get(src, {})
        return all(counts[src].get(c, 0) - comp_counts[cid].get(c, 0) >= n for c, n in need.items())

    for role in (r for r in ROLES if r != RESIDUAL_ROLE):
        donors = [RESIDUAL_ROLE, *(r for r in ROLES if r not in (RESIDUAL_ROLE, role))]
        for src in donors:
            if role not in deficits:
                break
            for cid in order:
                if role not in deficits:
                    break
                if role_of_comp[cid] != src:
                    continue
                need = deficits[role]
                if not any(comp_counts[cid].get(c, 0) > 0 for c in need):
                    continue
                if not _donor_ok(cid, src):
                    continue
                _move(cid, src, role)
                deficits = _deficits(counts, minimums)
    if deficits:
        raise SplitSizeError(
            "最小 n を満たす分割が作れない（役割 → クラス → 不足枚数）: "
            + json.dumps(deficits, ensure_ascii=False, sort_keys=True)
            + f"。現在のクラス別枚数: {json.dumps(counts, ensure_ascii=False, sort_keys=True)}"
        )

    role_of_item = {pid: role_of_comp[comp_of[pid]] for pid in comp_of}
    crossing = dedup.check_no_crossing(role_of_item, comp_of)
    if crossing:
        raise RuntimeError(f"成分が役割を跨いだ: {crossing[:5]}")

    ids = df["photo_id"].to_list()
    out = pl.DataFrame(
        {
            "photo_id": ids,
            "role": [role_of_item[p] for p in ids],
            "component_id": [comp_of[p] for p in ids],
        },
        schema={"photo_id": pl.Utf8, "role": pl.Utf8, "component_id": pl.Utf8},
    )
    per_class = sizes.get("eval_balanced_per_class") or 0
    classes = [str(c) for c in (sizes.get("classes") or CLASSES)]
    chosen = balanced_slice(out, label_of, int(per_class), seed_s, classes=classes)
    return out.with_columns(pl.col("photo_id").is_in(sorted(chosen)).alias("eval_balanced"))


def balanced_slice(
    splits: pl.DataFrame,
    label_of: Mapping[str, str | None],
    n_per_class: int,
    seed: str,
    *,
    role: str = "eval",
    classes: Sequence[str] = CLASSES,
) -> set[str]:
    """役割 ``role`` の中からクラス別に n_per_class 枚を決定的（sha1(seed + photo_id) 順）に選ぶ。

    そのクラスの枚数が n_per_class に満たなければ :class:`SplitSizeError`。n_per_class ≤ 0 なら空集合。
    """
    if n_per_class <= 0:
        return set()
    ids = splits.filter(pl.col("role") == role)["photo_id"].to_list()
    by_class: dict[str, list[str]] = defaultdict(list)
    for pid in ids:
        lb = label_of.get(pid)
        if lb is not None:
            by_class[lb].append(pid)
    chosen: set[str] = set()
    short: dict[str, int] = {}
    for c in classes:
        pool = sorted(
            by_class.get(c, []), key=lambda p: dedup._uniform_from_key(seed + ":balanced", p)
        )
        if len(pool) < n_per_class:
            short[c] = n_per_class - len(pool)
            continue
        chosen.update(pool[:n_per_class])
    if short:
        raise SplitSizeError(f"{role} の均等スライス（クラス別 {n_per_class}）に不足: {short}")
    return chosen


def split_summary(
    splits: pl.DataFrame, df: pl.DataFrame, classes: Sequence[str] = CLASSES
) -> pl.DataFrame:
    """役割 × クラスの枚数表。列: role, n_components, n_items, label, n（label=null は欠損）。"""
    j = splits.join(df.select("photo_id", "label"), on="photo_id", how="left")
    rows: list[dict[str, Any]] = []
    for role in ROLES:
        sub = j.filter(pl.col("role") == role)
        n_comp = sub["component_id"].n_unique() if sub.height else 0
        for c in [*classes, None]:
            n = sub.filter(pl.col("label").is_null() if c is None else pl.col("label") == c).height
            rows.append(
                {
                    "role": role,
                    "n_components": int(n_comp),
                    "n_items": int(sub.height),
                    "label": c,
                    "n": int(n),
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "role": pl.Utf8,
            "n_components": pl.Int64,
            "n_items": pl.Int64,
            "label": pl.Utf8,
            "n": pl.Int64,
        },
    )


# ---------------------------------------------------------------------------
# eval 凍結
# ---------------------------------------------------------------------------


def ids_sha256(ids: Iterable[str]) -> str:
    """sha256(ソート済み photo_id を "\\n" 連結)。"""
    return hashlib.sha256("\n".join(sorted(str(i) for i in ids)).encode("utf-8")).hexdigest()


ROLE_DEFINITIONS: dict[str, str] = {
    "tune_fit": "kNN バンク・クラス平均と共有共分散・ヘッド学習。food ≥ 2,000、他クラス各 ≥ 500（g0_nonfood_plan.md §3.1）",
    "tune_cal": "動作点（閾値）の決定。food ≥ 3,000（§7.1）",
    "work": "開発・追加手法の発動判断・記述分析の予備。残余",
    "eval": "最終報告のみ。natural スライス（Yelp のクラス分布のまま、food ≥ 3,000）",
    "eval_balanced": "eval 内のクラス別均等スライス（design.md §2 クラス別 400 目安）",
}


def freeze_eval_sets(
    splits: pl.DataFrame,
    *,
    prefix: str = "yelp-v1",
    seed: str | int = "",
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """役割ごと（＋eval 均等スライス）の eval_sets 行を返す（eval_set_id, role, photo_ids_hash, definition, created_at）。"""
    now = now or datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    for role in ROLES:
        ids = splits.filter(pl.col("role") == role)["photo_id"].to_list()
        rows.append(
            {
                "eval_set_id": f"{prefix}-{role}",
                "role": role,
                "photo_ids_hash": ids_sha256(ids),
                "definition": f"{ROLE_DEFINITIONS[role]}; n={len(ids)}; seed={seed}; key=business_id ∪ dup 群の連結成分",
                "created_at": now,
            }
        )
    if "eval_balanced" in splits.columns:
        ids = splits.filter(pl.col("eval_balanced"))["photo_id"].to_list()
        if ids:
            rows.append(
                {
                    "eval_set_id": f"{prefix}-eval-balanced",
                    "role": "eval",
                    "photo_ids_hash": ids_sha256(ids),
                    "definition": f"{ROLE_DEFINITIONS['eval_balanced']}; n={len(ids)}; seed={seed}",
                    "created_at": now,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# レポート
# ---------------------------------------------------------------------------


def _md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(lines)


def _fmt(v: Any, nd: int = 1) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def m0_report(stats: Mapping[str, Any], splits: pl.DataFrame, out_path: str | Path) -> str:
    """reports/m0_setup.md を描画して書く（集計値のみ）。

    ``stats`` のキー: layout（verify_layout の戻り値）, class_counts（list[dict]: label, n, share）,
    missing_label_rate, n_photos, n_businesses, image_stats（summarize_image_stats）, split_config（sizes）,
    split_summary（list[dict]）, crossing（list）, eval_sets（list[dict]）, dedup（dict, 任意）, stamps（dict）。
    """
    layout = stats.get("layout", {}) or {}
    record = layout.get("download_record_fields", {}) or {}
    stamps = stats.get("stamps", {}) or {}
    cc = stats.get("class_counts", []) or []
    img = stats.get("image_stats", {}) or {}
    cfg = stats.get("split_config", {}) or {}
    summ = stats.get("split_summary", []) or []
    crossing = list(stats.get("crossing", []) or [])
    eval_rows = stats.get("eval_sets", []) or []
    dd = stats.get("dedup") or {}

    n_photos = int(stats.get("n_photos", 0))
    role_n = {r: 0 for r in ROLES}
    for row in summ:
        role_n[row["role"]] = int(row["n_items"])
    conclusion = (
        f"Yelp photos {n_photos:,} 枚（店舗 {int(stats.get('n_businesses', 0)):,}）を business_id ∪ dup 群の連結成分単位で "
        f"tune_fit {role_n['tune_fit']:,} / tune_cal {role_n['tune_cal']:,} / work {role_n['work']:,} / eval {role_n['eval']:,} に分割し、"
        f"役割別・クラス別の最小 n を満たした。非跨ぎ検証: {'合格（跨いだ成分 0）' if not crossing else f'不合格（{len(crossing)} 成分）'}。"
        f"eval は photo_id リストの sha256 で凍結した（{len(eval_rows)} 行）。"
    )

    parts: list[str] = []
    parts.append("# M0 セットアップ — Yelp 検証と分割\n")
    parts.append(f"**結論**: {conclusion}\n")
    parts.append("## 1. 配置と規約\n")
    parts.append(
        _md_table(
            ["項目", "値"],
            [
                ["配置 root", layout.get("root")],
                [
                    "photos.json",
                    "あり" if Path(str(layout.get("photos_json", ""))).exists() else "なし",
                ],
                ["画像ファイル数", f"{int(layout.get('n_image_files', 0)):,}"],
                ["利用規約 PDF", ", ".join(layout.get("terms_pdfs", [])) or "なし"],
                ["DOWNLOAD.txt sha256", layout.get("download_record_sha256")],
                ["規約の版", record.get("terms_version")],
                ["DL 日", record.get("downloaded_at")],
                ["アーカイブ sha256", record.get("archive_sha256")],
                ["検証結果", "OK" if layout.get("ok") else "; ".join(layout.get("problems", []))],
            ],
        )
    )
    parts.append("\n## 2. データ統計（photos.json）\n")
    parts.append(
        f"- 総枚数 {n_photos:,}、店舗数 {int(stats.get('n_businesses', 0)):,}、"
        f"ラベル欠損率 {100 * float(stats.get('missing_label_rate', 0.0)):.2f}%\n"
    )
    parts.append("### 2.1 クラス分布\n")
    parts.append(
        _md_table(
            ["label", "n", "share"],
            [
                [
                    "(欠損)" if r["label"] is None else r["label"],
                    f"{int(r['n']):,}",
                    f"{100 * float(r['share']):.2f}%",
                ]
                for r in cc
            ],
        )
    )
    parts.append("\n### 2.2 画像サイズ分布（標本）\n")
    if img:
        rows = []
        for key, v in [("all", img.get("all", {})), *sorted((img.get("by_label") or {}).items())]:
            rows.append(
                [
                    key,
                    v.get("n"),
                    v.get("n_readable"),
                    _fmt(v.get("long_side_median"), 0),
                    f"{_fmt(v.get('long_side_min'))}–{_fmt(v.get('long_side_max'))}",
                    _fmt(v.get("jpeg_quality_median"), 0),
                    _fmt(v.get("bytes_median"), 0),
                ]
            )
        parts.append(
            _md_table(
                [
                    "群",
                    "n",
                    "読取可",
                    "長辺 中央値",
                    "長辺 範囲",
                    "JPEG 品質 中央値",
                    "bytes 中央値",
                ],
                rows,
            )
        )
    else:
        parts.append("（標本統計なし。`verify-yelp` を先に実行する）")
    parts.append("\n## 3. 分割の定義\n")
    parts.append(
        "- 分割キー: `business_id` ∪ 近重複群（pHash Hamming ≤ 8）の連結成分。同一店舗・同一群は役割を跨がない（design.md §2）\n"
        f"- seed: `{cfg.get('seed')}`、基礎比率: `{json.dumps(cfg.get('ratios', {}), ensure_ascii=False)}`\n"
        f"- 最小 n: `{json.dumps(cfg.get('minimums', {}), ensure_ascii=False)}`（不足分は不足クラスを含む成分をハッシュ順に移して補填。提供元は work（残余）を優先し、尽きたら自身の最小 n を保てる範囲で他の役割からも移す）\n"
        f"- eval 均等スライス: クラス別 {cfg.get('eval_balanced_per_class')} 枚。natural スライス = eval 全体\n"
        f"- 近重複: {('pHash 群 ' + str(dd.get('n_groups_multi', 0)) + ' 群（' + str(dd.get('n_in_groups', 0)) + ' 枚）') if dd else '未実施（business_id のみ。--with-dedup で pHash 群を加える）'}\n"
    )
    parts.append("### 3.1 役割 × クラス\n")
    by_role: dict[str, dict[str | None, int]] = defaultdict(dict)
    comps: dict[str, int] = {}
    for r in summ:
        by_role[r["role"]][r["label"]] = int(r["n"])
        comps[r["role"]] = int(r["n_components"])
    parts.append(
        _md_table(
            ["role", "成分数", "枚数", *CLASSES, "(欠損)"],
            [
                [
                    role,
                    f"{comps.get(role, 0):,}",
                    f"{role_n[role]:,}",
                    *[f"{by_role[role].get(c, 0):,}" for c in CLASSES],
                    f"{by_role[role].get(None, 0):,}",
                ]
                for role in ROLES
            ],
        )
    )
    parts.append("\n### 3.2 非跨ぎ検証\n")
    parts.append(
        "合格: 役割を跨いだ成分は 0（`cascade.data.dedup.check_no_crossing`）"
        if not crossing
        else f"**不合格**: 跨いだ成分 {len(crossing)}"
    )
    parts.append("\n## 4. eval の定義と凍結\n")
    parts.append(
        _md_table(
            ["eval_set_id", "role", "photo_ids_hash", "definition"],
            [
                [r["eval_set_id"], r["role"], r["photo_ids_hash"], r["definition"]]
                for r in eval_rows
            ],
        )
    )
    parts.append(
        "\n照合: `data/marts/splits.parquet` の photo_id を役割ごとに昇順で `\\n` 連結し sha256 を取ると上表に一致する。photo_id 一覧はリポジトリに置かない。\n"
    )
    parts.append("## フッタ\n")
    parts.append(
        _md_table(
            ["項目", "値"],
            [
                ["run_id", stamps.get("run_id")],
                ["git sha", stamps.get("git_sha")],
                ["taxonomy_version", stamps.get("taxonomy_version")],
                ["rulebook_version", stamps.get("rulebook_version", "未導入")],
                ["eval_set_id", stamps.get("eval_set_id")],
                ["概算 API コスト", stamps.get("api_cost", "0（API 呼び出しなし）")],
                ["W&B run URL", stamps.get("wandb_url") or "（未送出）"],
                ["Yelp 規約の版", record.get("terms_version")],
                ["Yelp DL 日", record.get("downloaded_at")],
                ["生成時刻", stamps.get("generated_at")],
            ],
        )
    )
    text = "\n".join(parts) + "\n"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text
