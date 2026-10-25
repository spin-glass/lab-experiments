"""Open Images V7 から非料理候補を抽出する（design.md §2 外部データ源、G0 層 A / 対照正例）。

手順:
1. 人手検証済み image-level ラベル CSV（``ImageID,Source,LabelName,Confidence``）を split ごとに走査し、
   画像ごとの positive（Confidence==1）/ negative（Confidence==0）MID 集合を作る
2. config の subject_type 順に、型の MID を持つ画像を ImageID 昇順で選ぶ（1 画像 1 型・先勝ち）
   - 料理系 MID（``food_mid`` ＋ ``food_exclusion_mids``。V7 の image-level ラベルは階層展開されていないため
     Pizza / Cake 等の下位クラスを config にピン留めしてある）が positive の画像は除外。
     例外: ``near_food_nonfood`` は Food の明示 negative を要求、``food_contrast``（ソース対照正例）は料理系 positive を要求
   - 型内はクラス（MID）別上限 ``per_class_cap``（**split をまたいで**数える）、型合計は源横断で配分した予算
3. 帰属 CSV（``*-images-with-rotation.csv``）から License / Author / OriginalLandingURL を引き、
   帰属行が無い・ライセンスが allowlist 外の画像は **ダウンロード前に棄却**（n_rejected に計上）
4. S3 公開バケット（HTTPS。任意で boto3 匿名。どちらも ThrottledClient のスロットル下）から画像を取得し、
   マニフェスト（再開の正本）へ N 件ごとに書く

Confidence の値は "1" / "1.0" / "0.0" が混在するため数値比較する。CSV は UTF-8（BOM 可）として復号する。
"""

from __future__ import annotations

import csv
import io
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cascade.data import fetch_common as fc

log = logging.getLogger(__name__)

SOURCE = "open_images"

_LICENSE_SHORT = {
    "by/2.0": "CC BY 2.0",
    "by-sa/2.0": "CC BY-SA 2.0",
    "by/3.0": "CC BY 3.0",
    "by-sa/3.0": "CC BY-SA 3.0",
    "by/4.0": "CC BY 4.0",
    "by-sa/4.0": "CC BY-SA 4.0",
}


def license_short_from_url(url: str) -> str:
    u = (url or "").strip().rstrip("/").lower()
    for key, short in _LICENSE_SHORT.items():
        if u.endswith("licenses/" + key):
            return short
    if "publicdomain" in u:
        return "Public domain"
    return "unknown"


# ---------------------------------------------------------------------------
# ラベル CSV の走査
# ---------------------------------------------------------------------------


def parse_image_labels(lines: Iterable[str]) -> dict[str, dict[str, set[str]]]:
    """ラベル CSV 行 → {ImageID: {"pos": {MID...}, "neg": {MID...}}}。ヘッダ有無どちらも可。"""
    out: dict[str, dict[str, set[str]]] = {}
    reader = csv.reader(lines)
    for row in reader:
        if len(row) < 4:
            continue
        image_id, _src, mid, conf = row[0], row[1], row[2], row[3]
        if image_id == "ImageID":
            continue
        try:
            c = float(conf)
        except ValueError:
            continue
        rec = out.setdefault(image_id, {"pos": set(), "neg": set()})
        if c >= 1.0:
            rec["pos"].add(mid)
        elif c <= 0.0:
            rec["neg"].add(mid)
    return out


def parse_attribution(lines: Iterable[str], wanted: set[str]) -> dict[str, dict[str, str]]:
    """帰属 CSV（ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,Author,Title,...）。

    ``wanted`` に含まれる ImageID の行のみ保持する（CSV は数十万行あるため）。
    """
    out: dict[str, dict[str, str]] = {}
    reader = csv.DictReader(lines)
    for row in reader:
        iid = row.get("ImageID")
        if iid in wanted:
            out[iid] = {k: (v or "") for k, v in row.items() if k}
    return out


# ---------------------------------------------------------------------------
# 選択（純関数）
# ---------------------------------------------------------------------------


def select_candidates(
    labels: dict[str, dict[str, set[str]]],
    type_mids: dict[str, dict[str, str]],
    *,
    food_mid: str,
    per_class_cap: int,
    budget_of: dict[str, int],
    split: str,
    food_mids: Iterable[str] | None = None,
    class_counts: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """型 → {MID: 表示名} の順に候補を決定的に選ぶ。

    - ``food_mids``: 料理系とみなす MID 集合（``food_mid`` は常に含む）
    - ``class_counts``: MID → これまでに採った枚数。呼び出し側が split をまたいで同じ dict を渡すと
      ``per_class_cap`` が全 split 合算で効く（渡さなければこの呼び出し内だけで数える）

    返す dict: ext_id / subject_type_hint / candidate_label（MID）/ candidate_name / split_hint。
    """
    fmids = {food_mid, *(food_mids or ())}
    counts = class_counts if class_counts is not None else {}
    assigned: set[str] = set()
    out: list[dict[str, Any]] = []
    ids_sorted = sorted(labels)
    for stype, mids in type_mids.items():
        budget = budget_of.get(stype, 0)
        picked: list[dict[str, Any]] = []
        for mid, name in mids.items():
            for image_id in ids_sorted:
                if len(picked) >= budget or counts.get(mid, 0) >= per_class_cap:
                    break
                if image_id in assigned:
                    continue
                rec = labels[image_id]
                if mid not in rec["pos"]:
                    continue
                if not _food_rule_ok(stype, rec, food_mid, fmids):
                    continue
                picked.append(
                    {
                        "ext_id": image_id,
                        "subject_type_hint": stype,
                        "candidate_label": mid,
                        "candidate_name": name,
                        "split_hint": split,
                    }
                )
                assigned.add(image_id)
                counts[mid] = counts.get(mid, 0) + 1
            if len(picked) >= budget:
                break
        out.extend(picked)
    return out


def _food_rule_ok(stype: str, rec: dict[str, set[str]], food_mid: str, food_mids: set[str]) -> bool:
    food_pos = bool(rec["pos"] & food_mids)
    food_neg = food_mid in rec["neg"]
    if stype == "food_contrast":
        return food_pos
    if stype == "near_food_nonfood":
        return food_neg and not food_pos
    return not food_pos


def type_mids_from_config(cfg: dict[str, Any]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for stype, srcs in cfg.get("subject_types", {}).items():
        mids = srcs.get("open_images")
        if mids:
            out[stype] = {str(k): str(v) for k, v in mids.items()}
    return out


def food_mids_from_config(cfg: dict[str, Any]) -> set[str]:
    """``open_images.food_mid`` ＋ ``open_images.food_exclusion_mids``（mapping か list）。"""
    oi = cfg["open_images"]
    excl = oi.get("food_exclusion_mids") or {}
    mids = set(excl.keys()) if isinstance(excl, dict) else set(excl)
    return {str(m) for m in mids} | {str(oi["food_mid"])}


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------


def _iter_lines(text: str) -> Iterable[str]:
    return io.StringIO(text)


def fetch(
    cfg: dict[str, Any],
    client: fc.ThrottledClient | fc.ClientLike,
    *,
    dry_run: bool = False,
    max_per_type: int | None = None,
    root: Path | None = None,
    budget_of: dict[str, int] | None = None,
) -> dict[str, Any]:
    """設定に従って候補を選び（dry_run でなければ）画像を取得し、マニフェストへ書く。

    ``budget_of`` を省略すると ``fetch_common.source_budgets``（源横断配分）を使う。
    """
    oi = cfg["open_images"]
    type_mids = type_mids_from_config(cfg)
    budgets = budget_of if budget_of is not None else fc.source_budgets(cfg, SOURCE, max_per_type)
    budget_of = {t: int(budgets.get(t, 0)) for t in type_mids}
    per_class_cap = int(cfg.get("per_class_cap", 40))
    food_mid = oi["food_mid"]
    food_mids = food_mids_from_config(cfg)
    allow = fc.compile_allowlist(cfg.get("license_allowlist", []))

    selected: list[dict[str, Any]] = []
    rejected: dict[str, int] = {"no_attribution": 0, "license": 0}
    remaining = dict(budget_of)
    class_counts: dict[str, int] = {}
    for split, spec in oi["splits"].items():
        if all(v <= 0 for v in remaining.values()):
            break
        labels = parse_image_labels(_iter_lines(_get_text(client, spec["labels_url"])))
        picks = select_candidates(
            labels,
            type_mids,
            food_mid=food_mid,
            per_class_cap=per_class_cap,
            budget_of=remaining,
            split=split,
            food_mids=food_mids,
            class_counts=class_counts,
        )
        if not picks:
            continue
        attrib = parse_attribution(
            _iter_lines(_get_text(client, spec["attribution_url"])), {p["ext_id"] for p in picks}
        )
        kept: list[dict[str, Any]] = []
        for p in picks:
            a = attrib.get(p["ext_id"])
            if a is None:
                rejected["no_attribution"] += 1
                _uncount(class_counts, p["candidate_label"])
                continue
            short = license_short_from_url(a.get("License", ""))
            if not fc.license_ok(short, allow):
                rejected["license"] += 1
                _uncount(class_counts, p["candidate_label"])
                continue
            p["_attrib"] = a
            p["_license_short"] = short
            p["_s3_prefix"] = spec.get("s3_prefix", split)
            kept.append(p)
        for p in kept:
            remaining[p["subject_type_hint"]] -= 1
        selected.extend(kept)

    summary: dict[str, Any] = {
        "source": SOURCE,
        "budget_by_type": budget_of,
        "n_selected": len(selected),
        "n_rejected": sum(rejected.values()),
        "rejected_by_reason": rejected,
        "by_type": _count_by(selected, "subject_type_hint"),
        "n_downloaded": 0,
        "n_skipped": 0,
        "n_failed": 0,
    }
    if dry_run:
        summary["candidates"] = [
            {k: v for k, v in p.items() if not k.startswith("_")} for p in selected
        ]
        return summary

    s3 = _S3Downloader(oi, client) if oi.get("download_backend", "https") == "s3" else None
    with fc.ManifestWriter(SOURCE, root, flush_every=fc.manifest_flush_every(cfg)) as mw:
        for p in selected:
            if mw.is_done(p["ext_id"]):
                summary["n_skipped"] += 1
                continue
            data = (
                s3.get(p["_s3_prefix"], p["ext_id"])
                if s3 is not None
                else _download_https(oi, client, p["_s3_prefix"], p["ext_id"])
            )
            if data is None:
                summary["n_failed"] += 1
                continue
            saved = fc.save_image(SOURCE, p["ext_id"], data, root)
            a = p["_attrib"]
            mw.add(
                fc.new_manifest_row(
                    source=SOURCE,
                    ext_id=p["ext_id"],
                    subject_type_hint=p["subject_type_hint"],
                    candidate_label=p["candidate_label"],
                    split_hint=p["split_hint"],
                    license_short=p["_license_short"],
                    license_url=a.get("License", ""),
                    attribution=_attribution(a),
                    source_url=a.get("OriginalLandingURL", "") or a.get("OriginalURL", ""),
                    **saved,
                )
            )
            summary["n_downloaded"] += 1
    summary["manifest"] = str(fc.manifest_path(SOURCE, root))
    return summary


def _uncount(counts: dict[str, int], mid: str) -> None:
    if counts.get(mid, 0) > 0:
        counts[mid] -= 1


def _attribution(a: dict[str, str]) -> str:
    author = fc.strip_html(a.get("Author", ""))
    landing = a.get("OriginalLandingURL", "")
    title = fc.strip_html(a.get("Title", ""))
    parts = [x for x in (author, title, landing) if x]
    return " / ".join(parts)


def _get_text(client: Any, url: str) -> str:
    """CSV を UTF-8（BOM 可）として読む。``resp.text`` は charset 無しの text/csv を ISO-8859-1 にするため使わない。"""
    if hasattr(client, "get_text"):
        return client.get_text(url)
    resp = client.get(url)
    if resp.status_code != 200:
        raise fc.FetchError(f"HTTP {resp.status_code}: {url}")
    content = getattr(resp, "content", None)
    if content:
        return fc.decode_utf8(content)
    return resp.text


def _download_https(oi: dict[str, Any], client: Any, split: str, image_id: str) -> bytes | None:
    url = oi["image_url"].format(split=split, image_id=image_id)
    resp = client.get(url)
    if resp.status_code != 200:
        log.warning("画像取得失敗 %s: HTTP %s", url, resp.status_code)
        return None
    return resp.content


class _S3Downloader:
    """boto3 匿名アクセス（任意経路）。クライアントは 1 回だけ生成し、取得前に ThrottledClient の間隔を守る。"""

    def __init__(self, oi: dict[str, Any], client: Any) -> None:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        self.bucket: str = oi["s3_bucket"]
        self._s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        self._throttle = getattr(client, "_wait_turn", None)

    def get(self, split: str, image_id: str) -> bytes | None:
        key = f"{split}/{image_id}.jpg"
        if callable(self._throttle):
            self._throttle()
        try:
            obj = self._s3.get_object(Bucket=self.bucket, Key=key)
        except Exception as e:  # noqa: BLE001 - 取得失敗はスキップして続行
            log.warning("S3 取得失敗 %s/%s: %s", self.bucket, key, e)
            return None
        return obj["Body"].read()


def _count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        out[it[key]] = out.get(it[key], 0) + 1
    return out
