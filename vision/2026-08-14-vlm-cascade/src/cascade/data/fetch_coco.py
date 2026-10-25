"""COCO 2017 val から「主被写体が人物・動物・乗り物」の候補を抽出する（design.md §2、G0 層 A）。

- ``val2017.zip``（815,585,330 bytes）と ``annotations_trainval2017.zip``（252,907,541 bytes）を
  ``paths.EXT/coco/zips/`` に置き（既存・サイズ一致なら再利用）、``zipfile`` で必要なメンバだけ読む
- 主被写体規則（plan v4）: 食関連カテゴリ {44, 46–61, 67} を 1 つでも含む画像は除外、``iscrowd==1`` を含む画像は除外、
  最大 bbox の面積比 (w*h)/(W*H) ≥ 0.2 でありその category が型のカテゴリ集合に属する、
  person 型は count(person) ≤ 3
- 1 画像 1 型（config の subject_type 順で先勝ち）、型内クラス別上限、ImageID 昇順の決定的順序
- ライセンス id（1–8）を annotation の ``licenses`` から名前・URL に解決し、``flickr_url`` を source_url に記録。
  ``license_ids_allowed`` のフィルタは **候補選択の前** に画像へ掛ける（後で掛けると型予算が黙って目減りする）。
  既定は 1–8 全部（NC / ND を含む）: 画像は内部評価にのみ使い再配布しない（configs/ood_sources.yaml の注記）
- zip は ``ThrottledClient.get_stream`` でストリーミング保存し、サイズと（config にあれば）sha256 を検証する。
  実測 sha256 は ``<zip>.sha256`` サイドカーに記録し、summary の ``zip_sha256`` に出す
- マニフェストが再開の正本（``fetch_common.ManifestWriter``）
"""

from __future__ import annotations

import json
import logging
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cascade.data import fetch_common as fc

log = logging.getLogger(__name__)

SOURCE = "coco"


# ---------------------------------------------------------------------------
# 選択（純関数）
# ---------------------------------------------------------------------------


def select_candidates(
    images: Iterable[dict[str, Any]],
    annotations: Iterable[dict[str, Any]],
    type_categories: dict[str, dict[int, str]],
    *,
    exclusion_ids: set[int],
    min_area_ratio: float,
    person_id: int,
    person_max_count: int,
    per_class_cap: int,
    budget_of: dict[str, int],
) -> list[dict[str, Any]]:
    """type_categories: {subject_type: {category_id: name}}（config 順）。返す dict は ext_id / subject_type_hint /
    candidate_label（カテゴリ名）/ image（元の画像 dict）。"""
    img_by_id = {int(im["id"]): im for im in images}
    anns_by_img: dict[int, list[dict[str, Any]]] = {}
    for a in annotations:
        anns_by_img.setdefault(int(a["image_id"]), []).append(a)

    main_of: dict[int, int] = {}  # image id → 主被写体カテゴリ
    for iid in sorted(img_by_id):
        anns = anns_by_img.get(iid, [])
        if not anns:
            continue
        cats = [int(a["category_id"]) for a in anns]
        if any(c in exclusion_ids for c in cats):
            continue
        if any(int(a.get("iscrowd", 0)) == 1 for a in anns):
            continue
        im = img_by_id[iid]
        area_img = float(im["width"]) * float(im["height"])
        if area_img <= 0:
            continue
        largest = max(anns, key=lambda a: float(a["bbox"][2]) * float(a["bbox"][3]))
        ratio = float(largest["bbox"][2]) * float(largest["bbox"][3]) / area_img
        if ratio < min_area_ratio:
            continue
        main_cat = int(largest["category_id"])
        if main_cat == person_id and cats.count(person_id) > person_max_count:
            continue
        main_of[iid] = main_cat

    assigned: set[int] = set()
    out: list[dict[str, Any]] = []
    for stype, cat_names in type_categories.items():
        budget = budget_of.get(stype, 0)
        picked: list[dict[str, Any]] = []
        n_by_cat: dict[int, int] = {}
        for iid in sorted(main_of):
            if len(picked) >= budget:
                break
            if iid in assigned:
                continue
            cat = main_of[iid]
            if cat not in cat_names or n_by_cat.get(cat, 0) >= per_class_cap:
                continue
            n_by_cat[cat] = n_by_cat.get(cat, 0) + 1
            assigned.add(iid)
            picked.append(
                {
                    "ext_id": f"{iid:012d}",
                    "subject_type_hint": stype,
                    "candidate_label": cat_names[cat],
                    "split_hint": "val2017",
                    "image": img_by_id[iid],
                }
            )
        out.extend(picked)
    return out


def type_categories_from_config(cfg: dict[str, Any]) -> dict[str, dict[int, str]]:
    name_to_id = {str(k): int(v) for k, v in cfg["coco"]["categories"].items()}
    out: dict[str, dict[int, str]] = {}
    for stype, srcs in cfg.get("subject_types", {}).items():
        names = srcs.get("coco")
        if names:
            unknown = [n for n in names if n not in name_to_id]
            if unknown:
                raise ValueError(f"coco.categories に無いカテゴリ名: {unknown}")
            out[stype] = {name_to_id[n]: n for n in names}
    return out


# ---------------------------------------------------------------------------
# zip の取得と読み出し
# ---------------------------------------------------------------------------


def ensure_zip(
    client: Any,
    url: str,
    dst: Path,
    expected_bytes: int | None,
    expected_sha256: str | None = None,
) -> Path:
    """zip が無い／サイズ不一致なら取得する。ストリーミングで書き、部分ファイルは残さない。

    取得後（または既存ファイルの初回検査時）に sha256 を計算して ``<dst>.sha256`` に記録し、
    ``expected_sha256`` があれば一致を要求する。
    """
    side = dst.with_name(dst.name + ".sha256")
    if dst.exists() and (expected_bytes is None or dst.stat().st_size == expected_bytes):
        digest = side.read_text(encoding="utf-8").strip() if side.exists() else ""
        if not digest:
            digest = fc.sha256_file(dst)
            side.write_text(digest + "\n", encoding="utf-8")
        if expected_sha256 and digest != expected_sha256.lower():
            raise fc.FetchError(
                f"既存 zip の sha256 不一致: {dst} got={digest} expected={expected_sha256}"
            )
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    if hasattr(client, "get_stream"):
        digest = client.get_stream(
            url, tmp, expected_bytes=expected_bytes, expected_sha256=expected_sha256
        )
    else:  # テスト用の偽クライアント（get_bytes のみ）
        data = client.get_bytes(url) if hasattr(client, "get_bytes") else client.get(url).content
        tmp.write_bytes(data)
        digest = fc.sha256_bytes(data)
        if expected_bytes is not None and len(data) != expected_bytes:
            tmp.unlink(missing_ok=True)
            raise fc.FetchError(f"サイズ不一致: {url} got={len(data)} expected={expected_bytes}")
        if expected_sha256 and digest != expected_sha256.lower():
            tmp.unlink(missing_ok=True)
            raise fc.FetchError(f"sha256 不一致: {url} got={digest} expected={expected_sha256}")
    tmp.replace(dst)
    side.write_text(digest + "\n", encoding="utf-8")
    return dst


def zip_sha256(dst: Path) -> str | None:
    """``ensure_zip`` が記録したサイドカーの sha256（無ければ None）。"""
    side = dst.with_name(dst.name + ".sha256")
    return side.read_text(encoding="utf-8").strip() if side.exists() else None


def filter_by_license(
    images: Iterable[dict[str, Any]], allowed: set[int]
) -> tuple[list[dict[str, Any]], int]:
    """許可ライセンス id の画像だけ残す。(残った画像, 落とした枚数)。"""
    kept: list[dict[str, Any]] = []
    n_drop = 0
    for im in images:
        if int(im.get("license", 0)) in allowed:
            kept.append(im)
        else:
            n_drop += 1
    return kept, n_drop


def load_instances(zip_path: Path, member: str) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as f:
        return json.load(f)


def read_image_bytes(zf: zipfile.ZipFile, member: str) -> bytes:
    return zf.read(member)


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------


def fetch(
    cfg: dict[str, Any],
    client: Any,
    *,
    dry_run: bool = False,
    max_per_type: int | None = None,
    root: Path | None = None,
    instances: dict[str, Any] | None = None,
    images_zip: Path | None = None,
    budget_of: dict[str, int] | None = None,
) -> dict[str, Any]:
    """``instances`` / ``images_zip`` を与えると zip の取得を省略する（テスト・再実行用）。"""
    cc = cfg["coco"]
    zips = fc.source_dir(SOURCE, root) / "zips"
    ann_zip = zips / "annotations_trainval2017.zip"
    img_zip = images_zip if images_zip is not None else zips / "val2017.zip"
    if instances is None:
        ann_zip = ensure_zip(
            client,
            cc["annotations_zip_url"],
            ann_zip,
            cc.get("annotations_zip_bytes"),
            cc.get("annotations_zip_sha256"),
        )
        instances = load_instances(ann_zip, cc["annotations_member"])

    type_cats = type_categories_from_config(cfg)
    budgets = budget_of if budget_of is not None else fc.source_budgets(cfg, SOURCE, max_per_type)
    budget_of = {t: int(budgets.get(t, 0)) for t in type_cats}
    licenses = {int(lic["id"]): lic for lic in instances.get("licenses", [])}
    allowed_lic = {int(x) for x in cc.get("license_ids_allowed", range(1, 9))}
    images, n_license_rejected = filter_by_license(instances["images"], allowed_lic)
    selected = select_candidates(
        images,
        instances["annotations"],
        type_cats,
        exclusion_ids={int(x) for x in cc["exclusion_category_ids"]},
        min_area_ratio=float(cc.get("min_area_ratio", 0.2)),
        person_id=int(cc["categories"]["person"]),
        person_max_count=int(cc.get("person_max_count", 3)),
        per_class_cap=int(cfg.get("per_class_cap", 40)),
        budget_of=budget_of,
    )

    summary: dict[str, Any] = {
        "source": SOURCE,
        "budget_by_type": budget_of,
        "license_ids_allowed": sorted(allowed_lic),
        "n_license_rejected": n_license_rejected,
        "n_selected": len(selected),
        "by_type": _count_by(selected, "subject_type_hint"),
        "n_downloaded": 0,
        "n_skipped": 0,
    }
    if dry_run:
        summary["candidates"] = [
            {k: s[k] for k in ("ext_id", "subject_type_hint", "candidate_label")} for s in selected
        ]
        return summary

    if images_zip is None:
        img_zip = ensure_zip(
            client,
            cc["images_zip_url"],
            img_zip,
            cc.get("images_zip_bytes"),
            cc.get("images_zip_sha256"),
        )
    summary["zip_sha256"] = {"images": zip_sha256(img_zip), "annotations": zip_sha256(ann_zip)}
    prefix = cc.get("images_member_prefix", "val2017/")
    with (
        zipfile.ZipFile(img_zip) as zf,
        fc.ManifestWriter(SOURCE, root, flush_every=fc.manifest_flush_every(cfg)) as mw,
    ):
        for s in selected:
            if mw.is_done(s["ext_id"]):
                summary["n_skipped"] += 1
                continue
            im = s["image"]
            data = read_image_bytes(zf, prefix + im["file_name"])
            saved = fc.save_image(SOURCE, s["ext_id"], data, root)
            lic = licenses.get(int(im.get("license", 0)), {})
            mw.add(
                fc.new_manifest_row(
                    source=SOURCE,
                    ext_id=s["ext_id"],
                    subject_type_hint=s["subject_type_hint"],
                    candidate_label=s["candidate_label"],
                    split_hint=s["split_hint"],
                    license_short=str(lic.get("name", f"coco_license_{im.get('license')}")),
                    license_url=str(lic.get("url", "")),
                    attribution=f"COCO 2017 val id={int(im['id'])} license_id={im.get('license')} flickr={im.get('flickr_url', '')}",
                    source_url=str(im.get("flickr_url") or im.get("coco_url") or ""),
                    **saved,
                )
            )
            summary["n_downloaded"] += 1
    summary["manifest"] = str(fc.manifest_path(SOURCE, root))
    return summary


def _count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        out[it[key]] = out.get(it[key], 0) + 1
    return out
