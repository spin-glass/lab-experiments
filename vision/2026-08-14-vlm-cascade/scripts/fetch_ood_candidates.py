"""外部候補画像の取得 CLI（G0 層 A / 対照正例 / 補助 far。design.md §2・§6 G0、計画 Phase B-a）。

    uv run python scripts/fetch_ood_candidates.py --config configs/ood_sources.yaml --dry-run
    uv run python scripts/fetch_ood_candidates.py --source open_images --select-only
    uv run python scripts/fetch_ood_candidates.py --source open_images --max-per-type 10

モード:
- ``--dry-run``: 取得計画（源・型・MID/カテゴリ・予算配分・URL・出力先）を JSON で出すだけ。
  ネットワークにもファイルにも触れない（``CASCADE_CONTACT`` 不要）
- ``--select-only``: 候補の選択まで実行して画像は取得しない（ラベル CSV / API 照会のためネットワークは使う）
- 既定: 選択して画像を取得し、マニフェストへ書く

予算: 型ごとの候補数（目標 × candidate_multiplier）を、その型に抽出ラベルを持つ源で均等に分ける
（``fetch_common.allocate_budgets``。合計 ≈ 目標の 2 倍。配分は config だけで決まり ``--source`` に依存しない）。
連絡先（``CASCADE_CONTACT``）は Commons を含むときだけ必須（GCS / S3 / COCO は不要）。

画像は data/ext/<source>/images/ に落ち git 管理外。取得後に scripts/write_ood_lock.py で lock を作る。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Annotated, Any

import typer

from cascade import paths
from cascade.config import load_config
from cascade.data import fetch_coco, fetch_common, fetch_commons, fetch_hf, fetch_openimages

app = typer.Typer(add_completion=False)
SOURCES = ("open_images", "commons", "coco", "hf")


def _resolve_targets(source: str) -> tuple[str, ...]:
    targets = SOURCES if source == "all" else tuple(source.split(","))
    unknown = [s for s in targets if s not in SOURCES]
    if unknown:
        raise typer.BadParameter(f"未知の source: {unknown}（候補: {SOURCES}）")
    return targets


def build_plan(
    cfg: dict[str, Any],
    targets: tuple[str, ...] | list[str],
    max_per_type: int | None = None,
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """ネットワークにもファイルにも触れずに取得計画を組み立てる（``--dry-run`` の出力）。

    各源の抽出ラベル（MID / カテゴリ）と型ごとの候補予算、取得 URL、出力先を列挙する。
    連絡先（``CASCADE_CONTACT``）は設定済みかどうかだけを報告し、値は出さない。
    """
    stypes = list(cfg.get("subject_types", {}))
    alloc = fetch_common.allocate_budgets(cfg, max_per_type)
    plan: dict[str, Any] = {
        "mode": "dry_run",
        "config": None if config_path is None else str(config_path),
        "data_root": str(paths.DATA),
        "contact_configured": bool(cfg.get("contact") or os.environ.get("CASCADE_CONTACT")),
        "throttle": dict(cfg.get("throttle", {})),
        "candidate_multiplier": int(cfg.get("candidate_multiplier", 2)),
        "per_class_cap": int(cfg.get("per_class_cap", 40)),
        "min_side": int(cfg.get("min_side", 256)),
        "budget_by_type": {t: fetch_common.type_budget(cfg, t, max_per_type) for t in stypes},
        "allocation": alloc,
        "contact_required": "commons" in targets,
        "sources": [],
    }
    for s in targets:
        entry: dict[str, Any] = {"source": s}
        if s == "open_images":
            oi = cfg["open_images"]
            mids = fetch_openimages.type_mids_from_config(cfg)
            entry.update(
                {
                    "images_dir": str(fetch_common.image_dir(s)),
                    "manifest": str(fetch_common.manifest_path(s)),
                    "download_backend": oi.get("download_backend", "https"),
                    "food_mid": oi.get("food_mid"),
                    "splits": {
                        split: {
                            "labels_url": spec["labels_url"],
                            "attribution_url": spec["attribution_url"],
                        }
                        for split, spec in oi.get("splits", {}).items()
                    },
                    "food_exclusion_mids": len(oi.get("food_exclusion_mids") or {}),
                    "by_type": {
                        t: {"budget": alloc["open_images"].get(t, 0), "mids": m}
                        for t, m in mids.items()
                    },
                }
            )
        elif s == "commons":
            cc = cfg["commons"]
            cats = fetch_commons.type_categories_from_config(cfg)
            entry.update(
                {
                    "images_dir": str(fetch_common.image_dir(s)),
                    "manifest": str(fetch_common.manifest_path(s)),
                    "api_url": cc.get("api_url"),
                    "max_depth": int(cc.get("max_depth", 2)),
                    "thumb_width": cc.get("thumb_width"),
                    "by_type": {
                        t: {"budget": alloc["commons"].get(t, 0), "categories": c}
                        for t, c in cats.items()
                    },
                }
            )
        elif s == "coco":
            co = cfg["coco"]
            cats_by_id = fetch_coco.type_categories_from_config(cfg)
            entry.update(
                {
                    "images_dir": str(fetch_common.image_dir(s)),
                    "manifest": str(fetch_common.manifest_path(s)),
                    "annotations_zip_url": co.get("annotations_zip_url"),
                    "images_zip_url": co.get("images_zip_url"),
                    "exclusion_category_ids": list(co.get("exclusion_category_ids", [])),
                    "images_zip_sha256": co.get("images_zip_sha256"),
                    "license_ids_allowed": list(co.get("license_ids_allowed", range(1, 9))),
                    "by_type": {
                        t: {
                            "budget": alloc["coco"].get(t, 0),
                            "categories": {str(k): v for k, v in c.items()},
                        }
                        for t, c in cats_by_id.items()
                    },
                }
            )
        elif s == "hf":
            entry["datasets"] = [
                {
                    "key": key,
                    "source": spec.get("source_name", key),
                    "enabled": bool(spec.get("enabled", True)),
                    "hf_id": spec.get("hf_id"),
                    "revision": spec.get("revision"),
                    "n_take": int(
                        max_per_type if max_per_type is not None else spec.get("n_take", 200)
                    ),
                    "subject_type": spec.get("subject_type"),
                    "splits": spec.get("splits") or "all",
                    "images_dir": str(fetch_common.image_dir(spec.get("source_name", key))),
                }
                for key, spec in cfg.get("hf", {}).items()
            ]
        plan["sources"].append(entry)
    return plan


@app.command()
def main(
    config: Annotated[Path, typer.Option("--config")] = Path("configs/ood_sources.yaml"),
    source: Annotated[
        str,
        typer.Option("--source", help="open_images | commons | coco | hf | all（カンマ区切り可）"),
    ] = "all",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="取得計画を表示するだけ。ネットワーク・ファイル書き込みなし"
        ),
    ] = False,
    select_only: Annotated[
        bool,
        typer.Option(
            "--select-only",
            help="候補の選択まで実行し画像は取得しない（ラベル CSV / API 照会にネットワークを使う）",
        ),
    ] = False,
    max_per_type: Annotated[
        int | None, typer.Option("--max-per-type", help="型ごとの候補数上限を上書き")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(config)  # ピン留め検証込み
    targets = _resolve_targets(source)

    if dry_run:
        plan = build_plan(cfg, targets, max_per_type, config_path=config)
        typer.echo(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    paths.ensure_dirs()
    results: list[dict[str, Any]] = []
    client = None
    alloc = fetch_common.allocate_budgets(cfg, max_per_type)
    for s in targets:
        if s == "hf":
            results.extend(fetch_hf.fetch(cfg, dry_run=select_only, max_per_type=max_per_type))
            continue
        if client is None:
            # 連絡先は Commons を含むときだけ必須（GCS / S3 / COCO は無くても動く）
            client = fetch_common.make_client(cfg, require_contact="commons" in targets)
        mod = {"open_images": fetch_openimages, "commons": fetch_commons, "coco": fetch_coco}[s]
        results.append(
            mod.fetch(
                cfg,
                client,
                dry_run=select_only,
                max_per_type=max_per_type,
                budget_of=alloc.get(s, {}),
            )
        )

    for r in results:
        if "candidates" in r:
            r["candidates_head"] = r.pop("candidates")[:20]
    typer.echo(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
