"""Hugging Face datasets（CORD-v2、任意で ScreenSpot）から補助 far 候補を取る（design.md §2、G0 補助 far）。

- ``huggingface_hub.snapshot_download(repo_type="dataset", revision=<コミットハッシュ>)`` で parquet だけ取得し、
  pyarrow で画像バイト列を読む（``datasets`` パッケージは使わない）
- revision は config にピン留め（``validate_pins`` が "main" / "latest" を拒否）
- 取り出しは決定的: parquet ファイル名の昇順 → 行番号順に先頭 ``n_take`` 件。
  ``splits``（例 ``[train]``）を config に置くと、その split 名で始まるファイルだけを使う（未指定なら全 split）
- マニフェストが再開の正本（``fetch_common.ManifestWriter``）
- 画像列は ``{"bytes": ..., "path": ...}`` の struct か bytes を想定
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from cascade.data import fetch_common as fc

log = logging.getLogger(__name__)


def list_parquets(snapshot_dir: Path, splits: list[str] | None = None) -> list[Path]:
    """snapshot 内の parquet を名前順で。``splits`` があれば ``<split>-`` / ``<split>.`` で始まるファイルに限る。"""
    files = sorted(p for p in Path(snapshot_dir).rglob("*.parquet"))
    if not splits:
        return files
    prefixes = tuple(str(sp) for sp in splits)
    return [
        p
        for p in files
        if any(p.name == f"{sp}.parquet" or p.name.startswith(f"{sp}-") for sp in prefixes)
    ]


def iter_images(parquet_files: list[Path], column: str, n_take: int):
    """(ext_id, bytes) を決定的順序で n_take 件 yield する。ext_id = ``<file stem>:<row idx>``。"""
    n = 0
    for pf in parquet_files:
        table = pq.read_table(pf, columns=[column])
        col = table.column(column).to_pylist()
        for i, v in enumerate(col):
            if n >= n_take:
                return
            data = v.get("bytes") if isinstance(v, dict) else v
            if not data:
                continue
            yield f"{pf.stem}:{i:06d}", bytes(data)
            n += 1


def snapshot(spec: dict[str, Any], cache_dir: Path | None = None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=spec["hf_id"],
            repo_type="dataset",
            revision=spec["revision"],
            allow_patterns=["*.parquet"],
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )


def fetch_one(
    key: str,
    spec: dict[str, Any],
    *,
    dry_run: bool = False,
    max_per_type: int | None = None,
    root: Path | None = None,
    snapshot_dir: Path | None = None,
    flush_every: int = 20,
) -> dict[str, Any]:
    source = spec.get("source_name", key)
    if source not in fc.SOURCES:
        raise ValueError(f"未知の source_name: {source!r}")
    n_take = int(max_per_type if max_per_type is not None else spec.get("n_take", 200))
    summary: dict[str, Any] = {
        "source": source,
        "hf_id": spec["hf_id"],
        "revision": spec["revision"],
        "n_take": n_take,
        "n_downloaded": 0,
        "n_skipped": 0,
    }
    if dry_run:
        return summary
    snap = snapshot_dir or snapshot(spec, cache_dir=fc.source_dir(source, root) / "hf_cache")
    splits = [str(x) for x in (spec.get("splits") or [])]
    files = list_parquets(snap, splits or None)
    if not files:
        raise fc.FetchError(f"parquet が見つからない: {snap} (splits={splits or 'all'})")
    summary["splits"] = splits or "all"
    with fc.ManifestWriter(source, root, flush_every=flush_every) as mw:
        for ext_id, data in iter_images(files, spec.get("image_column", "image"), n_take):
            if mw.is_done(ext_id):
                summary["n_skipped"] += 1
                continue
            saved = fc.save_image(source, ext_id, data, root)
            mw.add(
                fc.new_manifest_row(
                    source=source,
                    ext_id=ext_id,
                    subject_type_hint=spec.get("subject_type", "texture_document_screen"),
                    candidate_label=key,
                    split_hint=ext_id.split(":")[0].split("-")[0],
                    license_short=str(spec.get("license_short", "")),
                    license_url=str(spec.get("license_url", "")),
                    attribution=f"{spec['hf_id']}@{spec['revision']}",
                    source_url=str(spec.get("source_url", "")),
                    **saved,
                )
            )
            summary["n_downloaded"] += 1
    summary["manifest"] = str(fc.manifest_path(source, root))
    return summary


def fetch(
    cfg: dict[str, Any],
    *,
    dry_run: bool = False,
    max_per_type: int | None = None,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    out = []
    for key, spec in cfg.get("hf", {}).items():
        if not spec.get("enabled", True):
            continue
        out.append(
            fetch_one(
                key,
                spec,
                dry_run=dry_run,
                max_per_type=max_per_type,
                root=root,
                flush_every=fc.manifest_flush_every(cfg),
            )
        )
    return out
