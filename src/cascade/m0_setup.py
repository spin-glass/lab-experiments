"""M0 セットアップ CLI（design.md §6 M0、計画 Phase B-a）。

現時点は ``check-env`` のみ実装:
- torch 版、MPS の is_built / is_available、選択 device
- E1 が HF キャッシュにピン留め revision で揃っている（または ``CASCADE_NETWORK_TESTS`` が設定されている）場合のみ
  合成画像 256 枚のマイクロベンチ（img/s）と CPU vs MPS の余弦一致（16 枚、min cosine。期待 ≥ 0.999）
- ``--skip-model`` で torch / MPS の事実だけを出す

出力は JSON 1 個（stdout）。``--out`` で保存もできる。画像・生データは扱わない。

実行: ``uv run python -m cascade.m0_setup check-env [--skip-model] [--encoder e1]``
（計画 Phase B-a の表記 ``--check-env`` も同じ意味で受け付ける: ``python -m cascade.m0_setup --check-env --skip-model``）

Phase B-b（Yelp 到着後。Yelp Open Dataset を paths.YELP に手動配置してから実行）:
- ``verify-yelp``: レイアウト検査（photos.json / photos/ / 規約 PDF / DOWNLOAD.txt の sha256）、photos.json の読取、
  クラス別枚数・ラベル欠損率、標本画像のサイズ・JPEG 品質統計。JSON を marts/yelp_stats.json に保存
- ``build-splits``: business_id（``--with-dedup`` で pHash 群も連結。再開可能・バッチ）単位の
  tune_fit / tune_cal / work / eval 分割 → core/labels、core/eval_sets（役割別 sha256）、marts/splits.parquet、
  runs 行（tracking fan-out）、reports/m0_setup.md。photo_id 一覧はリポジトリに置かない
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import polars as pl
import typer

from cascade import paths
from cascade.config import load_config
from cascade.data import yelp as yelp_mod
from cascade.embed.encoder import find_spec, is_cached_locally, load_encoder, resolve_device

app = typer.Typer(help="M0 セットアップ（環境確認・データ準備）", no_args_is_help=True)

DEFAULT_CONFIG = paths.REPO_ROOT / "configs" / "encoders.yaml"


@app.callback()
def _main() -> None:
    """サブコマンド名（check-env 等）を維持するためのルート callback。"""


def torch_facts() -> dict[str, Any]:
    import torch

    mps_built = bool(torch.backends.mps.is_built())
    mps_avail = bool(torch.backends.mps.is_available())
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "mps_built": mps_built,
        "mps_available": mps_avail,
        "device": resolve_device("auto"),
        "cpu_threads": torch.get_num_threads(),
    }


def synthetic_images(n: int, size: int, seed: int = 0) -> list[Any]:
    from PIL import Image

    rng = np.random.default_rng(seed)
    return [
        Image.fromarray(rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8), "RGB")
        for _ in range(n)
    ]


def _min_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return float((a * b).sum(axis=1).min())


def benchmark_encoder(
    spec: dict[str, Any], n_bench: int = 256, n_agree: int = 16, batch_size: int = 64
) -> dict[str, Any]:
    """合成画像で img/s と CPU/MPS 余弦一致を測る。モデルはキャッシュ済み前提。"""
    out: dict[str, Any] = {
        "encoder": spec["name"],
        "hf_id": spec["hf_id"],
        "revision": spec["revision"],
    }
    size = int(spec.get("image_size", 224))
    enc = load_encoder(spec, batch_size=batch_size)
    out["model_key"] = enc.model_key
    out["device"] = enc.device
    t0 = time.perf_counter()
    enc.ensure_loaded()  # type: ignore[attr-defined]
    out["load_sec"] = round(time.perf_counter() - t0, 3)
    out["dim"] = enc.dim

    imgs = synthetic_images(n_bench, size)
    enc.encode_images(imgs[:batch_size])  # warm-up（MPS のカーネルコンパイル分を除く）
    t0 = time.perf_counter()
    emb = enc.encode_images(imgs)
    dt = time.perf_counter() - t0
    out["bench_n"] = n_bench
    out["bench_sec"] = round(dt, 3)
    out["img_per_sec"] = round(n_bench / dt, 2) if dt > 0 else None
    out["bench_batch_size"] = batch_size
    out["nan_or_inf"] = bool(~np.isfinite(emb).all())

    if enc.device != "cpu":
        cpu_spec = dict(spec, device="cpu")
        cpu_enc = load_encoder(cpu_spec, batch_size=batch_size)
        sub = imgs[:n_agree]
        a = enc.encode_images(sub)
        b = cpu_enc.encode_images(sub)
        mc = _min_cosine(a, b)
        out["agreement"] = {
            "n": n_agree,
            "devices": [enc.device, "cpu"],
            "min_cosine": round(mc, 6),
            "ok": bool(mc >= 0.999),
            "expected_min": 0.999,
        }
        est = None
        if out["img_per_sec"]:
            t0 = time.perf_counter()
            cpu_enc.encode_images(imgs[:batch_size])
            cpu_dt = time.perf_counter() - t0
            est = round(batch_size / cpu_dt, 2) if cpu_dt > 0 else None
        out["cpu_img_per_sec_estimate"] = est
    else:
        out["agreement"] = {"skipped": "device is cpu; nothing to compare"}
    # 所要見積り（G0 規模の目安。25k work + 数千 ext）
    if out["img_per_sec"]:
        out["eta_sec_per_10k_images"] = round(10_000 / out["img_per_sec"], 1)
    return out


@app.command("check-env")
def check_env(
    config: Annotated[Path, typer.Option(help="configs/encoders.yaml")] = DEFAULT_CONFIG,
    encoder: Annotated[str, typer.Option(help="ベンチ対象のエンコーダ name")] = "e1",
    skip_model: Annotated[
        bool, typer.Option("--skip-model", help="torch / MPS の事実のみ出す")
    ] = False,
    n_bench: Annotated[int, typer.Option(help="マイクロベンチの合成画像枚数")] = 256,
    batch_size: Annotated[int, typer.Option()] = 64,
    out: Annotated[Path | None, typer.Option(help="JSON の保存先（省略時は stdout のみ）")] = None,
) -> None:
    """環境確認: torch / MPS / device、E1 のマイクロベンチと CPU-MPS 余弦一致。"""
    result: dict[str, Any] = {"torch": torch_facts(), "config": str(config)}
    if skip_model:
        result["model"] = {"skipped": "--skip-model"}
    else:
        cfg = load_config(config)  # ピン留め検証込み
        spec = find_spec(cfg, encoder)
        cached = is_cached_locally(spec)
        allow_net = bool(os.environ.get("CASCADE_NETWORK_TESTS"))
        result["model_cached_locally"] = cached
        if not cached and not allow_net:
            result["model"] = {
                "skipped": (
                    f"{spec['hf_id']}@{spec['revision']} が HF キャッシュに無い。"
                    "ダウンロードを許可するには CASCADE_NETWORK_TESTS=1 を設定して再実行"
                )
            }
        else:
            try:
                result["model"] = benchmark_encoder(spec, n_bench=n_bench, batch_size=batch_size)
            except Exception as e:  # noqa: BLE001 - 環境確認コマンドは落とさず記録する
                result["model"] = {"failed": f"{type(e).__name__}: {e}"}
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    typer.echo(text)


DEFAULT_M0_CONFIG = paths.REPO_ROOT / "configs" / "m0.yaml"
DEFAULT_REPORT = paths.REPO_ROOT / "reports" / "m0_setup.md"


def _stats_path() -> Path:
    return paths.LAYERS["marts"] / "yelp_stats.json"


def _load_m0_config(config: Path) -> dict[str, Any]:
    cfg = load_config(config)  # ピン留め検証込み（taxonomy_version 等）
    tracking_path = paths.REPO_ROOT / "configs" / "tracking.yaml"
    if "tracking" not in cfg and tracking_path.exists():
        cfg.update(load_config(tracking_path, validate=False))
    return cfg


def verify_yelp_stats(
    cfg: dict[str, Any], root: Path | None, *, n_sample: int | None = None, seed: int | None = None
) -> dict[str, Any]:
    """verify-yelp の本体。レイアウト・photos.json の統計・標本画像統計を dict で返す（集計値のみ）。"""
    ycfg = cfg.get("yelp", {}) or {}
    layout = yelp_mod.verify_layout(root, cfg)
    out: dict[str, Any] = {"layout": layout}
    pj = Path(layout["photos_json"])
    if not pj.is_file():
        return out
    df = yelp_mod.load_photos_json(pj)
    out["photos_json_sha256"] = yelp_mod.dedup.sha256_file(pj)
    out["n_photos"] = int(df.height)
    out["n_businesses"] = int(df["business_id"].drop_nulls().n_unique())
    out["class_counts"] = yelp_mod.class_counts(
        df, cfg.get("splits", {}).get("classes") or yelp_mod.CLASSES
    ).to_dicts()
    out["missing_label_rate"] = yelp_mod.missing_label_rate(df)
    scfg = ycfg.get("sample_image_stats", {}) or {}
    n = int(scfg.get("n", 2000) if n_sample is None else n_sample)
    sd = int(scfg.get("seed", 0) if seed is None else seed)
    if n > 0 and Path(layout["photos_dir"]).is_dir():
        stats = yelp_mod.sample_image_stats(
            df,
            Path(layout["root"]),
            n=n,
            seed=sd,
            photos_dir=str(ycfg.get("photos_dir", "photos")),
            ext=str(ycfg.get("image_ext", ".jpg")),
        )
        out["image_stats"] = yelp_mod.summarize_image_stats(stats)
        out["image_stats"]["sample"] = {"n": n, "seed": sd}
    return out


@app.command("verify-yelp")
def verify_yelp(
    config: Annotated[Path, typer.Option(help="configs/m0.yaml")] = DEFAULT_M0_CONFIG,
    root: Annotated[Path | None, typer.Option(help="Yelp の配置先（既定 data/yelp）")] = None,
    n_sample: Annotated[int | None, typer.Option(help="標本画像の枚数（config 既定）")] = None,
    seed: Annotated[int | None, typer.Option(help="標本の seed（config 既定）")] = None,
    out: Annotated[
        Path | None, typer.Option(help="JSON 保存先（既定 marts/yelp_stats.json）")
    ] = None,
    strict: Annotated[bool, typer.Option("--strict", help="レイアウト不備で非 0 終了")] = False,
) -> None:
    """Yelp 配置の検証: レイアウト・規約記録・クラス別枚数・欠損率・標本画像統計（集計値のみ）。"""
    cfg = _load_m0_config(config)
    result = verify_yelp_stats(cfg, root, n_sample=n_sample, seed=seed)
    text = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    out = out or _stats_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    typer.echo(text)
    if strict and not result["layout"]["ok"]:
        raise typer.Exit(code=1)


def run_build_splits(
    cfg: dict[str, Any],
    root: Path | None,
    *,
    with_dedup: bool = False,
    report_path: Path | None = None,
    stats_path: Path | None = None,
) -> dict[str, Any]:
    """build-splits の本体。テーブル・runs 行・レポートを書き、集計 dict を返す。"""
    from cascade.io import tables
    from cascade.tracking.adapter import RunRecorder

    scfg = dict(cfg.get("splits", {}) or {})
    ycfg = cfg.get("yelp", {}) or {}
    dcfg = cfg.get("dedup", {}) or {}
    layout = yelp_mod.verify_layout(root, cfg)
    pj = Path(layout["photos_json"])
    if not pj.is_file():
        raise FileNotFoundError(
            f"photos.json が無い: {pj}（先に Yelp を配置し verify-yelp を実行）"
        )
    df = yelp_mod.load_photos_json(pj)
    pj_sha = yelp_mod.dedup.sha256_file(pj)

    groups: dict[str, str] | None = None
    dedup_info: dict[str, Any] | None = None
    if with_dedup:
        shard_dir = Path(layout["root"]) / str(dcfg.get("shard_dir", "_phash"))
        phashes = yelp_mod.compute_phash_shards(
            df,
            Path(layout["root"]),
            shard_dir,
            batch_size=int(dcfg.get("batch_size", 5000)),
            photos_dir=str(ycfg.get("photos_dir", "photos")),
            ext=str(ycfg.get("image_ext", ".jpg")),
        )
        groups = yelp_mod.build_dedup_groups(phashes, int(dcfg.get("phash_max", 8)))
        sizes: dict[str, int] = {}
        for g in groups.values():
            sizes[g] = sizes.get(g, 0) + 1
        multi = {g: n for g, n in sizes.items() if n > 1}
        dedup_info = {
            "n_hashed": len(phashes),
            "n_groups_multi": len(multi),
            "n_in_groups": int(sum(multi.values())),
            "phash_max": int(dcfg.get("phash_max", 8)),
        }

    splits = yelp_mod.build_splits(df, groups, scfg, scfg.get("seed", "yelp-splits-v1"))
    crossing = yelp_mod.dedup.check_no_crossing(
        dict(zip(splits["photo_id"].to_list(), splits["role"].to_list(), strict=True)),
        dict(zip(splits["photo_id"].to_list(), splits["component_id"].to_list(), strict=True)),
    )
    summary = yelp_mod.split_summary(splits, df, scfg.get("classes") or yelp_mod.CLASSES)
    now = datetime.now(UTC)
    eval_rows = yelp_mod.freeze_eval_sets(
        splits,
        prefix=str(scfg.get("eval_set_prefix", "yelp-v1")),
        seed=scfg.get("seed", ""),
        now=now,
    )

    # core/labels（Yelp 付与ラベル。来歴の混在した既存ラベル）
    labels = df.select(
        pl.col("photo_id"),
        pl.lit("yelp").alias("label_source"),
        pl.col("label"),
        pl.lit(f"photos.json sha256={pj_sha}").alias("provenance"),
        pl.lit(now).alias("ts"),
    )
    tables.write_table("labels", labels, "core")
    tables.write_table("eval_sets", pl.DataFrame(eval_rows), "core")
    marts_dir = paths.LAYERS["marts"]
    marts_dir.mkdir(parents=True, exist_ok=True)
    splits_path = marts_dir / "splits.parquet"
    splits.write_parquet(splits_path)

    stats: dict[str, Any] = {}
    stats_path = stats_path or _stats_path()
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats.update(
        {
            "layout": layout,
            "photos_json_sha256": pj_sha,
            "n_photos": int(df.height),
            "n_businesses": int(df["business_id"].drop_nulls().n_unique()),
            "class_counts": yelp_mod.class_counts(
                df, scfg.get("classes") or yelp_mod.CLASSES
            ).to_dicts(),
            "missing_label_rate": yelp_mod.missing_label_rate(df),
            "split_config": scfg,
            "split_summary": summary.to_dicts(),
            "crossing": crossing,
            "eval_sets": [{**r, "created_at": r["created_at"].isoformat()} for r in eval_rows],
            "dedup": dedup_info,
        }
    )

    eval_set_id = f"{scfg.get('eval_set_prefix', 'yelp-v1')}-eval"
    rec = RunRecorder(
        cfg,
        taxonomy_version=str(cfg.get("taxonomy_version")),
        rulebook_version=str(cfg.get("rulebook_version", "未導入")),
        eval_set_id=eval_set_id,
        model_id="none",
    )
    rec.log_metrics(
        {
            "n_photos": int(df.height),
            "n_businesses": stats["n_businesses"],
            "missing_label_rate": stats["missing_label_rate"],
            **{
                f"n_{r['role']}": int(r["n_items"])
                for r in summary.unique(subset=["role"]).to_dicts()
            },
            "n_crossing": len(crossing),
        }
    )
    rec.log_table("split_summary", summary)
    rec.log_table(
        "eval_sets", pl.DataFrame(eval_rows).with_columns(pl.col("created_at").cast(pl.Utf8))
    )
    finished = rec.finish()
    stats["stamps"] = {
        "run_id": rec.run_id,
        "git_sha": rec.stamps["git_sha"],
        "taxonomy_version": rec.stamps["taxonomy_version"],
        "rulebook_version": rec.stamps["rulebook_version"],
        "eval_set_id": eval_set_id,
        "api_cost": "0（API 呼び出しなし）",
        "wandb_url": finished.get("wandb_url"),
        "tracking": finished.get("backends"),
        "generated_at": now.isoformat(),
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    report_path = report_path or DEFAULT_REPORT
    yelp_mod.m0_report(stats, splits, report_path)
    return {
        "run_id": rec.run_id,
        "splits_parquet": str(splits_path),
        "report": str(report_path),
        "stats_json": str(stats_path),
        "crossing": crossing,
        "split_summary": summary.to_dicts(),
        "eval_sets": stats["eval_sets"],
        "tracking": finished.get("backends"),
    }


@app.command("build-splits")
def build_splits(
    config: Annotated[Path, typer.Option(help="configs/m0.yaml")] = DEFAULT_M0_CONFIG,
    root: Annotated[Path | None, typer.Option(help="Yelp の配置先（既定 data/yelp）")] = None,
    with_dedup: Annotated[
        bool,
        typer.Option("--with-dedup", help="pHash 近重複群を分割キーに加える（バッチ・再開可能）"),
    ] = False,
    report: Annotated[Path | None, typer.Option(help="reports/m0_setup.md の出力先")] = None,
) -> None:
    """tune_fit / tune_cal / work / eval 分割と eval 凍結、reports/m0_setup.md の生成。"""
    cfg = _load_m0_config(config)
    try:
        result = run_build_splits(cfg, root, with_dedup=with_dedup, report_path=report)
    except yelp_mod.SplitSizeError as e:
        typer.echo(f"SplitSizeError: {e}", err=True)
        raise typer.Exit(code=2) from e
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def main(argv: list[str] | None = None) -> None:
    """エントリポイント。``--check-env``（計画・CLAUDE.md の表記）をサブコマンド ``check-env`` に写像する。"""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--check-env" in args:
        args = ["check-env", *(a for a in args if a != "--check-env")]
    app(args=args)


if __name__ == "__main__":
    main()
