"""cascade.data.yelp と m0_setup の verify-yelp / build-splits（design.md §2、g0_nonfood_plan.md §3.1、Phase B-b）。

合成 photos.json（5,000 行・800 店舗・偏ったクラス分布、menu は少数）と小さな生成画像のみ。ネットワーク・モデル DL なし。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from cascade import m0_setup, paths
from cascade.data import dedup, yelp
from cascade.io import tables

N_ROWS = 5_000
N_BIZ = 800
CLASS_P = {"food": 0.55, "inside": 0.20, "outside": 0.10, "drink": 0.08, "menu": 0.04, None: 0.03}

# 5,000 行規模に縮めた最小 n（本番 configs/m0.yaml の比率を保つ）
SIZES = {
    "seed": "test-splits-v1",
    "ratios": {"tune_fit": 0.20, "tune_cal": 0.10, "work": 0.35, "eval": 0.35},
    "minimums": {
        "tune_fit": {"food": 500, "drink": 90, "menu": 90, "inside": 90, "outside": 90},
        "tune_cal": {"food": 400},
        "eval": {"food": 700, "drink": 60, "menu": 60, "inside": 60, "outside": 60},
    },
    "eval_balanced_per_class": 50,
    "classes": list(yelp.CLASSES),
    "include_unlabeled": True,
    "eval_set_prefix": "test-v1",
}


def _synthetic_photos(
    n: int = N_ROWS, n_biz: int = N_BIZ, seed: int = 0
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    labels = list(CLASS_P)
    p = np.array([CLASS_P[k] for k in labels])
    # 店舗サイズは偏らせる（少数の店舗に多くの写真）
    biz_w = rng.pareto(1.5, size=n_biz) + 1
    biz_idx = rng.choice(n_biz, size=n, p=biz_w / biz_w.sum())
    rows = []
    for i in range(n):
        lb = labels[int(rng.choice(len(labels), p=p))]
        row: dict[str, object] = {
            "photo_id": f"p{i:06d}",
            "business_id": f"b{int(biz_idx[i]):04d}",
            "caption": "" if rng.random() < 0.5 else f"caption {i}",
        }
        if lb is not None:
            row["label"] = lb
        elif rng.random() < 0.5:
            row["label"] = ""  # 空文字も欠損扱い
        rows.append(row)
    return rows


@pytest.fixture
def photos_df() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "photo_id": r["photo_id"],
                "business_id": r["business_id"],
                "caption": r["caption"],
                "label": (r.get("label") or None),
            }
            for r in _synthetic_photos()
        ],
        schema={"photo_id": pl.Utf8, "business_id": pl.Utf8, "caption": pl.Utf8, "label": pl.Utf8},
    )


@pytest.fixture
def layers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    d = {k: tmp_path / "data" / k for k in ("raw", "core", "marts")}
    monkeypatch.setattr(paths, "LAYERS", d)
    return d


_BASE = None


def _write_image(path: Path, seed: int, size: tuple[int, int] = (64, 48)) -> None:
    """seed で位相をずらした滑らかな模様の小さな JPEG（5,000 枚でも数秒で書ける）。"""
    global _BASE
    if _BASE is None or _BASE.shape[:2] != (size[1], size[0]):
        yy, xx = np.mgrid[0 : size[1], 0 : size[0]].astype(np.float32)
        _BASE = np.stack([xx / size[0] * (2 + c) + yy / size[1] * (1 + c) for c in range(3)], -1)
    arr = 127 + 120 * np.sin(_BASE + (seed % 97) * 0.13)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").save(
        path, format="JPEG", quality=85
    )


@pytest.fixture
def yelp_root(tmp_path: Path) -> Path:
    """photos.json（5,000 行）＋ 全行ぶんの小さな JPEG ＋ 規約 PDF ＋ DOWNLOAD.txt。"""
    root = tmp_path / "yelp"
    root.mkdir()
    rows = _synthetic_photos()
    with (root / "photos.json").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for i, r in enumerate(rows):
        _write_image(root / "photos" / f"{r['photo_id']}.jpg", seed=i)
    (root / "Dataset_User_Agreement.pdf").write_bytes(b"%PDF-1.4\n%stub\n")
    (root / "DOWNLOAD.txt").write_text(
        "# Yelp Open Dataset\nterms_version: 2024-01 Dataset License\ndownloaded_at: 2026-09-07\narchive_sha256: deadbeef\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def m0_config(tmp_path: Path) -> Path:
    cfg = {
        "taxonomy_version": "0.1.0",
        "rulebook_version": "未導入",
        "yelp": {
            "root": None,
            "photos_json": "photos.json",
            "photos_dir": "photos",
            "download_record": "DOWNLOAD.txt",
            "image_ext": ".jpg",
            "terms": {"required_fields": ["terms_version", "downloaded_at"]},
            "sample_image_stats": {"n": 30, "seed": 1},
        },
        "splits": SIZES,
        "dedup": {"phash_max": 8, "batch_size": 16, "shard_dir": "_phash"},
        "tracking": [],
    }
    p = tmp_path / "m0.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# photos.json とレイアウト
# ---------------------------------------------------------------------------


def test_load_photos_json_parses_and_normalizes_missing_labels(yelp_root: Path):
    df = yelp.load_photos_json(yelp_root / "photos.json")
    assert df.columns == ["photo_id", "business_id", "caption", "label"]
    assert df.height == N_ROWS
    assert df["photo_id"].n_unique() == N_ROWS
    assert set(df["label"].drop_nulls().unique().to_list()) == set(yelp.CLASSES)
    # 空文字ラベルと欠損キーはどちらも null
    assert df["label"].null_count() > 0
    assert "" not in df["label"].to_list()


def test_load_photos_json_rejects_duplicate_ids(tmp_path: Path):
    p = tmp_path / "dup.json"
    p.write_text('{"photo_id": "a", "label": "food"}\n{"photo_id": "a", "label": "menu"}\n')
    with pytest.raises(ValueError, match="重複"):
        yelp.load_photos_json(p)


def test_class_counts_and_missing_rate(photos_df: pl.DataFrame):
    cc = yelp.class_counts(photos_df)
    assert cc["label"].to_list() == [*yelp.CLASSES, None]
    assert int(cc["n"].sum()) == photos_df.height
    assert abs(float(cc["share"].sum()) - 1.0) < 1e-9
    food = cc.filter(pl.col("label") == "food")["n"][0]
    menu = cc.filter(pl.col("label") == "menu")["n"][0]
    assert food > 10 * menu  # 偏った分布
    rate = yelp.missing_label_rate(photos_df)
    assert 0.0 < rate < 0.1
    assert rate == pytest.approx(cc.filter(pl.col("label").is_null())["n"][0] / photos_df.height)


def test_verify_layout_ok_and_reports_record(yelp_root: Path, m0_config: Path):
    cfg = yaml.safe_load(m0_config.read_text(encoding="utf-8"))
    res = yelp.verify_layout(yelp_root, cfg)
    assert res["ok"], res["problems"]
    assert res["n_image_files"] == N_ROWS
    assert res["terms_pdfs"] == ["Dataset_User_Agreement.pdf"]
    assert res["download_record_fields"]["terms_version"] == "2024-01 Dataset License"
    assert res["download_record_fields"]["downloaded_at"] == "2026-09-07"
    assert res["download_record_sha256"] == dedup.sha256_file(yelp_root / "DOWNLOAD.txt")


def test_verify_layout_lists_problems(tmp_path: Path):
    root = tmp_path / "empty"
    root.mkdir()
    (root / "DOWNLOAD.txt").write_text("terms_version: x\n", encoding="utf-8")
    res = yelp.verify_layout(root, {})
    assert not res["ok"]
    joined = "\n".join(res["problems"])
    assert "photos.json" in joined and "photos/" in joined and "PDF" in joined
    assert res["missing_fields"] == ["downloaded_at"]


def test_sample_image_stats_reads_existing_images(yelp_root: Path):
    df = yelp.load_photos_json(yelp_root / "photos.json").head(40)
    stats = yelp.sample_image_stats(df, yelp_root, n=10, seed=3)
    assert stats.height == 10
    assert stats["width"].to_list() == [64] * 10
    assert stats["long_side"].to_list() == [64] * 10
    assert stats["format"].to_list() == ["JPEG"] * 10
    assert all(70 <= q <= 100 for q in stats["jpeg_quality_estimate"].to_list())
    summ = yelp.summarize_image_stats(stats)
    assert summ["all"]["long_side_median"] == 64
    assert summ["all"]["n_readable"] == 10
    # 同じ seed なら同じ標本
    again = yelp.sample_image_stats(df, yelp_root, n=10, seed=3)
    assert again["photo_id"].to_list() == stats["photo_id"].to_list()


# ---------------------------------------------------------------------------
# 分割
# ---------------------------------------------------------------------------


def _label_of(df: pl.DataFrame) -> dict[str, str | None]:
    return dict(zip(df["photo_id"].to_list(), df["label"].to_list(), strict=True))


def _counts(splits: pl.DataFrame, df: pl.DataFrame) -> dict[str, dict[str, int]]:
    j = splits.join(df.select("photo_id", "label"), on="photo_id")
    out: dict[str, dict[str, int]] = {r: {} for r in yelp.ROLES}
    for role, lb, n in j.group_by("role", "label").len().iter_rows():
        if lb is not None:
            out[role][lb] = int(n)
    return out


def test_build_splits_meets_minimums_and_businesses_do_not_cross(photos_df: pl.DataFrame):
    splits = yelp.build_splits(photos_df, None, SIZES, SIZES["seed"])
    assert splits.columns == ["photo_id", "role", "component_id", "eval_balanced"]
    assert splits.height == photos_df.height
    assert set(splits["role"].unique().to_list()) == set(yelp.ROLES)
    counts = _counts(splits, photos_df)
    for role, need in SIZES["minimums"].items():
        for c, n in need.items():
            assert counts[role].get(c, 0) >= n, (role, c, counts[role])
    # work は残余として残っている
    assert counts["work"].get("food", 0) > 0
    # business_id が役割を跨がない（check_no_crossing で検証）
    role_of = dict(zip(splits["photo_id"].to_list(), splits["role"].to_list(), strict=True))
    biz_of = dict(
        zip(photos_df["photo_id"].to_list(), photos_df["business_id"].to_list(), strict=True)
    )
    assert dedup.check_no_crossing(role_of, biz_of) == []
    comp_of = dict(zip(splits["photo_id"].to_list(), splits["component_id"].to_list(), strict=True))
    assert dedup.check_no_crossing(role_of, comp_of) == []
    # 欠損ラベル行も分割に含まれ、その店舗と同じ役割
    assert (
        splits.filter(
            pl.col("photo_id").is_in(
                photos_df.filter(pl.col("label").is_null())["photo_id"].to_list()
            )
        ).height
        > 0
    )


def test_build_splits_is_deterministic_and_seed_sensitive(photos_df: pl.DataFrame):
    a = yelp.build_splits(photos_df, None, SIZES, SIZES["seed"])
    b = yelp.build_splits(
        photos_df.sample(fraction=1.0, shuffle=True, seed=7), None, SIZES, SIZES["seed"]
    )
    assert a.sort("photo_id").equals(b.sort("photo_id"))  # 入力順に依存しない
    c = yelp.build_splits(photos_df, None, SIZES, "another-seed")
    assert not a.sort("photo_id")["role"].equals(c.sort("photo_id")["role"])


def test_build_splits_raises_clear_error_when_minimums_unreachable(photos_df: pl.DataFrame):
    sizes = {**SIZES, "minimums": {**SIZES["minimums"], "tune_fit": {"menu": 10_000}}}
    with pytest.raises(yelp.SplitSizeError) as ex:
        yelp.build_splits(photos_df, None, sizes, SIZES["seed"])
    msg = str(ex.value)
    assert "tune_fit" in msg and "menu" in msg and "不足" in msg


def test_balanced_slice_is_deterministic_and_within_eval(photos_df: pl.DataFrame):
    splits = yelp.build_splits(photos_df, None, SIZES, SIZES["seed"])
    bal = splits.filter(pl.col("eval_balanced"))
    assert set(bal["role"].unique().to_list()) == {"eval"}
    lab = _label_of(photos_df)
    per = {}
    for pid in bal["photo_id"].to_list():
        per[lab[pid]] = per.get(lab[pid], 0) + 1
    assert per == {c: 50 for c in yelp.CLASSES}
    again = yelp.balanced_slice(splits, lab, 50, SIZES["seed"])
    assert again == set(bal["photo_id"].to_list())
    with pytest.raises(yelp.SplitSizeError, match="均等スライス"):
        yelp.balanced_slice(splits, lab, 100_000, SIZES["seed"])


def test_dup_groups_across_businesses_share_role(photos_df: pl.DataFrame):
    """異なる店舗の 2 枚を同じ dup 群にすると、両店舗が同じ役割に入る。"""
    base = yelp.build_splits(photos_df, None, SIZES, SIZES["seed"])
    role_of = dict(zip(base["photo_id"].to_list(), base["role"].to_list(), strict=True))
    by_role: dict[str, str] = {}
    for pid, r in role_of.items():
        by_role.setdefault(r, pid)
    a, b = by_role["tune_fit"], by_role["eval"]
    groups = {a: "gdup", b: "gdup"}
    splits = yelp.build_splits(photos_df, groups, SIZES, SIZES["seed"])
    r2 = dict(zip(splits["photo_id"].to_list(), splits["role"].to_list(), strict=True))
    assert r2[a] == r2[b]
    biz_of = dict(
        zip(photos_df["photo_id"].to_list(), photos_df["business_id"].to_list(), strict=True)
    )
    assert dedup.check_no_crossing(r2, biz_of) == []
    assert dedup.check_no_crossing(r2, groups) == []


def test_compute_phash_shards_is_resumable_and_groups_duplicates(yelp_root: Path, tmp_path: Path):
    df = yelp.load_photos_json(yelp_root / "photos.json").head(40)
    # 完全な複製を 1 枚作る（別 photo_id）
    src = yelp_root / "photos" / f"{df['photo_id'][0]}.jpg"
    dup_id = "p_dup_of_first"
    (yelp_root / "photos" / f"{dup_id}.jpg").write_bytes(src.read_bytes())
    df = pl.concat(
        [
            df,
            pl.DataFrame(
                {"photo_id": [dup_id], "business_id": ["bZ"], "caption": [""], "label": ["food"]}
            ),
        ]
    )
    shard_dir = yelp_root / "_phash"
    h1 = yelp.compute_phash_shards(df, yelp_root, shard_dir, batch_size=16)
    assert len(h1) == 41
    shards = sorted(shard_dir.glob("shard_*.parquet"))
    assert len(shards) == 3
    # 再開: 既存シャードは読み直すだけ（mtime が変わらない）
    mtimes = [s.stat().st_mtime_ns for s in shards]
    h2 = yelp.compute_phash_shards(df, yelp_root, shard_dir, batch_size=16)
    assert h2 == h1
    assert [s.stat().st_mtime_ns for s in shards] == mtimes
    groups = yelp.build_dedup_groups(h1, phash_max=8)
    assert groups[dup_id] == groups[df["photo_id"][0]]


# ---------------------------------------------------------------------------
# eval 凍結とレポート
# ---------------------------------------------------------------------------


def test_ids_sha256_is_order_independent_and_matches_definition():
    import hashlib

    ids = ["b", "a", "c"]
    expect = hashlib.sha256(b"a\nb\nc").hexdigest()
    assert yelp.ids_sha256(ids) == expect
    assert yelp.ids_sha256(reversed(ids)) == expect


def test_freeze_eval_sets_hash_stable(photos_df: pl.DataFrame):
    s1 = yelp.build_splits(photos_df, None, SIZES, SIZES["seed"])
    s2 = yelp.build_splits(photos_df, None, SIZES, SIZES["seed"])
    r1 = yelp.freeze_eval_sets(s1, prefix="test-v1", seed=SIZES["seed"])
    r2 = yelp.freeze_eval_sets(s2, prefix="test-v1", seed=SIZES["seed"])
    assert [r["eval_set_id"] for r in r1] == [
        "test-v1-tune_fit",
        "test-v1-tune_cal",
        "test-v1-work",
        "test-v1-eval",
        "test-v1-eval-balanced",
    ]
    assert [r["photo_ids_hash"] for r in r1] == [r["photo_ids_hash"] for r in r2]
    assert len({r["photo_ids_hash"] for r in r1}) == 5
    ev = next(r for r in r1 if r["eval_set_id"] == "test-v1-eval")
    assert ev["photo_ids_hash"] == yelp.ids_sha256(
        s1.filter(pl.col("role") == "eval")["photo_id"].to_list()
    )
    assert ev["role"] == "eval" and "natural" in ev["definition"]
    # スキーマ準拠（eval_sets に書ける）
    tables._validate_and_cast("eval_sets", pl.DataFrame(r1))


def test_m0_report_renders_required_sections(photos_df: pl.DataFrame, tmp_path: Path):
    splits = yelp.build_splits(photos_df, None, SIZES, SIZES["seed"])
    rows = yelp.freeze_eval_sets(splits, prefix="test-v1", seed=SIZES["seed"])
    stats = {
        "layout": {
            "root": "/x",
            "photos_json": "/x/photos.json",
            "n_image_files": 40,
            "terms_pdfs": ["Dataset_User_Agreement.pdf"],
            "download_record_sha256": "abc",
            "download_record_fields": {"terms_version": "2024-01", "downloaded_at": "2026-09-07"},
            "ok": True,
            "problems": [],
        },
        "n_photos": photos_df.height,
        "n_businesses": N_BIZ,
        "class_counts": yelp.class_counts(photos_df).to_dicts(),
        "missing_label_rate": yelp.missing_label_rate(photos_df),
        "split_config": SIZES,
        "split_summary": yelp.split_summary(splits, photos_df).to_dicts(),
        "crossing": [],
        "eval_sets": rows,
        "stamps": {
            "run_id": "r1",
            "git_sha": "abc123",
            "taxonomy_version": "0.1.0",
            "rulebook_version": "未導入",
            "eval_set_id": "test-v1-eval",
            "wandb_url": None,
        },
    }
    out = tmp_path / "reports" / "m0_setup.md"
    text = yelp.m0_report(stats, splits, out)
    assert out.read_text(encoding="utf-8") == text
    assert text.startswith("# M0 セットアップ")
    assert "**結論**" in text
    for needle in [
        "## 1. 配置と規約",
        "## 2. データ統計",
        "## 3. 分割の定義",
        "### 3.2 非跨ぎ検証",
        "合格",
        "## 4. eval の定義と凍結",
        "test-v1-eval-balanced",
        "## フッタ",
        "| rulebook_version | 未導入 |",
        "| taxonomy_version | 0.1.0 |",
        "| 概算 API コスト |",
        "| Yelp 規約の版 | 2024-01 |",
        "| Yelp DL 日 | 2026-09-07 |",
    ]:
        assert needle in text, needle
    # photo_id を載せない
    assert "p000000" not in text


# ---------------------------------------------------------------------------
# CLI（end-to-end、tracking なし）
# ---------------------------------------------------------------------------


def test_cli_verify_yelp_and_build_splits(
    yelp_root: Path,
    m0_config: Path,
    layers: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    runner = CliRunner()
    stats_out = tmp_path / "stats.json"
    res = runner.invoke(
        m0_setup.app,
        [
            "verify-yelp",
            "--config",
            str(m0_config),
            "--root",
            str(yelp_root),
            "--out",
            str(stats_out),
            "--strict",
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(stats_out.read_text(encoding="utf-8"))
    assert data["layout"]["ok"]
    assert data["n_photos"] == N_ROWS
    assert (
        data["n_businesses"]
        == yelp.load_photos_json(yelp_root / "photos.json")["business_id"].n_unique()
    )
    assert data["image_stats"]["sample"] == {"n": 30, "seed": 1}
    assert data["image_stats"]["all"]["n"] == 30
    assert "photo_id" not in json.dumps(data["class_counts"])

    # 既定の stats パス（marts）へも書いてから build-splits
    (layers["marts"]).mkdir(parents=True, exist_ok=True)
    (layers["marts"] / "yelp_stats.json").write_text(
        stats_out.read_text(encoding="utf-8"), encoding="utf-8"
    )
    report = tmp_path / "reports" / "m0_setup.md"
    res = runner.invoke(
        m0_setup.app,
        [
            "build-splits",
            "--config",
            str(m0_config),
            "--root",
            str(yelp_root),
            "--report",
            str(report),
        ],
    )
    assert res.exit_code == 0, res.output
    out = json.loads(res.output)
    assert out["crossing"] == []
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "長辺 中央値" in text and "| 64 |" in text  # 標本統計が転記されている
    assert f"| run_id | {out['run_id']} |" in text

    labels = tables.read_table("labels", "core")
    assert labels.height == N_ROWS and labels["label_source"].unique().to_list() == ["yelp"]
    eval_sets = tables.read_table("eval_sets", "core")
    assert eval_sets.height == 5
    splits = pl.read_parquet(layers["marts"] / "splits.parquet")
    assert splits.columns == ["photo_id", "role", "component_id", "eval_balanced"]
    ev_hash = yelp.ids_sha256(splits.filter(pl.col("role") == "eval")["photo_id"].to_list())
    assert eval_sets.filter(pl.col("eval_set_id") == "test-v1-eval")["photo_ids_hash"][0] == ev_hash
    runs = tables.read_table("runs", "marts")
    assert runs.height == 1 and runs["eval_set_id"][0] == "test-v1-eval"
    assert runs["rulebook_version"][0] == "未導入"


def test_cli_build_splits_exits_2_on_split_size_error(
    yelp_root: Path, m0_config: Path, layers: dict[str, Path], tmp_path: Path
):
    cfg = yaml.safe_load(m0_config.read_text(encoding="utf-8"))
    cfg["splits"]["minimums"]["eval"]["menu"] = 10_000
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    res = CliRunner().invoke(
        m0_setup.app, ["build-splits", "--config", str(bad), "--root", str(yelp_root)]
    )
    assert res.exit_code == 2
    assert "SplitSizeError" in res.output


def test_check_env_still_works():
    res = CliRunner().invoke(m0_setup.app, ["check-env", "--skip-model"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["model"] == {"skipped": "--skip-model"}


def test_real_m0_config_is_pinned_and_consistent():
    from cascade.config import load_config

    cfg = load_config(paths.REPO_ROOT / "configs" / "m0.yaml")
    s = cfg["splits"]
    assert set(s["ratios"]) == set(yelp.ROLES)
    assert s["minimums"]["tune_fit"]["food"] >= 2000
    assert all(s["minimums"]["tune_fit"][c] >= 500 for c in ("drink", "menu", "inside", "outside"))
    assert s["minimums"]["tune_cal"]["food"] >= 3000
    assert s["minimums"]["eval"]["food"] >= 3000
    assert s["eval_balanced_per_class"] == 400
    assert all(s["minimums"]["eval"][c] >= 400 for c in ("drink", "menu", "inside", "outside"))
    assert cfg["rulebook_version"] == "未導入"
