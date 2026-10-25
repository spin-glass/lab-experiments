"""EmbeddingCache のオフラインテスト（FakeEncoder。ネットワーク・モデル不要）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from PIL import Image

from cascade.embed.cache import (
    INDEX_COLUMNS,
    CacheIntegrityError,
    EmbeddingCache,
    ModelKeyMismatch,
    _load_image,
    sha256_file,
)
from cascade.embed.fake import FakeEncoder


def _make_images(tmp: Path, n: int) -> list[tuple[str, Path]]:
    rng = np.random.default_rng(0)
    items = []
    for i in range(n):
        p = tmp / f"img_{i:03d}.png"
        Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(p)
        items.append((f"p{i:03d}", p))
    return items


@pytest.fixture
def items(tmp_path: Path) -> list[tuple[str, Path]]:
    d = tmp_path / "imgs"
    d.mkdir()
    return _make_images(d, 10)


def test_round_trip_shards_and_index(tmp_path: Path, items: list[tuple[str, Path]]) -> None:
    enc = FakeEncoder(model_key="abc123abc123")
    cache = EmbeddingCache(model_key=enc.model_key, root=tmp_path / "emb")
    emb = cache.encode_dataset("ds1", items, enc, batch_size=3, shard_size=4)
    assert emb.shape == (10, 8) and emb.dtype == np.float32

    ddir = tmp_path / "emb" / enc.model_key / "ds1"
    shards = sorted(p.name for p in ddir.glob("shard-*.npy"))
    assert shards == ["shard-00000.npy", "shard-00001.npy", "shard-00002.npy"]
    assert not list(ddir.glob("*.tmp"))

    got, idx = cache.get("ds1")
    np.testing.assert_allclose(got, emb)
    assert tuple(idx.columns) == INDEX_COLUMNS
    assert idx["photo_id"].to_list() == [pid for pid, _ in items]
    assert idx["shard_path"].to_list()[:5] == ["shard-00000.npy"] * 4 + ["shard-00001.npy"]
    assert idx["row_idx"].to_list()[:5] == [0, 1, 2, 3, 0]
    assert idx["sha256"][0] == sha256_file(items[0][1])
    assert idx["model_key"].unique().to_list() == [enc.model_key]
    assert idx["dataset_id"].unique().to_list() == ["ds1"]
    assert idx["embedding_id"].n_unique() == 10

    # 埋め込みは決定的で、直接エンコードした値と一致する
    direct = enc.encode_images([Image.open(p) for _, p in items])
    np.testing.assert_allclose(got, direct)

    m = json.loads((ddir / "manifest.json").read_text())
    assert m["complete"] and m["dim"] == 8 and len(m["shards"]) == 3


def test_resume_skips_completed_shards(tmp_path: Path, items: list[tuple[str, Path]]) -> None:
    enc = FakeEncoder(model_key="abc123abc123")
    cache = EmbeddingCache(model_key=enc.model_key, root=tmp_path / "emb")
    first = cache.encode_dataset("ds1", items, enc, batch_size=4, shard_size=4)
    n_first = enc.n_images_encoded
    assert n_first == 10

    # 最後のシャードを消して「途中停止」を模す → 再開で 2 枚だけ再エンコード
    ddir = tmp_path / "emb" / enc.model_key / "ds1"
    (ddir / "shard-00002.npy").unlink()
    second = cache.encode_dataset("ds1", items, enc, batch_size=4, shard_size=4, resume=True)
    assert enc.n_images_encoded - n_first == 2
    np.testing.assert_allclose(first, second)
    m = json.loads((ddir / "manifest.json").read_text())
    assert [s["reused"] for s in m["shards"]] == [True, True, False]

    # 完了済みなら再実行でエンコード 0 枚
    n = enc.n_images_encoded
    cache.encode_dataset("ds1", items, enc, batch_size=4, shard_size=4)
    assert enc.n_images_encoded == n

    # resume=False は全部やり直す
    cache.encode_dataset("ds1", items, enc, batch_size=4, shard_size=4, resume=False)
    assert enc.n_images_encoded == n + 10


def test_resume_reencodes_when_items_change(tmp_path: Path, items: list[tuple[str, Path]]) -> None:
    enc = FakeEncoder(model_key="abc123abc123")
    cache = EmbeddingCache(model_key=enc.model_key, root=tmp_path / "emb")
    cache.encode_dataset("ds1", items, enc, shard_size=4)
    n = enc.n_images_encoded
    # 先頭シャードの並びを変える → そのシャードだけ再エンコード
    changed = [items[1], items[0], *items[2:]]
    out = cache.encode_dataset("ds1", changed, enc, shard_size=4)
    assert enc.n_images_encoded - n == 4
    got, idx = cache.get("ds1")
    assert idx["photo_id"].to_list()[:2] == ["p001", "p000"]
    np.testing.assert_allclose(got, out)


def test_model_key_mismatch_raises(tmp_path: Path, items: list[tuple[str, Path]]) -> None:
    enc = FakeEncoder(model_key="abc123abc123")
    root = tmp_path / "emb"
    cache = EmbeddingCache(model_key=enc.model_key, root=root)
    cache.encode_dataset("ds1", items, enc, shard_size=4)

    other = EmbeddingCache(model_key="ffffffffffff", root=root)
    with pytest.raises(ModelKeyMismatch):
        other.encode_dataset("ds1", items, enc)

    # manifest を改ざん → get で検出
    mp = root / enc.model_key / "ds1" / "manifest.json"
    m = json.loads(mp.read_text())
    m["model_key"] = "deadbeef0000"
    mp.write_text(json.dumps(m))
    with pytest.raises(ModelKeyMismatch):
        cache.get("ds1")


def test_index_model_key_mismatch_raises(tmp_path: Path, items: list[tuple[str, Path]]) -> None:
    enc = FakeEncoder(model_key="abc123abc123")
    root = tmp_path / "emb"
    cache = EmbeddingCache(model_key=enc.model_key, root=root)
    cache.encode_dataset("ds1", items, enc, shard_size=4)
    ip = root / enc.model_key / "ds1" / "index.parquet"
    df = pl.read_parquet(ip).with_columns(pl.lit("deadbeef0000").alias("model_key"))
    df.write_parquet(ip)
    with pytest.raises(ModelKeyMismatch):
        cache.get("ds1")


def test_incomplete_cache_and_missing_raise(tmp_path: Path, items: list[tuple[str, Path]]) -> None:
    enc = FakeEncoder(model_key="abc123abc123")
    root = tmp_path / "emb"
    cache = EmbeddingCache(model_key=enc.model_key, root=root)
    with pytest.raises(FileNotFoundError):
        cache.get("nope")
    cache.encode_dataset("ds1", items, enc, shard_size=4)
    mp = root / enc.model_key / "ds1" / "manifest.json"
    m = json.loads(mp.read_text())
    m["complete"] = False
    mp.write_text(json.dumps(m))
    assert not cache.exists("ds1")
    with pytest.raises(RuntimeError):
        cache.get("ds1")


def test_duplicate_photo_ids_rejected(tmp_path: Path, items: list[tuple[str, Path]]) -> None:
    enc = FakeEncoder(model_key="abc123abc123")
    cache = EmbeddingCache(model_key=enc.model_key, root=tmp_path / "emb")
    with pytest.raises(ValueError):
        cache.encode_dataset("ds1", [items[0], items[0]], enc)


def test_empty_dataset(tmp_path: Path) -> None:
    enc = FakeEncoder(model_key="abc123abc123")
    cache = EmbeddingCache(model_key=enc.model_key, root=tmp_path / "emb")
    emb = cache.encode_dataset("empty", [], enc)
    assert emb.shape == (0, 8)
    got, idx = cache.get("empty")
    assert got.shape == (0, 8) and idx.height == 0


# --- レビュー指摘の回帰テスト（cache.py:168 / :244 / :230 / :78） -------------------------------


def _rewrite_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(path)


def test_resume_reencodes_when_bytes_change(tmp_path: Path, items: list[tuple[str, Path]]) -> None:
    """同 photo_id・別バイト → そのシャードだけ再エンコードされ、index の sha256 も更新される。"""
    enc = FakeEncoder(model_key="abc123abc123")
    cache = EmbeddingCache(model_key=enc.model_key, root=tmp_path / "emb")
    first = cache.encode_dataset("ds1", items, enc, shard_size=4)
    old_sha = sha256_file(items[5][1])
    ddir = tmp_path / "emb" / enc.model_key / "ds1"
    m0 = json.loads((ddir / "manifest.json").read_text())
    assert all("rows_sha1" in s for s in m0["shards"])

    _rewrite_image(items[5][1], seed=12345)  # shard-00001 の 2 番目
    assert sha256_file(items[5][1]) != old_sha
    n = enc.n_images_encoded
    second = cache.encode_dataset("ds1", items, enc, shard_size=4, resume=True)
    assert enc.n_images_encoded - n == 4  # shard-00001 のみ
    assert not np.allclose(first[5], second[5])
    np.testing.assert_allclose(first[:4], second[:4])
    got, idx = cache.get("ds1")
    np.testing.assert_allclose(got, second)
    assert idx["sha256"][5] == sha256_file(items[5][1])
    m1 = json.loads((ddir / "manifest.json").read_text())
    assert [s["reused"] for s in m1["shards"]] == [True, False, True]
    assert m1["shards"][1]["rows_sha1"] != m0["shards"][1]["rows_sha1"]
    assert m1["shards"][0]["rows_sha1"] == m0["shards"][0]["rows_sha1"]


def test_manifest_without_rows_sha1_is_not_reused(
    tmp_path: Path, items: list[tuple[str, Path]]
) -> None:
    enc = FakeEncoder(model_key="abc123abc123")
    cache = EmbeddingCache(model_key=enc.model_key, root=tmp_path / "emb")
    cache.encode_dataset("ds1", items, enc, shard_size=4)
    mp = tmp_path / "emb" / enc.model_key / "ds1" / "manifest.json"
    m = json.loads(mp.read_text())
    for s in m["shards"]:
        del s["rows_sha1"]
    mp.write_text(json.dumps(m))
    n = enc.n_images_encoded
    cache.encode_dataset("ds1", items, enc, shard_size=4)
    assert enc.n_images_encoded - n == 10


def _tamper_index(root: Path, model_key: str, fn) -> None:
    ip = root / model_key / "ds1" / "index.parquet"
    fn(pl.read_parquet(ip)).write_parquet(ip)


def test_get_detects_tampered_index_and_manifest(
    tmp_path: Path, items: list[tuple[str, Path]]
) -> None:
    enc = FakeEncoder(model_key="abc123abc123")
    root = tmp_path / "emb"
    cache = EmbeddingCache(model_key=enc.model_key, root=root)
    cache.encode_dataset("ds1", items, enc, shard_size=4)
    ip = root / enc.model_key / "ds1" / "index.parquet"
    pristine = pl.read_parquet(ip)

    # photo_id をシャード内で入れ替え（行数は同じ）→ shard ids_sha1 不一致
    def swap_ids(df: pl.DataFrame) -> pl.DataFrame:
        ids = df["photo_id"].to_list()
        ids[0], ids[1] = ids[1], ids[0]
        return df.with_columns(pl.Series("photo_id", ids))

    _tamper_index(root, enc.model_key, swap_ids)
    with pytest.raises(CacheIntegrityError):
        cache.get("ds1")
    pristine.write_parquet(ip)

    # row_idx を重複させる → 連続性違反
    def dup_row_idx(df: pl.DataFrame) -> pl.DataFrame:
        r = df["row_idx"].to_list()
        r[1] = 0
        return df.with_columns(pl.Series("row_idx", r, dtype=pl.Int64))

    _tamper_index(root, enc.model_key, dup_row_idx)
    with pytest.raises(CacheIntegrityError):
        cache.get("ds1")
    pristine.write_parquet(ip)

    # sha256 列の改変 → rows_sha1 不一致
    def alter_sha(df: pl.DataFrame) -> pl.DataFrame:
        s = df["sha256"].to_list()
        s[3] = "0" * 64
        return df.with_columns(pl.Series("sha256", s))

    _tamper_index(root, enc.model_key, alter_sha)
    with pytest.raises(CacheIntegrityError):
        cache.get("ds1")
    pristine.write_parquet(ip)

    # index の行を落とす → 行数・シャード集合不一致
    _tamper_index(root, enc.model_key, lambda df: df.head(8))
    with pytest.raises(CacheIntegrityError):
        cache.get("ds1")
    pristine.write_parquet(ip)

    # manifest の shard ids_sha1 改竄
    mp = root / enc.model_key / "ds1" / "manifest.json"
    m = json.loads(mp.read_text())
    m["shards"][2]["ids_sha1"] = "f" * 40
    mp.write_text(json.dumps(m))
    with pytest.raises(CacheIntegrityError):
        cache.get("ds1")


def test_exists_and_get_verify_items(tmp_path: Path, items: list[tuple[str, Path]]) -> None:
    enc = FakeEncoder(model_key="abc123abc123")
    cache = EmbeddingCache(model_key=enc.model_key, root=tmp_path / "emb")
    assert not cache.exists("ds1", items)
    cache.encode_dataset("ds1", items, enc, shard_size=4)
    assert cache.exists("ds1")
    assert cache.exists("ds1", items)
    assert cache.exists("ds1", [pid for pid, _ in items])  # photo_id 列だけでも可
    assert not cache.exists("ds1", items[:5])
    assert not cache.exists("ds1", list(reversed(items)))
    got, idx = cache.get("ds1", items)
    assert got.shape == (10, 8) and idx.height == 10
    with pytest.raises(CacheIntegrityError):
        cache.get("ds1", items[:5])


def test_stale_shards_removed_when_items_shrink(
    tmp_path: Path, items: list[tuple[str, Path]]
) -> None:
    enc = FakeEncoder(model_key="abc123abc123")
    cache = EmbeddingCache(model_key=enc.model_key, root=tmp_path / "emb")
    cache.encode_dataset("ds1", items, enc, shard_size=4)
    ddir = tmp_path / "emb" / enc.model_key / "ds1"
    assert (ddir / "shard-00002.npy").exists()
    (ddir / "shard-00001.npy.tmp").write_bytes(b"junk")
    out = cache.encode_dataset("ds1", items[:5], enc, shard_size=4)
    assert sorted(p.name for p in ddir.glob("shard-*")) == ["shard-00000.npy", "shard-00001.npy"]
    assert not list(ddir.glob("*.tmp"))
    got, idx = cache.get("ds1", items[:5])
    np.testing.assert_allclose(got, out)
    assert idx.height == 5


def test_load_image_applies_exif_transpose(make_image) -> None:
    # Orientation=6（時計回り 90° 回転が必要）: 256x192 を保存 → 読み込み後は 192x256
    p = make_image("rot.jpg", size=(256, 192), exif={0x0112: 6})
    with Image.open(p) as raw:
        assert raw.size == (256, 192)
    im = _load_image(p)
    assert im.size == (192, 256) and im.mode == "RGB"
