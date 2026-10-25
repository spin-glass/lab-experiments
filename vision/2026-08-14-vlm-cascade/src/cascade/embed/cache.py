"""埋め込みキャッシュ: npy シャード＋parquet index（計画 Phase B-a「npy シャード＋parquet index、再開可能」）。

配置（``paths.EMB`` 直下）::

    <root>/<model_key>/<dataset_id>/shard-00000.npy
    <root>/<model_key>/<dataset_id>/index.parquet
    <root>/<model_key>/<dataset_id>/manifest.json

- index 列: embedding_id, photo_id, dataset_id, model_key, shard_path, row_idx, sha256, ts
- manifest: model_key / dataset_id / dim / shard_size / n_items / ``ids_sha1``（全 photo_id 列の sha1）と、
  シャードごとの ``ids_sha1``（そのシャードの photo_id 列）・``rows_sha1``（``"{photo_id}|{sha256}"`` 列の sha1。
  画像バイトが変われば変わる）。再開時は npy が存在し ids_sha1 と rows_sha1 が両方一致するシャードだけ飛ばす
  （同 photo_id でもバイトが変わっていれば再エンコード）
- ``get()`` / ``exists()`` は整合性を検証する: シャードごとの ids_sha1、``row_idx == arange(n)``、
  manifest と index のシャード集合、行数。``items`` を渡せば全体 ``ids_sha1`` も突き合わせる
- 完了時に manifest に無い ``shard-*.npy`` と ``*.tmp`` を削除する（items が減った場合の残骸）
- ``model_key`` 不一致（キャッシュ vs エンコーダ、キャッシュ vs manifest）は必ず raise する
- 入力画像は ``ImageOps.exif_transpose`` で EXIF 向きを画素に反映してから渡す
  （``preprocess.normalize_image`` 済み画像と向きの扱いを揃える）
- 画像・生データはここに置かない。npy は数値ベクトルのみ（CLAUDE.md 絶対制約）
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from PIL import Image, ImageOps

from cascade import paths
from cascade.embed.encoder import Encoder

INDEX_COLUMNS = (
    "embedding_id",
    "photo_id",
    "dataset_id",
    "model_key",
    "shard_path",
    "row_idx",
    "sha256",
    "ts",
)
INDEX_SCHEMA = {
    "embedding_id": pl.Utf8,
    "photo_id": pl.Utf8,
    "dataset_id": pl.Utf8,
    "model_key": pl.Utf8,
    "shard_path": pl.Utf8,
    "row_idx": pl.Int64,
    "sha256": pl.Utf8,
    "ts": pl.Utf8,
}


class ModelKeyMismatch(ValueError):
    """キャッシュの model_key とエンコーダ／manifest の model_key が一致しない。"""


class CacheIntegrityError(RuntimeError):
    """manifest / index / シャードの整合性が取れていない（改変・破損・items 不一致）。"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ids_sha1(ids: Sequence[str]) -> str:
    return hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest()


def _rows_sha1(ids: Sequence[str], sha256s: Sequence[str]) -> str:
    """photo_id と画像 sha256 の対の sha1。画像バイトが変われば変わる。"""
    return hashlib.sha1(
        "\n".join(f"{pid}|{s}" for pid, s in zip(ids, sha256s, strict=True)).encode("utf-8")
    ).hexdigest()


def embedding_id(model_key: str, dataset_id: str, photo_id: str) -> str:
    return hashlib.sha1(f"{model_key}|{dataset_id}|{photo_id}".encode()).hexdigest()[:16]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _load_image(path: Path) -> Image.Image:
    """EXIF 向きを画素に反映し RGB に変換する（preprocess.normalize_image と同じ向きの扱い）。"""
    with Image.open(path) as im:
        oriented = ImageOps.exif_transpose(im)
        assert oriented is not None
        return oriented.convert("RGB")


class EmbeddingCache:
    def __init__(self, model_key: str, root: Path = paths.EMB) -> None:
        self.model_key = model_key
        self.root = Path(root)

    # --- パス -----------------------------------------------------------------
    def dataset_dir(self, dataset_id: str) -> Path:
        return self.root / self.model_key / dataset_id

    def manifest_path(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / "manifest.json"

    def index_path(self, dataset_id: str) -> Path:
        return self.dataset_dir(dataset_id) / "index.parquet"

    @staticmethod
    def shard_name(i: int) -> str:
        return f"shard-{i:05d}.npy"

    def _read_manifest(self, dataset_id: str) -> dict[str, Any] | None:
        p = self.manifest_path(dataset_id)
        if not p.exists():
            return None
        m = json.loads(p.read_text(encoding="utf-8"))
        if m.get("model_key") != self.model_key:
            raise ModelKeyMismatch(
                f"manifest の model_key={m.get('model_key')!r} がキャッシュの {self.model_key!r} と異なる: {p}"
            )
        return m

    def _write_manifest(self, dataset_id: str, m: dict[str, Any]) -> None:
        p = self.manifest_path(dataset_id)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(p)

    def _cleanup_stale(self, dataset_id: str, keep: set[str]) -> list[str]:
        """manifest に無い shard-*.npy と *.tmp を削除し、削除したファイル名を返す。"""
        ddir = self.dataset_dir(dataset_id)
        removed: list[str] = []
        for p in sorted(ddir.glob("shard-*.npy")):
            if p.name not in keep:
                p.unlink()
                removed.append(p.name)
        for p in sorted(ddir.glob("*.tmp")):
            p.unlink()
            removed.append(p.name)
        return removed

    # --- 書き込み -------------------------------------------------------------
    def encode_dataset(
        self,
        dataset_id: str,
        items: Sequence[tuple[str, Path]],
        encoder: Encoder,
        batch_size: int = 64,
        shard_size: int = 2048,
        resume: bool = True,
    ) -> np.ndarray:
        """items=(photo_id, path) を順に埋め込み、シャードと index を書く。戻り値は全行の [n, dim]。"""
        if encoder.model_key != self.model_key:
            raise ModelKeyMismatch(
                f"encoder.model_key={encoder.model_key!r} がキャッシュの {self.model_key!r} と異なる"
            )
        if shard_size <= 0 or batch_size <= 0:
            raise ValueError("shard_size / batch_size は正の整数")
        ids = [pid for pid, _ in items]
        if len(set(ids)) != len(ids):
            raise ValueError("photo_id が重複している")

        ddir = self.dataset_dir(dataset_id)
        ddir.mkdir(parents=True, exist_ok=True)
        prev = self._read_manifest(dataset_id) if resume else None
        prev_shards: dict[str, dict[str, Any]] = {
            s["path"]: s for s in (prev or {}).get("shards", [])
        }

        manifest: dict[str, Any] = {
            "model_key": self.model_key,
            "dataset_id": dataset_id,
            "encoder_name": getattr(encoder, "name", ""),
            "dim": None,
            "shard_size": shard_size,
            "n_items": len(items),
            "ids_sha1": _ids_sha1(ids),
            "shards": [],
            "complete": False,
            "updated_at": _now(),
        }
        rows: list[dict[str, Any]] = []
        outs: list[np.ndarray] = []
        n_shards = (len(items) + shard_size - 1) // shard_size
        for si in range(n_shards):
            chunk = list(items[si * shard_size : (si + 1) * shard_size])
            chunk_ids = [pid for pid, _ in chunk]
            chunk_sha = [sha256_file(Path(p)) for _, p in chunk]
            sname = self.shard_name(si)
            spath = ddir / sname
            ids_sha = _ids_sha1(chunk_ids)
            rows_sha = _rows_sha1(chunk_ids, chunk_sha)
            ps = prev_shards.get(sname)
            reused = (
                resume
                and ps is not None
                and ps.get("ids_sha1") == ids_sha
                and ps.get("rows_sha1") == rows_sha
                and spath.exists()
            )
            if reused:
                arr = np.load(spath)
                if arr.shape[0] != len(chunk):
                    reused = False
            if not reused:
                vecs: list[np.ndarray] = []
                for b in range(0, len(chunk), batch_size):
                    ims = [_load_image(Path(p)) for _, p in chunk[b : b + batch_size]]
                    vecs.append(np.asarray(encoder.encode_images(ims), dtype=np.float32))
                arr = np.concatenate(vecs, axis=0).astype(np.float32)
                tmp = spath.with_suffix(".npy.tmp")
                with tmp.open("wb") as f:  # ファイルオブジェクト渡しで ".npy" 付加を避ける
                    np.save(f, arr)
                tmp.replace(spath)
            if manifest["dim"] is None:
                manifest["dim"] = int(arr.shape[1])
            ts = _now()
            for ri, ((pid, _), sha) in enumerate(zip(chunk, chunk_sha, strict=True)):
                rows.append(
                    {
                        "embedding_id": embedding_id(self.model_key, dataset_id, pid),
                        "photo_id": pid,
                        "dataset_id": dataset_id,
                        "model_key": self.model_key,
                        "shard_path": sname,
                        "row_idx": ri,
                        "sha256": sha,
                        "ts": ts,
                    }
                )
            outs.append(arr)
            manifest["shards"].append(
                {
                    "path": sname,
                    "n": len(chunk),
                    "ids_sha1": ids_sha,
                    "rows_sha1": rows_sha,
                    "reused": bool(reused),
                }
            )
            manifest["updated_at"] = _now()
            # シャードごとに index / manifest を書き、途中停止でも再開できるようにする
            self._write_index(dataset_id, rows)
            self._write_manifest(dataset_id, manifest)

        if n_shards == 0:
            manifest["dim"] = int(encoder.dim)
            self._write_index(dataset_id, rows)
        manifest["complete"] = True
        self._write_manifest(dataset_id, manifest)
        self._cleanup_stale(dataset_id, {s["path"] for s in manifest["shards"]})
        if not outs:
            return np.zeros((0, int(manifest["dim"])), dtype=np.float32)
        return np.concatenate(outs, axis=0)

    def _write_index(self, dataset_id: str, rows: list[dict[str, Any]]) -> None:
        df = pl.DataFrame(rows, schema=INDEX_SCHEMA) if rows else pl.DataFrame(schema=INDEX_SCHEMA)
        p = self.index_path(dataset_id)
        tmp = p.with_suffix(".parquet.tmp")
        df.select(list(INDEX_COLUMNS)).write_parquet(tmp)
        tmp.replace(p)

    # --- 読み出し -------------------------------------------------------------
    @staticmethod
    def _check_items(m: dict[str, Any], items: Sequence[tuple[str, Path]] | Sequence[str]) -> None:
        ids = [it[0] if isinstance(it, tuple) else str(it) for it in items]
        if m.get("ids_sha1") != _ids_sha1(ids) or int(m.get("n_items", -1)) != len(ids):
            raise CacheIntegrityError(
                f"キャッシュの ids_sha1/n_items が渡された items（{len(ids)} 件）と一致しない: {m.get('dataset_id')}"
            )

    def exists(
        self, dataset_id: str, items: Sequence[tuple[str, Path]] | Sequence[str] | None = None
    ) -> bool:
        """完了済みキャッシュがあるか。``items`` を渡すと、その photo_id 列に対する完成品かも検証する。"""
        m = self._read_manifest(dataset_id)
        if not (m and m.get("complete")):
            return False
        if items is not None:
            try:
                self._check_items(m, items)
            except CacheIntegrityError:
                return False
        return True

    def index(self, dataset_id: str) -> pl.DataFrame:
        p = self.index_path(dataset_id)
        if not p.exists():
            raise FileNotFoundError(p)
        df = pl.read_parquet(p)
        bad = df.filter(pl.col("model_key") != self.model_key)
        if bad.height:
            raise ModelKeyMismatch(f"index に model_key 不一致の行がある: {p}")
        return df.sort(["shard_path", "row_idx"])

    def _verify(self, m: dict[str, Any], df: pl.DataFrame) -> None:
        """manifest と index の整合性（シャード集合・ids_sha1・row_idx 連続性・行数）。"""
        shards = list(m.get("shards", []))
        want = [s["path"] for s in shards]
        got = df["shard_path"].unique(maintain_order=True).to_list() if df.height else []
        if sorted(got) != sorted(want):
            raise CacheIntegrityError(f"index のシャード集合 {got} が manifest {want} と一致しない")
        if int(m.get("n_items", df.height)) != df.height:
            raise CacheIntegrityError(
                f"manifest n_items={m.get('n_items')} と index 行数 {df.height} が不一致"
            )
        if df.height and m.get("ids_sha1") != _ids_sha1(df["photo_id"].to_list()):
            raise CacheIntegrityError("index の photo_id 列が manifest の ids_sha1 と一致しない")
        for s in shards:
            part = df.filter(pl.col("shard_path") == s["path"]).sort("row_idx")
            if part.height != int(s["n"]):
                raise CacheIntegrityError(
                    f"{s['path']}: index 行数 {part.height} != manifest n={s['n']}"
                )
            if part["row_idx"].to_list() != list(range(part.height)):
                raise CacheIntegrityError(f"{s['path']}: row_idx が 0..n-1 の連続でない")
            if _ids_sha1(part["photo_id"].to_list()) != s.get("ids_sha1"):
                raise CacheIntegrityError(
                    f"{s['path']}: index の photo_id 列が manifest の ids_sha1 と一致しない"
                )
            if (
                "rows_sha1" in s
                and _rows_sha1(part["photo_id"].to_list(), part["sha256"].to_list())
                != s["rows_sha1"]
            ):
                raise CacheIntegrityError(
                    f"{s['path']}: index の sha256 列が manifest の rows_sha1 と一致しない"
                )

    def get(
        self, dataset_id: str, items: Sequence[tuple[str, Path]] | Sequence[str] | None = None
    ) -> tuple[np.ndarray, pl.DataFrame]:
        """index 順にシャードを連結して返す。行 i が index の行 i に対応する。

        manifest / index / シャードの整合性を検証し、不一致は CacheIntegrityError。
        ``items`` を渡すと、その photo_id 列に対する完成品であることも検証する。
        """
        m = self._read_manifest(dataset_id)
        if m is None:
            raise FileNotFoundError(self.manifest_path(dataset_id))
        if not m.get("complete"):
            raise RuntimeError(
                f"キャッシュが未完了（resume=True で encode_dataset を再実行）: {dataset_id}"
            )
        if items is not None:
            self._check_items(m, items)
        df = self.index(dataset_id)
        self._verify(m, df)
        ddir = self.dataset_dir(dataset_id)
        arrs = []
        for s in m["shards"]:
            a = np.load(ddir / s["path"])
            if a.shape[0] != int(s["n"]):
                raise CacheIntegrityError(
                    f"{s['path']}: npy 行数 {a.shape[0]} != manifest n={s['n']}"
                )
            arrs.append(a)
        emb = (
            np.concatenate(arrs, axis=0) if arrs else np.zeros((0, int(m["dim"])), dtype=np.float32)
        )
        if emb.shape[0] != df.height:
            raise CacheIntegrityError(
                f"シャード行数 {emb.shape[0]} と index 行数 {df.height} が不一致"
            )
        return emb.astype(np.float32, copy=False), df
