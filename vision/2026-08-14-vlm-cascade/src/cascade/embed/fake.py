"""テスト用ダミーエンコーダ（ネットワーク・モデル不要）。

画像はバイト列の sha256、テキストは UTF-8 の sha256 から決定的に dim=8 のベクトルを作る。
``cascade.embed.encoder.Encoder`` 契約を満たすので、他レーンのテスト（cache / zeroshot / gate）で共有する。
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence

import numpy as np
from PIL import Image


def _vec_from_digest(digest: bytes, dim: int) -> np.ndarray:
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


class FakeEncoder:
    """決定的ダミー。``text_vectors`` を渡すと特定テキストの埋め込みを固定できる。"""

    def __init__(
        self,
        name: str = "fake",
        dim: int = 8,
        model_key: str = "fake00000000",
        device: str = "cpu",
        text_vectors: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.name = name
        self._dim = dim
        self.model_key = model_key
        self.device = device
        self.text_vectors = dict(text_vectors or {})
        self.n_image_calls = 0
        self.n_images_encoded = 0

    @property
    def dim(self) -> int:
        return self._dim

    @staticmethod
    def image_bytes(im: Image.Image) -> bytes:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        self.n_image_calls += 1
        self.n_images_encoded += len(images)
        if len(images) == 0:
            return np.zeros((0, self._dim), dtype=np.float32)
        rows = [
            _vec_from_digest(hashlib.sha256(self.image_bytes(im)).digest(), self._dim)
            for im in images
        ]
        return np.stack(rows).astype(np.float32)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self._dim), dtype=np.float32)
        rows = []
        for t in texts:
            if t in self.text_vectors:
                rows.append(np.asarray(self.text_vectors[t], dtype=np.float32))
            else:
                rows.append(_vec_from_digest(hashlib.sha256(t.encode("utf-8")).digest(), self._dim))
        return np.stack(rows).astype(np.float32)
