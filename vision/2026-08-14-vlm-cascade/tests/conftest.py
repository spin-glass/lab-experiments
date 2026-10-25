"""共有フィクスチャ: 小さな決定的画像を書く工場、近重複・遠隔画像の生成。ネットワーク・モデル DL なし。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _pattern(seed: int, size: tuple[int, int]) -> Image.Image:
    """seed に依存する滑らかなグラデーション＋矩形パターン（pHash が安定する構造を持たせる）。"""
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        fx, fy = rng.uniform(0.5, 3.0, size=2)
        ph = rng.uniform(0, np.pi * 2)
        base[..., c] = 127 + 120 * np.sin(fx * xx / w * np.pi * 2 + fy * yy / h * np.pi + ph)
    # いくつかの矩形
    for _ in range(4):
        x0, y0 = rng.integers(0, w // 2), rng.integers(0, h // 2)
        x1, y1 = x0 + rng.integers(w // 8, w // 2), y0 + rng.integers(h // 8, h // 2)
        col = rng.integers(0, 255, size=3)
        base[y0:y1, x0:x1] = col
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")


@pytest.fixture
def make_image(tmp_path: Path) -> Callable[..., Path]:
    """make_image(name, seed=0, size=(256, 192), fmt="JPEG", quality=90, exif=None) -> Path"""

    def _make(
        name: str,
        seed: int = 0,
        size: tuple[int, int] = (256, 192),
        fmt: str = "JPEG",
        quality: int = 90,
        exif: dict[int, object] | None = None,
    ) -> Path:
        im = _pattern(seed, size)
        path = tmp_path / name
        kwargs: dict[str, object] = {}
        if fmt.upper() == "JPEG":
            kwargs["quality"] = quality
        if exif:
            ex = Image.Exif()
            for k, v in exif.items():
                ex[k] = v
            kwargs["exif"] = ex.tobytes()
        im.save(path, format=fmt.upper(), **kwargs)
        return path

    return _make


@pytest.fixture
def make_near_duplicate(tmp_path: Path) -> Callable[[Path, str], Path]:
    """元画像を 90% に縮小し品質 70 で再エンコードした近重複を作る。"""

    def _make(src: Path, name: str) -> Path:
        with Image.open(src) as im:
            w, h = im.size
            im2 = im.convert("RGB").resize((int(w * 0.9), int(h * 0.9)), Image.Resampling.BILINEAR)
        path = tmp_path / name
        im2.save(path, format="JPEG", quality=70)
        return path

    return _make


@pytest.fixture
def make_far_image(make_image: Callable[..., Path]) -> Callable[[str], Path]:
    """全く別のパターン（別 seed）の画像。"""

    def _make(name: str) -> Path:
        return make_image(name, seed=9_999)

    return _make
