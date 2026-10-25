"""外部画像の統一前処理と Yelp の画像サイズ分布の計測（design.md §2、計画 Phase A2 / B-b）。

外部画像は Yelp と同分布（長辺サイズ・JPEG 品質・EXIF 除去）に揃える。前処理の数値は config に固定し、
A2 で :func:`measure_image_stats` により Yelp 側の分布を測ってから決める。

固定する前処理の内容（``normalize_image``）:

- EXIF の向きを画素へ反映してから EXIF / ICC / XMP を捨てる
- ICC プロファイルがあれば sRGB へ変換してから除去（Adobe RGB / Display P3 の外部写真で色が変わらない）
- アルファ（RGBA / LA / 透過 P）は白（``background``）に合成する（黒背景にしない。外部 PNG だけ
  黒背景になると「ソース検出」の交絡を自ら作る）
- 16 bit（I;16 系）・32 bit 整数（I）・float（F）は 8 bit へスケール／クリップ
- JPEG の chroma subsampling / progressive は引数（既定 ``JPEG_DEFAULTS``）で固定し lock に記録する
- 画素数上限（``max_pixels``、既定 Pillow の ``MAX_IMAGE_PIXELS``）を超える画像は
  ``DecompressionBombError`` で拒否する（黙って処理しない）
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from PIL import Image, ImageCms, ImageOps

__all__ = [
    "JPEG_DEFAULTS",
    "JPEG_FORMATS",
    "estimate_jpeg_quality",
    "measure_image_stats",
    "normalize_image",
    "to_rgb8",
]

#: JPEG 互換として扱う Pillow の format 名（MPO = スマホ由来の multi-picture JPEG）
JPEG_FORMATS = frozenset({"JPEG", "MPO"})
#: JPEG 出力の固定パラメータ（Pillow 既定と同じ 4:2:0・非プログレッシブ。lock に記録する）
JPEG_DEFAULTS: dict[str, Any] = {"subsampling": "4:2:0", "progressive": False}
#: アルファ合成の既定背景（白）
DEFAULT_BACKGROUND: tuple[int, int, int] = (255, 255, 255)

# libjpeg の標準輝度量子化テーブル（IJG）。quality 推定の基準に使う
_STD_LUMA_Q = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]  # fmt: skip


def estimate_jpeg_quality(im: Image.Image) -> int | None:
    """JPEG の輝度量子化テーブルから IJG 流の quality（1–100）を逆算する。JPEG 以外は None。"""
    qt = getattr(im, "quantization", None)
    if not qt or 0 not in qt:
        return None
    table = list(qt[0])
    if len(table) != 64:
        return None
    # IJG: q_scaled = clip((std * scale + 50) / 100, 1, 255)、scale = 5000/Q (Q<50) or 200-2Q
    # 平均比から scale を逆算する。
    ratios = [t / s for t, s in zip(table, _STD_LUMA_Q, strict=True)]
    scale = 100.0 * sum(ratios) / len(ratios)
    if scale <= 0:
        return 100
    if scale < 100:
        q = (200.0 - scale) / 2.0
    else:
        q = 5000.0 / scale
    return int(max(1, min(100, round(q))))


def _has_alpha(im: Image.Image) -> bool:
    if im.mode in ("RGBA", "LA", "RGBa", "La", "PA"):
        return True
    return im.mode == "P" and "transparency" in im.info


def _to_8bit(im: Image.Image) -> Image.Image:
    """高ビット深度（I;16 系・I・F）を 8 bit "L" にスケール／クリップする。それ以外はそのまま。"""
    if im.mode.startswith("I;16"):
        arr = np.asarray(im).astype(np.float32) / 257.0
    elif im.mode == "I":
        arr = np.asarray(im).astype(np.float32)
        if arr.size and arr.max() > 255:  # 16 bit 由来とみなす
            arr = arr / 257.0
    elif im.mode == "F":
        arr = np.asarray(im).astype(np.float32)
        if arr.size and arr.max() <= 1.0:
            arr = arr * 255.0
    else:
        return im
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "L")


def _apply_icc(im: Image.Image) -> Image.Image:
    """ICC プロファイル付き RGB を sRGB へ変換する（変換できない壊れたプロファイルは無視）。"""
    icc = im.info.get("icc_profile")
    if not icc:
        return im
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        dst = ImageCms.createProfile("sRGB")
        out = ImageCms.profileToProfile(im, src, dst, outputMode="RGB")
    except (ImageCms.PyCMSError, OSError, ValueError):
        return im
    return out if out is not None else im


def to_rgb8(im: Image.Image, background: tuple[int, int, int] = DEFAULT_BACKGROUND) -> Image.Image:
    """任意モードの画像を 8 bit sRGB "RGB" にする（アルファは background に合成、ICC は sRGB へ）。"""
    icc = im.info.get("icc_profile")
    im = _to_8bit(im)
    if _has_alpha(im):
        rgba = im.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (*background, 255))
        canvas.alpha_composite(rgba)
        rgb = canvas.convert("RGB")
    else:
        rgb = im.convert("RGB")
    if icc:
        rgb.info["icc_profile"] = icc
        rgb = _apply_icc(rgb)
    return rgb


def normalize_image(
    src: Path,
    dst: Path,
    max_side: int,
    jpeg_quality: int,
    *,
    subsampling: str | int = JPEG_DEFAULTS["subsampling"],
    progressive: bool = JPEG_DEFAULTS["progressive"],
    background: tuple[int, int, int] = DEFAULT_BACKGROUND,
    max_pixels: int | None = Image.MAX_IMAGE_PIXELS,
) -> dict[str, Any]:
    """EXIF を捨て、sRGB 8 bit RGB に変換し、長辺が max_side を超えれば縮小して JPEG に再エンコードする。

    EXIF の向き情報は捨てる前に画素へ反映する（``ImageOps.exif_transpose``）。アルファは
    ``background``（既定は白）へ合成、ICC は sRGB へ変換、16 bit は 8 bit へスケールする。
    画素数が ``max_pixels`` を超える画像は ``PIL.Image.DecompressionBombError``。
    戻り値は ``{"width", "height", "bytes"}``（出力後の値）。
    """
    src = Path(src)
    dst = Path(dst)
    with Image.open(src) as im:
        if max_pixels is not None and im.width * im.height > max_pixels:
            raise Image.DecompressionBombError(
                f"{src}: {im.width}x{im.height} = {im.width * im.height} 画素は上限 {max_pixels} を超える"
            )
        im = ImageOps.exif_transpose(im)
        im = to_rgb8(im, background=background)
        w, h = im.size
        longer = max(w, h)
        if longer > max_side:
            scale = max_side / float(longer)
            new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
            im = im.resize(new_size, Image.Resampling.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # 新規 Image に画素だけコピーし、info（EXIF / ICC / XMP）を持ち越さない
        clean = Image.new("RGB", im.size)
        clean.paste(im)
        clean.save(
            dst,
            format="JPEG",
            quality=int(jpeg_quality),
            optimize=True,
            subsampling=subsampling,
            progressive=bool(progressive),
        )
        out_w, out_h = clean.size
    return {"width": out_w, "height": out_h, "bytes": dst.stat().st_size}


def measure_image_stats(paths: Sequence[str | Path]) -> pl.DataFrame:
    """画像の幅・高さ・バイト数・JPEG 品質推定値を返す（A2: Yelp のサイズ分布を特徴付ける）。

    列: path, width, height, bytes, format, jpeg_quality_estimate（JPEG 以外は null）, error。
    MPO は JPEG として扱う（format = "JPEG"）。開けない・画素数上限超過（DecompressionBomb）は
    ``error`` に記録して継続する。
    """
    rows: list[dict[str, Any]] = []
    for p in paths:
        p = Path(p)
        w = h = None
        fmt = None
        q = None
        err = None
        try:
            with Image.open(p) as im:
                w, h = im.size
                fmt = im.format
                if fmt in JPEG_FORMATS:
                    q = estimate_jpeg_quality(im)
                    fmt = "JPEG"
        except (OSError, ValueError, Image.DecompressionBombError) as e:
            err = f"{type(e).__name__}: {e}"
        rows.append(
            {
                "path": str(p),
                "width": w,
                "height": h,
                "bytes": p.stat().st_size if p.exists() else None,
                "format": fmt,
                "jpeg_quality_estimate": q,
                "error": err,
            }
        )
    schema = {
        "path": pl.Utf8,
        "width": pl.Int64,
        "height": pl.Int64,
        "bytes": pl.Int64,
        "format": pl.Utf8,
        "jpeg_quality_estimate": pl.Int64,
        "error": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema)
