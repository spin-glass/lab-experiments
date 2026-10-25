"""preprocess: EXIF 除去・長辺リサイズ・RGB JPEG 出力・サイズ統計（design.md §2）。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from cascade.data.preprocess import estimate_jpeg_quality, measure_image_stats, normalize_image

EXIF_MAKE = 0x010F
EXIF_ORIENTATION = 0x0112


def test_strips_exif_and_outputs_rgb_jpeg(make_image, tmp_path: Path):
    src = make_image("src.jpg", seed=1, size=(300, 200), exif={EXIF_MAKE: "TestCam"})
    with Image.open(src) as im:
        assert im.getexif().get(EXIF_MAKE) == "TestCam"
    dst = tmp_path / "out" / "dst.jpg"
    info = normalize_image(src, dst, max_side=1024, jpeg_quality=85)
    with Image.open(dst) as out:
        assert out.format == "JPEG"
        assert out.mode == "RGB"
        assert len(out.getexif()) == 0
        assert out.info.get("exif") is None
        assert out.size == (300, 200)
    assert info == {"width": 300, "height": 200, "bytes": dst.stat().st_size}


def test_respects_max_side(make_image, tmp_path: Path):
    src = make_image("big.png", seed=2, size=(800, 400), fmt="PNG")
    dst = tmp_path / "small.jpg"
    info = normalize_image(src, dst, max_side=200, jpeg_quality=80)
    assert (info["width"], info["height"]) == (200, 100)
    with Image.open(dst) as out:
        assert out.size == (200, 100) and out.mode == "RGB" and out.format == "JPEG"


def test_no_upscale_when_smaller(make_image, tmp_path: Path):
    src = make_image("s.jpg", seed=3, size=(120, 90))
    info = normalize_image(src, tmp_path / "o.jpg", max_side=1024, jpeg_quality=80)
    assert (info["width"], info["height"]) == (120, 90)


def test_rgba_png_converted_to_rgb(tmp_path: Path):
    src = tmp_path / "rgba.png"
    Image.new("RGBA", (64, 32), (255, 0, 0, 128)).save(src)
    info = normalize_image(src, tmp_path / "rgb.jpg", max_side=64, jpeg_quality=90)
    assert info["width"] == 64
    with Image.open(tmp_path / "rgb.jpg") as out:
        assert out.mode == "RGB"


def test_orientation_applied_before_strip(make_image, tmp_path: Path):
    # Orientation=6 (回転 90°) は画素へ反映し、タグ自体は捨てる
    src = make_image("rot.jpg", seed=4, size=(300, 200), exif={EXIF_ORIENTATION: 6})
    info = normalize_image(src, tmp_path / "rot_out.jpg", max_side=1024, jpeg_quality=85)
    assert (info["width"], info["height"]) == (200, 300)
    with Image.open(tmp_path / "rot_out.jpg") as out:
        assert len(out.getexif()) == 0


def test_estimate_jpeg_quality_monotone(make_image):
    lo = make_image("q50.jpg", seed=5, quality=50)
    hi = make_image("q95.jpg", seed=5, quality=95)
    with Image.open(lo) as a, Image.open(hi) as b:
        qa, qb = estimate_jpeg_quality(a), estimate_jpeg_quality(b)
    assert qa is not None and qb is not None
    assert abs(qa - 50) <= 3 and abs(qb - 95) <= 3


def test_measure_image_stats(make_image, tmp_path: Path):
    a = make_image("a.jpg", seed=6, size=(320, 240), quality=75)
    b = make_image("b.png", seed=7, size=(100, 50), fmt="PNG")
    df = measure_image_stats([a, b, tmp_path / "missing.jpg"])
    assert df.columns == ["path", "width", "height", "bytes", "format", "jpeg_quality_estimate"]
    assert df.height == 3
    row_a = df.filter(df["path"] == str(a)).row(0, named=True)
    assert (row_a["width"], row_a["height"], row_a["format"]) == (320, 240, "JPEG")
    assert abs(row_a["jpeg_quality_estimate"] - 75) <= 3
    row_b = df.filter(df["path"] == str(b)).row(0, named=True)
    assert row_b["format"] == "PNG" and row_b["jpeg_quality_estimate"] is None
    row_m = df.filter(df["path"] == str(tmp_path / "missing.jpg")).row(0, named=True)
    assert row_m["width"] is None and row_m["bytes"] is None
