"""Offline tests for src.data.download.

Covers tar/zip safety checks and the public ``download_ednet_kt1`` happy path
by staging a fake KT1 archive locally and invoking the function with
``skip_contents=True`` so no network is touched.
"""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import pytest

from src.data.download import (
    EDNET_KT1_ARCHIVE_NAME,
    EDNET_KT1_EXTRACTED_DIR,
    _detect_archive_kind,
    _safe_extract_tar,
    _safe_extract_zip,
    download_ednet_kt1,
)


def _stage_kt1_csvs(dest_dir: Path, num_users: int) -> Path:
    staging = dest_dir / "_staging_kt1"
    kt1 = staging / EDNET_KT1_EXTRACTED_DIR
    kt1.mkdir(parents=True, exist_ok=True)
    header = "timestamp,solving_id,question_id,user_answer,elapsed_time\n"
    row = "1565332027449,1,q100,a,45000\n"
    for uid in range(100000, 100000 + num_users):
        (kt1 / f"u{uid}.csv").write_text(header + row, encoding="utf-8")
    return kt1


def _make_fake_kt1_archive(dest_dir: Path, num_users: int = 3, kind: str = "tar.gz") -> Path:
    """Stage ``EdNet-KT1.tar.gz`` with ``num_users`` synthetic user CSVs.

    ``kind`` is either ``"tar.gz"`` or ``"zip"`` — both are exercised, because
    the real Drive distribution is a zip with a ``.tar.gz`` filename.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    kt1 = _stage_kt1_csvs(dest_dir, num_users)
    archive = dest_dir / EDNET_KT1_ARCHIVE_NAME
    if kind == "tar.gz":
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(kt1, arcname=EDNET_KT1_EXTRACTED_DIR)
    elif kind == "zip":
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
            for p in kt1.iterdir():
                zf.write(p, arcname=f"{EDNET_KT1_EXTRACTED_DIR}/{p.name}")
    else:
        raise ValueError(f"unknown kind: {kind}")
    return archive


def test_safe_extract_tar_rejects_path_traversal(tmp_path: Path) -> None:
    evil = tmp_path / "evil.tar.gz"
    outside = tmp_path / "outside.txt"
    outside.write_text("nope", encoding="utf-8")
    with tarfile.open(evil, "w:gz") as tar:
        tar.add(outside, arcname="../escaped.txt")

    target = tmp_path / "out"
    with pytest.raises(RuntimeError, match="unsafe tar entry"):
        _safe_extract_tar(evil, target)


def test_safe_extract_zip_rejects_path_traversal(tmp_path: Path) -> None:
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../escaped.txt", "nope")

    target = tmp_path / "out"
    with pytest.raises(RuntimeError, match="unsafe zip entry"):
        _safe_extract_zip(evil, target)


@pytest.mark.parametrize("kind", ["tar.gz", "zip"])
def test_download_ednet_kt1_uses_existing_archive(tmp_path: Path, kind: str) -> None:
    _make_fake_kt1_archive(tmp_path, num_users=5, kind=kind)
    result = download_ednet_kt1(dest=tmp_path, skip_contents=True)
    assert result.extracted_dir.is_dir()
    assert result.num_user_files == 5
    assert (result.extracted_dir / "u100000.csv").exists()
    assert result.contents_archive is None
    assert result.questions_csv is None


def test_download_ednet_kt1_force_extract(tmp_path: Path) -> None:
    _make_fake_kt1_archive(tmp_path, num_users=2)
    first = download_ednet_kt1(dest=tmp_path, skip_contents=True)
    stray = first.extracted_dir / "stray.csv"
    stray.write_text("x", encoding="utf-8")
    second = download_ednet_kt1(dest=tmp_path, skip_contents=True, force_extract=True)
    assert second.num_user_files == 2
    assert not stray.exists()


def test_detect_archive_kind(tmp_path: Path) -> None:
    tar_archive = _make_fake_kt1_archive(tmp_path / "a", num_users=1, kind="tar.gz")
    zip_archive = _make_fake_kt1_archive(tmp_path / "b", num_users=1, kind="zip")
    assert _detect_archive_kind(tar_archive) == "tar.gz"
    assert _detect_archive_kind(zip_archive) == "zip"
    bogus = tmp_path / "bogus.bin"
    bogus.write_bytes(b"not-an-archive")
    with pytest.raises(RuntimeError, match="unknown archive format"):
        _detect_archive_kind(bogus)
