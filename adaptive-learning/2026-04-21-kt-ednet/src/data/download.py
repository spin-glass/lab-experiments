"""EdNet-KT1 dataset download utility.

EdNet-KT1 (Riiid / KAIST) is distributed via Google Drive as of 2026:

* KT1 interactions (``EdNet-KT1.tar.gz``, ~1.2 GB compressed, ~5.6 GB extracted)
  contains one CSV per user under ``KT1/u{uid}.csv``.
* The Contents archive (``contents.zip``) carries the metadata tables —
  in particular ``questions.csv``, which we need to join correct answers and
  tags onto interactions.

The GitHub releases under ``riiid/ednet`` that the original README linked
no longer serve these assets; the canonical ``bit.ly/ednet_kt1`` /
``bit.ly/ednet-content`` short links now redirect to Drive.

Resulting layout under ``<dest>``::

    <dest>/
      EdNet-KT1.tar.gz           (downloaded)
      KT1/                       (extracted)
        u100000.csv
        ...
      contents.zip               (downloaded)
      questions.csv              (extracted, alongside lectures.csv etc.)

Usage
-----

From the CLI::

    uv run python -m src.data.download --dest data/raw

Programmatic::

    from src.data.download import download_ednet_kt1
    download_ednet_kt1(dest="data/raw")
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import gdown

# --- Public dataset constants --------------------------------------------------

EDNET_KT1_GDRIVE_ID = "1AmGcOs5U31wIIqvthn9ARqJMrMTFTcaw"
EDNET_KT1_ARCHIVE_NAME = "EdNet-KT1.tar.gz"
EDNET_KT1_EXTRACTED_DIR = "KT1"

EDNET_CONTENTS_GDRIVE_ID = "117aYJAWG3GU48suS66NPaB82HwFj6xWS"
EDNET_CONTENTS_ARCHIVE_NAME = "contents.zip"
EDNET_CONTENTS_SENTINEL = "questions.csv"  # extraction is considered done iff this exists


# --- Result types --------------------------------------------------------------


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of :func:`download_ednet_kt1`.

    Attributes
    ----------
    archive_path
        Path to the downloaded KT1 ``.tar.gz``.
    extracted_dir
        Path to the extracted ``KT1/`` directory with per-user CSVs.
    num_user_files
        Count of ``u*.csv`` files under ``extracted_dir``.
    contents_archive
        Path to the downloaded ``contents.zip`` (``None`` if skipped).
    questions_csv
        Path to the extracted ``questions.csv`` (``None`` if contents skipped).
    """

    archive_path: Path
    extracted_dir: Path
    num_user_files: int
    contents_archive: Path | None
    questions_csv: Path | None


# --- Helpers -------------------------------------------------------------------


def _gdown_to(file_id: str, dest: Path) -> None:
    """Download a Google Drive file by id to ``dest``, with progress and resume.

    ``gdown`` prints a tqdm bar itself; we rely on that for UX parity with the
    old urllib-based implementation.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(dest), quiet=False, resume=True)
    if not dest.exists() or dest.stat().st_size == 0:
        raise RuntimeError(f"download failed or produced empty file: {dest}")


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _safe_extract_tar(archive: Path, dest: Path) -> None:
    """Extract a tarball, rejecting entries that escape ``dest`` (CVE-2007-4559)."""
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for m in members:
            target = (dest / m.name).resolve()
            if not str(target).startswith(str(dest)):
                raise RuntimeError(f"unsafe tar entry: {m.name}")
        tar.extractall(dest)


def _detect_archive_kind(archive: Path) -> str:
    """Return ``"tar.gz"`` or ``"zip"`` based on file magic bytes.

    The KT1 archive distributed via Google Drive is a ``.zip`` despite the
    ``.tar.gz`` filename inherited from the original release. Older copies
    (pre-2024) are real gzip'd tarballs. Detect by content, not name.
    """
    with archive.open("rb") as f:
        head = f.read(4)
    if head.startswith(b"\x1f\x8b"):
        return "tar.gz"
    if head.startswith(b"PK"):
        return "zip"
    raise RuntimeError(
        f"unknown archive format for {archive}: first bytes = {head!r}",
    )


def _safe_extract_archive(archive: Path, dest: Path) -> None:
    """Dispatch to tar or zip extraction based on the archive's magic bytes."""
    kind = _detect_archive_kind(archive)
    if kind == "tar.gz":
        _safe_extract_tar(archive, dest)
    else:
        _safe_extract_zip(archive, dest)


def _safe_extract_zip(archive: Path, dest: Path) -> None:
    """Extract a zip file, rejecting entries that escape ``dest``."""
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            target = (dest / name).resolve()
            if not str(target).startswith(str(dest)):
                raise RuntimeError(f"unsafe zip entry: {name}")
        zf.extractall(dest)


# --- Component downloaders -----------------------------------------------------


def _download_kt1(
    dest: Path,
    *,
    force_download: bool,
    force_extract: bool,
    expected_sha256: str | None,
) -> tuple[Path, Path, int]:
    archive = dest / EDNET_KT1_ARCHIVE_NAME
    extracted = dest / EDNET_KT1_EXTRACTED_DIR

    if archive.exists() and not force_download:
        print(f"[skip] archive already exists: {archive}")
    else:
        print(f"[download] gdrive:{EDNET_KT1_GDRIVE_ID} -> {archive}")
        _gdown_to(EDNET_KT1_GDRIVE_ID, archive)

    if expected_sha256:
        print("[verify] computing sha256...")
        actual = _sha256(archive)
        if actual.lower() != expected_sha256.lower():
            raise RuntimeError(f"sha256 mismatch: expected={expected_sha256} actual={actual}")
        print("[verify] ok")

    if extracted.exists() and not force_extract:
        print(f"[skip] extract already done: {extracted}")
    else:
        if extracted.exists():
            shutil.rmtree(extracted)
        print(f"[extract] {archive} -> {dest}")
        _safe_extract_archive(archive, dest)

    user_files = sorted(extracted.glob("u*.csv"))
    print(f"[done] KT1: {len(user_files)} user CSVs under {extracted}")
    return archive, extracted, len(user_files)


def _download_contents(
    dest: Path,
    *,
    force_download: bool,
    force_extract: bool,
) -> tuple[Path, Path | None]:
    archive = dest / EDNET_CONTENTS_ARCHIVE_NAME
    questions = dest / EDNET_CONTENTS_SENTINEL

    if archive.exists() and not force_download:
        print(f"[skip] contents archive already exists: {archive}")
    else:
        print(f"[download] gdrive:{EDNET_CONTENTS_GDRIVE_ID} -> {archive}")
        _gdown_to(EDNET_CONTENTS_GDRIVE_ID, archive)

    if questions.exists() and not force_extract:
        print(f"[skip] contents already extracted: {questions}")
    else:
        print(f"[extract] {archive} -> {dest}")
        _safe_extract_zip(archive, dest)
        # The zip may extract into a subdirectory; promote questions.csv to dest.
        if not questions.exists():
            candidates = list(dest.rglob(EDNET_CONTENTS_SENTINEL))
            if not candidates:
                print(
                    "[warn] contents archive did not contain questions.csv at any level; "
                    "check archive contents manually"
                )
                return archive, None
            picked = candidates[0]
            if picked != questions:
                shutil.copy2(picked, questions)
    print(f"[done] contents: {questions}")
    return archive, questions


# --- Public API ----------------------------------------------------------------


def download_ednet_kt1(
    dest: str | Path = "data/raw",
    *,
    force_download: bool = False,
    force_extract: bool = False,
    expected_sha256: str | None = None,
    skip_contents: bool = False,
) -> DownloadResult:
    """Download and extract EdNet-KT1 and (by default) the Contents metadata.

    Parameters
    ----------
    dest
        Directory under which all artifacts are placed.
    force_download
        If True, re-download even when archives already exist.
    force_extract
        If True, wipe and re-extract even when extracted data already exists.
    expected_sha256
        Optional SHA-256 to verify the KT1 archive.
    skip_contents
        If True, skip ``contents.zip`` / ``questions.csv``. Primarily for tests
        that stage raw data manually.

    Returns
    -------
    DownloadResult
    """
    dest = Path(dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    archive, extracted, n = _download_kt1(
        dest,
        force_download=force_download,
        force_extract=force_extract,
        expected_sha256=expected_sha256,
    )

    contents_archive: Path | None = None
    questions_csv: Path | None = None
    if not skip_contents:
        contents_archive, questions_csv = _download_contents(
            dest,
            force_download=force_download,
            force_extract=force_extract,
        )

    return DownloadResult(
        archive_path=archive,
        extracted_dir=extracted,
        num_user_files=n,
        contents_archive=contents_archive,
        questions_csv=questions_csv,
    )


# --- CLI -----------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download EdNet-KT1 dataset from Google Drive.")
    p.add_argument(
        "--dest",
        type=str,
        default="data/raw",
        help="Destination directory for archives + extracted files (default: data/raw)",
    )
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--force-extract", action="store_true")
    p.add_argument(
        "--sha256",
        type=str,
        default=None,
        help="Optional expected sha256 of the KT1 archive.",
    )
    p.add_argument(
        "--skip-contents",
        action="store_true",
        help="Do not fetch contents.zip / questions.csv.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        download_ednet_kt1(
            dest=args.dest,
            force_download=args.force_download,
            force_extract=args.force_extract,
            expected_sha256=args.sha256,
            skip_contents=args.skip_contents,
        )
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
