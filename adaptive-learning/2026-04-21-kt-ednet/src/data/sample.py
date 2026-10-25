"""Seeded user-level sampling with a parquet cache.

For iterative EDA and unit-tested pipelines, loading all 780k users of
EdNet-KT1 is wasteful. This module selects a reproducible subset of users
(by deterministic seeded shuffle), loads their interactions via
:mod:`src.data.ednet`, and caches the result to parquet under
``<processed_dir>/ednet_sample_{n_users}_seed{seed}.parquet``.

A second call with the same ``(n_users, seed)`` re-uses the cache; pass
``force=True`` to rebuild.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from src.data.ednet import list_user_files, load_interactions

__all__ = ["SampleResult", "sample_cache_path", "build_sample"]


@dataclass(frozen=True)
class SampleResult:
    """Return value of :func:`build_sample`.

    Attributes
    ----------
    path
        Parquet file on disk containing the sampled interactions.
    df
        The loaded DataFrame.
    n_users
        Number of users actually included (may be less than requested if the
        raw directory has fewer files).
    n_rows
        Total interaction count in the sample.
    """

    path: Path
    df: pl.DataFrame
    n_users: int
    n_rows: int


def sample_cache_path(processed_dir: str | Path, n_users: int, seed: int) -> Path:
    """Return the canonical parquet path for a ``(n_users, seed)`` sample."""
    return Path(processed_dir).expanduser().resolve() / f"ednet_sample_{n_users}_seed{seed}.parquet"


def _pick_users(all_files: list[Path], n_users: int, seed: int) -> list[Path]:
    """Pick ``n_users`` user files using a seeded Fisher-Yates shuffle.

    Uses ``random.Random(seed).sample`` so the selection is reproducible and
    independent of the platform's default RNG state.
    """
    if n_users >= len(all_files):
        return list(all_files)
    rng = random.Random(seed)
    return sorted(rng.sample(all_files, k=n_users))


def build_sample(
    raw_dir: str | Path,
    *,
    n_users: int,
    seed: int = 42,
    processed_dir: str | Path = "data/processed",
    force: bool = False,
    questions_csv: str | Path | None = None,
) -> SampleResult:
    """Build (or reuse) a ``n_users``-user sample of EdNet-KT1 interactions.

    Parameters
    ----------
    raw_dir
        Directory containing ``KT1/`` (see :mod:`src.data.download`).
    n_users
        Number of users to include. If greater than the available count, all
        users are used.
    seed
        Seed for the user-level shuffle. Fixed default = 42 so CI and EDA
        share a sample unless overridden.
    processed_dir
        Where to store the parquet cache.
    force
        If True, rebuild even when the cache file already exists.
    questions_csv
        Optional override for ``questions.csv``; defaults to
        ``<raw_dir>/questions.csv`` when present.

    Returns
    -------
    SampleResult
    """
    processed_dir = Path(processed_dir).expanduser().resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)
    cache_path = sample_cache_path(processed_dir, n_users, seed)

    if cache_path.exists() and not force:
        df = pl.read_parquet(cache_path)
        n_u = int(df.select(pl.col("user_id").n_unique()).item())
        return SampleResult(path=cache_path, df=df, n_users=n_u, n_rows=df.height)

    all_files = list_user_files(raw_dir)
    picked = _pick_users(all_files, n_users, seed)

    df = load_interactions(
        raw_dir,
        user_files=picked,
        questions=questions_csv,
    )
    df.write_parquet(cache_path, compression="zstd")
    n_u = int(df.select(pl.col("user_id").n_unique()).item())
    return SampleResult(path=cache_path, df=df, n_users=n_u, n_rows=df.height)
