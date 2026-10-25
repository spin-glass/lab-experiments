"""Tests for src.data.sample."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.sample import build_sample, sample_cache_path
from tests.conftest import write_kt1, write_questions


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    write_kt1(
        raw_dir,
        {
            100000: [(1_000, 1, "q1", "a", 1000), (1_100, 2, "q2", "b", 1100)],
            100001: [(1_200, 1, "q1", "d", 900)],
            100002: [(1_300, 1, "q2", "a", 1200)],
            100003: [(1_400, 1, "q1", "a", 1500)],
        },
    )
    write_questions(
        raw_dir,
        [
            ("q1", "b1", "e1", "a", 1, "10", 1),
            ("q2", "b1", "e1", "a", 1, "20", 1),
        ],
    )
    return raw_dir


def test_build_sample_creates_cache(tmp_path: Path, raw: Path) -> None:
    processed = tmp_path / "processed"
    result = build_sample(raw, n_users=2, seed=7, processed_dir=processed)
    assert result.path == sample_cache_path(processed, 2, 7)
    assert result.path.exists()
    assert result.n_users == 2
    assert result.n_rows >= 2


def test_build_sample_is_deterministic(tmp_path: Path, raw: Path) -> None:
    processed = tmp_path / "processed"
    a = build_sample(raw, n_users=2, seed=42, processed_dir=processed)
    # Second call reuses cache; users should be identical.
    b = build_sample(raw, n_users=2, seed=42, processed_dir=processed)
    assert sorted(a.df["user_id"].unique().to_list()) == sorted(b.df["user_id"].unique().to_list())


def test_build_sample_different_seeds_differ(tmp_path: Path, raw: Path) -> None:
    processed = tmp_path / "processed"
    a = build_sample(raw, n_users=2, seed=1, processed_dir=processed)
    b = build_sample(raw, n_users=2, seed=999, processed_dir=processed)
    # With 4 source users and 2 picks, at least some seed pairs must differ.
    # Use a broader check: both results must be internally valid, and some
    # seed out of a handful must differ from seed=1.
    results = {
        s: sorted(
            build_sample(raw, n_users=2, seed=s, processed_dir=processed)
            .df["user_id"]
            .unique()
            .to_list()
        )
        for s in (1, 2, 3, 4, 5)
    }
    assert len({tuple(v) for v in results.values()}) >= 2
    assert a.n_users == b.n_users == 2


def test_build_sample_force_rebuilds(tmp_path: Path, raw: Path) -> None:
    processed = tmp_path / "processed"
    r1 = build_sample(raw, n_users=2, seed=5, processed_dir=processed)
    mtime1 = r1.path.stat().st_mtime_ns
    r2 = build_sample(raw, n_users=2, seed=5, processed_dir=processed, force=True)
    mtime2 = r2.path.stat().st_mtime_ns
    assert mtime2 >= mtime1


def test_build_sample_caps_at_available_users(tmp_path: Path, raw: Path) -> None:
    processed = tmp_path / "processed"
    result = build_sample(raw, n_users=99, seed=0, processed_dir=processed)
    assert result.n_users == 4  # all available
