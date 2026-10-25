"""Tests for src.features.preprocess."""

from __future__ import annotations

import polars as pl

from src.features.preprocess import (
    SplitResult,
    clip_elapsed_time,
    drop_null_correct,
    explode_tags,
    filter_short_users,
    preprocess_pipeline,
    split_by_user,
    split_within_user,
    truncate_sequences,
)


def _make_df(n_users: int = 5, rows_per_user: int = 15) -> pl.DataFrame:
    """Build a small synthetic DataFrame mimicking EdNet-KT1 columns."""
    rows: list[dict[str, object]] = []
    ts = 1_000_000
    for uid in range(1, n_users + 1):
        for i in range(rows_per_user):
            rows.append(
                {
                    "user_id": uid,
                    "timestamp": ts + i * 1000,
                    "solving_id": uid * 1000 + i,
                    "question_id": i + 1,
                    "elapsed_time": 5_000 + i * 1000,
                    "correct": i % 2,
                    "part": (i % 7) + 1,
                    "tags": [i % 5, (i + 1) % 5],
                }
            )
    return pl.DataFrame(rows).cast(
        {
            "user_id": pl.UInt32,
            "timestamp": pl.Int64,
            "solving_id": pl.Int32,
            "question_id": pl.UInt32,
            "elapsed_time": pl.Int32,
            "correct": pl.UInt8,
            "part": pl.UInt8,
            "tags": pl.List(pl.UInt16),
        }
    )


# --- filter_short_users ---


def test_filter_short_users_removes_short() -> None:
    df = _make_df(n_users=3, rows_per_user=5)
    # Add a user with only 2 rows
    extra = pl.DataFrame(
        {
            "user_id": pl.Series([99, 99], dtype=pl.UInt32),
            "timestamp": pl.Series([1, 2], dtype=pl.Int64),
            "solving_id": pl.Series([991, 992], dtype=pl.Int32),
            "question_id": pl.Series([1, 2], dtype=pl.UInt32),
            "elapsed_time": pl.Series([5000, 6000], dtype=pl.Int32),
            "correct": pl.Series([1, 0], dtype=pl.UInt8),
            "part": pl.Series([1, 2], dtype=pl.UInt8),
            "tags": [[0, 1], [2, 3]],
        }
    ).cast({"tags": pl.List(pl.UInt16)})
    combined = pl.concat([df, extra])
    result = filter_short_users(combined, min_len=5)
    assert 99 not in result["user_id"].to_list()
    assert result.select(pl.col("user_id").n_unique()).item() == 3


def test_filter_short_users_keeps_long() -> None:
    df = _make_df(n_users=2, rows_per_user=20)
    result = filter_short_users(df, min_len=10)
    assert result.select(pl.col("user_id").n_unique()).item() == 2


# --- clip_elapsed_time ---


def test_clip_elapsed_time_drops_and_caps() -> None:
    df = pl.DataFrame(
        {
            "user_id": pl.Series([1, 1, 1, 1], dtype=pl.UInt32),
            "elapsed_time": pl.Series([500, 2000, 400_000, 100_000], dtype=pl.Int32),
        }
    )
    result = clip_elapsed_time(df, min_ms=1000, max_ms=300_000)
    assert result.height == 3  # 500 dropped
    assert result["elapsed_time"].max() == 300_000


# --- drop_null_correct ---


def test_drop_null_correct() -> None:
    df = pl.DataFrame(
        {
            "correct": pl.Series([1, None, 0, None], dtype=pl.UInt8),
        }
    )
    result = drop_null_correct(df)
    assert result.height == 2


# --- truncate_sequences ---


def test_truncate_sequences() -> None:
    df = _make_df(n_users=1, rows_per_user=20)
    result = truncate_sequences(df, max_len=10)
    assert result.height == 10


# --- explode_tags ---


def test_explode_tags_multiplies_rows() -> None:
    df = pl.DataFrame(
        {
            "user_id": pl.Series([1, 1], dtype=pl.UInt32),
            "tags": [[0, 1, 2], [3, 4]],
        }
    ).cast({"tags": pl.List(pl.UInt16)})
    result = explode_tags(df)
    assert result.height == 5
    assert "concept" in result.columns
    assert "tags" not in result.columns


def test_explode_tags_drops_empty() -> None:
    df = pl.DataFrame(
        {
            "user_id": pl.Series([1, 2], dtype=pl.UInt32),
            "tags": [[], [1, 2]],
        }
    ).cast({"tags": pl.List(pl.UInt16)})
    result = explode_tags(df)
    assert result.height == 2  # only user 2's tags


# --- split_by_user ---


def test_split_by_user_no_overlap() -> None:
    df = _make_df(n_users=20, rows_per_user=10)
    df = explode_tags(df)
    result = split_by_user(df, train_frac=0.7, val_frac=0.15, seed=42)
    train_users = set(result.train["user_id"].unique().to_list())
    val_users = set(result.val["user_id"].unique().to_list())
    test_users = set(result.test["user_id"].unique().to_list())
    assert train_users.isdisjoint(val_users)
    assert train_users.isdisjoint(test_users)
    assert val_users.isdisjoint(test_users)
    assert len(train_users | val_users | test_users) == 20


def test_split_by_user_deterministic() -> None:
    df = _make_df(n_users=20, rows_per_user=10)
    df = explode_tags(df)
    r1 = split_by_user(df, seed=123)
    r2 = split_by_user(df, seed=123)
    assert r1.train["user_id"].unique().sort().to_list() == r2.train["user_id"].unique().sort().to_list()


def test_split_by_user_different_seeds() -> None:
    df = _make_df(n_users=20, rows_per_user=10)
    df = explode_tags(df)
    r1 = split_by_user(df, seed=1)
    r2 = split_by_user(df, seed=2)
    # Very unlikely to have the same train set with different seeds
    assert r1.train["user_id"].unique().sort().to_list() != r2.train["user_id"].unique().sort().to_list()


# --- split_within_user ---


def test_split_within_user_no_overlap() -> None:
    df = _make_df(n_users=3, rows_per_user=20)
    early, late = split_within_user(df, train_frac=0.7)
    # No row should appear in both
    assert early.height + late.height == df.height
    # Each user should have rows in both splits
    for uid in df["user_id"].unique().to_list():
        assert early.filter(pl.col("user_id") == uid).height > 0
        assert late.filter(pl.col("user_id") == uid).height > 0


def test_split_within_user_temporal_order() -> None:
    df = _make_df(n_users=2, rows_per_user=10)
    early, late = split_within_user(df, train_frac=0.5)
    for uid in df["user_id"].unique().to_list():
        e_max = early.filter(pl.col("user_id") == uid)["timestamp"].max()
        l_min = late.filter(pl.col("user_id") == uid)["timestamp"].min()
        assert e_max <= l_min  # early timestamps <= late timestamps


# --- preprocess_pipeline ---


def test_preprocess_pipeline_returns_split_result() -> None:
    df = _make_df(n_users=10, rows_per_user=15)
    result = preprocess_pipeline(df, min_seq_len=5, seed=42)
    assert isinstance(result, SplitResult)
    assert result.train.height > 0
    assert result.n_concepts > 0
    assert "concept" in result.train.columns
