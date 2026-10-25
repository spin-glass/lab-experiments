"""Preprocessing pipeline for Knowledge Tracing.

Implements the filtering, clipping, truncation, tag-explosion, and
user-level train/val/test split recommended by the EDA (notebook 01).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

__all__ = [
    "SplitResult",
    "filter_short_users",
    "clip_elapsed_time",
    "drop_null_correct",
    "truncate_sequences",
    "explode_tags",
    "split_by_user",
    "split_within_user",
    "preprocess_pipeline",
]

# ---------------------------------------------------------------------------
# Defaults (from EDA section 8)
# ---------------------------------------------------------------------------
SEED: int = 42
MIN_SEQ_LEN: int = 10
ELAPSED_MIN_MS: int = 1_000
ELAPSED_MAX_MS: int = 300_000
MAX_SEQ_LEN: int = 2_000
TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15
# TEST_FRAC is implicitly 1 - TRAIN_FRAC - VAL_FRAC


@dataclass(frozen=True)
class SplitResult:
    """Train / validation / test DataFrames after preprocessing.

    Attributes
    ----------
    train : pl.DataFrame
    val : pl.DataFrame
    test : pl.DataFrame
    n_concepts : int
        Number of unique concepts across all splits.
    """

    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame
    n_concepts: int


# ---------------------------------------------------------------------------
# Individual preprocessing steps
# ---------------------------------------------------------------------------


def filter_short_users(df: pl.DataFrame, min_len: int = MIN_SEQ_LEN) -> pl.DataFrame:
    """Remove users with fewer than *min_len* interactions."""
    counts = df.group_by("user_id").agg(pl.len().alias("_n"))
    keep = counts.filter(pl.col("_n") >= min_len).select("user_id")
    return df.join(keep, on="user_id", how="semi")


def clip_elapsed_time(
    df: pl.DataFrame,
    min_ms: int = ELAPSED_MIN_MS,
    max_ms: int = ELAPSED_MAX_MS,
) -> pl.DataFrame:
    """Drop rows with ``elapsed_time < min_ms``; cap at *max_ms*."""
    return df.filter(pl.col("elapsed_time") >= min_ms).with_columns(
        pl.col("elapsed_time").clip(upper_bound=max_ms)
    )


def drop_null_correct(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows where ``correct`` is null."""
    return df.filter(pl.col("correct").is_not_null())


def truncate_sequences(df: pl.DataFrame, max_len: int = MAX_SEQ_LEN) -> pl.DataFrame:
    """Keep only the first *max_len* interactions per user (by timestamp)."""
    return (
        df.sort("user_id", "timestamp")
        .with_columns(
            pl.col("timestamp")
            .rank(method="ordinal")
            .over("user_id")
            .alias("_seq_rank")
        )
        .filter(pl.col("_seq_rank") <= max_len)
        .drop("_seq_rank")
    )


def explode_tags(df: pl.DataFrame) -> pl.DataFrame:
    """Explode the ``tags`` list column into one row per (interaction, tag).

    Adds a ``concept`` column (UInt16) from the exploded tag value.
    Rows with null or empty ``tags`` are dropped.
    """
    return (
        df.filter(pl.col("tags").list.len() > 0)
        .explode("tags")
        .rename({"tags": "concept"})
        .filter(pl.col("concept").is_not_null())
        .cast({"concept": pl.UInt16})
    )


def split_by_user(
    df: pl.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    seed: int = SEED,
) -> SplitResult:
    """User-level train / validation / test split.

    All interactions for a given user go entirely into one split.

    Parameters
    ----------
    df
        DataFrame with a ``user_id`` column.
    train_frac, val_frac
        Fractions for train and validation; test gets the remainder.
    seed
        Seed for the permutation.
    """
    user_ids = df.select("user_id").unique().sort("user_id").to_series().to_numpy()
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(user_ids))

    n_train = int(len(user_ids) * train_frac)
    n_val = int(len(user_ids) * val_frac)

    train_ids = set(user_ids[perm[:n_train]].tolist())
    val_ids = set(user_ids[perm[n_train : n_train + n_val]].tolist())
    # test_ids = remainder

    train_df = df.filter(pl.col("user_id").is_in(train_ids))
    val_df = df.filter(pl.col("user_id").is_in(val_ids))
    test_df = df.filter(
        ~pl.col("user_id").is_in(train_ids) & ~pl.col("user_id").is_in(val_ids)
    )

    n_concepts = int(
        pl.concat([train_df, val_df, test_df])
        .select(pl.col("concept").n_unique())
        .item()
    )

    return SplitResult(
        train=train_df,
        val=val_df,
        test=test_df,
        n_concepts=n_concepts,
    )


def split_within_user(
    df: pl.DataFrame,
    train_frac: float = 0.7,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split each user's sequence temporally into early/late portions.

    For IRT/BKT evaluation: the model estimates parameters from the
    early portion and predicts on the late portion of the **same** users.

    Parameters
    ----------
    df
        DataFrame with ``user_id`` and ``timestamp`` columns.
    train_frac
        Fraction of each user's interactions to use as "early" (train).

    Returns
    -------
    (early, late)
        Two DataFrames with no row overlap.
    """
    ranked = (
        df.sort("user_id", "timestamp")
        .with_columns(
            pl.col("timestamp")
            .rank(method="ordinal")
            .over("user_id")
            .alias("_rank"),
            pl.len().over("user_id").alias("_total"),
        )
        .with_columns(
            (pl.col("_total") * train_frac).floor().cast(pl.UInt32).alias("_cutoff")
        )
    )
    early = ranked.filter(pl.col("_rank") <= pl.col("_cutoff")).drop("_rank", "_total", "_cutoff")
    late = ranked.filter(pl.col("_rank") > pl.col("_cutoff")).drop("_rank", "_total", "_cutoff")
    return early, late


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def preprocess_pipeline(
    df: pl.DataFrame,
    *,
    min_seq_len: int = MIN_SEQ_LEN,
    elapsed_min_ms: int = ELAPSED_MIN_MS,
    elapsed_max_ms: int = ELAPSED_MAX_MS,
    max_seq_len: int = MAX_SEQ_LEN,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    seed: int = SEED,
) -> SplitResult:
    """Run the full preprocessing pipeline.

    Steps (in order): filter short users → clip elapsed_time →
    drop null correct → truncate sequences → explode tags → split by user.
    """
    df = filter_short_users(df, min_len=min_seq_len)
    df = clip_elapsed_time(df, min_ms=elapsed_min_ms, max_ms=elapsed_max_ms)
    df = drop_null_correct(df)
    df = truncate_sequences(df, max_len=max_seq_len)
    df = explode_tags(df)
    return split_by_user(df, train_frac=train_frac, val_frac=val_frac, seed=seed)
