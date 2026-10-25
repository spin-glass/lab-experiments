"""Unit tests for src.data.ednet using synthetic CSVs."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.data.ednet import (
    list_user_files,
    load_interactions,
    load_questions,
    user_id_from_path,
)
from tests.conftest import write_kt1, write_questions


def test_user_id_from_path() -> None:
    assert user_id_from_path(Path("u100000.csv")) == 100000
    with pytest.raises(ValueError):
        user_id_from_path(Path("not_a_user.csv"))


def test_list_user_files_sorted(tmp_path: Path) -> None:
    write_kt1(
        tmp_path,
        {
            100002: [(1_000, 1, "q1", "a", 1000)],
            100000: [(1_000, 1, "q1", "a", 1000)],
            100001: [(1_000, 1, "q1", "a", 1000)],
        },
    )
    files = list_user_files(tmp_path)
    assert [user_id_from_path(p) for p in files] == [100000, 100001, 100002]


def test_load_questions_parses_tags_and_strips_prefixes(tmp_path: Path) -> None:
    write_questions(
        tmp_path,
        [
            ("q2319", "b1707", "e1707", "a", 3, "179;53;183;184", 1571279008033),
            ("q2320", "b1707", "e1707", "d", 3, "52;183", 1571279009205),
        ],
    )
    df = load_questions(tmp_path / "questions.csv")
    assert df.columns == ["question_id", "bundle_id", "correct_answer", "part", "tags"]
    assert df["question_id"].to_list() == [2319, 2320]
    assert df["bundle_id"].to_list() == [1707, 1707]
    assert df["tags"].to_list() == [[179, 53, 183, 184], [52, 183]]


def test_load_interactions_joins_correct_label(tmp_path: Path) -> None:
    write_kt1(
        tmp_path,
        {
            100000: [
                (1_000, 1, "q1", "a", 1000),  # correct
                (2_000, 2, "q2", "b", 1500),  # wrong (correct='a')
            ],
            100001: [
                (1_500, 1, "q1", "d", 900),  # wrong
            ],
        },
    )
    write_questions(
        tmp_path,
        [
            ("q1", "b1", "e1", "a", 1, "10;20", 1),
            ("q2", "b1", "e1", "a", 1, "10", 1),
        ],
    )
    df = load_interactions(tmp_path)
    assert df.height == 3
    # sorted by (user_id, timestamp)
    assert df["user_id"].to_list() == [100000, 100000, 100001]
    assert df["correct"].to_list() == [1, 0, 0]
    # joined metadata present
    assert df["part"].to_list() == [1, 1, 1]
    assert df["bundle_id"].to_list() == [1, 1, 1]


def test_load_interactions_without_questions(tmp_path: Path) -> None:
    write_kt1(tmp_path, {100000: [(1_000, 1, "q1", "a", 1000)]})
    df = load_interactions(tmp_path)
    assert "correct" not in df.columns
    assert df["question_id"].dtype == pl.UInt32


def test_load_interactions_respects_user_files_override(tmp_path: Path) -> None:
    write_kt1(
        tmp_path,
        {
            100000: [(1, 1, "q1", "a", 1)],
            100001: [(1, 1, "q1", "a", 1)],
            100002: [(1, 1, "q1", "a", 1)],
        },
    )
    files = list_user_files(tmp_path)
    picked = [files[0], files[2]]
    df = load_interactions(tmp_path, user_files=picked)
    assert sorted(df["user_id"].unique().to_list()) == [100000, 100002]
