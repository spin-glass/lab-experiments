"""Tests for src.models.bkt.

Full BKT fitting requires substantial data and time; these tests
verify the import patch, format conversion, and parameter extraction.
"""

from __future__ import annotations

import polars as pl


def test_pybkt_import_succeeds() -> None:
    """The sklearn compatibility patch should let pyBKT import cleanly."""
    from src.models.bkt import _BKTModel

    assert _BKTModel is not None


def test_to_pybkt_format_columns() -> None:
    from src.models.bkt import to_pybkt_format

    df = pl.DataFrame(
        {
            "user_id": [1, 1, 2],
            "concept": [10, 20, 10],
            "correct": [1, 0, 1],
            "solving_id": [1, 2, 3],
        }
    ).cast(
        {
            "user_id": pl.UInt32,
            "concept": pl.UInt16,
            "correct": pl.UInt8,
            "solving_id": pl.Int32,
        }
    )
    pdf = to_pybkt_format(df)
    assert set(pdf.columns) == {"order_id", "skill_name", "correct", "user_id"}
    assert "str" in str(pdf["skill_name"].dtype).lower()  # string type in pandas
    assert pdf["correct"].dtype.name.startswith("int")


def test_extract_params_schema() -> None:
    from src.models.bkt import BKTResult, extract_params

    result = BKTResult(
        model=None,
        skills=["10", "20"],
        params={
            "10": {"prior": 0.1, "learn": 0.2, "guess": 0.3, "slip": 0.1, "forget": 0.0},
            "20": {"prior": 0.2, "learn": 0.3, "guess": 0.2, "slip": 0.15, "forget": 0.01},
        },
    )
    pdf = extract_params(result)
    assert pdf.columns == ["skill", "prior", "learn", "guess", "slip", "forget"]
    assert pdf.height == 2


def test_predict_bkt_mastery_empty_when_no_skills() -> None:
    """When no skills match, return an empty DataFrame with correct schema."""
    from src.models.bkt import BKTResult, predict_bkt_mastery

    result = BKTResult(model=None, skills=[], params={})
    df = pl.DataFrame(
        {
            "user_id": [1],
            "concept": [999],
            "correct": [1],
            "solving_id": [1],
        }
    ).cast(
        {
            "user_id": pl.UInt32,
            "concept": pl.UInt16,
            "correct": pl.UInt8,
            "solving_id": pl.Int32,
        }
    )
    out = predict_bkt_mastery(result, df)
    assert out.height == 0
    assert "state_predictions" in out.columns
