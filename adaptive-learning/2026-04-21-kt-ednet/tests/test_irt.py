"""Tests for src.models.irt (data preparation and model building only).

Full MCMC fitting is too slow for CI; we test ``prepare_irt_data``
and ``build_irt_2pl`` here.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pymc as pm

from src.models.irt import IRTResult, build_irt_2pl, predict_irt, prepare_irt_data


def _make_concept_df() -> pl.DataFrame:
    """Tiny exploded DataFrame for IRT tests."""
    return pl.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3, 3],
            "concept": [10, 20, 10, 30, 20, 30, 10],
            "correct": [1, 0, 1, 1, 0, 0, 1],
        }
    ).cast(
        {
            "user_id": pl.UInt32,
            "concept": pl.UInt16,
            "correct": pl.UInt8,
        }
    )


def test_prepare_irt_data_indices_contiguous() -> None:
    df = _make_concept_df()
    user_idx, concept_idx, correct, uids, cids = prepare_irt_data(df)
    assert user_idx.min() == 0
    assert user_idx.max() == len(uids) - 1
    assert concept_idx.min() == 0
    assert concept_idx.max() == len(cids) - 1
    assert len(correct) == len(df)


def test_prepare_irt_data_unique_sorted() -> None:
    df = _make_concept_df()
    _, _, _, uids, cids = prepare_irt_data(df)
    assert list(uids) == sorted(uids)
    assert list(cids) == sorted(cids)


def test_build_irt_2pl_creates_model() -> None:
    df = _make_concept_df()
    user_idx, concept_idx, correct, uids, cids = prepare_irt_data(df)
    model = build_irt_2pl(user_idx, concept_idx, correct, len(uids), len(cids))
    assert isinstance(model, pm.Model)
    var_names = {v.name for v in model.free_RVs}
    assert "theta" in var_names
    assert "difficulty" in var_names
    assert "log_disc" in var_names


def test_predict_irt_shape() -> None:
    """Test prediction with a mock IRTResult (no actual MCMC)."""
    df = _make_concept_df()
    result = IRTResult(
        trace=None,  # type: ignore[arg-type]
        user_ids=np.array([1, 2, 3], dtype=np.int64),
        concept_ids=np.array([10, 20, 30], dtype=np.int64),
        theta=np.array([0.5, -0.3, 0.1]),
        difficulty=np.array([0.0, 0.2, -0.1]),
        discrimination=np.array([1.0, 1.2, 0.8]),
    )
    probs = predict_irt(result, df)
    assert probs.shape == (len(df),)
    assert np.all((probs >= 0) & (probs <= 1))


def test_predict_irt_unseen_user() -> None:
    """Unseen users should get population-mean theta (0)."""
    df = pl.DataFrame(
        {"user_id": [999], "concept": [10], "correct": [1]}
    ).cast({"user_id": pl.UInt32, "concept": pl.UInt16, "correct": pl.UInt8})

    result = IRTResult(
        trace=None,  # type: ignore[arg-type]
        user_ids=np.array([1], dtype=np.int64),
        concept_ids=np.array([10], dtype=np.int64),
        theta=np.array([0.0]),
        difficulty=np.array([0.0]),
        discrimination=np.array([1.0]),
    )
    probs = predict_irt(result, df)
    # sigmoid(1.0 * (0.0 - 0.0)) = 0.5
    assert abs(probs[0] - 0.5) < 1e-6
