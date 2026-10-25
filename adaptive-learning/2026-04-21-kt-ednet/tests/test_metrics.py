"""Tests for src.eval.metrics."""

from __future__ import annotations

import numpy as np

from src.eval.metrics import EvalResult, calibration_data, evaluate_predictions


def test_perfect_predictions() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_prob = np.array([0.0, 0.1, 0.9, 1.0], dtype=np.float64)
    result = evaluate_predictions(y_true, y_prob)
    assert isinstance(result, EvalResult)
    assert result.auc_roc == 1.0
    assert result.accuracy == 1.0
    assert result.n_samples == 4


def test_random_predictions_auc_near_half() -> None:
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, size=10_000).astype(np.int64)
    y_prob = rng.uniform(0, 1, size=10_000).astype(np.float64)
    result = evaluate_predictions(y_true, y_prob)
    assert 0.45 < result.auc_roc < 0.55


def test_calibration_data_shape() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1], dtype=np.int64)
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)
    frac_pos, mean_pred = calibration_data(y_true, y_prob, n_bins=5)
    assert len(frac_pos) == len(mean_pred)
    assert len(frac_pos) <= 5
