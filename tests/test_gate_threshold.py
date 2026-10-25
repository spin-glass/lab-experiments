"""gate.threshold: 動作点の解析解・同値処理・小 n・NaN（g0_nonfood_plan.md §7.1）。"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from cascade.gate.threshold import (
    LOSSES_DEFAULT,
    ThresholdResult,
    apply_threshold,
    fit_threshold,
    fit_thresholds,
    k_index_for,
)


def _distinct(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(np.linspace(-3.0, 3.0, n))


# ---------------------------------------------------------------------------
# 解析解
# ---------------------------------------------------------------------------


def test_k_index_3000_at_1pct_is_31():
    assert k_index_for(3000, 0.01) == 31
    assert k_index_for(3000, 0.005) == 16
    assert k_index_for(3000, 0.02) == 61


@pytest.mark.parametrize(
    ("n", "loss", "k"),
    [(100, 0.01, 2), (100, 0.29, 30), (100, 0.07, 8), (3001, 0.01, 31), (250, 0.004, 2)],
)
def test_k_index_is_floor_plus_one_without_float_drift(n, loss, k):
    # 0.29 * 100 = 28.999… の丸めで 29 にずれないこと
    assert k_index_for(n, loss) == k


def test_fit_threshold_n3000_distinct_rejects_exactly_30():
    scores = _distinct(3000)
    res = fit_threshold(scores, 0.01)
    assert isinstance(res, ThresholdResult)
    ordered = np.sort(scores)
    assert res.k_index == 31
    assert res.t == ordered[30]  # 31 番目の最小値
    rejected = apply_threshold(scores, res.t)
    assert int(rejected.sum()) == 30
    assert res.n_rejected == 30
    assert res.achieved_loss == pytest.approx(30 / 3000)
    assert res.n == 3000
    assert res.n_ties_at_t == 1
    # 棄却されたのは最小 30 個
    assert set(np.flatnonzero(rejected)) == set(np.argsort(scores)[:30])


@pytest.mark.parametrize(("loss", "n_rej"), [(0.005, 15), (0.02, 60)])
def test_secondary_losses(loss, n_rej):
    scores = _distinct(3000, seed=1)
    res = fit_threshold(scores, loss)
    assert res.k_index == n_rej + 1
    assert int(apply_threshold(scores, res.t).sum()) == n_rej
    assert res.achieved_loss == pytest.approx(n_rej / 3000)


def test_fit_thresholds_covers_default_losses():
    scores = _distinct(3000, seed=2)
    out = fit_thresholds(scores)
    assert set(out) == set(LOSSES_DEFAULT)
    assert out[0.005].t <= out[0.01].t <= out[0.02].t


def test_to_dict_roundtrip_keys():
    res = fit_threshold(_distinct(500), 0.01)
    d = res.to_dict()
    assert set(d) == {"t", "k_index", "achieved_loss", "n", "loss", "n_rejected", "n_ties_at_t"}


# ---------------------------------------------------------------------------
# 同値: 棄却は s < t のみ。達成損失は目標を超えない
# ---------------------------------------------------------------------------


def test_ties_at_threshold_are_not_rejected():
    # 最小 40 個が同値 0.0、残りは相異なる正の値。1% 位置（31 番目）は同値の中
    scores = np.concatenate([np.zeros(40), np.linspace(1.0, 2.0, 2960)])
    res = fit_threshold(scores, 0.01)
    assert res.t == 0.0
    assert res.k_index == 31
    assert res.n_rejected == 0
    assert res.achieved_loss == 0.0
    assert res.n_ties_at_t == 40
    assert not apply_threshold(scores, res.t).any()


def test_partial_ties_reduce_but_never_exceed_target():
    # 20 個の -1.0 と 20 個の 0.0、残り正。31 番目 = 0.0 → 棄却は -1.0 の 20 個のみ
    scores = np.concatenate([np.full(20, -1.0), np.zeros(20), np.linspace(1.0, 2.0, 2960)])
    res = fit_threshold(scores, 0.01)
    assert res.t == 0.0
    assert res.n_rejected == 20
    assert res.achieved_loss <= 0.01


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("loss", [0.005, 0.01, 0.02, 0.29])
def test_achieved_loss_never_exceeds_target_with_random_ties(seed, loss):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(100, 2000))
    # 少数の離散値に丸めて同値を大量に作る
    scores = np.round(rng.standard_normal(n), 1)
    res = fit_threshold(scores, loss)
    assert res.achieved_loss <= loss
    assert res.n_rejected <= math.floor(loss * n) + 1e-9
    assert int(apply_threshold(scores, res.t).sum()) == res.n_rejected
    assert res.achieved_loss == pytest.approx(np.mean(scores < res.t))


def test_scores_order_does_not_matter():
    scores = _distinct(1000, seed=3)
    a = fit_threshold(scores, 0.01)
    b = fit_threshold(np.sort(scores)[::-1], 0.01)
    assert a == b


# ---------------------------------------------------------------------------
# 小 n・入力検証
# ---------------------------------------------------------------------------


def test_small_n_warns_but_computes():
    scores = _distinct(50)
    with pytest.warns(UserWarning, match="較正集合が小さい"):
        res = fit_threshold(scores, 0.01)
    assert res.k_index == 1  # floor(0.5) + 1
    assert res.t == np.min(scores)
    assert res.n_rejected == 0


def test_n_1_edge():
    with pytest.warns(UserWarning):
        res = fit_threshold(np.array([0.7]), 0.01)
    assert res.t == 0.7 and res.k_index == 1 and res.achieved_loss == 0.0


def test_n_100_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = fit_threshold(_distinct(100), 0.01)
    assert res.k_index == 2 and res.n_rejected == 1


def test_high_loss_k_never_exceeds_n():
    scores = _distinct(100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fit_threshold(scores, 0.999)
    assert res.k_index == 100
    assert res.t == np.max(scores)
    assert res.n_rejected == 99


@pytest.mark.parametrize("loss", [0.0, 1.0, -0.1, 1.5])
def test_invalid_loss_raises(loss):
    with pytest.raises(ValueError, match="loss"):
        fit_threshold(_distinct(200), loss)


def test_empty_scores_raise():
    with pytest.raises(ValueError, match="空"):
        fit_threshold(np.array([]), 0.01)


def test_nan_in_calibration_is_rejected():
    scores = _distinct(300)
    scores[7] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fit_threshold(scores, 0.01)


def test_apply_threshold_rejects_nan_and_strict_less():
    scores = np.array([np.nan, 0.5, 1.0, 1.5, -np.inf, np.inf])
    rej = apply_threshold(scores, 1.0)
    assert rej.dtype == bool
    assert rej.tolist() == [True, True, False, False, True, False]


def test_apply_threshold_non_finite_t_raises():
    with pytest.raises(ValueError):
        apply_threshold(np.array([0.1]), float("nan"))


def test_accepts_lists_and_2d_inputs():
    res = fit_threshold([0.1, 0.2, 0.3] * 100, 0.01)
    assert res.n == 300
    assert apply_threshold([[0.05, 0.5]], res.t).tolist() == [True, False]
