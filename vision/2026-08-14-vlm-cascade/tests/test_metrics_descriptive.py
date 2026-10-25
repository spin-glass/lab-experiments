"""cascade.metrics.descriptive: AUROC・FPR@TPR・重なり係数・KL・4 象限・ヒストグラム表（plan §6、§7.3）。"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from cascade.metrics.descriptive import (
    auroc,
    fpr_at_tpr,
    histogram_table,
    kl_divergence_hist,
    overlap_coefficient,
    quadrant_counts,
)


def test_auroc_separable_identical_and_ties() -> None:
    pos = np.array([2.0, 3.0, 4.0])
    neg = np.array([-1.0, 0.0, 1.0])
    assert auroc(pos, neg) == 1.0
    assert auroc(neg, pos) == 0.0
    same = np.arange(10, dtype=float)
    assert auroc(same, same) == pytest.approx(0.5)
    assert auroc([1.0, 1.0], [1.0, 1.0]) == pytest.approx(0.5)
    assert math.isnan(auroc([], [1.0]))


def test_auroc_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    pos = rng.normal(1.0, 1.0, 300)
    neg = rng.normal(0.0, 1.0, 200)
    brute = np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :])
    assert auroc(pos, neg) == pytest.approx(float(brute), rel=1e-12)


def test_fpr_at_tpr_monotone_and_bounds() -> None:
    rng = np.random.default_rng(1)
    pos = rng.normal(1.0, 1.0, 1000)
    neg = rng.normal(0.0, 1.0, 1000)
    tprs = [0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
    vals = [fpr_at_tpr(pos, neg, t) for t in tprs]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert all(a <= b for a, b in zip(vals[:-1], vals[1:], strict=True))
    # 完全分離なら FPR@95 = 0
    assert fpr_at_tpr([5.0, 6.0, 7.0], [0.0, 1.0]) == 0.0
    # TPR 制約が実際に満たされている
    srt = np.sort(pos)
    t = srt[int(np.floor(0.05 * pos.size))]
    assert np.mean(pos >= t) >= 0.95
    with pytest.raises(ValueError):
        fpr_at_tpr(pos, neg, 0.0)


def test_overlap_and_kl_identical_and_disjoint() -> None:
    bins = np.linspace(0.0, 1.0, 11)
    a = np.linspace(0.0, 1.0, 100)
    assert overlap_coefficient(a, a, bins) == pytest.approx(1.0)
    assert kl_divergence_hist(a, a, bins) == pytest.approx(0.0, abs=1e-12)
    left = np.full(50, 0.05)
    right = np.full(50, 0.95)
    assert overlap_coefficient(left, right, bins) == pytest.approx(0.0)
    assert kl_divergence_hist(left, right, bins) > 10.0
    # 範囲外は両端ビンに寄せるので落ちない
    assert overlap_coefficient([-5.0, 5.0], [0.0, 1.0], bins) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        overlap_coefficient(a, a, [1.0, 0.0])


def test_quadrant_counts_sums_and_boundaries() -> None:
    p = np.array([0.9, 0.9, 0.1, 0.1, 0.5])
    d = np.array([2.0, 0.0, 2.0, 0.0, 1.0])
    q = quadrant_counts(p, d, p_thr=0.5, d_thr=1.0)
    assert q["n"] == 5
    assert q["high_far"] + q["high_near"] + q["low_far"] + q["low_near"] == 5
    # high は ≥、far は > （境界の 0.5 / 1.0 は high_near）
    assert q == {"high_far": 1, "high_near": 2, "low_far": 1, "low_near": 1, "n": 5}
    with pytest.raises(ValueError):
        quadrant_counts([0.1], [0.1, 0.2], 0.5, 0.5)


def test_histogram_table_shape_and_counts() -> None:
    bins = np.array([0.0, 0.5, 1.0])
    df = histogram_table(
        {"food": np.array([0.1, 0.2, 0.7]), "layer_a": np.array([]), "b": [2.0]}, bins
    )
    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["group", "bin_index", "bin_left", "bin_right", "count", "frac", "n"]
    assert df.height == 3 * 2
    food = df.filter(pl.col("group") == "food")
    assert food["count"].to_list() == [2, 1]
    assert food["frac"].to_list() == pytest.approx([2 / 3, 1 / 3])
    assert food["n"].to_list() == [3, 3]
    assert df.filter(pl.col("group") == "layer_a")["count"].sum() == 0
    # 範囲外 (2.0) は右端ビンへ
    assert df.filter(pl.col("group") == "b")["count"].to_list() == [0, 1]
    totals = df.group_by("group").agg(pl.col("count").sum(), pl.col("n").first())
    assert (totals["count"] == totals["n"]).all()
