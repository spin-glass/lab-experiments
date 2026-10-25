"""cascade.metrics.binom: Clopper–Pearson・片側正確二項検定の既知値と境界（plan §7.4、§8.2）。"""

from __future__ import annotations

import pytest
from scipy.stats import binom

from cascade.metrics.binom import binom_test_one_sided_greater, clopper_pearson, wilson


def test_cp_two_sided_known_value_20_of_2000() -> None:
    lo, hi = clopper_pearson(20, 2000)
    assert lo == pytest.approx(0.0061, abs=5e-5)
    assert hi == pytest.approx(0.0154, abs=5e-5)


def test_cp_one_sided_upper_zero_of_100() -> None:
    lo, hi = clopper_pearson(0, 100, side="upper")
    assert lo == 0.0
    assert hi == pytest.approx(0.0295, abs=5e-5)
    # 解析解 1 − 0.05^(1/n)
    assert hi == pytest.approx(1 - 0.05 ** (1 / 100), rel=1e-9)


def test_cp_one_sided_lower_returns_one_as_upper() -> None:
    lo, hi = clopper_pearson(50, 100, side="lower")
    assert hi == 1.0
    assert 0.4 < lo < 0.5
    # 片側 95% 下限は両側 90% 下限と一致（§8.2 の同値表現）
    lo90, _ = clopper_pearson(50, 100, alpha=0.10)
    assert lo == pytest.approx(lo90, rel=1e-12)


def test_cp_boundaries_and_ordering() -> None:
    assert clopper_pearson(0, 10)[0] == 0.0
    assert clopper_pearson(10, 10)[1] == 1.0
    assert clopper_pearson(0, 0) == (0.0, 1.0)
    lo2, hi2 = clopper_pearson(7, 30)
    lo1, _ = clopper_pearson(7, 30, side="lower")
    _, hi1 = clopper_pearson(7, 30, side="upper")
    assert lo2 < lo1 < 7 / 30 < hi1 < hi2


def test_cp_coverage_property_at_bounds() -> None:
    # 上限 p_U では P[X ≤ k | p_U] = alpha/2、下限 p_L では P[X ≥ k | p_L] = alpha/2
    k, n = 20, 2000
    lo, hi = clopper_pearson(k, n)
    assert binom.cdf(k, n, hi) == pytest.approx(0.025, rel=1e-6)
    assert binom.sf(k - 1, n, lo) == pytest.approx(0.025, rel=1e-6)


@pytest.mark.parametrize(
    ("k", "n", "alpha", "side"),
    [(-1, 10, 0.05, "two"), (11, 10, 0.05, "two"), (1, 10, 0.0, "two"), (1, 10, 0.05, "left")],
)
def test_cp_invalid_inputs(k: int, n: int, alpha: float, side: str) -> None:
    with pytest.raises(ValueError):
        clopper_pearson(k, n, alpha, side)


def test_binom_test_known_value() -> None:
    assert binom_test_one_sided_greater(8, 8, 0.5) == pytest.approx(0.0039, abs=5e-5)
    assert binom_test_one_sided_greater(8, 8, 0.5) == pytest.approx(1 / 256, rel=1e-12)


def test_binom_test_primary_hypothesis_h0_quarter() -> None:
    # §8.2: H0: p ≤ 0.25。r = n_before/2 で強く棄却、r = n_before/4 で棄却できない
    assert binom_test_one_sided_greater(20, 40, 0.25) < 0.001
    assert binom_test_one_sided_greater(10, 40, 0.25) > 0.5
    assert binom_test_one_sided_greater(0, 40, 0.25) == 1.0
    assert binom_test_one_sided_greater(0, 0, 0.25) == 1.0


def test_binom_test_matches_cp_lower_bound_equivalence() -> None:
    # 「片側 CP 下限 ≥ p0」と「片側検定 p ≤ alpha」は同値
    for r, n in [(15, 40), (16, 40), (20, 40), (10, 40)]:
        lo, _ = clopper_pearson(r, n, alpha=0.05, side="lower")
        p = binom_test_one_sided_greater(r, n, 0.25)
        assert (lo >= 0.25) == (p <= 0.05)


def test_wilson_contains_point_and_is_inside_unit() -> None:
    lo, hi = wilson(20, 2000)
    assert 0.0 <= lo < 0.01 < hi <= 1.0
    assert wilson(0, 0) == (0.0, 1.0)
    assert wilson(0, 100)[0] == 0.0
    assert wilson(100, 100)[1] == 1.0
