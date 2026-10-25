"""二項比率の正確 CI と片側正確二項検定（g0_nonfood_plan.md §7.4、§8.2）。

- :func:`clopper_pearson`              Clopper–Pearson 正確区間。判定は片側 95%、報告は両側 95%（§7.4）
- :func:`binom_test_one_sided_greater` H0: p ≤ p0 に対する片側正確二項検定 P[X ≥ r | n, p0]（§8.2）
- :func:`wilson`                       Wilson 区間（補助。判定には使わない）

scipy.stats の beta 分位点と binom の生存関数を用いる。n = 0 のときは無情報区間 (0, 1) を返す。
"""

from __future__ import annotations

import math

from scipy.stats import beta, binom

__all__ = ["binom_test_one_sided_greater", "clopper_pearson", "wilson"]

_SIDES = ("two", "lower", "upper")


def _check_counts(k: int, n: int) -> None:
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"require 0 <= k <= n, got k={k}, n={n}")


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _cp_lower(k: int, n: int, q: float) -> float:
    """P[X ≥ k | p] = q を満たす p（k = 0 なら 0）。"""
    if k == 0:
        return 0.0
    return float(beta.ppf(q, k, n - k + 1))


def _cp_upper(k: int, n: int, q: float) -> float:
    """P[X ≤ k | p] = 1 - q を満たす p（k = n なら 1）。"""
    if k == n:
        return 1.0
    return float(beta.ppf(q, k + 1, n - k))


def clopper_pearson(k: int, n: int, alpha: float = 0.05, side: str = "two") -> tuple[float, float]:
    """Clopper–Pearson 正確区間を返す。

    Args:
        k: 成功数（0 ≤ k ≤ n）
        n: 試行数
        alpha: 有意水準。両側なら各裾 alpha/2、片側なら片裾 alpha
        side: ``"two"``（両側 1-alpha）／``"lower"``（片側下限。返り値は (lower, 1.0)）／
            ``"upper"``（片側上限。返り値は (0.0, upper)）

    Returns:
        (lower, upper)。n = 0 なら (0.0, 1.0)。
    """
    _check_counts(k, n)
    _check_alpha(alpha)
    if side not in _SIDES:
        raise ValueError(f"side must be one of {_SIDES}, got {side!r}")
    if n == 0:
        return 0.0, 1.0
    if side == "two":
        return _cp_lower(k, n, alpha / 2), _cp_upper(k, n, 1 - alpha / 2)
    if side == "lower":
        return _cp_lower(k, n, alpha), 1.0
    return 0.0, _cp_upper(k, n, 1 - alpha)


def binom_test_one_sided_greater(r: int, n: int, p0: float) -> float:
    """H0: p ≤ p0 に対する片側正確二項検定の p 値 P[X ≥ r | n, p0]。

    §8.2 の主検定（r ～ Binom(n_before, p)、H0: p ≤ 0.25）に用いる。
    n = 0 のときは検定不能として 1.0 を返す（§8.1 の評価不能は呼び出し側で判定する）。
    """
    _check_counts(r, n)
    if not 0.0 <= p0 <= 1.0:
        raise ValueError(f"p0 must be in [0, 1], got {p0}")
    if n == 0:
        return 1.0
    if r == 0:
        return 1.0
    return float(binom.sf(r - 1, n, p0))


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson スコア区間（両側 1-alpha）。補助用。n = 0 なら (0.0, 1.0)。"""
    _check_counts(k, n)
    _check_alpha(alpha)
    if n == 0:
        return 0.0, 1.0
    from scipy.stats import norm

    z = float(norm.ppf(1 - alpha / 2))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return lo, hi
