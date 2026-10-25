"""McNemar 正確検定（g0_nonfood_plan.md §8.4「P1 で足りるか」の対応比較）。

同一 eval 層 A 上で P1 と P2 の棄却集合を対応比較する。不一致セル b（P1 のみ棄却）と
c（P2 のみ棄却）だけが情報を持ち、H0 のもとで b ～ Binom(b + c, 1/2)。

- :func:`mcnemar_exact` 正確二項検定に基づく p 値。両側は 2·min(両裾) を 1 で打ち切る
"""

from __future__ import annotations

from scipy.stats import binom

__all__ = ["mcnemar_exact"]

_ALTERNATIVES = ("two", "greater", "less")


def mcnemar_exact(b: int, c: int, alternative: str = "two") -> float:
    """不一致セル (b, c) に対する McNemar 正確検定の p 値。

    Args:
        b: 一方のみが陽性（例: P1 のみ棄却）の枚数
        c: 他方のみが陽性（例: P2 のみ棄却）の枚数
        alternative: ``"two"``／``"greater"``（H1: b の側が多い）／``"less"``（H1: c の側が多い）

    Returns:
        p 値。b + c = 0 なら 1.0。
    """
    if b < 0 or c < 0:
        raise ValueError(f"counts must be non-negative, got b={b}, c={c}")
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}, got {alternative!r}")
    n = b + c
    if n == 0:
        return 1.0
    if alternative == "greater":
        return float(binom.sf(b - 1, n, 0.5)) if b > 0 else 1.0
    if alternative == "less":
        return float(binom.cdf(b, n, 0.5))
    lo = min(b, c)
    return float(min(1.0, 2.0 * binom.cdf(lo, n, 0.5)))
