"""ブートストラップとデルタ法の補助（g0_nonfood_plan.md §7.2、§7.4）。

- :func:`cluster_bootstrap`             店舗（business_id）単位ブートストラップ。Yelp 側指標の感度分析
                                        （判定には使わない、§7.4）
- :func:`delta_method_ratio`            比 a/b のデルタ法 SE
- :func:`food_loss_confirmed_bootstrap` 確認済み損失のパラメトリック（画像）ブートストラップ。
                                        デルタ法（:func:`cascade.metrics.rates.food_loss_confirmed`）
                                        の突き合わせ用

乱数は ``numpy.random.default_rng(seed)`` で固定し、決定的に動く。
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

__all__ = ["cluster_bootstrap", "delta_method_ratio", "food_loss_confirmed_bootstrap"]

_STATS = ("ratio", "mean_of_ratios")


def cluster_bootstrap(
    values_by_cluster: Mapping[str, tuple[int, int]],
    stat: str = "ratio",
    B: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """クラスタ（店舗）単位の再標本化によるパーセンタイル CI。

    Args:
        values_by_cluster: {cluster_id: (k, n)}。k = 事象数、n = クラスタ内の枚数
        stat: ``"ratio"``（Σk/Σn、比率のプール推定）／``"mean_of_ratios"``（クラスタ別比率の平均）
        B: 再標本化回数
        seed: 乱数 seed
        alpha: 両側水準

    Returns:
        (point, lo, hi)。クラスタが無ければ (nan, nan, nan)。
    """
    if stat not in _STATS:
        raise ValueError(f"stat must be one of {_STATS}, got {stat!r}")
    if B < 1:
        raise ValueError("B must be >= 1")
    ids = sorted(values_by_cluster)
    if not ids:
        return math.nan, math.nan, math.nan
    k = np.asarray([values_by_cluster[i][0] for i in ids], dtype=np.float64)
    n = np.asarray([values_by_cluster[i][1] for i in ids], dtype=np.float64)
    if np.any(n < 0) or np.any(k < 0) or np.any(k > n):
        raise ValueError("require 0 <= k <= n for every cluster")
    c = len(ids)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, c, size=(B, c))

    def _stat(kk: np.ndarray, nn: np.ndarray) -> np.ndarray:
        if stat == "ratio":
            num = kk.sum(axis=-1)
            den = nn.sum(axis=-1)
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(den > 0, num / den, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(nn > 0, kk / nn, np.nan)
        return np.nanmean(r, axis=-1)

    point = float(_stat(k, n))
    draws = _stat(k[idx], n[idx])
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return point, math.nan, math.nan
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def delta_method_ratio(
    num: float, den: float, var_num: float, var_den: float, cov: float = 0.0
) -> tuple[float, float]:
    """比 R = num/den の点推定とデルタ法 SE。

    Var(R) ≈ Var(num)/den² + num²·Var(den)/den⁴ − 2·num·Cov/den³。den ≤ 0 なら (nan, nan)。
    """
    if den <= 0:
        return math.nan, math.nan
    point = num / den
    var = var_num / den**2 + num**2 * var_den / den**4 - 2 * num * cov / den**3
    return point, math.sqrt(max(var, 0.0))


def food_loss_confirmed_bootstrap(
    r_true: int,
    n_pass: int,
    q_hat: float,
    n_q: int,
    B: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float | tuple[float, float]]:
    """確認済み損失 L = R/(R + N·q̂) のパラメトリック（画像）ブートストラップ。

    真の food 総数 T = R + N·q̂、L = R/T として、各反復で
    R* ～ Binom(round(T), L)、q* ～ Binom(n_q, q̂)/n_q を引き L* = R*/(R* + N·q*) を得る。

    Returns:
        ``point``、``ci_95``（両側パーセンタイル）、``upper_one_sided_95``、``lower_one_sided_95``。
    """
    if r_true < 0 or n_pass < 0 or n_q <= 0 or B < 1:
        raise ValueError("invalid counts")
    if not 0.0 <= q_hat <= 1.0:
        raise ValueError(f"q_hat must be in [0, 1], got {q_hat}")
    total = r_true + n_pass * q_hat
    if total <= 0:
        nan = math.nan
        return {
            "point": nan,
            "ci_95": (nan, nan),
            "upper_one_sided_95": nan,
            "lower_one_sided_95": nan,
        }
    point = r_true / total
    rng = np.random.default_rng(seed)
    t_int = max(int(round(total)), 1)
    r_star = rng.binomial(t_int, point, size=B).astype(np.float64)
    q_star = rng.binomial(n_q, q_hat, size=B).astype(np.float64) / n_q
    den = r_star + n_pass * q_star
    with np.errstate(invalid="ignore", divide="ignore"):
        l_star = np.where(den > 0, r_star / den, np.nan)
    l_star = l_star[np.isfinite(l_star)]
    lo, hi = np.quantile(l_star, [alpha / 2, 1 - alpha / 2])
    return {
        "point": point,
        "ci_95": (float(lo), float(hi)),
        "lower_one_sided_95": float(np.quantile(l_star, alpha)),
        "upper_one_sided_95": float(np.quantile(l_star, 1 - alpha)),
    }
