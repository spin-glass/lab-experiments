"""主指標の比率と CI（g0_nonfood_plan.md §7.2、§7.4）。

各比率は :class:`Rate`（k, n, p、両側 95% CP、片側 95% 下限／上限）として返す。
判定は片側 95%、報告は両側 95% の Clopper–Pearson（§7.4）。

- :func:`false_acceptance`   非料理→food 誤受理率 = 通過して food 確定した枚数 / 層 A eval 総数
- :func:`reduction`          誤混入削減率 r / n_before（n_before を条件付けた二項比率）
- :func:`food_loss_nominal`  料理損失（名目）= 正しく food のうち棄却された率
- :func:`food_loss_confirmed` 料理損失（確認済み）R_true / (R_true + N_pass·q̂) とデルタ法 CI
- :func:`pass_rate`          通過率（層 B の P2 通過率など。片側 95% 下限で判定）
- :func:`other_class_frr`    P2 の他クラス FRR（名目）

確認済み損失のデルタ法: L = f(R, q) = R / (R + N q)。R は棄却された名目 food の全数目視で
得る計数（Var(R) ≈ R(1 − L)）、q̂ は通過側無作為 n_q 枚の content = food 率
（Var(q̂) = q̂(1 − q̂)/n_q）。両者を独立として
Var(L) ≈ (∂f/∂R)²·Var(R) + (∂f/∂q)²·Var(q̂)。画像ブートストラップは
:mod:`cascade.metrics.bootstrap` に置く。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from scipy.stats import norm

from cascade.metrics.binom import clopper_pearson

__all__ = [
    "Rate",
    "false_acceptance",
    "food_loss_confirmed",
    "food_loss_nominal",
    "other_class_frr",
    "pass_rate",
    "rate",
    "reduction",
]

ALPHA = 0.05


@dataclass(frozen=True)
class Rate:
    """二項比率 k/n と Clopper–Pearson 区間。

    Attributes:
        k: 分子
        n: 分母
        p: 点推定 k/n（n = 0 なら nan）
        ci: 両側 95% CP 区間（報告用）
        lower95: 片側 95% 下限（判定用）
        upper95: 片側 95% 上限（判定用）
    """

    k: int
    n: int
    p: float
    ci: tuple[float, float]
    lower95: float
    upper95: float
    name: str = field(default="", compare=False)

    # 契約上の別名
    @property
    def ci_two_95(self) -> tuple[float, float]:
        return self.ci

    @property
    def ci_lower_one_sided_95(self) -> float:
        return self.lower95

    @property
    def ci_upper_one_sided_95(self) -> float:
        return self.upper95

    @property
    def evaluable(self) -> bool:
        return self.n > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "k": self.k,
            "n": self.n,
            "p": self.p,
            "ci_lo": self.ci[0],
            "ci_hi": self.ci[1],
            "lower95": self.lower95,
            "upper95": self.upper95,
        }


def rate(k: int, n: int, name: str = "", alpha: float = ALPHA) -> Rate:
    """k/n を :class:`Rate` にする。"""
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"require 0 <= k <= n, got k={k}, n={n}")
    p = k / n if n > 0 else math.nan
    ci = clopper_pearson(k, n, alpha, "two")
    lo, _ = clopper_pearson(k, n, alpha, "lower")
    _, hi = clopper_pearson(k, n, alpha, "upper")
    return Rate(k=k, n=n, p=p, ci=ci, lower95=lo, upper95=hi, name=name)


def false_acceptance(n_pass_food: int, n_total: int) -> Rate:
    """非料理→food 誤受理率（§7.2）。P0 と P1/P2 を並記する。"""
    return rate(n_pass_food, n_total, "false_acceptance")


def reduction(r: int, n_before: int) -> Rate:
    """誤混入削減率 r / n_before。n_before は P0 が food 確定した層 A 枚数（§7.2、§7.4）。"""
    return rate(r, n_before, "reduction")


def food_loss_nominal(rejected: int, total_correct_food: int) -> Rate:
    """料理損失（名目）。Yelp ラベル基準の「正しく food」のうち棄却された率。"""
    return rate(rejected, total_correct_food, "food_loss_nominal")


def pass_rate(passed: int, total: int) -> Rate:
    """通過率。層 B の P2 通過率（片側 95% 下限で判定、§8.3 (d)）など。"""
    return rate(passed, total, "pass_rate")


def other_class_frr(rejected: int, total_correct: int) -> Rate:
    """P2 の他クラス FRR（名目）。片側 95% 上限 ≤ 3% で判定（§8.3 (d)）。"""
    return rate(rejected, total_correct, "other_class_frr")


def food_loss_confirmed(
    r_true: int, n_pass: int, q_hat: float, n_q: int, alpha: float = ALPHA
) -> dict[str, Any]:
    """料理損失（確認済み）R_true / (R_true + N_pass·q̂) とデルタ法 CI（§7.2）。

    Args:
        r_true: 棄却された名目 food のうち目視で content = food だった枚数（全数目視）
        n_pass: 通過した名目 food の枚数
        q_hat: 通過側無作為 n_q 枚の content = food 率
        n_q: q̂ の推定に用いた枚数（200、標準 300）

    Returns:
        ``point``、``se``、``ci_delta_95``（両側 1-alpha）、``ci_upper_one_sided_95``、
        ``ci_lower_one_sided_95``、および入力値。n_q = 0 または分母 0 のときは nan。
    """
    if r_true < 0 or n_pass < 0 or n_q < 0:
        raise ValueError("counts must be non-negative")
    if not 0.0 <= q_hat <= 1.0:
        raise ValueError(f"q_hat must be in [0, 1], got {q_hat}")
    denom = r_true + n_pass * q_hat
    base: dict[str, Any] = {
        "r_true": r_true,
        "n_pass": n_pass,
        "q_hat": q_hat,
        "n_q": n_q,
        "true_food_total": denom,
    }
    if denom <= 0 or n_q == 0:
        return {
            **base,
            "point": math.nan,
            "se": math.nan,
            "ci_delta_95": (math.nan, math.nan),
            "ci_lower_one_sided_95": math.nan,
            "ci_upper_one_sided_95": math.nan,
        }
    point = r_true / denom
    d_r = n_pass * q_hat / denom**2
    d_q = -r_true * n_pass / denom**2
    var_r = r_true * (1 - point)
    var_q = q_hat * (1 - q_hat) / n_q
    var = d_r**2 * var_r + d_q**2 * var_q
    se = math.sqrt(max(var, 0.0))
    z2 = float(norm.ppf(1 - alpha / 2))
    z1 = float(norm.ppf(1 - alpha))
    return {
        **base,
        "point": point,
        "se": se,
        "ci_delta_95": (max(0.0, point - z2 * se), min(1.0, point + z2 * se)),
        "ci_lower_one_sided_95": max(0.0, point - z1 * se),
        "ci_upper_one_sided_95": min(1.0, point + z1 * se),
    }
