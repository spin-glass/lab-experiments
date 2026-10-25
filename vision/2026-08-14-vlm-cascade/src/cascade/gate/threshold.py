"""動作点（閾値）の決定と適用（g0_nonfood_plan.md §7.1、design.md §6 G0）。

- tune_cal の「正しく food」（Yelp ラベル food かつ P0 = food）のスコアを昇順に並べ、
  ``loss`` 位置の値 t を動作点とする。food 3,000・loss 1% なら 31 番目の最小値
  （``k_index = floor(loss * n) + 1``、1 始まり）
- 棄却は ``s < t``（同値は棄却しない）。したがって同値の有無に依らず
  ``achieved_loss = mean(scores < t) ≤ loss`` が成り立つ（テストで解析解と突き合わせる）
- 副の動作点は 0.5% / 2%（:data:`LOSSES_DEFAULT`）
- スコアは大 = 対象内（food）で統一（§5.3）。NaN を含む較正スコアは拒否し、適用側では
  NaN スコアを「確定できない」として棄却扱いにする

閾値の決定に eval を渡してはならない。役割のガードは :mod:`cascade.gate.splits` が担う。
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

import numpy as np

__all__ = [
    "LOSSES_DEFAULT",
    "MIN_N_WARN",
    "ThresholdResult",
    "apply_threshold",
    "fit_threshold",
    "fit_thresholds",
    "k_index_for",
]

LOSSES_DEFAULT: tuple[float, ...] = (0.01, 0.005, 0.02)
"""主 1%、副 0.5% / 2%（§7.1）。"""

MIN_N_WARN = 100
"""較正集合がこれ未満だと警告する（計算は行う）。1% 位置が 1 番目に落ちる規模。"""


@dataclass(frozen=True)
class ThresholdResult:
    """動作点の決定結果。

    - ``t``: 閾値（棄却は ``s < t``）
    - ``k_index``: 昇順で何番目の最小値を t としたか（1 始まり）
    - ``achieved_loss``: 較正集合で実際に棄却された率 ``mean(scores < t)``。常に ``≤ loss``
    - ``n``: 較正集合のサイズ
    - ``loss``: 目標損失
    - ``n_rejected``: 較正集合で棄却された枚数
    - ``n_ties_at_t``: t と同値のスコア枚数（1 以上。同値があると achieved_loss < loss になる）
    """

    t: float
    k_index: int
    achieved_loss: float
    n: int
    loss: float
    n_rejected: int
    n_ties_at_t: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def k_index_for(n: int, loss: float) -> int:
    """``floor(loss * n) + 1`` を返す（1 始まり）。

    ``loss`` は 10 進表記から有理数に直して掛けるので、``0.29 * 100 = 28.999…`` のような
    浮動小数の丸めで位置がずれない。
    """
    if n <= 0:
        raise ValueError(f"n は正でなければならない: {n}")
    if not (0.0 < loss < 1.0):
        raise ValueError(f"loss は (0, 1) の範囲: {loss}")
    exact = Fraction(repr(float(loss))) * n
    k = math.floor(exact) + 1
    return min(k, n)  # loss < 1 なので k ≤ n だが、契約として明示する


def _as_scores(scores: np.ndarray | Iterable[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    return arr


def fit_threshold(
    scores_correct_food: np.ndarray | Iterable[float], loss: float
) -> ThresholdResult:
    """較正スコア（正しく food）から動作点 t を決める。

    ``t`` は昇順 ``k_index = floor(loss * n) + 1`` 番目の値。棄却は ``s < t`` なので、
    スコアが全て相異なるとき丁度 ``k_index - 1 = floor(loss * n)`` 枚が棄却され、
    同値があるときはそれ以下になる。NaN を含む入力は拒否する。
    """
    arr = _as_scores(scores_correct_food)
    n = int(arr.size)
    if n == 0:
        raise ValueError("較正スコアが空")
    if np.isnan(arr).any():
        raise ValueError(f"較正スコアに NaN が {int(np.isnan(arr).sum())} 件ある")
    if n < MIN_N_WARN:
        warnings.warn(
            f"較正集合が小さい (n={n} < {MIN_N_WARN})。loss={loss} の位置は粗い",
            stacklevel=2,
        )
    k = k_index_for(n, loss)
    ordered = np.sort(arr, kind="stable")
    t = float(ordered[k - 1])
    rejected = ordered < t
    n_rejected = int(rejected.sum())
    achieved = n_rejected / n
    n_ties = int((ordered == t).sum())
    # 契約: 同値の扱いにより達成損失は目標を超えない
    assert n_rejected <= math.floor(Fraction(repr(float(loss))) * n), (n_rejected, loss, n)
    assert achieved <= loss + 1e-12, (achieved, loss)
    return ThresholdResult(
        t=t,
        k_index=k,
        achieved_loss=float(achieved),
        n=n,
        loss=float(loss),
        n_rejected=n_rejected,
        n_ties_at_t=n_ties,
    )


def fit_thresholds(
    scores_correct_food: np.ndarray | Iterable[float],
    losses: Iterable[float] = LOSSES_DEFAULT,
) -> dict[float, ThresholdResult]:
    """主・副の動作点をまとめて決める（loss → 結果）。"""
    arr = _as_scores(scores_correct_food)
    return {float(loss): fit_threshold(arr, float(loss)) for loss in losses}


def apply_threshold(scores: np.ndarray | Iterable[float], t: float) -> np.ndarray:
    """棄却マスク（True = 棄却）。``s < t`` を棄却し、NaN も棄却する。

    NaN は検出器がスコアを出せなかった画像であり、「food として確定しない」側に倒す。
    """
    arr = _as_scores(scores)
    if not np.isfinite(t):
        raise ValueError(f"閾値が有限でない: {t}")
    with np.errstate(invalid="ignore"):
        rejected = arr < float(t)
    return rejected | np.isnan(arr)
