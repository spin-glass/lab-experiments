"""Holm の step-down 多重比較補正（g0_nonfood_plan.md §8.2）。

主構成 E1 × kNN × P1 は α = 0.05 で単独検定し、残り 7 構成を Holm（m = 7）で調整する。
本関数は渡された p 値の集合のみを補正するため、m は ``len(pvals)`` になる。主構成を除いた
辞書を渡すのは呼び出し側の責任。

- :func:`holm` 各構成に p / p_adj / reject / level（その構成に適用された名目水準 α/(m-i)）を返す
"""

from __future__ import annotations

__all__ = ["holm"]


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, float | bool]]:
    """Holm step-down 補正。

    昇順に並べた i 番目（0 始まり）の p 値を水準 α/(m-i) と比較し、最初に棄却できなかった時点で
    残りを全て非棄却とする。``p_adj`` は step-down 単調化した調整 p 値
    （p_adj_i = max_{j ≤ i} min(1, (m-j)·p_j)）で、``reject`` は ``p_adj ≤ alpha`` と同値。

    Returns:
        {name: {"p": p, "p_adj": p_adj, "reject": bool, "level": α/(m-i)}}。
        入力が空なら空辞書。
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    for name, p in pvals.items():
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-value for {name!r} must be in [0, 1], got {p}")
    m = len(pvals)
    if m == 0:
        return {}
    order = sorted(pvals.items(), key=lambda kv: (kv[1], kv[0]))
    out: dict[str, dict[str, float | bool]] = {}
    running = 0.0
    still_rejecting = True
    for i, (name, p) in enumerate(order):
        level = alpha / (m - i)
        running = max(running, min(1.0, (m - i) * p))
        reject = still_rejecting and p <= level
        if not reject:
            still_rejecting = False
        out[name] = {"p": float(p), "p_adj": float(running), "reject": bool(reject), "level": level}
    # 入力順で返す
    return {name: out[name] for name in pvals}
