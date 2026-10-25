"""記述分析（g0_nonfood_plan.md §6）と副指標（§7.3）の分布要約。

スコアの向きは「大 = 対象内（food）」で統一する（§5.3）。距離は大 = 遠い。

- :func:`auroc`               陽性（対象内）スコア vs 陰性（層 A）スコアの AUROC（同値は 0.5）
- :func:`fpr_at_tpr`          陽性の TPR を tpr 以上に保つ最大の閾値での陰性の通過率（FPR@95）
- :func:`overlap_coefficient` ヒストグラム質量の重なり係数 Σ min(p_a, p_b)
- :func:`kl_divergence_hist`  ヒストグラム質量の KL(p_a ‖ p_b)（eps で平滑化）
- :func:`quadrant_counts`     p(food) × 距離の 4 象限表（境界は Stage1 argmax 相当の p_thr と動作点 d_thr）
- :func:`histogram_table`     群別ヒストグラムを縦持ち polars.DataFrame にする

ビン幅は事前に固定し §12 に記録する（§6）。ビンの外側の値は両端ビンに寄せて数える（外れ値を
落とさない）。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import polars as pl
from scipy.stats import rankdata

__all__ = [
    "auroc",
    "fpr_at_tpr",
    "histogram_table",
    "kl_divergence_hist",
    "overlap_coefficient",
    "quadrant_counts",
]


def _as_1d(x: object) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size and not np.all(np.isfinite(a)):
        raise ValueError("scores must be finite")
    return a


def auroc(pos_scores: object, neg_scores: object) -> float:
    """AUROC = P[s_pos > s_neg] + 0.5·P[s_pos = s_neg]（Mann–Whitney）。

    陽性 = 対象内（P1 は正しく food、P2 は対象内全体）、陰性 = 層 A eval（§7.3）。
    どちらかが空なら nan。
    """
    pos, neg = _as_1d(pos_scores), _as_1d(neg_scores)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def fpr_at_tpr(pos_scores: object, neg_scores: object, tpr: float = 0.95) -> float:
    """TPR ≥ tpr を満たす最大の閾値 t（受理 = s ≥ t）における陰性の受理率。

    t は陽性スコアの昇順で floor((1 − tpr)·n_pos) 番目（0 始まり）の値。tpr に対して単調非減少。
    """
    if not 0.0 < tpr <= 1.0:
        raise ValueError(f"tpr must be in (0, 1], got {tpr}")
    pos, neg = _as_1d(pos_scores), _as_1d(neg_scores)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    srt = np.sort(pos)
    idx = int(np.floor((1.0 - tpr) * pos.size))
    idx = min(max(idx, 0), pos.size - 1)
    t = srt[idx]
    return float(np.mean(neg >= t))


def _mass(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """ビン質量（Σ = 1）。範囲外は両端ビンに寄せる。空なら零ベクトル。"""
    if values.size == 0:
        return np.zeros(bins.size - 1, dtype=np.float64)
    clipped = np.clip(values, bins[0], bins[-1])
    counts, _ = np.histogram(clipped, bins=bins)
    return counts.astype(np.float64) / counts.sum()


def _check_bins(bins: object) -> np.ndarray:
    b = np.asarray(bins, dtype=np.float64).ravel()
    if b.size < 2 or np.any(np.diff(b) <= 0):
        raise ValueError("bins must be strictly increasing with at least 2 edges")
    return b


def overlap_coefficient(a: object, b: object, bins: object) -> float:
    """重なり係数 Σ_i min(p_a,i, p_b,i) ∈ [0, 1]。同一分布で 1、分離で 0。"""
    edges = _check_bins(bins)
    pa, pb = _mass(_as_1d(a), edges), _mass(_as_1d(b), edges)
    return float(np.minimum(pa, pb).sum())


def kl_divergence_hist(a: object, b: object, bins: object, eps: float = 1e-9) -> float:
    """KL(p_a ‖ p_b)（自然対数）。両質量に eps を足して再正規化してから計算する。"""
    if eps <= 0:
        raise ValueError("eps must be positive")
    edges = _check_bins(bins)
    pa, pb = _mass(_as_1d(a), edges) + eps, _mass(_as_1d(b), edges) + eps
    pa /= pa.sum()
    pb /= pb.sum()
    return float(np.sum(pa * np.log(pa / pb)))


def quadrant_counts(p_food: object, distance: object, p_thr: float, d_thr: float) -> dict[str, int]:
    """p(food) × 距離の 4 象限の枚数。

    high = p_food ≥ p_thr、far = distance > d_thr（動作点で棄却 s < t と同値。距離 = −s、
    d_thr = −t を渡す）。返り値は ``high_far`` / ``high_near`` / ``low_far`` / ``low_near`` / ``n``。
    """
    p, d = _as_1d(p_food), _as_1d(distance)
    if p.size != d.size:
        raise ValueError(f"length mismatch: {p.size} vs {d.size}")
    high = p >= p_thr
    far = d > d_thr
    return {
        "high_far": int(np.sum(high & far)),
        "high_near": int(np.sum(high & ~far)),
        "low_far": int(np.sum(~high & far)),
        "low_near": int(np.sum(~high & ~far)),
        "n": int(p.size),
    }


def histogram_table(values_by_group: Mapping[str, np.ndarray], bins: np.ndarray) -> pl.DataFrame:
    """群別ヒストグラムの縦持ち表。

    列: group, bin_index, bin_left, bin_right, count, frac, n。count の群内合計は n に一致する
    （範囲外は両端ビンに寄せる）。
    """
    edges = _check_bins(bins)
    rows: list[dict[str, object]] = []
    for group, vals in values_by_group.items():
        v = _as_1d(vals)
        if v.size:
            counts, _ = np.histogram(np.clip(v, edges[0], edges[-1]), bins=edges)
        else:
            counts = np.zeros(edges.size - 1, dtype=np.int64)
        n = int(v.size)
        for i, c in enumerate(counts.tolist()):
            rows.append(
                {
                    "group": str(group),
                    "bin_index": i,
                    "bin_left": float(edges[i]),
                    "bin_right": float(edges[i + 1]),
                    "count": int(c),
                    "frac": (c / n) if n else 0.0,
                    "n": n,
                }
            )
    schema = {
        "group": pl.Utf8,
        "bin_index": pl.Int64,
        "bin_left": pl.Float64,
        "bin_right": pl.Float64,
        "count": pl.Int64,
        "frac": pl.Float64,
        "n": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema, orient="row")
