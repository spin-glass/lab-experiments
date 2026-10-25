"""cascade.gate.scorers — G0 の検出器（g0_nonfood_plan.md §5、design.md §6 G0）。

主構成 4 種（msp / knn / mahalanobis_pp / c3_binary）と副次 3 種（mahalanobis / c1_oe / c2_sixway）。
すべて :class:`~cascade.gate.scorers.base.Scorer` 契約（大 = 対象内、fit 前 score は NotFittedError、
seed 決定性、P1 / P2 両モード）を満たす。生成は :func:`make_scorer`。
"""

from __future__ import annotations

from typing import Any

from cascade.gate.scorers.base import (
    DEFAULT_FOOD_LABEL,
    MODES,
    BaseScorer,
    NotFittedError,
    Scorer,
    l2_normalize,
)
from cascade.gate.scorers.binary_head import BinaryHeadScorer
from cascade.gate.scorers.knn import KNNScorer
from cascade.gate.scorers.mahalanobis import MahalanobisScorer
from cascade.gate.scorers.msp import MSPScorer
from cascade.gate.scorers.oe_head import OEHeadScorer
from cascade.gate.scorers.sixway_head import OUT_LABEL, SixWayHeadScorer

PRIMARY_SCORERS: tuple[str, ...] = ("msp", "knn", "mahalanobis_pp", "c3_binary")
SECONDARY_SCORERS: tuple[str, ...] = ("mahalanobis", "c1_oe", "c2_sixway")
SCORER_NAMES: tuple[str, ...] = PRIMARY_SCORERS + SECONDARY_SCORERS
NEGATIVE_REQUIRING: tuple[str, ...] = ("c3_binary", "c1_oe", "c2_sixway")


def make_scorer(name: str, mode: str, **params: Any) -> Scorer:
    """名前とモードから検出器を作る。未知の名前は ValueError。

    params は各クラスのコンストラクタ引数（msp: tau / zeroshot / text_emb、knn: k / metric、
    mahalanobis*: ridge、c3_binary: C / max_iter、c1_oe: lam / epochs / lr / weight_decay、
    c2_sixway: C / max_iter / class_weight）。
    """
    if name == "msp":
        return MSPScorer(mode, **params)
    if name == "knn":
        return KNNScorer(mode, **params)
    if name == "mahalanobis_pp":
        return MahalanobisScorer(mode, normalize=True, **params)
    if name == "mahalanobis":
        return MahalanobisScorer(mode, normalize=False, **params)
    if name == "c3_binary":
        return BinaryHeadScorer(mode, **params)
    if name == "c1_oe":
        return OEHeadScorer(mode, **params)
    if name == "c2_sixway":
        return SixWayHeadScorer(mode, **params)
    raise ValueError(f"未知の検出器: {name!r}（候補: {SCORER_NAMES}）")


__all__ = [
    "DEFAULT_FOOD_LABEL",
    "MODES",
    "NEGATIVE_REQUIRING",
    "OUT_LABEL",
    "PRIMARY_SCORERS",
    "SCORER_NAMES",
    "SECONDARY_SCORERS",
    "BaseScorer",
    "BinaryHeadScorer",
    "KNNScorer",
    "MSPScorer",
    "MahalanobisScorer",
    "NotFittedError",
    "OEHeadScorer",
    "Scorer",
    "SixWayHeadScorer",
    "l2_normalize",
    "make_scorer",
]
