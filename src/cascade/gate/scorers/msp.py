"""MSP（MCM）検出器（g0_nonfood_plan.md §5.1 表 1 行目）。

- P2 スコア = max softmax（τ = 1 が主。τ = 0.01 はアブレーション §5.2）
- P1 スコア = p(food)
- softmax は Stage1 zero-shot（:class:`cascade.stage1.zeroshot.ZeroShot`）の確率そのもの。
  fit ではテキスト埋め込みを **学習しない**（凍結）。fit の役割は class_order の整合確認と
  P0 と同じ確率を返すことの保証だけ
- テキスト側は ``zeroshot=``（ZeroShot 互換: ``probs(X, tau)`` と ``class_order``）か
  ``text_emb=``（[C, dim]、class_order 順）のどちらかをコンストラクタ、または fit のキーワードで渡す
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from cascade.gate.scorers.base import DEFAULT_FOOD_LABEL, BaseScorer, as_2d
from cascade.stage1.zeroshot import ZeroShot


class MSPScorer(BaseScorer):
    name = "msp"
    normalize = False  # ZeroShot が内部で L2 正規化する

    def __init__(
        self,
        mode: str,
        tau: float = 1.0,
        zeroshot: Any | None = None,
        text_emb: np.ndarray | None = None,
    ) -> None:
        super().__init__(mode)
        if tau <= 0:
            raise ValueError("tau は正")
        self.tau = float(tau)
        self._zeroshot = zeroshot
        self._text_emb = None if text_emb is None else np.asarray(text_emb, dtype=np.float32)
        self.zeroshot_: Any | None = None

    def fit(
        self,
        X_id: np.ndarray,
        y_id: Sequence[str] | np.ndarray,
        X_neg: np.ndarray | None = None,
        *,
        class_order: Sequence[str],
        food_label: str = DEFAULT_FOOD_LABEL,
        seed: int = 0,
        zeroshot: Any | None = None,
        text_emb: np.ndarray | None = None,
    ) -> None:
        if zeroshot is not None:
            self._zeroshot = zeroshot
        if text_emb is not None:
            self._text_emb = np.asarray(text_emb, dtype=np.float32)
        super().fit(X_id, y_id, X_neg, class_order=class_order, food_label=food_label, seed=seed)

    def _fit(self, X: np.ndarray, y: np.ndarray, X_neg: np.ndarray | None) -> None:
        if self._zeroshot is not None:
            zs = self._zeroshot
            zs_order = [str(c) for c in getattr(zs, "class_order", [])]
            if zs_order != self.class_order_:
                raise ValueError(
                    f"zeroshot.class_order と class_order が不一致: {zs_order} vs {self.class_order_}"
                )
            if not hasattr(zs, "probs"):
                raise TypeError("zeroshot は probs(X, tau) を持つ必要がある")
        elif self._text_emb is not None:
            t = as_2d(self._text_emb, "text_emb")
            if t.shape[0] != len(self.class_order_):
                raise ValueError(
                    f"text_emb の行数 {t.shape[0]} が class_order の長さ {len(self.class_order_)} と不一致"
                )
            if t.shape[1] != self.dim_:
                raise ValueError(
                    f"text_emb の次元 {t.shape[1]} が X_id の次元 {self.dim_} と不一致"
                )
            zs = ZeroShot({}, t.astype(np.float32), self.class_order_)
        else:
            raise ValueError(
                "msp は zeroshot= か text_emb= のどちらかが必要（凍結テキスト埋め込み）"
            )
        self.zeroshot_ = zs
        self.food_index_ = self.class_order_.index(self.food_label_)

    def _score(self, X: np.ndarray) -> np.ndarray:
        assert self.zeroshot_ is not None
        p = np.asarray(self.zeroshot_.probs(X.astype(np.float32), self.tau), dtype=np.float64)
        if self.mode == "P2":
            return p.max(axis=1)
        return p[:, self.food_index_]

    def params(self) -> dict[str, Any]:
        return {**super().params(), "tau": self.tau}
