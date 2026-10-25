"""C2 6-way ヘッド（g0_nonfood_plan.md §5.2「C2 6-way: 第 6 クラス = 対象外」）。

- LogisticRegression（多項）で class_order の 5 クラス + ``out_of_scope``（X_neg）を学習。入力は L2 正規化
- P2 スコア = 1 − p(out)、P1 スコア = p(food)（大 = 対象内）
- X_neg 無しの fit は ValueError
- ``class_weight`` は既定 None（計画に指定なし）。負例が少ない場合は "balanced" を明示できる
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from cascade.gate.scorers.base import BaseScorer

OUT_LABEL = "out_of_scope"


class SixWayHeadScorer(BaseScorer):
    name = "c2_sixway"
    requires_negatives = True
    normalize = True

    def __init__(
        self,
        mode: str,
        C: float = 1.0,
        max_iter: int = 2000,
        class_weight: str | None = None,
    ) -> None:
        super().__init__(mode)
        if C <= 0:
            raise ValueError("C は正")
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.class_weight = class_weight
        self._clf: LogisticRegression | None = None
        self.labels_: list[str] = []

    def _fit(self, X: np.ndarray, y: np.ndarray, X_neg: np.ndarray | None) -> None:
        assert X_neg is not None
        if OUT_LABEL in self.class_order_:
            raise ValueError(
                f"class_order に {OUT_LABEL!r} を含めてはならない（第 6 クラスは内部で追加）"
            )
        self.labels_ = [*self.class_order_, OUT_LABEL]
        idx = {c: i for i, c in enumerate(self.labels_)}
        y_int = np.asarray([idx[v] for v in y.tolist()], dtype=np.int64)
        Xall = np.vstack([X, X_neg])
        yall = np.concatenate([y_int, np.full(X_neg.shape[0], idx[OUT_LABEL], dtype=np.int64)])
        clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
            class_weight=self.class_weight,
            random_state=self.seed_,
        )
        clf.fit(Xall, yall)
        self._clf = clf

    def _proba_full(self, X: np.ndarray) -> np.ndarray:
        """[n, 6]（labels_ 順。fit 集合に無かったクラスは 0）。"""
        assert self._clf is not None
        p = self._clf.predict_proba(X)
        full = np.zeros((X.shape[0], len(self.labels_)), dtype=np.float64)
        for j, c in enumerate(self._clf.classes_):
            full[:, int(c)] = p[:, j]
        return full

    def _score(self, X: np.ndarray) -> np.ndarray:
        p = self._proba_full(X)
        if self.mode == "P2":
            return 1.0 - p[:, self.labels_.index(OUT_LABEL)]
        return p[:, self.labels_.index(self.food_label_)]

    def params(self) -> dict[str, Any]:
        return {
            **super().params(),
            "C": self.C,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
        }
