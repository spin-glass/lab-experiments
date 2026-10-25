"""C3 二値ヘッド（g0_nonfood_plan.md §5.1 表 4 行目、§5.4 の LOTO / LOSO で使う）。

- LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)、入力は L2 正規化
- P2: 正例 = ID（tune_fit 全体） vs 負例 = 外部 tune。P1: 正例 = food 行のみ vs 負例
- スコア = p(in)（大 = 対象内）
- X_neg 無しの fit は ValueError（外部負例で学習する方式なので）
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from cascade.gate.scorers.base import BaseScorer


class BinaryHeadScorer(BaseScorer):
    name = "c3_binary"
    requires_negatives = True
    normalize = True

    def __init__(self, mode: str, C: float = 1.0, max_iter: int = 2000) -> None:
        super().__init__(mode)
        if C <= 0:
            raise ValueError("C は正")
        self.C = float(C)
        self.max_iter = int(max_iter)
        self._clf: LogisticRegression | None = None

    def _fit(self, X: np.ndarray, y: np.ndarray, X_neg: np.ndarray | None) -> None:
        assert X_neg is not None
        pos = X if self.mode == "P2" else X[self._food_mask(y)]
        Xall = np.vstack([pos, X_neg])
        yall = np.concatenate(
            [np.ones(pos.shape[0], dtype=np.int64), np.zeros(X_neg.shape[0], dtype=np.int64)]
        )
        clf = LogisticRegression(
            C=self.C,
            class_weight="balanced",
            max_iter=self.max_iter,
            solver="lbfgs",
            random_state=self.seed_,
        )
        clf.fit(Xall, yall)
        self._clf = clf
        self.n_pos_ = int(pos.shape[0])
        self.n_neg_ = int(X_neg.shape[0])

    def _score(self, X: np.ndarray) -> np.ndarray:
        assert self._clf is not None
        idx = list(self._clf.classes_).index(1)
        return self._clf.predict_proba(X)[:, idx]

    def params(self) -> dict[str, Any]:
        return {**super().params(), "C": self.C, "max_iter": self.max_iter}
