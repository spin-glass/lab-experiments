"""kNN 検出器（g0_nonfood_plan.md §5.1 表 2 行目。Sun et al. 2022 "Out-of-Distribution Detection with Deep Nearest Neighbors"）。

- L2 正規化した埋め込み上の k 近傍距離。スコア = −（k 番目の近傍までの距離）（大 = 対象内）
- P2: バンク = tune_fit の ID 全体、P1: バンク = tune_fit の food 行のみ
- k = 50 が主。k ∈ {10, 200} は work のみ（§5.2）。バンクが k より小さければ k = len(bank)
- metric: ``cosine``（既定）または ``euclidean``。単位ベクトル上では単調変換の関係
  （euclid = sqrt(2 − 2 cos）なので順位は同じ
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from cascade.gate.scorers.base import BaseScorer

_METRICS = ("cosine", "euclidean")


class KNNScorer(BaseScorer):
    name = "knn"
    normalize = True

    def __init__(self, mode: str, k: int = 50, metric: str = "cosine") -> None:
        super().__init__(mode)
        if int(k) < 1:
            raise ValueError("k は 1 以上")
        if metric not in _METRICS:
            raise ValueError(f"metric は {_METRICS} のいずれか: {metric!r}")
        self.k = int(k)
        self.metric = metric
        self.k_eff_: int = 0
        self.bank_size_: int = 0
        self._nn: NearestNeighbors | None = None

    def _fit(self, X: np.ndarray, y: np.ndarray, X_neg: np.ndarray | None) -> None:
        bank = X if self.mode == "P2" else X[self._food_mask(y)]
        if bank.shape[0] == 0:
            raise ValueError("kNN バンクが空")
        self.bank_size_ = int(bank.shape[0])
        self.k_eff_ = min(self.k, self.bank_size_)
        self._nn = NearestNeighbors(n_neighbors=self.k_eff_, metric=self.metric, algorithm="brute")
        self._nn.fit(bank)

    def _score(self, X: np.ndarray) -> np.ndarray:
        assert self._nn is not None
        dist, _ = self._nn.kneighbors(X, n_neighbors=self.k_eff_, return_distance=True)
        return -dist[:, -1]

    def params(self) -> dict[str, Any]:
        return {**super().params(), "k": self.k, "metric": self.metric}
