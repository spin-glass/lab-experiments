"""Mahalanobis++ / 素の Mahalanobis 検出器（g0_nonfood_plan.md §5.1 表 3 行目、§5.2 アブレーション）。

- Mahalanobis++（Mueller & Hein 2025）: L2 正規化 → クラス平均 + 共有共分散（pooled within-class）
  + リッジ 1e-6·I。P2 スコア = −min_c d_c、P1 スコア = −d_food（大 = 対象内）
- 素の Mahalanobis（Lee et al. 2018）: L2 正規化なし、それ以外は同じ（``normalize=False``）
- d_c は平方根を取った距離（二乗ではない）。順位は二乗と同じだが記述分析のヒストグラム（§6）で距離として読める
- 共有共分散は Σ = (1/N) Σ_c Σ_{i∈c} (x_i − μ_c)(x_i − μ_c)^T（ML 推定）。n < dim でもリッジで正定
- 計算は float64。Cholesky 分解で d² = ||L⁻¹ (x − μ_c)||²
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import linalg

from cascade.gate.scorers.base import BaseScorer


class MahalanobisScorer(BaseScorer):
    """``normalize=True`` で Mahalanobis++（name="mahalanobis_pp"）、False で素の Mahalanobis（name="mahalanobis"）。"""

    def __init__(self, mode: str, normalize: bool = True, ridge: float = 1e-6) -> None:
        super().__init__(mode)
        if ridge < 0:
            raise ValueError("ridge は非負")
        self.normalize = bool(normalize)
        self.name = "mahalanobis_pp" if self.normalize else "mahalanobis"
        self.ridge = float(ridge)
        self.means_: np.ndarray | None = None  # [C_present, dim]
        self.classes_: list[str] = []
        self._chol: tuple[np.ndarray, bool] | None = None

    def _fit(self, X: np.ndarray, y: np.ndarray, X_neg: np.ndarray | None) -> None:
        present = [c for c in self.class_order_ if (y == c).any()]
        if self.mode == "P1":
            if self.food_label_ not in present:
                raise ValueError("P1 には food 行が必要")
        means = []
        dim = X.shape[1]
        scatter = np.zeros((dim, dim), dtype=np.float64)
        n_total = 0
        for c in present:
            Xc = X[y == c]
            mu = Xc.mean(axis=0)
            means.append(mu)
            R = Xc - mu
            scatter += R.T @ R
            n_total += Xc.shape[0]
        cov = scatter / max(n_total, 1) + self.ridge * np.eye(dim)
        self.classes_ = present
        self.means_ = np.stack(means)
        self.cov_ = cov
        self._chol = linalg.cho_factor(cov, lower=True, check_finite=False)

    def distances(self, X: np.ndarray) -> np.ndarray:
        """全クラスへの Mahalanobis 距離 [n, C_present]（入力は fit 時と同じ正規化前提で受ける）。"""
        assert self.means_ is not None and self._chol is not None
        out = np.empty((X.shape[0], len(self.classes_)), dtype=np.float64)
        for j, mu in enumerate(self.means_):
            R = (X - mu).T  # [dim, n]
            Z = linalg.solve_triangular(self._chol[0], R, lower=True, check_finite=False)
            out[:, j] = np.sqrt(np.maximum((Z * Z).sum(axis=0), 0.0))
        return out

    def _score(self, X: np.ndarray) -> np.ndarray:
        d = self.distances(X)
        if self.mode == "P2":
            return -d.min(axis=1)
        return -d[:, self.classes_.index(self.food_label_)]

    def params(self) -> dict[str, Any]:
        return {**super().params(), "normalize": self.normalize, "ridge": self.ridge}
