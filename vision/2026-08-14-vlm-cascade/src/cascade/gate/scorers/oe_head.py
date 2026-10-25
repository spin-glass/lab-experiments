"""C1 5-way + Outlier Exposure ヘッド（g0_nonfood_plan.md §5.2「C1 5-way + OE: 既存 5 クラス上の一様目標、λ = 0.5」。
Hendrycks et al. 2019 "Deep Anomaly Detection with Outlier Exposure"）。

- 線形ヘッド dim → C を torch で学習（CPU、full-batch Adam、≤ 200 epoch、seed 固定）
- lr の既定は 0.05（単位ベクトル入力では 1e-2 だと 200 epoch で収束しきらないため。合成データで確認）
- 損失 = CE(ID) + λ · mean_neg[ −(1/C) Σ_c log softmax_c ]（負例の予測を一様分布に近づける交差エントロピー）
- スコア P2 = max softmax、P1 = p(food)（大 = 対象内）
- 入力は L2 正規化（C 系で統一）。torch は fit 時に遅延 import する（import コストと
  他レーンのテストに torch を要求しないため）
- X_neg 無しの fit は ValueError
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cascade.gate.scorers.base import BaseScorer


def _softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class OEHeadScorer(BaseScorer):
    name = "c1_oe"
    requires_negatives = True
    normalize = True

    def __init__(
        self,
        mode: str,
        lam: float = 0.5,
        epochs: int = 200,
        lr: float = 0.05,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(mode)
        if lam < 0:
            raise ValueError("lam は非負")
        if not (1 <= int(epochs) <= 200):
            raise ValueError("epochs は 1..200（計画: ≤ 200 epoch）")
        self.lam = float(lam)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.W_: np.ndarray | None = None  # [dim, C]
        self.b_: np.ndarray | None = None  # [C]
        self.loss_history_: list[float] = []

    def _fit(self, X: np.ndarray, y: np.ndarray, X_neg: np.ndarray | None) -> None:
        assert X_neg is not None
        import torch

        torch.manual_seed(self.seed_)
        n_cls = len(self.class_order_)
        y_idx = self._class_index(y)
        Xt = torch.as_tensor(X, dtype=torch.float32, device="cpu")
        yt = torch.as_tensor(y_idx, dtype=torch.long, device="cpu")
        Nt = torch.as_tensor(X_neg, dtype=torch.float32, device="cpu")
        gen = torch.Generator(device="cpu").manual_seed(self.seed_)
        W = (torch.randn(X.shape[1], n_cls, generator=gen) * 0.01).requires_grad_(True)
        b = torch.zeros(n_cls).requires_grad_(True)
        opt = torch.optim.Adam([W, b], lr=self.lr, weight_decay=self.weight_decay)
        ce = torch.nn.CrossEntropyLoss()
        history: list[float] = []
        for _ in range(self.epochs):
            opt.zero_grad()
            logits_id = Xt @ W + b
            loss = ce(logits_id, yt)
            if self.lam > 0:
                logp_neg = torch.log_softmax(Nt @ W + b, dim=1)
                loss = loss + self.lam * (-(logp_neg.mean(dim=1))).mean()
            loss.backward()
            opt.step()
            history.append(float(loss.detach()))
        self.W_ = W.detach().cpu().numpy().astype(np.float64)
        self.b_ = b.detach().cpu().numpy().astype(np.float64)
        self.loss_history_ = history

    def probs(self, X: np.ndarray) -> np.ndarray:
        """[n, C] softmax 確率（入力は L2 正規化済みを想定。score 経由なら自動）。"""
        assert self.W_ is not None and self.b_ is not None
        return _softmax_np(X @ self.W_ + self.b_)

    def _score(self, X: np.ndarray) -> np.ndarray:
        p = self.probs(X)
        if self.mode == "P2":
            return p.max(axis=1)
        return p[:, self.class_order_.index(self.food_label_)]

    def params(self) -> dict[str, Any]:
        return {
            **super().params(),
            "lam": self.lam,
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }
