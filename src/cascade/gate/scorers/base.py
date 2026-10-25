"""検出器の共通契約（g0_nonfood_plan.md §5.3、design.md §6 G0）。

- スコアの向きは **大 = 対象内**（OOD スコアは s_ood = −s_in）
- fit 前の ``score`` は :class:`NotFittedError`
- P1（food 候補のみ再判定）と P2（全画像ゲート）の両モードを同一クラスで持つ（``mode`` で切替）
- seed を与えれば決定的に動く

各検出器は :class:`BaseScorer` を継承し ``_fit`` / ``_score`` を実装する。入力検証・fit 状態の管理・
L2 正規化のヘルパはここに置く。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

MODES: tuple[str, ...] = ("P1", "P2")
DEFAULT_FOOD_LABEL = "food"


class NotFittedError(RuntimeError):
    """fit 前に score を呼んだ。"""


@runtime_checkable
class Scorer(Protocol):
    """検出器の構造的インターフェース（他レーンはこれに依存する）。"""

    name: str
    mode: str

    def fit(
        self,
        X_id: np.ndarray,
        y_id: np.ndarray,
        X_neg: np.ndarray | None = None,
        *,
        class_order: list[str],
        food_label: str = DEFAULT_FOOD_LABEL,
        seed: int = 0,
    ) -> None: ...

    def score(self, X: np.ndarray) -> np.ndarray: ...


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """行ごとの L2 正規化（float64）。ゼロベクトルは eps で保護。"""
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def as_2d(x: np.ndarray, name: str = "X") -> np.ndarray:
    """[n, dim] の float64 配列に整える。1 次元は 1 行として扱う。"""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError(f"{name} は [n, dim] が必要: shape={a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} に NaN / inf が含まれる")
    return a


def check_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"mode は {MODES} のいずれか: {mode!r}")
    return mode


def validate_fit_inputs(
    X_id: np.ndarray,
    y_id: Sequence[str] | np.ndarray,
    X_neg: np.ndarray | None,
    *,
    class_order: Sequence[str],
    food_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str]]:
    """fit 入力の共通検証。返り値は (X_id[float64], y_id[str], X_neg[float64]|None, class_order)。"""
    order = [str(c) for c in class_order]
    if len(order) == 0:
        raise ValueError("class_order が空")
    if len(set(order)) != len(order):
        raise ValueError(f"class_order に重複: {order}")
    if food_label not in order:
        raise ValueError(f"food_label={food_label!r} が class_order に無い: {order}")
    X = as_2d(X_id, "X_id")
    y = np.asarray([str(v) for v in np.asarray(y_id).ravel()], dtype=object)
    if X.shape[0] == 0:
        raise ValueError("X_id が空")
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"X_id と y_id の行数が不一致: {X.shape[0]} vs {y.shape[0]}")
    unknown = sorted(set(y.tolist()) - set(order))
    if unknown:
        raise ValueError(f"y_id に class_order 外のラベル: {unknown}")
    neg: np.ndarray | None = None
    if X_neg is not None:
        neg = as_2d(X_neg, "X_neg")
        if neg.shape[1] != X.shape[1]:
            raise ValueError(f"X_neg の次元が X_id と不一致: {neg.shape[1]} vs {X.shape[1]}")
    return X, y, neg, order


class BaseScorer:
    """fit 状態と入力検証を共通化した基底。サブクラスは ``_fit`` と ``_score`` を実装する。

    - ``requires_negatives=True`` のサブクラス（C 系）は X_neg 無しの fit で ValueError
    - ``normalize=True`` のサブクラスは fit / score の入力を L2 正規化してから ``_fit`` / ``_score`` に渡す
    """

    name: str = "base"
    requires_negatives: bool = False
    normalize: bool = True

    def __init__(self, mode: str) -> None:
        self.mode = check_mode(mode)
        self._fitted = False
        self.dim_: int | None = None
        self.class_order_: list[str] = []
        self.food_label_: str = DEFAULT_FOOD_LABEL
        self.seed_: int = 0

    # -- 公開 API -------------------------------------------------------------
    def fit(
        self,
        X_id: np.ndarray,
        y_id: Sequence[str] | np.ndarray,
        X_neg: np.ndarray | None = None,
        *,
        class_order: Sequence[str],
        food_label: str = DEFAULT_FOOD_LABEL,
        seed: int = 0,
    ) -> None:
        X, y, neg, order = validate_fit_inputs(
            X_id, y_id, X_neg, class_order=class_order, food_label=food_label
        )
        if self.requires_negatives and (neg is None or neg.shape[0] == 0):
            raise ValueError(f"{self.name} は X_neg（外部負例）を必要とする")
        if self.mode == "P1" and not (y == food_label).any():
            raise ValueError(f"P1 には food_label={food_label!r} の行が必要")
        if self.normalize:
            X = l2_normalize(X)
            neg = l2_normalize(neg) if neg is not None else None
        self.dim_ = int(X.shape[1])
        self.class_order_ = order
        self.food_label_ = food_label
        self.seed_ = int(seed)
        self._fit(X, y, neg)
        self._fitted = True

    def score(self, X: np.ndarray) -> np.ndarray:
        """スコア [n]（大 = 対象内）。fit 前は NotFittedError。"""
        if not self._fitted:
            raise NotFittedError(f"{self.name}({self.mode}) は fit 前")
        Xa = as_2d(X)
        if Xa.shape[1] != self.dim_:
            raise ValueError(f"次元が fit 時と不一致: {Xa.shape[1]} vs {self.dim_}")
        if Xa.shape[0] == 0:
            return np.zeros((0,), dtype=np.float64)
        if self.normalize:
            Xa = l2_normalize(Xa)
        s = np.asarray(self._score(Xa), dtype=np.float64).reshape(-1)
        if s.shape[0] != Xa.shape[0]:
            raise RuntimeError(f"{self.name}: スコア長が入力と不一致")
        return s

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def params(self) -> dict[str, Any]:
        """再現用の設定（レポート・ood_scores の method_id 構成に使う）。"""
        return {"name": self.name, "mode": self.mode}

    # -- サブクラス実装 -------------------------------------------------------
    def _fit(self, X: np.ndarray, y: np.ndarray, X_neg: np.ndarray | None) -> None:
        raise NotImplementedError

    def _score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # -- 補助 ---------------------------------------------------------------
    def _food_mask(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y == self.food_label_, dtype=bool)

    def _class_index(self, y: np.ndarray) -> np.ndarray:
        lookup = {c: i for i, c in enumerate(self.class_order_)}
        return np.asarray([lookup[v] for v in y.tolist()], dtype=np.int64)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params()})"
