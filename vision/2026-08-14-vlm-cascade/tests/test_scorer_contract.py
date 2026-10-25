"""検出器の共通契約テスト（g0_nonfood_plan.md §5.3）: 全名前 × 両モードをパラメトライズ。

合成データ: 単位球上に 5 つの ID ガウス群（class_order 順）＋ 遠い負例群 1 つ。ネットワーク・モデル不要。
契約: (1) ID > 負例（AUROC > 0.9）、(2) 形状・有限、(3) seed 決定性、(4) fit 前 score は NotFittedError、
(5) C 系は X_neg 無しで ValueError、(6) mode / name の検証。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from cascade.gate.scorers import (
    MODES,
    NEGATIVE_REQUIRING,
    SCORER_NAMES,
    NotFittedError,
    Scorer,
    make_scorer,
)

CLASS_ORDER = ["food", "drink", "menu", "inside", "outside"]
DIM = 32


@dataclass
class Synthetic:
    X_id: np.ndarray
    y_id: np.ndarray
    X_neg: np.ndarray
    X_id_test: np.ndarray
    y_id_test: np.ndarray
    X_neg_test: np.ndarray
    centers: np.ndarray  # [5, dim]（クラス中心 = MSP のテキスト埋め込み代替）


def _clusters(
    rng: np.random.Generator, centers: np.ndarray, n_per: int, spread: float
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i, c in enumerate(centers):
        xs.append(c + spread * rng.standard_normal((n_per, centers.shape[1])))
        ys.append(np.full(n_per, i))
    return np.vstack(xs), np.concatenate(ys)


def make_synthetic(seed: int = 0, n_per: int = 120, n_neg: int = 150) -> Synthetic:
    rng = np.random.default_rng(seed)
    # 6 つの互いに離れた中心（直交基底に乗せてから少し混ぜる）。最後が負例の中心
    base = np.zeros((6, DIM))
    for i in range(6):
        base[i, i] = 1.0
    base += 0.05 * rng.standard_normal(base.shape)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    centers, neg_center = base[:5], base[5:]
    X, yi = _clusters(rng, centers, n_per, 0.08)
    Xt, yit = _clusters(rng, centers, n_per // 3, 0.08)
    Xn, _ = _clusters(rng, neg_center, n_neg, 0.08)
    Xnt, _ = _clusters(rng, neg_center, n_neg // 3, 0.08)
    labels = np.asarray(CLASS_ORDER, dtype=object)
    return Synthetic(
        X_id=X.astype(np.float32),
        y_id=labels[yi],
        X_neg=Xn.astype(np.float32),
        X_id_test=Xt.astype(np.float32),
        y_id_test=labels[yit],
        X_neg_test=Xnt.astype(np.float32),
        centers=centers.astype(np.float32),
    )


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann–Whitney による AUROC（同値は 0.5）。"""
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


@pytest.fixture(scope="module")
def synth() -> Synthetic:
    return make_synthetic()


def _params(name: str, synth: Synthetic) -> dict:
    if name == "msp":
        return {"text_emb": synth.centers}
    if name == "c1_oe":
        return {"epochs": 100}
    return {}


def _fit(name: str, mode: str, synth: Synthetic, seed: int = 0, with_neg: bool = True) -> Scorer:
    s = make_scorer(name, mode, **_params(name, synth))
    s.fit(
        synth.X_id,
        synth.y_id,
        synth.X_neg if with_neg else None,
        class_order=CLASS_ORDER,
        food_label="food",
        seed=seed,
    )
    return s


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", SCORER_NAMES)
def test_id_scores_above_negatives(name: str, mode: str, synth: Synthetic) -> None:
    s = _fit(name, mode, synth)
    assert s.name == name and s.mode == mode
    if mode == "P2":
        pos = s.score(synth.X_id_test)
    else:
        pos = s.score(synth.X_id_test[synth.y_id_test == "food"])
    neg = s.score(synth.X_neg_test)
    assert pos.ndim == 1 and neg.ndim == 1
    assert pos.shape[0] == (
        synth.X_id_test.shape[0] if mode == "P2" else int((synth.y_id_test == "food").sum())
    )
    assert neg.shape[0] == synth.X_neg_test.shape[0]
    assert np.isfinite(pos).all() and np.isfinite(neg).all()
    a = auroc(pos, neg)
    assert a > 0.9, f"{name}/{mode}: AUROC={a:.3f}"
    assert pos.mean() > neg.mean()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", SCORER_NAMES)
def test_deterministic_given_seed(name: str, mode: str, synth: Synthetic) -> None:
    a = _fit(name, mode, synth, seed=7).score(synth.X_id_test)
    b = _fit(name, mode, synth, seed=7).score(synth.X_id_test)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-7)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", SCORER_NAMES)
def test_score_before_fit_raises(name: str, mode: str, synth: Synthetic) -> None:
    s = make_scorer(name, mode, **_params(name, synth))
    with pytest.raises(NotFittedError):
        s.score(synth.X_id_test)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", SCORER_NAMES)
def test_shapes_single_row_and_empty(name: str, mode: str, synth: Synthetic) -> None:
    s = _fit(name, mode, synth)
    one = s.score(synth.X_id_test[0])
    assert one.shape == (1,)
    empty = s.score(np.zeros((0, DIM), dtype=np.float32))
    assert empty.shape == (0,)
    with pytest.raises(ValueError):
        s.score(np.zeros((3, DIM + 1), dtype=np.float32))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", NEGATIVE_REQUIRING)
def test_c_series_require_negatives(name: str, mode: str, synth: Synthetic) -> None:
    with pytest.raises(ValueError, match="X_neg"):
        _fit(name, mode, synth, with_neg=False)
    # 空の負例も不可
    s = make_scorer(name, mode)
    with pytest.raises(ValueError):
        s.fit(synth.X_id, synth.y_id, np.zeros((0, DIM)), class_order=CLASS_ORDER)


@pytest.mark.parametrize("name", [n for n in SCORER_NAMES if n not in NEGATIVE_REQUIRING])
def test_non_c_series_fit_without_negatives(name: str, synth: Synthetic) -> None:
    s = _fit(name, "P2", synth, with_neg=False)
    assert s.score(synth.X_id_test).shape[0] == synth.X_id_test.shape[0]


@pytest.mark.parametrize("name", SCORER_NAMES)
def test_invalid_mode_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        make_scorer(name, "P0")
    with pytest.raises(ValueError):
        make_scorer(name, "p1")


def test_unknown_name_rejected() -> None:
    with pytest.raises(ValueError, match="未知"):
        make_scorer("energy", "P2")


@pytest.mark.parametrize("name", SCORER_NAMES)
def test_fit_input_validation(name: str, synth: Synthetic) -> None:
    s = make_scorer(name, "P1", **_params(name, synth))
    kw = {"class_order": CLASS_ORDER}
    with pytest.raises(ValueError):  # food_label が class_order に無い
        s.fit(synth.X_id, synth.y_id, synth.X_neg, class_order=CLASS_ORDER, food_label="soup")
    with pytest.raises(ValueError):  # ラベルが class_order 外
        y_bad = synth.y_id.copy()
        y_bad[0] = "cake"
        s.fit(synth.X_id, y_bad, synth.X_neg, **kw)
    with pytest.raises(ValueError):  # 行数不一致
        s.fit(synth.X_id[:-1], synth.y_id, synth.X_neg, **kw)
    with pytest.raises(ValueError):  # P1 に food 行が無い
        y_nofood = np.where(synth.y_id == "food", "drink", synth.y_id)
        s.fit(synth.X_id, y_nofood, synth.X_neg, **kw)
    with pytest.raises(ValueError):  # X_neg の次元不一致
        s.fit(synth.X_id, synth.y_id, np.zeros((5, DIM + 1)), **kw)


@pytest.mark.parametrize("name", SCORER_NAMES)
def test_protocol_and_params(name: str, synth: Synthetic) -> None:
    s = make_scorer(name, "P2", **_params(name, synth))
    assert isinstance(s, Scorer)
    p = s.params()
    assert p["name"] == name and p["mode"] == "P2"
