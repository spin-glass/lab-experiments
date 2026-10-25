"""検出器ごとの仕様テスト（g0_nonfood_plan.md §5.1 / §5.2 の固定値と定義）。"""

from __future__ import annotations

import numpy as np
import pytest

from cascade.gate.scorers import (
    OUT_LABEL,
    BinaryHeadScorer,
    KNNScorer,
    MahalanobisScorer,
    MSPScorer,
    OEHeadScorer,
    SixWayHeadScorer,
    l2_normalize,
    make_scorer,
)
from cascade.stage1.zeroshot import ZeroShot
from tests.test_scorer_contract import CLASS_ORDER, DIM, Synthetic, auroc, make_synthetic


@pytest.fixture(scope="module")
def synth() -> Synthetic:
    return make_synthetic(seed=1)


# ---- MSP -----------------------------------------------------------------


def test_msp_p2_is_max_softmax_and_p1_is_p_food(synth: Synthetic) -> None:
    zs = ZeroShot({}, synth.centers, CLASS_ORDER)
    X = synth.X_id_test
    for tau in (1.0, 0.01):
        p2 = MSPScorer("P2", tau=tau, zeroshot=zs)
        p2.fit(synth.X_id, synth.y_id, class_order=CLASS_ORDER)
        p1 = MSPScorer("P1", tau=tau, zeroshot=zs)
        p1.fit(synth.X_id, synth.y_id, class_order=CLASS_ORDER)
        probs = zs.probs(X, tau=tau)
        np.testing.assert_allclose(p2.score(X), probs.max(axis=1), atol=1e-6)
        np.testing.assert_allclose(p1.score(X), probs[:, 0], atol=1e-6)
    # τ=1 は平坦、τ=0.01 は尖る（MSP が大きい）
    m1 = MSPScorer("P2", tau=1.0, text_emb=synth.centers)
    m1.fit(synth.X_id, synth.y_id, class_order=CLASS_ORDER)
    m001 = MSPScorer("P2", tau=0.01, text_emb=synth.centers)
    m001.fit(synth.X_id, synth.y_id, class_order=CLASS_ORDER)
    assert m001.score(X).mean() > m1.score(X).mean()


def test_msp_text_emb_via_fit_kwarg_and_validation(synth: Synthetic) -> None:
    s = MSPScorer("P2")
    with pytest.raises(ValueError, match="zeroshot"):
        s.fit(synth.X_id, synth.y_id, class_order=CLASS_ORDER)
    s.fit(synth.X_id, synth.y_id, class_order=CLASS_ORDER, text_emb=synth.centers)
    assert s.score(synth.X_id_test).shape == (synth.X_id_test.shape[0],)
    # class_order の不一致
    zs = ZeroShot({}, synth.centers, list(reversed(CLASS_ORDER)))
    with pytest.raises(ValueError, match="class_order"):
        MSPScorer("P2", zeroshot=zs).fit(synth.X_id, synth.y_id, class_order=CLASS_ORDER)
    with pytest.raises(ValueError):
        MSPScorer("P2", text_emb=synth.centers[:4]).fit(
            synth.X_id, synth.y_id, class_order=CLASS_ORDER
        )
    with pytest.raises(ValueError):
        MSPScorer("P2", tau=0.0)


def test_msp_is_frozen_ignores_fit_data(synth: Synthetic) -> None:
    """fit データが変わってもテキスト埋め込みが同じなら同じ確率（凍結）。"""
    a = MSPScorer("P1", text_emb=synth.centers)
    a.fit(synth.X_id, synth.y_id, class_order=CLASS_ORDER)
    b = MSPScorer("P1", text_emb=synth.centers)
    b.fit(synth.X_id[:10], synth.y_id[:10], class_order=CLASS_ORDER)
    np.testing.assert_allclose(a.score(synth.X_neg_test), b.score(synth.X_neg_test))


# ---- kNN -----------------------------------------------------------------


def test_knn_score_is_negative_kth_distance_on_unit_vectors(synth: Synthetic) -> None:
    k = 5
    s = KNNScorer("P2", k=k, metric="euclidean")
    s.fit(synth.X_id, synth.y_id, class_order=CLASS_ORDER)
    X = synth.X_neg_test[:7]
    bank = l2_normalize(synth.X_id)
    q = l2_normalize(X)
    d = np.sqrt(((q[:, None, :] - bank[None, :, :]) ** 2).sum(-1))
    kth = np.sort(d, axis=1)[:, k - 1]
    np.testing.assert_allclose(s.score(X), -kth, atol=1e-6)
    assert s.k_eff_ == k and s.bank_size_ == synth.X_id.shape[0]


def test_knn_cosine_and_euclidean_same_ranking(synth: Synthetic) -> None:
    X = np.vstack([synth.X_id_test[:20], synth.X_neg_test[:20]])
    sc = KNNScorer("P2", k=10, metric="cosine")
    sc.fit(synth.X_id, synth.y_id, class_order=CLASS_ORDER)
    se = KNNScorer("P2", k=10, metric="euclidean")
    se.fit(synth.X_id, synth.y_id, class_order=CLASS_ORDER)
    assert np.array_equal(np.argsort(sc.score(X)), np.argsort(se.score(X)))


def test_knn_scale_invariant(synth: Synthetic) -> None:
    s = KNNScorer("P2", k=10)
    s.fit(synth.X_id * 3.0, synth.y_id, class_order=CLASS_ORDER)
    np.testing.assert_allclose(
        s.score(synth.X_neg_test), s.score(synth.X_neg_test * 0.1), atol=1e-6
    )


def test_knn_p1_bank_is_food_only(synth: Synthetic) -> None:
    p1 = KNNScorer("P1", k=10)
    p1.fit(synth.X_id, synth.y_id, class_order=CLASS_ORDER)
    p2 = KNNScorer("P2", k=10)
    p2.fit(synth.X_id, synth.y_id, class_order=CLASS_ORDER)
    assert p1.bank_size_ == int((synth.y_id == "food").sum())
    drink = synth.X_id_test[synth.y_id_test == "drink"]
    # drink 群は food バンクからは遠い（P1 スコアが低い）が ID 全体バンクには近い
    assert (p1.score(drink) < p2.score(drink)).all()
    food = synth.X_id_test[synth.y_id_test == "food"]
    np.testing.assert_allclose(p1.score(food), p2.score(food), atol=1e-6)


def test_knn_k_clamped_to_bank_size() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((8, DIM))
    y = np.asarray(["food"] * 5 + ["drink"] * 3, dtype=object)
    s = KNNScorer("P1", k=50)
    s.fit(X, y, class_order=CLASS_ORDER)
    assert s.k_eff_ == 5
    out = s.score(rng.standard_normal((3, DIM)))
    assert out.shape == (3,) and np.isfinite(out).all()
    with pytest.raises(ValueError):
        KNNScorer("P2", k=0)
    with pytest.raises(ValueError):
        KNNScorer("P2", metric="manhattan")


# ---- Mahalanobis ---------------------------------------------------------


def _pooled_cov(
    X: np.ndarray, y: np.ndarray, classes: list[str]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    mus, S, n = {}, np.zeros((X.shape[1], X.shape[1])), 0
    for c in classes:
        Xc = X[y == c]
        mus[c] = Xc.mean(0)
        R = Xc - mus[c]
        S += R.T @ R
        n += len(Xc)
    return mus, S / n


def test_mahalanobis_pp_matches_closed_form(synth: Synthetic) -> None:
    ridge = 1e-6
    s = MahalanobisScorer("P2", ridge=ridge)
    s.fit(synth.X_id, synth.y_id, class_order=CLASS_ORDER)
    assert s.name == "mahalanobis_pp"
    Xn = l2_normalize(synth.X_id)
    mus, cov = _pooled_cov(Xn, synth.y_id, CLASS_ORDER)
    P = np.linalg.inv(cov + ridge * np.eye(DIM))
    q = l2_normalize(synth.X_neg_test[:5])
    d = np.stack(
        [np.sqrt(np.einsum("ij,jk,ik->i", q - mus[c], P, q - mus[c])) for c in CLASS_ORDER], 1
    )
    np.testing.assert_allclose(s.score(synth.X_neg_test[:5]), -d.min(1), rtol=1e-5)
    p1 = MahalanobisScorer("P1", ridge=ridge)
    p1.fit(synth.X_id, synth.y_id, class_order=CLASS_ORDER)
    np.testing.assert_allclose(p1.score(synth.X_neg_test[:5]), -d[:, 0], rtol=1e-5)


def test_mahalanobis_class_mean_has_zero_distance(synth: Synthetic) -> None:
    s = MahalanobisScorer("P1")
    s.fit(synth.X_id, synth.y_id, class_order=CLASS_ORDER)
    mu_food = s.means_[s.classes_.index("food")]
    # 平均は正規化後の平均なので、そのまま（正規化で少し動く）ではなく distances で直接確認
    assert s.distances(mu_food[None, :])[0, 0] < 1e-6
    assert s.score(synth.X_id_test[synth.y_id_test == "food"]).max() <= 0.0


def test_mahalanobis_raw_vs_pp_and_scale(synth: Synthetic) -> None:
    pp = make_scorer("mahalanobis_pp", "P2")
    raw = make_scorer("mahalanobis", "P2")
    assert raw.name == "mahalanobis" and pp.name == "mahalanobis_pp"
    X = synth.X_id.astype(np.float64) * np.linspace(0.5, 2.0, synth.X_id.shape[0])[:, None]
    pp.fit(X, synth.y_id, class_order=CLASS_ORDER)
    raw.fit(X, synth.y_id, class_order=CLASS_ORDER)
    q = synth.X_neg_test[:6]
    np.testing.assert_allclose(
        pp.score(q), pp.score(q * 5.0), rtol=1e-6
    )  # ++ は正規化でスケール不変
    assert not np.allclose(raw.score(q), raw.score(q * 5.0))  # 素は不変でない
    # 両方とも ID > 負例
    assert auroc(pp.score(synth.X_id_test), pp.score(synth.X_neg_test)) > 0.95
    assert auroc(raw.score(synth.X_id_test), raw.score(synth.X_neg_test)) > 0.95


def test_mahalanobis_ridge_handles_n_smaller_than_dim() -> None:
    rng = np.random.default_rng(3)
    X = rng.standard_normal((12, 64))
    y = np.asarray(["food"] * 6 + ["drink"] * 6, dtype=object)
    s = MahalanobisScorer("P2", ridge=1e-6)
    s.fit(X, y, class_order=CLASS_ORDER)
    out = s.score(rng.standard_normal((4, 64)))
    assert np.isfinite(out).all()
    assert s.classes_ == ["food", "drink"]  # 存在するクラスのみ、class_order 順
    with pytest.raises(ValueError):
        MahalanobisScorer("P2", ridge=-1.0)


# ---- C3 binary -----------------------------------------------------------


def test_c3_binary_probability_and_modes(synth: Synthetic) -> None:
    p2 = BinaryHeadScorer("P2")
    p2.fit(synth.X_id, synth.y_id, synth.X_neg, class_order=CLASS_ORDER, seed=0)
    s = p2.score(np.vstack([synth.X_id_test, synth.X_neg_test]))
    assert ((s >= 0) & (s <= 1)).all()
    assert p2.n_pos_ == synth.X_id.shape[0] and p2.n_neg_ == synth.X_neg.shape[0]
    p1 = BinaryHeadScorer("P1")
    p1.fit(synth.X_id, synth.y_id, synth.X_neg, class_order=CLASS_ORDER, seed=0)
    assert p1.n_pos_ == int((synth.y_id == "food").sum())
    # P1 では drink 群は正例集合に入っていないので、P2 より p(in) が低い傾向
    drink = synth.X_id_test[synth.y_id_test == "drink"]
    assert p1.score(drink).mean() < p2.score(drink).mean()
    assert p2.params()["C"] == 1.0 and p2.params()["max_iter"] == 2000
    with pytest.raises(ValueError):
        BinaryHeadScorer("P2", C=0)


def test_c3_scale_invariant_via_normalisation(synth: Synthetic) -> None:
    s = BinaryHeadScorer("P2")
    s.fit(synth.X_id, synth.y_id, synth.X_neg, class_order=CLASS_ORDER)
    q = synth.X_neg_test[:5]
    np.testing.assert_allclose(s.score(q), s.score(q * 7.0), atol=1e-8)


# ---- C2 six-way ----------------------------------------------------------


def test_c2_sixway_scores(synth: Synthetic) -> None:
    p2 = SixWayHeadScorer("P2")
    p2.fit(synth.X_id, synth.y_id, synth.X_neg, class_order=CLASS_ORDER)
    assert p2.labels_ == [*CLASS_ORDER, OUT_LABEL]
    full = p2._proba_full(l2_normalize(synth.X_neg_test))
    np.testing.assert_allclose(full.sum(1), 1.0, atol=1e-8)
    out_idx = p2.labels_.index(OUT_LABEL)
    assert full[:, out_idx].mean() > 0.9  # 負例は out_of_scope に寄る
    np.testing.assert_allclose(p2.score(synth.X_neg_test), 1.0 - full[:, out_idx])
    p1 = SixWayHeadScorer("P1")
    p1.fit(synth.X_id, synth.y_id, synth.X_neg, class_order=CLASS_ORDER)
    food = synth.X_id_test[synth.y_id_test == "food"]
    drink = synth.X_id_test[synth.y_id_test == "drink"]
    assert p1.score(food).mean() > 0.8 and p1.score(drink).mean() < 0.2
    with pytest.raises(ValueError, match=OUT_LABEL):
        SixWayHeadScorer("P2").fit(
            synth.X_id, synth.y_id, synth.X_neg, class_order=[*CLASS_ORDER, OUT_LABEL]
        )


def test_c2_sixway_missing_class_in_fit_is_zero_prob() -> None:
    rng = np.random.default_rng(5)
    X = rng.standard_normal((40, DIM))
    y = np.asarray(["food"] * 20 + ["drink"] * 20, dtype=object)
    neg = rng.standard_normal((20, DIM)) + 3.0
    s = SixWayHeadScorer("P2", class_weight="balanced")
    s.fit(X, y, neg, class_order=CLASS_ORDER)
    full = s._proba_full(l2_normalize(X[:3]))
    assert (full[:, s.labels_.index("menu")] == 0).all()
    assert s.params()["class_weight"] == "balanced"


# ---- C1 OE ---------------------------------------------------------------


def test_c1_oe_trains_and_uniformises_negatives(synth: Synthetic) -> None:
    s = OEHeadScorer("P2", lam=0.5, epochs=150)
    s.fit(synth.X_id, synth.y_id, synth.X_neg, class_order=CLASS_ORDER, seed=0)
    assert s.W_.shape == (DIM, 5) and s.b_.shape == (5,)
    assert len(s.loss_history_) == 150
    assert s.loss_history_[-1] < s.loss_history_[0]
    p_neg = s.probs(l2_normalize(synth.X_neg_test))
    p_id = s.probs(l2_normalize(synth.X_id_test))
    assert p_neg.max(1).mean() < 0.5  # 負例は一様に近い（MSP 低）
    assert p_id.max(1).mean() > 0.8
    # ID の argmax は正解クラス
    pred = np.asarray(CLASS_ORDER, dtype=object)[p_id.argmax(1)]
    assert (pred == synth.y_id_test).mean() > 0.95
    # P1 = p(food)
    p1 = OEHeadScorer("P1", epochs=150)
    p1.fit(synth.X_id, synth.y_id, synth.X_neg, class_order=CLASS_ORDER, seed=0)
    np.testing.assert_allclose(
        p1.score(synth.X_id_test), p1.probs(l2_normalize(synth.X_id_test))[:, 0]
    )


def test_c1_oe_lambda_zero_has_higher_negative_msp_than_oe(synth: Synthetic) -> None:
    oe = OEHeadScorer("P2", lam=0.5, epochs=150)
    oe.fit(synth.X_id, synth.y_id, synth.X_neg, class_order=CLASS_ORDER, seed=0)
    plain = OEHeadScorer("P2", lam=0.0, epochs=150)
    plain.fit(synth.X_id, synth.y_id, synth.X_neg, class_order=CLASS_ORDER, seed=0)
    assert oe.score(synth.X_neg_test).mean() < plain.score(synth.X_neg_test).mean()


def test_c1_oe_seed_changes_init_but_epochs_bounded() -> None:
    with pytest.raises(ValueError):
        OEHeadScorer("P2", epochs=201)
    with pytest.raises(ValueError):
        OEHeadScorer("P2", epochs=0)
    with pytest.raises(ValueError):
        OEHeadScorer("P2", lam=-0.1)


def test_make_scorer_passes_params(synth: Synthetic) -> None:
    s = make_scorer("knn", "P1", k=10, metric="euclidean")
    assert s.params() == {"name": "knn", "mode": "P1", "k": 10, "metric": "euclidean"}
    m = make_scorer("msp", "P2", tau=0.01, text_emb=synth.centers)
    assert m.params()["tau"] == 0.01
    o = make_scorer("c1_oe", "P2", lam=0.25, epochs=10, lr=0.05)
    assert o.params()["lam"] == 0.25 and o.params()["epochs"] == 10
