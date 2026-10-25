"""cascade.metrics.rates / bootstrap: 主指標の Rate、確認済み損失のデルタ法とブートストラップ（plan §7.2、§7.4）。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cascade.metrics.binom import clopper_pearson
from cascade.metrics.bootstrap import (
    cluster_bootstrap,
    delta_method_ratio,
    food_loss_confirmed_bootstrap,
)
from cascade.metrics.rates import (
    Rate,
    false_acceptance,
    food_loss_confirmed,
    food_loss_nominal,
    other_class_frr,
    pass_rate,
    rate,
    reduction,
)


def test_rate_fields_and_aliases() -> None:
    r = rate(20, 2000, "x")
    assert isinstance(r, Rate)
    assert (r.k, r.n) == (20, 2000)
    assert r.p == pytest.approx(0.01)
    assert r.ci == clopper_pearson(20, 2000)
    assert r.ci_two_95 == r.ci
    assert r.lower95 == clopper_pearson(20, 2000, side="lower")[0]
    assert r.upper95 == clopper_pearson(20, 2000, side="upper")[1]
    assert r.ci_lower_one_sided_95 == r.lower95
    assert r.ci_upper_one_sided_95 == r.upper95
    assert r.ci[0] < r.lower95 < r.p < r.upper95 < r.ci[1]
    d = r.to_dict()
    assert d["name"] == "x" and d["k"] == 20 and d["ci_hi"] == r.ci[1]


def test_rate_empty_is_not_evaluable() -> None:
    r = rate(0, 0)
    assert math.isnan(r.p)
    assert r.ci == (0.0, 1.0)
    assert r.lower95 == 0.0 and r.upper95 == 1.0
    assert not r.evaluable
    with pytest.raises(ValueError):
        rate(3, 2)


def test_named_helpers() -> None:
    assert false_acceptance(5, 100).name == "false_acceptance"
    assert false_acceptance(5, 100).p == pytest.approx(0.05)
    red = reduction(30, 40)
    assert red.name == "reduction" and red.p == pytest.approx(0.75)
    # §8.2 の主検定同値表現: 片側 CP 下限 ≥ 25%
    assert red.lower95 > 0.25
    assert reduction(10, 40).lower95 < 0.25
    fl = food_loss_nominal(30, 3000)
    assert fl.p == pytest.approx(0.01)
    assert fl.upper95 < 0.025  # §8.3 (b) 名目上限
    assert pass_rate(195, 200).lower95 > 0.90  # §8.3 (d)
    assert other_class_frr(10, 2000).upper95 < 0.03


def test_food_loss_confirmed_point_and_ci_shape() -> None:
    out = food_loss_confirmed(r_true=30, n_pass=2970, q_hat=0.9, n_q=200)
    assert out["point"] == pytest.approx(30 / (30 + 2970 * 0.9))
    lo, hi = out["ci_delta_95"]
    assert lo < out["point"] < hi
    assert out["ci_lower_one_sided_95"] < out["point"] < out["ci_upper_one_sided_95"]
    assert lo < out["ci_lower_one_sided_95"]
    assert out["ci_upper_one_sided_95"] < hi
    # q̂ を小さく見積もるほど分母が減り損失は増える
    assert food_loss_confirmed(30, 2970, 0.5, 200)["point"] > out["point"]


def test_food_loss_confirmed_delta_vs_bootstrap() -> None:
    args = dict(r_true=30, n_pass=2970, q_hat=0.9, n_q=200)
    delta = food_loss_confirmed(**args)
    boot = food_loss_confirmed_bootstrap(**args, B=4000, seed=1)
    assert boot["point"] == pytest.approx(delta["point"])
    assert boot["upper_one_sided_95"] == pytest.approx(delta["ci_upper_one_sided_95"], abs=1.5e-3)
    assert boot["ci_95"][0] == pytest.approx(delta["ci_delta_95"][0], abs=1.5e-3)
    assert boot["ci_95"][1] == pytest.approx(delta["ci_delta_95"][1], abs=1.5e-3)
    # 決定的
    again = food_loss_confirmed_bootstrap(**args, B=4000, seed=1)
    assert again == boot


def test_food_loss_confirmed_degenerate() -> None:
    out = food_loss_confirmed(0, 0, 0.9, 200)
    assert math.isnan(out["point"])
    zero = food_loss_confirmed(0, 1000, 0.9, 200)
    assert zero["point"] == 0.0 and zero["se"] == 0.0
    with pytest.raises(ValueError):
        food_loss_confirmed(1, 10, 1.5, 10)


def test_delta_method_ratio_matches_manual() -> None:
    p, se = delta_method_ratio(10.0, 100.0, 9.0, 90.0)
    assert p == 0.1
    assert se == pytest.approx(math.sqrt(9 / 100**2 + 100 * 90 / 100**4))
    assert all(math.isnan(v) for v in delta_method_ratio(1.0, 0.0, 1.0, 1.0))


def test_cluster_bootstrap_ratio_and_determinism() -> None:
    rng = np.random.default_rng(0)
    clusters: dict[str, tuple[int, int]] = {}
    for i in range(200):
        n = int(rng.integers(5, 30))
        k = int(rng.binomial(n, 0.1))
        clusters[f"b{i}"] = (k, n)
    point, lo, hi = cluster_bootstrap(clusters, B=1000, seed=3)
    tot_k = sum(k for k, _ in clusters.values())
    tot_n = sum(n for _, n in clusters.values())
    assert point == pytest.approx(tot_k / tot_n)
    assert lo <= point <= hi
    assert 0.0 < hi - lo < 0.1
    assert cluster_bootstrap(clusters, B=1000, seed=3) == (point, lo, hi)
    assert cluster_bootstrap(clusters, B=1000, seed=4) != (point, lo, hi)


def test_cluster_bootstrap_mean_of_ratios_and_edge_cases() -> None:
    clusters = {"a": (1, 2), "b": (0, 10)}
    point, lo, hi = cluster_bootstrap(clusters, stat="mean_of_ratios", B=200)
    assert point == pytest.approx(0.25)
    assert lo <= point <= hi
    assert all(math.isnan(v) for v in cluster_bootstrap({}))
    with pytest.raises(ValueError):
        cluster_bootstrap(clusters, stat="median")
    with pytest.raises(ValueError):
        cluster_bootstrap({"a": (3, 2)})
