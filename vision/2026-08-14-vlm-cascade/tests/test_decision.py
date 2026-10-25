"""D14 ルールエンジンのテスト（docs/g0_nonfood_plan.md §8、design.md §8 D14）。

手で組んだ指標 dict に対して、成立条件・主検定（α / Holm）・帰結表 (a)〜(f)・判定順序・
「P1 で足りるか」を検証する。数値はすべて合成。
"""

from __future__ import annotations

from typing import Any

import pytest

from cascade.gate.decision import (
    CONDITION_KEYS,
    OUTCOMES,
    conditions_for,
    decide,
    evaluability,
    outcome_for,
    p1_sufficiency,
    primary_tests,
)
from cascade.metrics.binom import binom_test_one_sided_greater, clopper_pearson


def _rate(k: int, n: int) -> dict[str, Any]:
    lo = clopper_pearson(k, n, side="lower")[0]
    hi = clopper_pearson(k, n, side="upper")[1]
    return {
        "k": k,
        "n": n,
        "p": k / n,
        "ci": list(clopper_pearson(k, n)),
        "lower95": lo,
        "upper95": hi,
    }


def good(
    key: str = "e1/knn/P1",
    mode: str = "P1",
    family: str = "distance",
    n_before: int = 100,
    r: int = 80,
    **over: Any,
) -> dict[str, Any]:
    """採用候補になる指標（P1・距離系）。``over`` で上書き。"""
    m: dict[str, Any] = {
        "config": key,
        "method": key.split("/")[1],
        "mode": mode,
        "family": family,
        "n_before": n_before,
        "r": r,
        "reduction": _rate(r, n_before)
        if n_before
        else {"p": None, "lower95": None, "upper95": None},
        "by_type": {
            "doll_character_statue": {"n_before": 40, "r": 30, "p": 0.75},
            "person_animal": {"n_before": 30, "r": 20, "p": 0.667},
            "vehicle_street": {"n_before": 3, "r": 0, "p": 0.0},  # n_before < 10 → 評価しない
        },
        "food_loss_nominal": _rate(25, 3000),  # 0.83%、片側上限 ≈ 1.2%
        "food_loss_confirmed": {"point": 0.006, "upper95": 0.012},
        "other_frr": _rate(10, 2000) if mode == "P2" else None,  # 0.5%、上限 < 3%
        "layer_b_pass": _rate(195, 200) if mode == "P2" else None,  # 97.5%、下限 > 90%
        "layer_b_removal": _rate(5, 7) if mode == "P1" else None,
        "heldout": (
            {
                "loto": {"r": 70, "n_before": 100, "lower95": 0.62},
                "loso": {"r": 65, "n_before": 100, "lower95": 0.57},
            }
            if family == "c_series"
            else None
        ),
        "source_flag": False,
        "total_inscope_loss": {"p": 0.005},
        "mcnemar": {"b": 3, "c": 5},
    }
    m.update(over)
    return m


def passing_test() -> dict[str, Any]:
    return {"p": 1e-6, "p_adj": 1e-5, "reject": True, "level": 0.05, "method": "primary"}


def failing_test() -> dict[str, Any]:
    return {"p": 0.2, "p_adj": 0.9, "reject": False, "level": 0.05, "method": "holm(m=7)"}


# ---------------------------------------------------------------------------
# §8.1 成立条件
# ---------------------------------------------------------------------------


def test_evaluability_pooled_and_types():
    ev = evaluability(39, {"a": 10, "b": 9})
    assert ev["pooled"] is False
    assert ev["types"] == {"a": True, "b": False}
    assert ev["evaluable_types"] == ["a"]
    assert evaluability(40)["pooled"] is True


# ---------------------------------------------------------------------------
# §8.2 主検定
# ---------------------------------------------------------------------------


def _eight(n_before: int = 100, r_by: dict[str, int] | None = None) -> dict[str, dict[str, Any]]:
    keys = [
        f"e1/{m}/{mode}"
        for m in ("msp", "knn", "mahalanobis_pp", "c3_binary")
        for mode in ("P1", "P2")
    ]
    r_by = r_by or {}
    return {k: good(k, mode=k[-2:], n_before=n_before, r=r_by.get(k, 80)) for k in keys}


def test_primary_config_uses_raw_alpha_and_others_holm():
    # 主構成と msp/P1 は同じ 33/100（p ≈ 0.044）。残り 6 構成は 10/100（p ≈ 1）
    keys = list(_eight())
    r_by = dict.fromkeys(keys, 10)
    r_by["e1/knn/P1"] = 33
    r_by["e1/msp/P1"] = 33
    metrics = _eight(r_by=r_by)
    tests = primary_tests(metrics, primary_config="e1/knn/P1", alpha=0.05, h0_p=0.25, holm_m=7)
    p_raw = binom_test_one_sided_greater(33, 100, 0.25)
    assert 0.03 < p_raw < 0.05
    assert tests["e1/knn/P1"]["method"] == "primary"
    assert tests["e1/knn/P1"]["p"] == pytest.approx(p_raw)
    assert tests["e1/knn/P1"]["reject"] is True  # α = 0.05 の素の検定
    assert tests["e1/msp/P1"]["method"] == "holm(m=7)"
    assert tests["e1/msp/P1"]["p"] == pytest.approx(p_raw)
    # 7 構成中で最小 p なので水準は α/7 ≈ 0.0071 < 0.044 → Holm では不合格
    assert tests["e1/msp/P1"]["reject"] is False
    assert tests["e1/msp/P1"]["level"] == pytest.approx(0.05 / 7)
    assert tests["e1/msp/P1"]["p_adj"] == pytest.approx(min(1.0, p_raw * 7))
    others = [k for k in keys if k not in ("e1/knn/P1", "e1/msp/P1")]
    assert not any(tests[k]["reject"] for k in others)
    assert len(tests) == 8


def test_holm_step_down_rejects_when_all_are_strong():
    metrics = _eight(r_by=dict.fromkeys(_eight(), 80))  # 80/100 → p ≈ 0 で全構成合格
    tests = primary_tests(metrics, primary_config="e1/knn/P1", holm_m=7)
    assert all(t["reject"] for t in tests.values())


def test_holm_pads_to_m_when_fewer_configs():
    metrics = {"e1/knn/P1": good(r=80), "e1/msp/P1": good("e1/msp/P1", r=33)}
    tests = primary_tests(metrics, primary_config="e1/knn/P1", holm_m=7)
    # 1 構成しか無くても m = 7 で調整される（p × 7）
    assert tests["e1/msp/P1"]["method"] == "holm(m=7)"
    assert tests["e1/msp/P1"]["p_adj"] == pytest.approx(min(1.0, tests["e1/msp/P1"]["p"] * 7))


def test_zero_n_before_gives_p_one():
    tests = primary_tests(
        {"e1/knn/P1": good(n_before=0, r=0, reduction={"p": None})}, primary_config="e1/knn/P1"
    )
    assert tests["e1/knn/P1"]["p"] == 1.0
    assert tests["e1/knn/P1"]["reject"] is False


# ---------------------------------------------------------------------------
# §8.3 帰結表
# ---------------------------------------------------------------------------


def test_conditions_applicability_by_mode_and_family():
    c_p1 = conditions_for(good(), passing_test())
    assert set(c_p1) == set(CONDITION_KEYS)
    assert c_p1["d"] is None and c_p1["e"] is None
    assert all(c_p1[k] for k in ("a", "b", "c", "f"))
    c_p2 = conditions_for(good("e1/knn/P2", mode="P2"), passing_test())
    assert c_p2["d"] is True
    c_c3 = conditions_for(good("e1/c3_binary/P1", family="c_series"), passing_test())
    assert c_c3["e"] is True


def test_inevaluable_precedes_everything():
    res = outcome_for(good(), passing_test(), evaluable=False)
    assert res["outcome"] == "評価不能"


@pytest.mark.parametrize(
    "over",
    [
        {"reduction": _rate(20, 100)},  # 片側上限 ≈ 28% < 50%
        {"food_loss_confirmed": {"point": 0.045, "upper95": 0.06}},  # 確認済み点推定 > 4%
        {"food_loss_nominal": _rate(180, 3000)},  # 名目点推定 6% > 5%
    ],
)
def test_reject_branches(over):
    assert outcome_for(good(**over), passing_test())["outcome"] == "不採用"


def test_reject_precedes_candidate_even_if_other_conditions_hold():
    # 不整合な dict（点 0.8 だが上限 0.4）でも判定順序どおり不採用が先に来る
    m = good()
    m["reduction"] = {**m["reduction"], "upper95": 0.4}
    assert outcome_for(m, passing_test())["outcome"] == "不採用"


def test_candidate_when_all_applicable_conditions_hold():
    assert outcome_for(good(), passing_test())["outcome"] == "採用候補"
    assert outcome_for(good("e1/knn/P2", mode="P2"), passing_test())["outcome"] == "採用候補"
    assert (
        outcome_for(good("e1/c3_binary/P2", mode="P2", family="c_series"), passing_test())[
            "outcome"
        ]
        == "採用候補"
    )


@pytest.mark.parametrize(
    "over, test, failed",
    [
        ({}, failing_test(), "a"),  # 点推定は満たすが検定不合格
        ({"reduction": _rate(45, 100)}, passing_test(), "a"),  # 点推定 < 50%（上限は > 50%）
        ({"food_loss_confirmed": None}, passing_test(), "b"),  # 目視未実施
        ({"food_loss_nominal": _rate(70, 3000)}, passing_test(), "b"),  # 名目 2.3%、上限 > 2.5%
        (
            {"by_type": {"doll_character_statue": {"n_before": 40, "r": 4, "p": 0.1}}},
            passing_test(),
            "c",
        ),
        ({"source_flag": True}, passing_test(), "f"),
    ],
)
def test_hold_branches(over, test, failed):
    res = outcome_for(good(**over), test)
    assert res["outcome"] == "保留", res
    assert res["conditions"][failed] is False


def test_hold_p2_specific_and_c_series_specific():
    p2 = good("e1/knn/P2", mode="P2", layer_b_pass=_rate(170, 200))  # 85% → 下限 < 90%
    res = outcome_for(p2, passing_test())
    assert res["outcome"] == "保留" and res["conditions"]["d"] is False
    c3 = good(
        "e1/c3_binary/P1",
        family="c_series",
        heldout={"loto": {"lower95": 0.1}, "loso": {"lower95": 0.6}},
    )
    res = outcome_for(c3, passing_test())
    assert res["outcome"] == "保留" and res["conditions"]["e"] is False


def test_condition_c_ignores_small_types_and_is_vacuous_without_evaluable_types():
    m = good(by_type={"x": {"n_before": 5, "r": 0, "p": 0.0}})
    assert conditions_for(m, passing_test())["c"] is True


def test_decide_table_eight_rows_exhaustive_and_ordered():
    metrics = _eight()
    metrics["e1/msp/P1"]["reduction"] = _rate(20, 100)  # 不採用
    metrics["e1/msp/P2"]["food_loss_confirmed"] = None  # 保留
    table = decide(metrics, primary_config="e1/knn/P1", holm_m=7)
    assert table.height == 8
    assert table["config"].to_list() == list(metrics)
    assert set(table["outcome"].to_list()) <= set(OUTCOMES)
    out = dict(zip(table["config"].to_list(), table["outcome"].to_list(), strict=True))
    assert out["e1/msp/P1"] == "不採用"
    assert out["e1/msp/P2"] == "保留"
    assert out["e1/knn/P1"] == "採用候補"
    assert table.filter(table["test_method"] == "primary").height == 1
    assert {f"cond_{c}" for c in CONDITION_KEYS} <= set(table.columns)


def test_decide_all_inevaluable_when_pooled_small():
    table = decide(
        _eight(n_before=39, r_by=dict.fromkeys(_eight(), 30)), primary_config="e1/knn/P1"
    )
    assert table["outcome"].to_list() == ["評価不能"] * 8


# ---------------------------------------------------------------------------
# §8.4 P1 で足りるか
# ---------------------------------------------------------------------------


def test_p1_sufficiency_verdicts_and_mcnemar():
    metrics = {
        "e1/knn/P1": good(
            "e1/knn/P1", r=85, total_inscope_loss={"p": 0.004}, mcnemar={"b": 2, "c": 9}
        ),
        "e1/knn/P2": good("e1/knn/P2", mode="P2", r=85, total_inscope_loss={"p": 0.03}),
        "e1/msp/P1": good(
            "e1/msp/P1", r=40, total_inscope_loss={"p": 0.004}, mcnemar={"b": 0, "c": 0}
        ),
        "e1/msp/P2": good("e1/msp/P2", mode="P2", r=88, total_inscope_loss={"p": 0.003}),
        "e1/c3_binary/P1": good(
            "e1/c3_binary/P1", family="c_series", reduction={"p": None, "lower95": None}
        ),
        "e1/c3_binary/P2": good("e1/c3_binary/P2", mode="P2", family="c_series"),
    }
    pairs = [
        ("knn", "e1/knn/P1", "e1/knn/P2"),
        ("msp", "e1/msp/P1", "e1/msp/P2"),
        ("c3_binary", "e1/c3_binary/P1", "e1/c3_binary/P2"),
    ]
    df = p1_sufficiency(metrics, pairs=pairs, factor=0.9)
    rows = {r["method"]: r for r in df.to_dicts()}
    assert rows["knn"]["verdict"] == "全画像ゲート不要"
    assert rows["knn"]["p2_reduction_point_x_factor"] == pytest.approx(0.85 * 0.9)
    assert rows["knn"]["p1_reduction_lower95"] >= 0.85 * 0.9
    assert 0 < rows["knn"]["mcnemar_p"] < 0.1  # b=2, c=9
    assert rows["msp"]["verdict"] == "P1 では不足"
    assert rows["msp"]["mcnemar_p"] == 1.0  # 不一致なし
    assert rows["c3_binary"]["verdict"] == "評価不能"
