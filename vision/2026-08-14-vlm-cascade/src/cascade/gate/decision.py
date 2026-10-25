"""D14 判定規則のルールエンジン（g0_nonfood_plan.md §8、design.md §8 D14）。

入力は構成ごとの指標 dict（:mod:`cascade.gate.pipeline` が作る。テストは手で組む）で、
数値だけを見て帰結を返す。判定に使う CI は片側 95% Clopper–Pearson（§7.4）。

- §8.1 成立条件: 層 A eval の合算 n_before ≥ 40（未満なら全構成「評価不能」）、型別は n_before ≥ 10 のみ評価
- §8.2 主検定: r ～ Binom(n_before, p)、H0: p ≤ 0.25 の片側正確二項検定。主構成（E1 × kNN × P1）は
  α = 0.05、残り 7 構成は Holm（m = 7）
- §8.3 帰結表: 評価不能 → 不採用 → 採用候補 → 保留 の順に判定し、必ずどれか 1 つに落ちる
- §8.4 「P1 で足りるか」: 検出器ごとに P1 と P2 を対応比較（McNemar は報告のみ）

指標 dict の主要キー（欠けたキーは None として扱い、該当条件は「未充足」になる）::

    config, method, mode, family ("distance" | "msp" | "c_series"), n_before, r,
    reduction: {p, lower95, upper95}, by_type: {type: {n_before, r, p}},
    food_loss_nominal: {p, upper95}, food_loss_confirmed: {point, upper95} | None,
    other_frr: {p, upper95} (P2), layer_b_pass: {p, lower95} (P2), layer_b_removal: {p} (P1),
    heldout: {loto: {r, n_before, lower95}, loso: {...}} (C 系), source_flag: bool,
    total_inscope_loss: {p}, mcnemar: {b, c} (P1 行に P1 vs P2 の不一致セル)
"""

from __future__ import annotations

from typing import Any

import polars as pl

from cascade.metrics.binom import binom_test_one_sided_greater
from cascade.metrics.holm import holm
from cascade.metrics.mcnemar import mcnemar_exact

__all__ = [
    "CONDITION_KEYS",
    "DEFAULT_THRESHOLDS",
    "OUTCOMES",
    "decide",
    "evaluability",
    "outcome_for",
    "p1_sufficiency",
    "primary_tests",
]

OUTCOMES: tuple[str, ...] = ("評価不能", "不採用", "採用候補", "保留")
CONDITION_KEYS: tuple[str, ...] = ("a", "b", "c", "d", "e", "f")

DEFAULT_THRESHOLDS: dict[str, float] = {
    "a_reduction_point": 0.50,
    "b_confirmed_upper": 0.02,
    "b_nominal_upper": 0.025,
    "c_type_reduction_point": 0.20,
    "d_other_frr_upper": 0.03,
    "d_layer_b_pass_lower": 0.90,
    "e_heldout_lower": 0.25,
    "f_source_ratio": 3.0,
    "f_source_diff": 0.05,
    "reject_reduction_upper": 0.50,
    "reject_confirmed_point": 0.04,
    "reject_nominal_point": 0.05,
    "p1_sufficiency_factor": 0.9,
}


def _get(d: dict[str, Any] | None, key: str) -> Any:
    return None if d is None else d.get(key)


def _is_num(x: Any) -> bool:
    return isinstance(x, int | float) and x == x  # NaN を除く


# ---------------------------------------------------------------------------
# §8.1 成立条件
# ---------------------------------------------------------------------------


def evaluability(
    n_before_pooled: int,
    n_before_by_type: dict[str, int] | None = None,
    *,
    min_pooled: int = 40,
    min_type: int = 10,
) -> dict[str, Any]:
    """合算と型別の評価可否。"""
    by_type = {t: int(n) >= min_type for t, n in (n_before_by_type or {}).items()}
    return {
        "pooled": int(n_before_pooled) >= min_pooled,
        "n_before": int(n_before_pooled),
        "min_pooled": min_pooled,
        "min_type": min_type,
        "types": by_type,
        "evaluable_types": sorted(t for t, ok in by_type.items() if ok),
    }


# ---------------------------------------------------------------------------
# §8.2 主検定
# ---------------------------------------------------------------------------


def primary_tests(
    metrics: dict[str, dict[str, Any]],
    *,
    primary_config: str,
    alpha: float = 0.05,
    h0_p: float = 0.25,
    holm_m: int | None = 7,
    tested: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """構成ごとの片側正確二項検定。主構成は α、残りは Holm。

    ``tested`` は検定対象の構成キー（既定は metrics の全キー）。副次構成は含めないこと（§8.2）。
    Holm の m は ``holm_m`` が None なら残り構成数、指定があればその値以上の構成数を要求する
    （構成が m より少ない場合は不足分を p = 1 のダミーで埋め、m で調整する）。
    """
    keys = list(tested) if tested is not None else list(metrics)
    out: dict[str, dict[str, Any]] = {}
    pvals: dict[str, float] = {}
    for k in keys:
        m = metrics[k]
        n_before = int(m.get("n_before") or 0)
        r = int(m.get("r") or 0)
        pvals[k] = binom_test_one_sided_greater(r, n_before, h0_p) if n_before > 0 else 1.0
    if primary_config in pvals:
        p = pvals.pop(primary_config)
        out[primary_config] = {
            "p": p,
            "p_adj": p,
            "reject": bool(p <= alpha),
            "level": alpha,
            "method": "primary",
        }
    if pvals:
        m_target = holm_m if holm_m is not None else len(pvals)
        padded = dict(pvals)
        for i in range(max(0, m_target - len(pvals))):
            padded[f"__pad{i}"] = 1.0
        adj = holm(padded, alpha=alpha)
        for k in pvals:
            row = dict(adj[k])
            row["method"] = f"holm(m={len(padded)})"
            row["reject"] = bool(row["reject"])
            out[k] = row
    return out


# ---------------------------------------------------------------------------
# §8.3 帰結表
# ---------------------------------------------------------------------------


def conditions_for(
    m: dict[str, Any],
    test: dict[str, Any] | None,
    thr: dict[str, float] | None = None,
    *,
    min_type: int = 10,
) -> dict[str, bool | None]:
    """(a)〜(f) の充足。適用外は None（P1 の (d)、非 C 系の (e)）。"""
    t = {**DEFAULT_THRESHOLDS, **(thr or {})}
    mode = m.get("mode")
    family = m.get("family")
    red = m.get("reduction") or {}

    # (a) 主検定合格かつ削減率の点推定 ≥ 50%
    a = (
        bool(test and test.get("reject"))
        and _is_num(red.get("p"))
        and red["p"] >= t["a_reduction_point"]
    )

    # (b) 確認済み損失の片側上限 ≤ 2% かつ名目の片側上限 ≤ 2.5%
    nom = m.get("food_loss_nominal") or {}
    conf = m.get("food_loss_confirmed")
    b_nom = _is_num(nom.get("upper95")) and nom["upper95"] <= t["b_nominal_upper"]
    b_conf = (
        conf is not None
        and _is_num(_get(conf, "upper95"))
        and conf["upper95"] <= t["b_confirmed_upper"]
    )
    b = bool(b_nom and b_conf)

    # (c) 評価可能な全型で削減率の点推定 ≥ 20%（評価可能な型が無ければ空条件として True）
    by_type = m.get("by_type") or {}
    ev = {k: v for k, v in by_type.items() if int(v.get("n_before") or 0) >= min_type}
    c = all(_is_num(v.get("p")) and v["p"] >= t["c_type_reduction_point"] for v in ev.values())

    # (d) P2 のみ: 他クラス FRR 上限 ≤ 3% かつ層 B 通過率 下限 ≥ 90%
    d: bool | None
    if mode == "P2":
        frr = m.get("other_frr") or {}
        lb = m.get("layer_b_pass") or {}
        d = bool(
            _is_num(frr.get("upper95"))
            and frr["upper95"] <= t["d_other_frr_upper"]
            and _is_num(lb.get("lower95"))
            and lb["lower95"] >= t["d_layer_b_pass_lower"]
        )
    else:
        d = None

    # (e) C 系のみ: LOTO・LOSO の held-out 合算 下限 ≥ 25%
    e: bool | None
    if family == "c_series":
        ho = m.get("heldout") or {}
        ok = []
        for axis in ("loto", "loso"):
            h = ho.get(axis) or {}
            ok.append(_is_num(h.get("lower95")) and h["lower95"] >= t["e_heldout_lower"])
        e = bool(ok and all(ok))
    else:
        e = None

    # (f) ソース検出フラグなし
    f = not bool(m.get("source_flag", False))
    return {"a": bool(a), "b": b, "c": bool(c), "d": d, "e": e, "f": f}


def outcome_for(
    m: dict[str, Any],
    test: dict[str, Any] | None,
    thr: dict[str, float] | None = None,
    *,
    evaluable: bool = True,
    min_type: int = 10,
) -> dict[str, Any]:
    """1 構成の帰結。判定順序: 評価不能 → 不採用 → 採用候補 → 保留。"""
    t = {**DEFAULT_THRESHOLDS, **(thr or {})}
    cond = conditions_for(m, test, t, min_type=min_type)
    reasons: list[str] = []
    if not evaluable:
        return {"outcome": "評価不能", "conditions": cond, "reasons": ["合算 n_before < 最小値"]}

    red = m.get("reduction") or {}
    nom = m.get("food_loss_nominal") or {}
    conf = m.get("food_loss_confirmed")
    if _is_num(red.get("upper95")) and red["upper95"] < t["reject_reduction_upper"]:
        reasons.append(f"削減率の片側上限 {red['upper95']:.3f} < {t['reject_reduction_upper']:.2f}")
    if (
        conf is not None
        and _is_num(_get(conf, "point"))
        and conf["point"] > t["reject_confirmed_point"]
    ):
        reasons.append(
            f"確認済み損失の点推定 {conf['point']:.3f} > {t['reject_confirmed_point']:.2f}"
        )
    if _is_num(nom.get("p")) and nom["p"] > t["reject_nominal_point"]:
        reasons.append(f"名目損失の点推定 {nom['p']:.3f} > {t['reject_nominal_point']:.2f}")
    if reasons:
        return {"outcome": "不採用", "conditions": cond, "reasons": reasons}

    applicable = [v for v in cond.values() if v is not None]
    if all(applicable):
        return {
            "outcome": "採用候補",
            "conditions": cond,
            "reasons": ["(a)〜(f) の適用条件を全て満たす"],
        }
    failed = [k for k, v in cond.items() if v is False]
    if conf is None:
        reasons.append("確認済み損失が未計測")
    reasons.append("未充足: " + ", ".join(f"({k})" for k in failed))
    return {"outcome": "保留", "conditions": cond, "reasons": reasons}


def decide(
    metrics: dict[str, dict[str, Any]],
    *,
    primary_config: str,
    alpha: float = 0.05,
    h0_p: float = 0.25,
    holm_m: int | None = 7,
    min_n_before_pooled: int = 40,
    min_n_before_type: int = 10,
    thresholds: dict[str, float] | None = None,
    tested: list[str] | None = None,
) -> pl.DataFrame:
    """主構成の帰結表（§13.3 の列）。行順は ``tested``（既定は metrics の順）。"""
    keys = list(tested) if tested is not None else list(metrics)
    tests = primary_tests(
        metrics, primary_config=primary_config, alpha=alpha, h0_p=h0_p, holm_m=holm_m, tested=keys
    )
    rows: list[dict[str, Any]] = []
    for k in keys:
        m = metrics[k]
        ev = evaluability(int(m.get("n_before") or 0), min_pooled=min_n_before_pooled)
        res = outcome_for(
            m, tests.get(k), thresholds, evaluable=ev["pooled"], min_type=min_n_before_type
        )
        cond = res["conditions"]
        rows.append(
            {
                "config": k,
                "mode": m.get("mode"),
                "method": m.get("method"),
                "n_before": int(m.get("n_before") or 0),
                "r": int(m.get("r") or 0),
                "reduction_p": _get(m.get("reduction"), "p"),
                "reduction_lower95": _get(m.get("reduction"), "lower95"),
                "p_value": tests[k]["p"],
                "p_adj": tests[k]["p_adj"],
                "test_reject": bool(tests[k]["reject"]),
                "test_method": tests[k]["method"],
                **{f"cond_{c}": cond[c] for c in CONDITION_KEYS},
                "outcome": res["outcome"],
                "reasons": "; ".join(res["reasons"]),
            }
        )
    schema = {
        "config": pl.Utf8,
        "mode": pl.Utf8,
        "method": pl.Utf8,
        "n_before": pl.Int64,
        "r": pl.Int64,
        "reduction_p": pl.Float64,
        "reduction_lower95": pl.Float64,
        "p_value": pl.Float64,
        "p_adj": pl.Float64,
        "test_reject": pl.Boolean,
        "test_method": pl.Utf8,
        **{f"cond_{c}": pl.Boolean for c in CONDITION_KEYS},
        "outcome": pl.Utf8,
        "reasons": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema)


# ---------------------------------------------------------------------------
# §8.4 P1 で足りるか
# ---------------------------------------------------------------------------


def p1_sufficiency(
    metrics: dict[str, dict[str, Any]],
    *,
    pairs: list[tuple[str, str, str]],
    factor: float = 0.9,
) -> pl.DataFrame:
    """検出器ごとの P1 vs P2 対応比較（§13.4 の列）。

    ``pairs`` は (method, P1 のキー, P2 のキー)。判定は
    「P1 削減率の片側 95% 下限 ≥ P2 削減率の点推定 × factor かつ P1 対象内総損失 ≤ P2 対象内総損失」。
    McNemar は P1 の指標に ``mcnemar: {b, c}`` があれば計算し、報告のみに使う。
    """
    rows: list[dict[str, Any]] = []
    for method, k1, k2 in pairs:
        m1, m2 = metrics.get(k1) or {}, metrics.get(k2) or {}
        lower1 = _get(m1.get("reduction"), "lower95")
        point2 = _get(m2.get("reduction"), "p")
        loss1 = _get(m1.get("total_inscope_loss"), "p")
        loss2 = _get(m2.get("total_inscope_loss"), "p")
        mc = m1.get("mcnemar") or m2.get("mcnemar") or {}
        p_mc = (
            mcnemar_exact(int(mc["b"]), int(mc["c"]), alternative="two")
            if _is_num(mc.get("b")) and _is_num(mc.get("c"))
            else None
        )
        ok_red = _is_num(lower1) and _is_num(point2) and lower1 >= point2 * factor
        ok_loss = _is_num(loss1) and _is_num(loss2) and loss1 <= loss2
        if not (_is_num(lower1) and _is_num(point2) and _is_num(loss1) and _is_num(loss2)):
            verdict = "評価不能"
        else:
            verdict = "全画像ゲート不要" if (ok_red and ok_loss) else "P1 では不足"
        rows.append(
            {
                "method": method,
                "p1_config": k1,
                "p2_config": k2,
                "p1_reduction_lower95": lower1,
                "p2_reduction_point_x_factor": None if not _is_num(point2) else point2 * factor,
                "p1_total_inscope_loss": loss1,
                "p2_total_inscope_loss": loss2,
                "mcnemar_b": None if not _is_num(mc.get("b")) else int(mc["b"]),
                "mcnemar_c": None if not _is_num(mc.get("c")) else int(mc["c"]),
                "mcnemar_p": p_mc,
                "verdict": verdict,
            }
        )
    schema = {
        "method": pl.Utf8,
        "p1_config": pl.Utf8,
        "p2_config": pl.Utf8,
        "p1_reduction_lower95": pl.Float64,
        "p2_reduction_point_x_factor": pl.Float64,
        "p1_total_inscope_loss": pl.Float64,
        "p2_total_inscope_loss": pl.Float64,
        "mcnemar_b": pl.Int64,
        "mcnemar_c": pl.Int64,
        "mcnemar_p": pl.Float64,
        "verdict": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema)
