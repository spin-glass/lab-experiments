"""G0 パイプライン（g0_nonfood_plan.md §4〜§8、design.md §6 G0）。

1 回の :func:`run_g0` で以下を行う。

1. P0 = Stage1 zero-shot argmax（τ = 1）。「正しく food」= Yelp ラベル food ∧ P0 = food
2. 主 8 構成（E1 × {MSP, kNN, Mahalanobis++, C3} × {P1, P2}）と副次構成の検出器を
   ``yelp_tune_fit``（C 系の負例は ``ext_tune`` のみ）で fit
3. 動作点は ``yelp_tune_cal`` の正しく food のスコアだけで決める（§7.1。主 1%、副 0.5% / 2%）
4. stage（work / eval）の集合にゲートを適用し、§7.2 / §7.3 の指標を構成ごとに計算
5. C 系は LOTO / LOSO（held-out は tune + work (+ eval) の外部集合全体で評価。§5.4）
6. 記述分析（§6: 距離のヒストグラム・4 象限表・AUROC / 重なり / KL）
7. D14 帰結表（§8.3）と「P1 で足りるか」（§8.4）

eval を fit・閾値決定に渡す経路は存在しない（役割はブロック単位で固定）。
出力の ``ood_scores`` / ``gate_decisions`` は design.md §5 のスキーマに一致する。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl

from cascade.gate.data_loader import (
    AUX_FAR_TYPE,
    CONTRAST_TYPE,
    LAYER_A_TYPES,
    NEAR_FOOD_TYPE,
    OUT_OF_SCOPE,
    Block,
    G0Data,
    attach_p0,
    concat_blocks,
)
from cascade.gate.decision import decide, evaluability, p1_sufficiency
from cascade.gate.scorers import make_scorer
from cascade.gate.splits import fold_evaluable, loso_folds, loto_folds
from cascade.gate.threshold import ThresholdResult, apply_threshold, fit_threshold
from cascade.metrics.bootstrap import cluster_bootstrap
from cascade.metrics.descriptive import (
    auroc,
    fpr_at_tpr,
    histogram_table,
    kl_divergence_hist,
    overlap_coefficient,
    quadrant_counts,
)
from cascade.metrics.rates import (
    false_acceptance,
    food_loss_confirmed,
    food_loss_nominal,
    pass_rate,
    reduction,
)
from cascade.tracking.adapter import RunRecorder, new_run_id

__all__ = ["G0Result", "Spec", "build_specs", "run_g0"]

MODES: tuple[str, ...] = ("P1", "P2")


# ---------------------------------------------------------------------------
# 構成
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """1 構成（encoder × method × mode）。``main`` の 8 構成だけが検定対象。"""

    key: str
    encoder: str
    method: str
    method_id: str
    mode: str
    params: dict[str, Any]
    family: str
    main: bool
    work_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "encoder": self.encoder,
            "method": self.method,
            "method_id": self.method_id,
            "mode": self.mode,
            "params": dict(self.params),
            "family": self.family,
            "main": self.main,
            "work_only": self.work_only,
        }


def _family(method: str, c_series: list[str]) -> str:
    if method in c_series:
        return "c_series"
    return "msp" if method == "msp" else "distance"


def build_specs(config: dict[str, Any], encoder_name: str, stage: str) -> list[Spec]:
    """config の detectors から構成一覧を作る。eval では work_only の副次構成を落とす。"""
    det = dict(config.get("detectors", {}))
    main = dict(det.get("main", {}))
    params_all = dict(det.get("params", {}))
    c_series = [str(x) for x in det.get("c_series", ("c3_binary", "c1_oe", "c2_sixway"))]
    specs: list[Spec] = []
    for method in main.get("methods", ("msp", "knn", "mahalanobis_pp", "c3_binary")):
        for mode in main.get("modes", MODES):
            specs.append(
                Spec(
                    key=f"{encoder_name}/{method}/{mode}",
                    encoder=encoder_name,
                    method=str(method),
                    method_id=str(method),
                    mode=str(mode),
                    params=dict(params_all.get(method, {})),
                    family=_family(str(method), c_series),
                    main=True,
                )
            )
    for sec in det.get("secondary", []) or []:
        method = str(sec["method"])
        sid = str(sec.get("id", method))
        work_only = bool(sec.get("work_only", False))
        if work_only and stage == "eval":
            continue
        params = {**params_all.get(method, {}), **(sec.get("params") or {})}
        for mode in sec.get("modes", MODES):
            specs.append(
                Spec(
                    key=f"{encoder_name}/{sid}/{mode}",
                    encoder=encoder_name,
                    method=method,
                    method_id=sid,
                    mode=str(mode),
                    params=params,
                    family=_family(method, c_series),
                    main=False,
                    work_only=work_only,
                )
            )
    return specs


# ---------------------------------------------------------------------------
# 結果
# ---------------------------------------------------------------------------


@dataclass
class G0Result:
    """:func:`run_g0` の出力。大きな表は DataFrame、判定に使う数値は dict（JSON 化可能）。"""

    stage: str
    encoder: str
    run_id: str
    class_order: list[str]
    primary_loss: float
    loss_levels: list[float]
    specs: list[Spec]
    thresholds: dict[str, dict[str, dict[str, Any]]]
    metrics: dict[str, dict[str, dict[str, Any]]]
    tests: dict[str, dict[str, Any]]
    evaluability: dict[str, Any]
    decision_table: pl.DataFrame
    p1_sufficiency: pl.DataFrame
    heldout: pl.DataFrame
    descriptive: dict[str, Any]
    secondary: pl.DataFrame
    sizes: pl.DataFrame
    ood_scores: pl.DataFrame
    gate_decisions: pl.DataFrame
    timings: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def primary_key(self) -> str:
        return _loss_key(self.primary_loss)

    def metrics_at(self, key: str, loss: float | None = None) -> dict[str, Any]:
        return self.metrics[key][_loss_key(self.primary_loss if loss is None else loss)]

    def main_specs(self) -> list[Spec]:
        return [s for s in self.specs if s.main]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "encoder": self.encoder,
            "run_id": self.run_id,
            "primary_loss": self.primary_loss,
            "loss_levels": list(self.loss_levels),
            "specs": [s.to_dict() for s in self.specs],
            "thresholds": self.thresholds,
            "metrics": self.metrics,
            "tests": self.tests,
            "evaluability": self.evaluability,
            "decision_table": self.decision_table.to_dicts(),
            "p1_sufficiency": self.p1_sufficiency.to_dicts(),
            "heldout": self.heldout.to_dicts(),
            "descriptive": {
                "bins": self.descriptive.get("bins", {}),
                "separation": self.descriptive["separation"].to_dicts(),
                "quadrants": self.descriptive["quadrants"].to_dicts(),
                "n_histogram_rows": int(self.descriptive["histograms"].height),
            },
            "secondary": self.secondary.to_dicts(),
            "sizes": self.sizes.to_dicts(),
            "n_ood_scores": int(self.ood_scores.height),
            "n_gate_decisions": int(self.gate_decisions.height),
            "timings": self.timings,
            "notes": list(self.notes),
        }

    def flat_metrics(self) -> dict[str, Any]:
        """トラッカー向けのスカラー群（主動作点の主指標のみ）。"""
        out: dict[str, Any] = {
            "stage": self.stage,
            "evaluable": self.evaluability["pooled"],
            "n_before_pooled": self.evaluability["n_before"],
        }
        for s in self.specs:
            m = self.metrics_at(s.key)
            pre = s.key.replace("/", "_")
            out[f"{pre}/n_before"] = m["n_before"]
            out[f"{pre}/r"] = m["r"]
            out[f"{pre}/reduction_p"] = m["reduction"]["p"]
            out[f"{pre}/reduction_lower95"] = m["reduction"]["lower95"]
            out[f"{pre}/food_loss_nominal_p"] = m["food_loss_nominal"]["p"]
            out[f"{pre}/auroc"] = m["auroc"]
        for row in self.decision_table.to_dicts():
            out[f"decision/{row['config'].replace('/', '_')}"] = row["outcome"]
        return out


# ---------------------------------------------------------------------------
# 補助
# ---------------------------------------------------------------------------


def _loss_key(loss: float) -> str:
    return f"loss{loss:g}"


def _rate_dict(rate: Any) -> dict[str, Any]:
    ci = getattr(rate, "ci", None)
    if ci is None:
        ci = getattr(rate, "ci_two_95", (None, None))
    lo = getattr(rate, "lower95", None)
    if lo is None:
        lo = getattr(rate, "ci_lower_one_sided_95", None)
    hi = getattr(rate, "upper95", None)
    if hi is None:
        hi = getattr(rate, "ci_upper_one_sided_95", None)
    return {
        "k": int(rate.k),
        "n": int(rate.n),
        "p": _f(rate.p),
        "ci": [_f(ci[0]), _f(ci[1])],
        "lower95": _f(lo),
        "upper95": _f(hi),
    }


def _f(x: Any) -> float | None:
    if x is None:
        return None
    v = float(x)
    return None if v != v else v


def _rate(fn: Any, k: int, n: int) -> dict[str, Any]:
    if n <= 0:
        return {
            "k": int(k),
            "n": 0,
            "p": None,
            "ci": [None, None],
            "lower95": None,
            "upper95": None,
        }
    return _rate_dict(fn(int(k), int(n)))


def _empty_scores() -> np.ndarray:
    return np.zeros(0, dtype=np.float64)


@dataclass
class _Views:
    """stage の評価対象ブロックと、そこから導いたマスク。"""

    yelp: Block
    ext: Block
    layer_b: Block
    contrast: Block
    aux_far: Block
    # yelp
    y_label: np.ndarray
    y_pred: np.ndarray
    y_correct: np.ndarray
    y_correct_food: np.ndarray
    y_correct_other: np.ndarray
    y_business: np.ndarray
    # ext（層 A + near-food）
    x_type: np.ndarray
    x_pred_food: np.ndarray
    x_layer_a: np.ndarray
    x_near: np.ndarray
    # 層 B / 対照 / 補助 far
    b_pred_food: np.ndarray
    c_pred_food: np.ndarray
    f_pred_food: np.ndarray


def _pred(block: Block, class_order: list[str]) -> np.ndarray:
    if block.n == 0:
        return np.zeros(0, dtype=object)
    assert block.probs is not None
    return np.asarray(class_order, dtype=object)[block.probs.argmax(axis=1)]


def _by_role(block: Block, role: str) -> Block:
    """role 列に値があれば絞る。無ければ全体を返す（旧データ互換）。"""
    if block.n == 0:
        return block
    roles = block.col("role")
    if all(r is None or r == "" for r in roles):
        return block
    return block.subset(np.array([r == role for r in roles]))


def _views(data: G0Data, stage: str, sets: dict[str, Any], food: str) -> _Views:
    order = data.class_order
    yelp = data.yelp_work if stage == "work" else data.yelp_eval
    ext = data.ext_work if stage == "work" else data.ext_eval
    layer_b = _by_role(data.layer_b, stage)
    contrast = _by_role(data.food_contrast, stage)
    aux = _by_role(data.aux_far, stage)
    layer_a_types = [str(t) for t in sets.get("layer_a_types", LAYER_A_TYPES)]
    near = str(sets.get("near_food_type", NEAR_FOOD_TYPE))

    y_label = yelp.col("label") if yelp.n else np.zeros(0, dtype=object)
    y_pred = _pred(yelp, order)
    y_correct = (y_label == y_pred) if yelp.n else np.zeros(0, bool)
    y_correct_food = y_correct & (y_label == food) if yelp.n else np.zeros(0, bool)
    y_correct_other = y_correct & (y_label != food) if yelp.n else np.zeros(0, bool)
    x_type = ext.col("subject_type") if ext.n else np.zeros(0, dtype=object)
    x_pred = _pred(ext, order)
    x_label = ext.col("label") if ext.n else np.zeros(0, dtype=object)
    oos = (x_label == OUT_OF_SCOPE) if ext.n else np.zeros(0, bool)
    return _Views(
        yelp=yelp,
        ext=ext,
        layer_b=layer_b,
        contrast=contrast,
        aux_far=aux,
        y_label=y_label,
        y_pred=y_pred,
        y_correct=y_correct,
        y_correct_food=y_correct_food,
        y_correct_other=y_correct_other,
        y_business=yelp.col("business_id") if yelp.n else np.zeros(0, dtype=object),
        x_type=x_type,
        x_pred_food=(x_pred == food) if ext.n else np.zeros(0, bool),
        x_layer_a=np.isin(x_type, layer_a_types) & oos if ext.n else np.zeros(0, bool),
        x_near=(x_type == near) & oos if ext.n else np.zeros(0, bool),
        b_pred_food=(_pred(layer_b, order) == food) if layer_b.n else np.zeros(0, bool),
        c_pred_food=(_pred(contrast, order) == food) if contrast.n else np.zeros(0, bool),
        f_pred_food=(_pred(aux, order) == food) if aux.n else np.zeros(0, bool),
    )


def _fit_scorer(spec: Spec, data: G0Data, X_neg: np.ndarray, seed: int, food: str) -> Any:
    params = dict(spec.params)
    if spec.method == "msp":
        params.setdefault("text_emb", data.text_emb)
    scorer = make_scorer(spec.method, spec.mode, **params)
    scorer.fit(
        data.yelp_tune_fit.X,
        data.yelp_tune_fit.col("label"),
        X_neg if spec.family == "c_series" else None,
        class_order=list(data.class_order),
        food_label=food,
        seed=seed,
    )
    return scorer


def _score(scorer: Any, block: Block) -> np.ndarray:
    if block.n == 0:
        return _empty_scores()
    return np.asarray(scorer.score(block.X), dtype=np.float64).reshape(-1)


def _cal_scores(scorer: Any, data: G0Data, food: str) -> np.ndarray:
    """tune_cal の「正しく food」のスコア（動作点の唯一の入力）。"""
    cal = data.yelp_tune_cal
    pred = _pred(cal, data.class_order)
    mask = (cal.col("label") == food) & (pred == food)
    return _score(scorer, cal.subset(mask))


def _reject(mode: str, scores: np.ndarray, t: float, pred_food: np.ndarray) -> np.ndarray:
    """棄却マスク。P1 は P0 = food の画像だけを再判定し、他は不変（棄却しない）。"""
    if scores.size == 0:
        return np.zeros(0, bool)
    rej = apply_threshold(scores, t)
    return rej & pred_food if mode == "P1" else rej


# ---------------------------------------------------------------------------
# 指標（1 構成 × 1 動作点）
# ---------------------------------------------------------------------------


def _metrics_for(
    spec: Spec,
    thr: ThresholdResult,
    sc: dict[str, np.ndarray],
    v: _Views,
    data: G0Data,
    cfg: dict[str, Any],
    food: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    t = thr.t
    rej_y = _reject(spec.mode, sc["yelp"], t, v.y_pred == food)
    rej_x = _reject(spec.mode, sc["ext"], t, v.x_pred_food)
    rej_b = _reject(spec.mode, sc["layer_b"], t, v.b_pred_food)
    rej_c = _reject(spec.mode, sc["contrast"], t, v.c_pred_food)
    rej_f = _reject(spec.mode, sc["aux_far"], t, v.f_pred_food)
    sets = dict(cfg.get("sets", {}))
    thr_cfg = {**dict(cfg.get("decision", {}).get("thresholds", {}))}
    layer_a_types = [str(x) for x in sets.get("layer_a_types", LAYER_A_TYPES)]

    # --- 層 A: 誤受理と削減 --------------------------------------------------
    a = v.x_layer_a
    n_a = int(a.sum())
    before = a & v.x_pred_food
    n_before = int(before.sum())
    r = int((before & rej_x).sum())
    by_type: dict[str, dict[str, Any]] = {}
    for ty in layer_a_types:
        m_t = before & (v.x_type == ty)
        nb = int(m_t.sum())
        rt = int((m_t & rej_x).sum())
        by_type[ty] = {
            "n_before": nb,
            "r": rt,
            "p": (rt / nb) if nb else None,
            "n": int((a & (v.x_type == ty)).sum()),
        }

    # --- 料理損失 -------------------------------------------------------------
    cf = v.y_correct_food
    n_cf = int(cf.sum())
    k_loss = int((cf & rej_y).sum())
    nominal = _rate(food_loss_nominal, k_loss, n_cf)
    confirmed = _confirmed_loss(data, v, cf, rej_y)

    # --- 層 B ----------------------------------------------------------------
    n_b = int(v.layer_b.n)
    nb_food = int(v.b_pred_food.sum())
    layer_b_removal = (
        _rate(pass_rate, int((v.b_pred_food & rej_b).sum()), nb_food) if spec.mode == "P1" else None
    )
    layer_b_pass = _rate(pass_rate, int((~rej_b).sum()), n_b) if spec.mode == "P2" else None

    # --- P2 の他クラス FRR ---------------------------------------------------
    other_frr = None
    other_frr_by_class: dict[str, dict[str, Any]] = {}
    if spec.mode == "P2":
        co = v.y_correct_other
        other_frr = _rate(food_loss_nominal, int((co & rej_y).sum()), int(co.sum()))
        for c in data.class_order:
            if c == food:
                continue
            m_c = co & (v.y_label == c)
            other_frr_by_class[c] = _rate(
                food_loss_nominal, int((m_c & rej_y).sum()), int(m_c.sum())
            )

    # --- ソース対照正例（§3.4）----------------------------------------------
    c_pop = v.c_pred_food if spec.mode == "P1" else np.ones(v.contrast.n, bool)
    contrast_rej = _rate(food_loss_nominal, int((c_pop & rej_c).sum()), int(c_pop.sum()))
    source_flag = False
    if contrast_rej["p"] is not None and nominal["p"] is not None:
        ratio = float(thr_cfg.get("f_source_ratio", 3.0))
        diff = float(thr_cfg.get("f_source_diff", 0.05))
        source_flag = (
            contrast_rej["p"] >= ratio * nominal["p"] and (contrast_rej["p"] - nominal["p"]) >= diff
        )

    # --- near-food / 補助 far の誤受理 --------------------------------------
    near = v.x_near
    n_near = int(near.sum())
    near_fa = {
        "P0": _rate(false_acceptance, int((near & v.x_pred_food).sum()), n_near),
        "gate": _rate(false_acceptance, int((near & v.x_pred_food & ~rej_x).sum()), n_near),
    }
    n_far = int(v.aux_far.n)
    far_fa = {
        "P0": _rate(false_acceptance, int(v.f_pred_food.sum()), n_far),
        "gate": _rate(false_acceptance, int((v.f_pred_food & ~rej_f).sum()), n_far),
    }

    # --- 対象内総損失（§8.4）: Yelp の正しく分類された 5 クラス＋層 B ------------
    n_inscope = int(v.y_correct.sum()) + n_b
    if spec.mode == "P1":
        k_inscope = k_loss
    else:
        k_inscope = int((v.y_correct & rej_y).sum()) + int(rej_b.sum())
    total_loss = _rate(food_loss_nominal, k_inscope, n_inscope)

    # --- AUROC / FPR@95（§7.3）------------------------------------------------
    pos = sc["yelp"][cf] if spec.mode == "P1" else sc["yelp"]
    neg = sc["ext"][a]
    au = _f(auroc(pos, neg)) if pos.size and neg.size else None
    fpr = _f(fpr_at_tpr(pos, neg, tpr=0.95)) if pos.size and neg.size else None

    # --- 店舗単位ブートストラップ（感度分析。判定には使わない）-----------------
    boot = _cluster_boot(v, cf, rej_y, cfg)

    m: dict[str, Any] = {
        "config": spec.key,
        "encoder": spec.encoder,
        "method": spec.method,
        "method_id": spec.method_id,
        "mode": spec.mode,
        "family": spec.family,
        "main": spec.main,
        "loss": thr.loss,
        "threshold": thr.to_dict(),
        "n_layer_a": n_a,
        "n_before": n_before,
        "r": r,
        "false_acceptance": {
            "P0": _rate(false_acceptance, n_before, n_a),
            "gate": _rate(false_acceptance, n_before - r, n_a),
        },
        "reduction": _rate(reduction, r, n_before),
        "by_type": by_type,
        "food_loss_nominal": nominal,
        "food_loss_confirmed": confirmed,
        "food_loss_nominal_bootstrap": boot,
        "layer_b_removal": layer_b_removal,
        "layer_b_pass": layer_b_pass,
        "n_layer_b": n_b,
        "n_layer_b_p0_food": nb_food,
        "other_frr": other_frr,
        "other_frr_by_class": other_frr_by_class,
        "contrast_reject": contrast_rej,
        "source_flag": bool(source_flag),
        "near_food_false_acceptance": near_fa,
        "aux_far_false_acceptance": far_fa,
        "total_inscope_loss": total_loss,
        "auroc": au,
        "fpr_at_95tpr": fpr,
        "heldout": None,
        "mcnemar": None,
    }
    masks = {"yelp": rej_y, "ext": rej_x, "layer_b": rej_b, "contrast": rej_c, "aux_far": rej_f}
    return m, masks


def _confirmed_loss(
    data: G0Data, v: _Views, cf: np.ndarray, rej_y: np.ndarray
) -> dict[str, Any] | None:
    """料理損失（確認済み）: R_true / (R_true + N_pass · q̂)（§7.2）。目視が無ければ None。"""
    if data.review is None or data.review.height == 0 or v.yelp.n == 0:
        return None
    ids = v.yelp.col("photo_id")
    rev = dict(
        zip(data.review["photo_id"].to_list(), data.review["content"].to_list(), strict=True)
    )
    rej_ids = ids[cf & rej_y]
    pass_ids = ids[cf & ~rej_y]
    n_rej = int(rej_ids.size)
    rej_reviewed = [rev[p] for p in rej_ids if p in rev]
    pass_reviewed = [rev[p] for p in pass_ids if p in rev]
    n_q = len(pass_reviewed)
    if n_q == 0:
        return None
    q_hat = sum(1 for c in pass_reviewed if c == "food") / n_q
    n_rej_reviewed = len(rej_reviewed)
    r_true_reviewed = sum(1 for c in rej_reviewed if c == "food")
    # 棄却側は全数目視が前提（§3.1）。未目視分があれば目視済みの food 率で外挿する
    r_true = (
        r_true_reviewed
        if n_rej_reviewed == n_rej
        else (r_true_reviewed * n_rej / n_rej_reviewed if n_rej_reviewed else n_rej)
    )
    est = food_loss_confirmed(int(round(r_true)), int(pass_ids.size), float(q_hat), int(n_q))
    return {
        "point": _f(est["point"]),
        "ci_delta_95": [_f(x) for x in est["ci_delta_95"]],
        "upper95": _f(est["ci_upper_one_sided_95"]),
        "r_true": int(round(r_true)),
        "n_rejected": n_rej,
        "n_rejected_reviewed": n_rej_reviewed,
        "n_pass": int(pass_ids.size),
        "q_hat": q_hat,
        "n_q": n_q,
    }


def _cluster_boot(
    v: _Views, cf: np.ndarray, rej_y: np.ndarray, cfg: dict[str, Any]
) -> dict[str, Any] | None:
    bcfg = dict(cfg.get("bootstrap", {}))
    if v.yelp.n == 0 or not cf.any():
        return None
    biz = v.y_business
    if all(b is None or b == "" for b in biz):
        return None
    groups: dict[str, tuple[int, int]] = {}
    for b, is_cf, is_rej in zip(biz, cf, rej_y, strict=True):
        if not is_cf:
            continue
        k, n = groups.get(str(b), (0, 0))
        groups[str(b)] = (k + int(is_rej), n + 1)
    point, lo, hi = cluster_bootstrap(
        groups, stat="ratio", B=int(bcfg.get("B", 2000)), seed=int(cfg.get("seed", 0))
    )
    return {"point": _f(point), "ci95": [_f(lo), _f(hi)], "n_clusters": len(groups)}


# ---------------------------------------------------------------------------
# 表の行
# ---------------------------------------------------------------------------


def _score_rows(
    run_id: str,
    spec: Spec,
    block: Block,
    scores: np.ndarray,
    fold_axis: str,
    fold_id: str,
    ts: datetime,
) -> pl.DataFrame:
    if block.n == 0:
        return pl.DataFrame()
    return pl.DataFrame(
        {
            "run_id": [run_id] * block.n,
            "photo_id": block.meta["photo_id"],
            "source": block.meta["source"],
            "mode": [spec.mode] * block.n,
            "method_id": [spec.method_id] * block.n,
            "encoder_id": [spec.encoder] * block.n,
            "score": scores.astype(np.float64),
            "role": block.meta["role"],
            "fold_axis": [fold_axis] * block.n,
            "fold_id": [fold_id] * block.n,
            "ts": [ts] * block.n,
        }
    )


def _decision_rows(
    run_id: str,
    mode: str,
    method_id: str,
    threshold_id: str,
    block: Block,
    pred: np.ndarray,
    rejected: np.ndarray | None,
    ts: datetime,
    food: str,
) -> pl.DataFrame:
    if block.n == 0:
        return pl.DataFrame()
    if rejected is None:
        decision = ["pass"] * block.n
        final = [str(p) for p in pred]
    elif mode == "P1":
        decision = [
            "reject" if r else ("pass" if p == food else "bypass")
            for p, r in zip(pred, rejected, strict=True)
        ]
        final = ["rejected" if r else str(p) for p, r in zip(pred, rejected, strict=True)]
    else:
        decision = ["reject" if r else "pass" for r in rejected]
        final = ["rejected" if r else str(p) for p, r in zip(pred, rejected, strict=True)]
    return pl.DataFrame(
        {
            "run_id": [run_id] * block.n,
            "photo_id": block.meta["photo_id"],
            "mode": [mode] * block.n,
            "method_id": [method_id] * block.n,
            "threshold_id": [threshold_id] * block.n,
            "decision": decision,
            "stage1_pred": [str(p) for p in pred],
            "final_label": final,
            "ts": [ts] * block.n,
        }
    )


# ---------------------------------------------------------------------------
# LOTO / LOSO（C 系）
# ---------------------------------------------------------------------------


def _heldout(
    spec: Spec,
    data: G0Data,
    stage: str,
    cfg: dict[str, Any],
    loss: float,
    food: str,
    seed: int,
    run_id: str,
    ts: datetime,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[pl.DataFrame]]:
    sets = dict(cfg.get("sets", {}))
    dec = dict(cfg.get("decision", {}))
    min_type = int(dec.get("min_n_before_type", 10))
    roles = ("tune", "work") if stage == "work" else ("tune", "work", "eval")
    ext_all = data.ext_all(roles)
    out: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    score_rows: list[pl.DataFrame] = []
    if ext_all.n == 0 or data.ext_tune.n == 0:
        return {"loto": None, "loso": None}, rows, score_rows
    pred_food = _pred(ext_all, data.class_order) == food
    oos = ext_all.col("label") == OUT_OF_SCOPE
    tune_ids = data.ext_tune.col("photo_id")
    all_ids = ext_all.col("photo_id")
    exclude = tuple(
        str(x) for x in sets.get("loto_exclude_types", (CONTRAST_TYPE, AUX_FAR_TYPE, "other"))
    )
    sources = tuple(str(x) for x in sets.get("loso_sources", ("open_images", "commons", "coco")))
    axes = {
        "loto": loto_folds(ext_all.meta, type_col="subject_type", exclude_types=exclude),
        "loso": loso_folds(ext_all.meta, source_col="source", sources=sources),
    }
    for axis, folds in axes.items():
        pooled_r = pooled_n = 0
        n_folds = 0
        for held, train_mask, test_mask in folds:
            train_ids = set(all_ids[np.asarray(train_mask, bool)].tolist())
            neg_mask = np.array([p in train_ids for p in tune_ids], dtype=bool)
            if neg_mask.sum() == 0:
                continue
            scorer = _fit_scorer(spec, data, data.ext_tune.X[neg_mask], seed, food)
            t = fit_threshold(_cal_scores(scorer, data, food), loss).t
            test = ext_all.subset(np.asarray(test_mask, bool))
            sc = _score(scorer, test)
            tm = np.asarray(test_mask, bool)
            pf = pred_food[tm]
            rej = _reject(spec.mode, sc, t, pf)
            before = pf & oos[tm]
            nb = int(before.sum())
            rf = int((before & rej).sum())
            ok = bool(fold_evaluable(nb, min_n=min_type))
            rows.append(
                {
                    "config": spec.key,
                    "fold_axis": axis,
                    "fold_id": str(held),
                    "n_test": int(test.n),
                    "n_before": nb,
                    "r": rf,
                    "reduction_p": (rf / nb) if nb else None,
                    "evaluable": ok,
                    "n_neg_train": int(neg_mask.sum()),
                }
            )
            score_rows.append(_score_rows(run_id, spec, test, sc, axis, str(held), ts))
            if ok:
                pooled_r += rf
                pooled_n += nb
                n_folds += 1
        out[axis] = {
            **_rate(reduction, pooled_r, pooled_n),
            "r": pooled_r,
            "n_before": pooled_n,
            "n_folds": n_folds,
        }
    return out, rows, score_rows


# ---------------------------------------------------------------------------
# 記述分析（§6）
# ---------------------------------------------------------------------------


def _descriptive(
    specs_scores: dict[str, dict[str, np.ndarray]],
    thresholds: dict[str, dict[str, ThresholdResult]],
    v: _Views,
    cfg: dict[str, Any],
    encoder: str,
    primary_loss: float,
) -> dict[str, Any]:
    dcfg = dict(cfg.get("descriptive", {}))
    sets = dict(cfg.get("sets", {}))
    methods = [str(m) for m in dcfg.get("hist_methods", ("knn", "mahalanobis_pp"))]
    n_bins = int(dcfg.get("n_bins", 40))
    eps = float(dcfg.get("kl_eps", 1e-6))
    au_min = float(dcfg.get("separated_auroc_min", 0.90))
    ov_max = float(dcfg.get("separated_overlap_max", 0.30))
    layer_a_types = [str(t) for t in sets.get("layer_a_types", LAYER_A_TYPES)]
    hist_parts: list[pl.DataFrame] = []
    sep_rows: list[dict[str, Any]] = []
    quad_rows: list[dict[str, Any]] = []
    bins_out: dict[str, list[float]] = {}
    for method in methods:
        key = f"{encoder}/{method}/P1"
        if key not in specs_scores:
            continue
        sc = specs_scores[key]
        d_food = -sc["yelp"][v.y_correct_food]
        groups: dict[str, np.ndarray] = {"food_eval": d_food}
        for ty in layer_a_types:
            m = v.x_layer_a & (v.x_type == ty)
            if m.any():
                groups[f"layer_a:{ty}"] = -sc["ext"][m]
        if v.x_near.any():
            groups["near_food"] = -sc["ext"][v.x_near]
        if v.layer_b.n:
            groups["layer_b"] = -sc["layer_b"]
        pooled = np.concatenate([g for g in groups.values() if g.size]) if groups else np.zeros(0)
        if pooled.size == 0:
            continue
        lo, hi = float(pooled.min()), float(pooled.max())
        if hi <= lo:
            hi = lo + 1e-6
        bins = np.linspace(lo, hi, n_bins + 1)
        bins_out[method] = [float(b) for b in bins]
        ht = histogram_table(groups, bins).with_columns(pl.lit(method).alias("method"))
        hist_parts.append(ht)
        for name, vals in groups.items():
            if name == "food_eval" or vals.size == 0 or d_food.size == 0:
                continue
            au = _f(auroc(-d_food, -vals))  # 大 = 対象内 で AUROC（陽性 = food）
            ov = _f(overlap_coefficient(d_food, vals, bins))
            kl = _f(kl_divergence_hist(vals, d_food, bins, eps))
            verdict = (
                "遠く分離できる"
                if (au is not None and ov is not None and au >= au_min and ov <= ov_max)
                else "料理側に入り込む"
            )
            sep_rows.append(
                {
                    "method": method,
                    "group": name,
                    "n": int(vals.size),
                    "auroc": au,
                    "overlap": ov,
                    "kl": kl,
                    "verdict": verdict,
                }
            )
        # 4 象限表: 横軸は P0 の food argmax（1/0 の指標）、縦軸は距離。境界 d_thr = 動作点 t の距離表現
        thr = thresholds[key][_loss_key(primary_loss)]
        d_thr = -thr.t
        for name, dist, hi_ind in (
            ("food_eval", d_food, np.ones(d_food.size)),
            ("layer_a", -sc["ext"][v.x_layer_a], v.x_pred_food[v.x_layer_a].astype(float)),
            (
                "layer_a_p0_food",
                -sc["ext"][v.x_layer_a & v.x_pred_food],
                np.ones(int((v.x_layer_a & v.x_pred_food).sum())),
            ),
            (
                "layer_b",
                -sc["layer_b"] if v.layer_b.n else np.zeros(0),
                v.b_pred_food.astype(float),
            ),
        ):
            if dist.size == 0:
                continue
            q = quadrant_counts(hi_ind, dist, 0.5, d_thr)
            quad_rows.append(
                {"method": method, "group": name, "d_thr": float(d_thr), "n": int(dist.size)}
                | {k: int(q[k]) for k in ("high_far", "high_near", "low_far", "low_near")}
            )
    hist_schema = {
        "group": pl.Utf8,
        "bin_index": pl.Int64,
        "bin_left": pl.Float64,
        "bin_right": pl.Float64,
        "count": pl.Int64,
        "frac": pl.Float64,
        "n": pl.Int64,
        "method": pl.Utf8,
    }
    hist = pl.concat(hist_parts, how="vertical") if hist_parts else pl.DataFrame(schema=hist_schema)
    sep_schema = {
        "method": pl.Utf8,
        "group": pl.Utf8,
        "n": pl.Int64,
        "auroc": pl.Float64,
        "overlap": pl.Float64,
        "kl": pl.Float64,
        "verdict": pl.Utf8,
    }
    quad_schema = {
        "method": pl.Utf8,
        "group": pl.Utf8,
        "d_thr": pl.Float64,
        "high_far": pl.Int64,
        "high_near": pl.Int64,
        "low_far": pl.Int64,
        "low_near": pl.Int64,
        "n": pl.Int64,
    }
    return {
        "histograms": hist,
        "separation": pl.DataFrame(sep_rows, schema=sep_schema),
        "quadrants": pl.DataFrame(quad_rows, schema=quad_schema),
        "bins": bins_out,
    }


def _source_auroc(data: G0Data, v: _Views, seed: int) -> float | None:
    """Yelp vs 外部の 2 値 AUROC（tune で logistic、stage 集合で評価。§3.4）。"""
    if data.ext_tune.n == 0 or v.ext.n == 0 or v.yelp.n == 0:
        return None
    from sklearn.linear_model import LogisticRegression

    def l2(x: np.ndarray) -> np.ndarray:
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    X = np.concatenate([l2(data.yelp_tune_fit.X), l2(data.ext_tune.X)])
    y = np.r_[np.ones(data.yelp_tune_fit.n), np.zeros(data.ext_tune.n)]
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed)
    clf.fit(X, y)
    return _f(auroc(clf.decision_function(l2(v.yelp.X)), clf.decision_function(l2(v.ext.X))))


def _resolve_primary(primary_config: str, encoder: str, notes: list[str]) -> str:
    """事前登録の主構成（E1 × kNN × P1）を実行中のエンコーダのキーに写す。

    エンコーダが異なる run（E2、合成データ）では主構成は存在しないので、同じ method × mode を
    そのエンコーダで主として扱い、注記を残す（E2 は副次なので採否には使わない。§5.2）。
    """
    parts = primary_config.split("/")
    if len(parts) != 3:
        raise ValueError(f"primary_config は encoder/method/mode: {primary_config!r}")
    if parts[0] == encoder:
        return primary_config
    resolved = f"{encoder}/{parts[1]}/{parts[2]}"
    notes.append(
        f"主構成 {primary_config} はエンコーダ {encoder} に無いため {resolved} を主構成として扱った（事前登録の主構成は E1）"
    )
    return resolved


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------


def run_g0(
    config: dict[str, Any],
    *,
    stage: str,
    data: G0Data,
    recorder: RunRecorder | None = None,
) -> G0Result:
    """G0 を 1 回実行する。``stage`` は work（開発）または eval（最終報告。最後の 1 回）。"""
    if stage not in ("work", "eval"):
        raise ValueError(f"stage は work | eval: {stage!r}")
    t0 = time.perf_counter()
    cfg = config
    food = str(cfg.get("food_label", "food"))
    seed = int(cfg.get("seed", 0))
    op = dict(cfg.get("operating_point", {}))
    loss_levels = [float(x) for x in op.get("loss_levels", (0.01, 0.005, 0.02))]
    primary_loss = float(op.get("primary_loss", 0.01))
    if primary_loss not in loss_levels:
        loss_levels.insert(0, primary_loss)
    dec = dict(cfg.get("decision", {}))
    run_id = recorder.run_id if recorder is not None else new_run_id()
    ts = datetime.now(UTC)
    encoder = data.encoder_name
    notes: list[str] = []

    attach_p0(data)
    v = _views(data, stage, dict(cfg.get("sets", {})), food)
    specs = build_specs(cfg, encoder, stage)

    if data.yelp_tune_cal.n == 0 or data.yelp_tune_fit.n == 0:
        raise ValueError("yelp_tune_fit / yelp_tune_cal が空。動作点と学習ができない")

    thresholds: dict[str, dict[str, ThresholdResult]] = {}
    metrics: dict[str, dict[str, dict[str, Any]]] = {}
    rej_masks: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    specs_scores: dict[str, dict[str, np.ndarray]] = {}
    score_frames: list[pl.DataFrame] = []
    decision_frames: list[pl.DataFrame] = []
    heldout_rows: list[dict[str, Any]] = []
    timings: dict[str, float] = {}

    blocks = {
        "yelp": v.yelp,
        "ext": v.ext,
        "layer_b": v.layer_b,
        "contrast": v.contrast,
        "aux_far": v.aux_far,
    }
    preds = {name: _pred(b, data.class_order) for name, b in blocks.items()}

    # P0（ゲートなし）の決定行
    for name, b in blocks.items():
        decision_frames.append(
            _decision_rows(run_id, "P0", "none", "none", b, preds[name], None, ts, food)
        )

    for spec in specs:
        t_s = time.perf_counter()
        scorer = _fit_scorer(spec, data, data.ext_tune.X, seed, food)
        cal = _cal_scores(scorer, data, food)
        if cal.size == 0:
            raise ValueError("tune_cal に「正しく food」が無い")
        thresholds[spec.key] = {_loss_key(loss): fit_threshold(cal, loss) for loss in loss_levels}
        sc = {name: _score(scorer, b) for name, b in blocks.items()}
        specs_scores[spec.key] = sc
        for name, b in blocks.items():
            score_frames.append(_score_rows(run_id, spec, b, sc[name], "none", "", ts))
        metrics[spec.key] = {}
        rej_masks[spec.key] = {}
        for loss in loss_levels:
            lk = _loss_key(loss)
            m, masks = _metrics_for(spec, thresholds[spec.key][lk], sc, v, data, cfg, food)
            metrics[spec.key][lk] = m
            rej_masks[spec.key][lk] = masks
            for name, b in blocks.items():
                decision_frames.append(
                    _decision_rows(
                        run_id, spec.mode, spec.method_id, lk, b, preds[name], masks[name], ts, food
                    )
                )
        if spec.family == "c_series" and spec.main:
            ho, rows, frames = _heldout(
                spec, data, stage, cfg, primary_loss, food, seed, run_id, ts
            )
            metrics[spec.key][_loss_key(primary_loss)]["heldout"] = ho
            heldout_rows.extend(rows)
            score_frames.extend(frames)
        timings[spec.key] = time.perf_counter() - t_s

    # McNemar 用の不一致セル（同一検出器の P1 vs P2。層 A ∩ P0 = food 上）
    before = v.x_layer_a & v.x_pred_food
    pk = _loss_key(primary_loss)
    pairs: list[tuple[str, str, str]] = []
    for s in specs:
        if s.mode != "P1":
            continue
        k2 = s.key[: -len("P1")] + "P2"
        if k2 not in metrics:
            continue
        pairs.append((s.method_id, s.key, k2))
        for lk in (_loss_key(loss) for loss in loss_levels):
            r1 = rej_masks[s.key][lk]["ext"] & before
            r2 = rej_masks[k2][lk]["ext"] & before
            cell = {
                "b": int((r1 & ~r2).sum()),
                "c": int((r2 & ~r1).sum()),
                "both": int((r1 & r2).sum()),
                "neither": int((~r1 & ~r2 & before).sum()),
            }
            metrics[s.key][lk]["mcnemar"] = cell
            metrics[k2][lk]["mcnemar"] = cell

    # 判定（主 8 構成、主動作点）
    main_keys = [s.key for s in specs if s.main]
    primary_metrics = {k: metrics[k][pk] for k in main_keys}
    n_before_pooled = int(before.sum())
    by_type_counts = {
        ty: int((before & (v.x_type == ty)).sum())
        for ty in [str(t) for t in cfg.get("sets", {}).get("layer_a_types", LAYER_A_TYPES)]
    }
    ev = evaluability(
        n_before_pooled,
        by_type_counts,
        min_pooled=int(dec.get("min_n_before_pooled", 40)),
        min_type=int(dec.get("min_n_before_type", 10)),
    )
    primary_config = _resolve_primary(str(dec.get("primary_config", "e1/knn/P1")), encoder, notes)
    table = decide(
        primary_metrics,
        primary_config=primary_config,
        alpha=float(dec.get("alpha", 0.05)),
        h0_p=float(dec.get("h0_p", 0.25)),
        holm_m=dec.get("holm_m", 7),
        min_n_before_pooled=int(dec.get("min_n_before_pooled", 40)),
        min_n_before_type=int(dec.get("min_n_before_type", 10)),
        thresholds=dict(dec.get("thresholds", {})),
        tested=main_keys,
    )
    tests = {
        row["config"]: {
            "p": row["p_value"],
            "p_adj": row["p_adj"],
            "reject": row["test_reject"],
            "method": row["test_method"],
        }
        for row in table.to_dicts()
    }
    factor = float(dec.get("thresholds", {}).get("p1_sufficiency_factor", 0.9))
    main_pairs = [p for p in pairs if p[1] in main_keys]
    p1_table = p1_sufficiency(primary_metrics, pairs=main_pairs, factor=factor)

    # 記述分析・副指標
    desc = _descriptive(specs_scores, thresholds, v, cfg, encoder, primary_loss)
    src_auroc = _source_auroc(data, v, seed)
    sec_rows = []
    for s in specs:
        m = metrics[s.key][pk]
        sec_rows.append(
            {
                "config": s.key,
                "main": s.main,
                "auroc": m["auroc"],
                "fpr_at_95tpr": m["fpr_at_95tpr"],
                "near_food_fa_p0": m["near_food_false_acceptance"]["P0"]["p"],
                "near_food_fa_gate": m["near_food_false_acceptance"]["gate"]["p"],
                "aux_far_fa_p0": m["aux_far_false_acceptance"]["P0"]["p"],
                "aux_far_fa_gate": m["aux_far_false_acceptance"]["gate"]["p"],
                "contrast_reject_p": m["contrast_reject"]["p"],
                "source_flag": m["source_flag"],
                "yelp_vs_ext_auroc": src_auroc,
                "reduction_p": m["reduction"]["p"],
                "food_loss_nominal_p": m["food_loss_nominal"]["p"],
                "fit_score_sec": timings[s.key],
            }
        )
    sec_schema = {
        "config": pl.Utf8,
        "main": pl.Boolean,
        "auroc": pl.Float64,
        "fpr_at_95tpr": pl.Float64,
        "near_food_fa_p0": pl.Float64,
        "near_food_fa_gate": pl.Float64,
        "aux_far_fa_p0": pl.Float64,
        "aux_far_fa_gate": pl.Float64,
        "contrast_reject_p": pl.Float64,
        "source_flag": pl.Boolean,
        "yelp_vs_ext_auroc": pl.Float64,
        "reduction_p": pl.Float64,
        "food_loss_nominal_p": pl.Float64,
        "fit_score_sec": pl.Float64,
    }
    secondary = pl.DataFrame(sec_rows, schema=sec_schema)

    sizes = pl.DataFrame(
        [
            {"set": "yelp_tune_fit", "n": data.yelp_tune_fit.n},
            {"set": "yelp_tune_cal", "n": data.yelp_tune_cal.n},
            {"set": f"yelp_{stage}", "n": v.yelp.n},
            {"set": f"yelp_{stage}_correct_food", "n": int(v.y_correct_food.sum())},
            {"set": "ext_tune", "n": data.ext_tune.n},
            {"set": f"layer_a_{stage}", "n": int(v.x_layer_a.sum())},
            {"set": f"layer_a_{stage}_p0_food", "n": n_before_pooled},
            {"set": f"near_food_{stage}", "n": int(v.x_near.sum())},
            {"set": f"layer_b_{stage}", "n": v.layer_b.n},
            {"set": f"food_contrast_{stage}", "n": v.contrast.n},
            {"set": f"aux_far_{stage}", "n": v.aux_far.n},
        ],
        schema={"set": pl.Utf8, "n": pl.Int64},
    )
    heldout_schema = {
        "config": pl.Utf8,
        "fold_axis": pl.Utf8,
        "fold_id": pl.Utf8,
        "n_test": pl.Int64,
        "n_before": pl.Int64,
        "r": pl.Int64,
        "reduction_p": pl.Float64,
        "evaluable": pl.Boolean,
        "n_neg_train": pl.Int64,
    }
    ood_scores = (
        pl.concat([f for f in score_frames if f.height], how="vertical")
        if any(f.height for f in score_frames)
        else pl.DataFrame()
    )
    gate_decisions = (
        pl.concat([f for f in decision_frames if f.height], how="vertical")
        if any(f.height for f in decision_frames)
        else pl.DataFrame()
    )
    if not ev["pooled"]:
        notes.append(
            f"合算 n_before={n_before_pooled} < {ev['min_pooled']}: 主比較は評価不能（Stage1 が非料理を food に流さないという記述的知見として報告）"
        )
    if data.review is None:
        notes.append("目視（review）が無いため料理損失（確認済み）は未計測。(b) は未充足として扱う")
    timings["total"] = time.perf_counter() - t0

    result = G0Result(
        stage=stage,
        encoder=encoder,
        run_id=run_id,
        class_order=list(data.class_order),
        primary_loss=primary_loss,
        loss_levels=loss_levels,
        specs=specs,
        thresholds={k: {lk: t.to_dict() for lk, t in d.items()} for k, d in thresholds.items()},
        metrics=metrics,
        tests=tests,
        evaluability=ev,
        decision_table=table,
        p1_sufficiency=p1_table,
        heldout=pl.DataFrame(heldout_rows, schema=heldout_schema),
        descriptive=desc,
        secondary=secondary,
        sizes=sizes,
        ood_scores=ood_scores,
        gate_decisions=gate_decisions,
        timings=timings,
        notes=notes,
    )
    if recorder is not None:
        recorder.log_metrics(result.flat_metrics())
        recorder.log_table("decision_table", table)
        recorder.log_table("p1_sufficiency", p1_table)
        recorder.log_table("secondary", secondary)
        recorder.log_table("sizes", sizes)
    return result


def concat_for_stage(data: G0Data, stage: str) -> Block:
    """stage の評価対象（Yelp + 外部 + 層 B）を 1 ブロックに連結する補助（分析用）。"""
    v = _views(data, stage, {}, "food")
    return concat_blocks([v.yelp, v.ext, v.layer_b], data.dim)
