"""reports/g0_nonfood_gate.md の生成（g0_nonfood_plan.md §13、§15、CLAUDE.md 完了条件）。

構成は事前登録 §13 の表に揃える:

- 冒頭に結論（採用候補集合・P1 で足りるか・成立条件）
- 13.1 記述分析の要約（型別の分離、4 象限表）
- 13.2 主 8 構成の主指標
- 13.3 帰結表
- 13.4 P1 で足りるか
- 13.5 副指標（AUROC / FPR@95、near-food・補助 far・ソース対照、LOTO / LOSO、副次構成、時間）
- フッタ: run_id / git sha / taxonomy_version / rulebook_version（未導入）/ eval_set_id / 概算 API コスト /
  W&B run URL / Yelp 規約の版・DL 日（§15）

集計値のみを載せる。画像・photo_id・キャプションは載せない（§15、CLAUDE.md 絶対制約）。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import polars as pl

from cascade.gate.pipeline import G0Result

__all__ = ["FOOTER_KEYS", "render", "write_report"]

FOOTER_KEYS: tuple[str, ...] = (
    "run_id",
    "git_sha",
    "taxonomy_version",
    "rulebook_version",
    "eval_set_id",
    "api_cost",
    "wandb_url",
    "yelp_terms_version",
    "yelp_download_date",
)

_METHOD_LABEL = {
    "msp": "MSP",
    "knn": "kNN",
    "mahalanobis_pp": "Mahalanobis++",
    "c3_binary": "C3",
    "mahalanobis": "Mahalanobis（素）",
    "c1_oe": "C1 5-way+OE",
    "c2_sixway": "C2 6-way",
    "msp_tau0.01": "MSP τ=0.01",
    "knn_k10": "kNN k=10",
    "knn_k200": "kNN k=200",
}


# ---------------------------------------------------------------------------
# 書式
# ---------------------------------------------------------------------------


def _pct(x: Any, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{100 * float(x):.{nd}f}%"


def _num(x: Any, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{float(x):.{nd}f}"


def _pval(x: Any) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    x = float(x)
    return f"{x:.2e}" if x < 1e-3 else f"{x:.3f}"


def _pair(rate: dict[str, Any] | None, second: str = "lower95") -> str:
    """点／片側限界（(k/n)）の表記。"""
    if not rate or rate.get("p") is None:
        return "—"
    return f"{_pct(rate['p'])}／{_pct(rate.get(second))}（{rate['k']}/{rate['n']}）"


def _bool(x: Any) -> str:
    if x is None:
        return "—"
    return "○" if x else "×"


def _md_table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(str(h) for h in header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def _df_table(df: pl.DataFrame, cols: Sequence[tuple[str, str, Any]]) -> str:
    """(列名, 見出し, 書式関数) の並びで DataFrame を表にする。"""
    header = [h for _, h, _ in cols]
    rows = []
    for row in df.to_dicts():
        rows.append([fmt(row.get(c)) for c, _, fmt in cols])
    return _md_table(header, rows) if rows else "（該当なし）"


def _label(spec_key: str, method_id: str, mode: str) -> str:
    enc = spec_key.split("/")[0].upper()
    return f"{enc} × {_METHOD_LABEL.get(method_id, method_id)} × {mode}"


# ---------------------------------------------------------------------------
# 節
# ---------------------------------------------------------------------------


def _conclusion(result: G0Result) -> str:
    table = result.decision_table
    ev = result.evaluability
    candidates = table.filter(pl.col("outcome") == "採用候補")["config"].to_list()
    counts = {
        o: int(table.filter(pl.col("outcome") == o).height)
        for o in ("採用候補", "保留", "不採用", "評価不能")
    }
    lines = ["## 結論", ""]
    if not ev["pooled"]:
        lines.append(
            f"- **主比較は評価不能**: 層 A {result.stage} の合算 n_before = {ev['n_before']} < {ev['min_pooled']}。"
            "Stage1（P0）が非料理を food に流さないという記述的知見として報告する（§8.1）"
        )
    else:
        cand = "、".join(candidates) if candidates else "（空集合）"
        lines.append(f"- **採用候補集合**: {cand}")
        lines.append(
            f"- 帰結の内訳: 採用候補 {counts['採用候補']}／保留 {counts['保留']}／不採用 {counts['不採用']}／評価不能 {counts['評価不能']}（主 8 構成、動作点 {_pct(result.primary_loss, 1)}）"
        )
    p1 = result.p1_sufficiency
    if p1.height:
        ok = p1.filter(pl.col("verdict") == "全画像ゲート不要")["method"].to_list()
        lines.append(
            "- **P1 で足りるか**（§8.4）: "
            + (
                f"全画像ゲート不要と結論した検出器 = {', '.join(ok)}"
                if ok
                else "全画像ゲート不要と結論できた検出器なし"
            )
        )
    dec_primary = [r for r in table.to_dicts() if r["test_method"] == "primary"]
    if dec_primary:
        r = dec_primary[0]
        lines.append(
            f"- 主構成 {r['config']}: n_before = {r['n_before']}、r = {r['r']}、削減率 {_pct(r['reduction_p'])}（片側 95% 下限 {_pct(r['reduction_lower95'])}）、p = {_pval(r['p_value'])}、帰結 = {r['outcome']}"
        )
    if result.stage != "eval":
        lines.append(
            f"- 本レポートは **{result.stage} 集合**での動作確認であり、最終報告（eval）ではない"
        )
    for n in result.notes:
        lines.append(f"- 注: {n}")
    return "\n".join(lines)


def _sec_descriptive(result: G0Result) -> str:
    d = result.descriptive
    sep = d["separation"]
    quad = d["quadrants"]
    parts = ["## 13.1 記述分析の要約", ""]
    parts.append(
        "型ごとの分離（陽性 = eval の正しく food、陰性 = 各群。距離 = −スコア。判定には使わない）:"
    )
    parts.append("")
    parts.append(
        _df_table(
            sep,
            [
                ("method", "検出器", str),
                ("group", "群", str),
                ("n", "n", str),
                ("auroc", "AUROC", _num),
                ("overlap", "重なり係数", _num),
                ("kl", "KL(群‖food)", _num),
                ("verdict", "所見", str),
            ],
        )
    )
    parts.append("")
    parts.append(
        "4 象限表（横軸 = P0 の food argmax、縦軸 = 距離。境界 d_thr = 主動作点 t の距離表現。layer_a_p0_food は P0 で food に誤分類された層 A）:"
    )
    parts.append("")
    parts.append(
        _df_table(
            quad,
            [
                ("method", "検出器", str),
                ("group", "群", str),
                ("high_far", "高スコア×遠い", str),
                ("high_near", "高スコア×近い", str),
                ("low_far", "低スコア×遠い", str),
                ("low_near", "低スコア×近い", str),
                ("n", "n", str),
            ],
        )
    )
    bins = d.get("bins", {})
    if bins:
        parts.append("")
        parts.append(
            "ヒストグラムのビン: "
            + "、".join(
                f"{m}: {len(b) - 1} ビン [{b[0]:.4f}, {b[-1]:.4f}]" for m, b in bins.items()
            )
            + "（表は ood_scores から再現可能。凍結値は §12）"
        )
    return "\n".join(parts)


def _sec_main_metrics(result: G0Result) -> str:
    rows = []
    for s in result.main_specs():
        m = result.metrics_at(s.key)
        conf = m.get("food_loss_confirmed")
        conf_s = "—" if not conf else f"{_pct(conf['point'])}／{_pct(conf['upper95'])}"
        if s.mode == "P1":
            lb = _pair(m.get("layer_b_removal"), "lower95") if m.get("layer_b_removal") else "—"
            frr = "—"
        else:
            lb = _pair(m.get("layer_b_pass"), "lower95") if m.get("layer_b_pass") else "—"
            frr = _pair(m.get("other_frr"), "upper95") if m.get("other_frr") else "—"
        rows.append(
            [
                _label(s.key, s.method_id, s.mode),
                m["n_before"],
                m["r"],
                _pair(m["reduction"], "lower95"),
                _pair(m["food_loss_nominal"], "upper95"),
                conf_s,
                lb,
                frr,
                _pair(m["contrast_reject"], "upper95") + ("（疑い）" if m["source_flag"] else ""),
            ]
        )
    header = [
        "構成",
        "n_before",
        "r",
        "削減率（点／片側下限）",
        "料理損失 名目（点／上限）",
        "料理損失 確認済み（点／上限）",
        "層 B（P1 除去率／P2 通過率）",
        "他クラス FRR（点／上限）",
        "ソース対照 棄却率",
    ]
    fa_rows = []
    for s in result.main_specs():
        m = result.metrics_at(s.key)
        fa_rows.append(
            [
                _label(s.key, s.method_id, s.mode),
                _pct(m["false_acceptance"]["P0"]["p"]),
                _pct(m["false_acceptance"]["gate"]["p"]),
                _num(m["threshold"]["t"], 4),
                _pct(m["threshold"]["achieved_loss"], 2),
            ]
        )
    return "\n".join(
        [
            "## 13.2 主 8 構成の主指標",
            "",
            f"動作点 = tune_cal の正しく food の {_pct(result.primary_loss, 1)} 位置（§7.1）。CI は片側 95% Clopper–Pearson（判定用）。",
            "",
            _md_table(header, rows),
            "",
            "非料理→food 誤受理率（層 A、P0 → ゲート後）と動作点:",
            "",
            _md_table(["構成", "誤受理率 P0", "誤受理率 ゲート後", "t", "達成 cal 損失"], fa_rows),
        ]
    )


def _sec_decision(result: G0Result) -> str:
    rows = []
    for r in result.decision_table.to_dicts():
        rows.append(
            [
                _label(r["config"], r["method"], r["mode"]),
                f"{_pval(r['p_adj'])}（{r['test_method']}）",
                *[_bool(r[f"cond_{c}"]) for c in ("a", "b", "c", "d", "e", "f")],
                r["outcome"],
                r["reasons"],
            ]
        )
    header = [
        "構成",
        "主検定 p（調整後）",
        "(a)",
        "(b)",
        "(c)",
        "(d)",
        "(e)",
        "(f)",
        "帰結",
        "根拠",
    ]
    return "\n".join(
        [
            "## 13.3 帰結表",
            "",
            "判定順序: 評価不能 → 不採用 → 採用候補 → 保留（§8.3）。○ = 充足、× = 未充足、— = 適用外。",
            "",
            _md_table(header, rows),
        ]
    )


def _sec_p1(result: G0Result) -> str:
    return "\n".join(
        [
            "## 13.4 P1 で足りるか",
            "",
            "同一 eval 層 A 上の対応比較（§8.4）。McNemar は報告のみ（b = P1 のみ棄却、c = P2 のみ棄却）。",
            "",
            _df_table(
                result.p1_sufficiency,
                [
                    ("method", "検出器", lambda x: _METHOD_LABEL.get(x, x)),
                    ("p1_reduction_lower95", "P1 削減率 下限", _pct),
                    ("p2_reduction_point_x_factor", "P2 削減率 点推定 × 0.9", _pct),
                    ("p1_total_inscope_loss", "P1 対象内総損失", _pct),
                    ("p2_total_inscope_loss", "P2 対象内総損失", _pct),
                    ("mcnemar_b", "b", lambda x: "—" if x is None else str(x)),
                    ("mcnemar_c", "c", lambda x: "—" if x is None else str(x)),
                    ("mcnemar_p", "McNemar p", _pval),
                    ("verdict", "結論", str),
                ],
            ),
        ]
    )


def _sec_secondary(result: G0Result) -> str:
    parts = ["## 13.5 副指標", ""]
    parts.append(
        "AUROC・FPR@95（P1: 陽性 = 正しく food、P2: 陽性 = 対象内全体。陰性 = 層 A）、near-food・補助 far の誤受理、ソース対照:"
    )
    parts.append("")
    parts.append(
        _df_table(
            result.secondary,
            [
                ("config", "構成", str),
                ("main", "主", _bool),
                ("auroc", "AUROC", _num),
                ("fpr_at_95tpr", "FPR@95", _num),
                ("near_food_fa_p0", "near-food 誤受理 P0", _pct),
                ("near_food_fa_gate", "near-food 誤受理 ゲート後", _pct),
                ("aux_far_fa_p0", "補助 far 誤受理 P0", _pct),
                ("aux_far_fa_gate", "補助 far 誤受理 ゲート後", _pct),
                ("contrast_reject_p", "対照正例 棄却率", _pct),
                ("source_flag", "ソース検出", _bool),
                ("reduction_p", "削減率", _pct),
                ("food_loss_nominal_p", "料理損失 名目", _pct),
            ],
        )
    )
    src = result.secondary["yelp_vs_ext_auroc"].drop_nulls()
    parts.append("")
    parts.append(
        f"Yelp vs 外部の 2 値 AUROC（tune で logistic、{result.stage} で評価）: {_num(src[0]) if src.len() else '—'}"
    )
    if result.heldout.height:
        parts.append("")
        parts.append(
            "LOTO / LOSO（C 系。held-out は外部集合全体で評価。n_before < 10 の fold は評価不能）:"
        )
        parts.append("")
        parts.append(
            _df_table(
                result.heldout,
                [
                    ("config", "構成", str),
                    ("fold_axis", "軸", str),
                    ("fold_id", "held-out", str),
                    ("n_test", "n", str),
                    ("n_before", "n_before", str),
                    ("r", "r", str),
                    ("reduction_p", "削減率", _pct),
                    ("evaluable", "評価可", _bool),
                ],
            )
        )
        rows = []
        for s in result.main_specs():
            ho = result.metrics_at(s.key).get("heldout") or {}
            for axis in ("loto", "loso"):
                h = ho.get(axis)
                if h:
                    rows.append(
                        [
                            _label(s.key, s.method_id, s.mode),
                            axis,
                            h["n_before"],
                            h["r"],
                            _pair(h, "lower95"),
                        ]
                    )
        if rows:
            parts.append("")
            parts.append(
                _md_table(["構成", "軸", "合算 n_before", "合算 r", "削減率（点／片側下限）"], rows)
            )
    # 副動作点
    rows = []
    for s in result.main_specs():
        for loss in result.loss_levels:
            m = result.metrics_at(s.key, loss)
            rows.append(
                [
                    _label(s.key, s.method_id, s.mode),
                    _pct(loss, 1),
                    m["r"],
                    _pct(m["reduction"]["p"]),
                    _pct(m["food_loss_nominal"]["p"]),
                ]
            )
    parts.append("")
    parts.append("動作点別（主 1%、副 0.5% / 2%）:")
    parts.append("")
    parts.append(_md_table(["構成", "動作点", "r", "削減率", "料理損失 名目"], rows))
    # 店舗単位ブートストラップ
    boot_rows = []
    for s in result.main_specs():
        b = result.metrics_at(s.key).get("food_loss_nominal_bootstrap")
        if b:
            boot_rows.append(
                [
                    _label(s.key, s.method_id, s.mode),
                    _pct(b["point"]),
                    f"[{_pct(b['ci95'][0])}, {_pct(b['ci95'][1])}]",
                    b["n_clusters"],
                ]
            )
    if boot_rows:
        parts.append("")
        parts.append(
            "料理損失（名目）の店舗単位ブートストラップ 95% CI（感度分析。判定には使わない）:"
        )
        parts.append("")
        parts.append(_md_table(["構成", "点", "95% CI", "クラスタ数"], boot_rows))
    parts.append("")
    parts.append("集合サイズ:")
    parts.append("")
    parts.append(_df_table(result.sizes, [("set", "集合", str), ("n", "n", str)]))
    parts.append("")
    parts.append(
        f"実行時間（fit + score、秒）: 合計 {result.timings.get('total', 0):.1f}。埋め込みはキャッシュ済み（埋め込み時間は m0_setup の check-env を参照）。"
    )
    return "\n".join(parts)


def _footer(stamps: dict[str, Any]) -> str:
    s = {k: stamps.get(k) for k in FOOTER_KEYS}

    def v(x: Any, default: str = "未記入") -> str:
        return default if x in (None, "") else str(x)

    return "\n".join(
        [
            "---",
            "",
            f"- run_id: {v(s['run_id'])}",
            f"- git sha: {v(s['git_sha'])}",
            f"- taxonomy_version: {v(s['taxonomy_version'])}",
            f"- rulebook_version: {v(s['rulebook_version'], '未導入')}",
            f"- eval_set_id: {v(s['eval_set_id'])}",
            f"- 概算 API コスト: {v(s['api_cost'], '0')}（G0 は VLM を使わない）",
            f"- W&B run URL: {v(s['wandb_url'], 'なし')}",
            f"- Yelp 規約の版: {v(s['yelp_terms_version'])}／DL 日: {v(s['yelp_download_date'])}",
        ]
    )


def render(result: G0Result, stamps: dict[str, Any]) -> str:
    """G0Result → Markdown。``stamps`` はフッタの刻印（:data:`FOOTER_KEYS`）。"""
    title = "# G0 非料理画像の距離信号による識別 — 結果" + (
        "（work 集合での動作確認）" if result.stage != "eval" else ""
    )
    head = [
        title,
        "",
        f"事前登録: docs/g0_nonfood_plan.md（§13 の表に対応）。stage = {result.stage}、encoder = {result.encoder}、run_id = {result.run_id}。",
        "集計値のみを載せる（画像・ID・キャプションは載せない）。",
        "",
    ]
    body = [
        _conclusion(result),
        "",
        _sec_descriptive(result),
        "",
        _sec_main_metrics(result),
        "",
        _sec_decision(result),
        "",
        _sec_p1(result),
        "",
        _sec_secondary(result),
        "",
        _footer({**stamps, "run_id": stamps.get("run_id", result.run_id)}),
        "",
    ]
    return "\n".join(head + body)


def write_report(path: str | Path, result: G0Result, stamps: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(result, stamps), encoding="utf-8")
    return out
