"""G0 パイプラインの end-to-end テスト（合成データ。docs/g0_nonfood_plan.md §4〜§8、§13）。

ネットワーク・モデル DL なし。``make_synthetic_g0_data`` と FakeEncoder のみ。

- run_g0 が work / eval の両 stage で全テーブル（ood_scores / gate_decisions はスキーマ一致）を作る
- 帰結表が主 8 構成 8 行で、帰結が §8.3 の 4 値に収まる（work は n_before < 40 → 評価不能）
- 動作点は tune_cal の正しく food だけで決まり、P1 は P0 = food 以外を棄却しない
- レポートが §13 の節とフッタを持つ
- CLI ``--synthetic`` が完走し、eval は ``--confirm-eval`` なしでは実行されない
- load_g0_data が埋め込みキャッシュ＋core テーブルから G0Data を組める（FakeEncoder）
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from typer.testing import CliRunner

from cascade import paths
from cascade.config import load_config
from cascade.gate.data_loader import (
    BLOCK_NAMES,
    META_COLUMNS,
    G0Data,
    make_synthetic_g0_data,
)
from cascade.gate.decision import OUTCOMES
from cascade.gate.pipeline import G0Result, build_specs, run_g0
from cascade.io import tables
from cascade.report import gate_report

CONFIG = paths.REPO_ROOT / "configs" / "g0_nonfood_gate.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def data() -> G0Data:
    return make_synthetic_g0_data(seed=0)


@pytest.fixture(scope="module")
def result_eval(cfg: dict, data: G0Data) -> G0Result:
    return run_g0(cfg, stage="eval", data=data)


@pytest.fixture(scope="module")
def result_work(cfg: dict, data: G0Data) -> G0Result:
    return run_g0(cfg, stage="work", data=data)


@pytest.fixture
def isolated_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """data/ 配下をテスト用に隔離し、トラッカーの認証情報を消す。"""
    layers = {k: tmp_path / k for k in ("raw", "core", "marts")}
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(paths, "LAYERS", layers)
    monkeypatch.setattr(paths, "MARTS", layers["marts"])
    monkeypatch.setattr(paths, "CORE", layers["core"])
    monkeypatch.setattr(paths, "EMB", tmp_path / "embeddings")
    for k in ("WANDB_API_KEY", "GOOGLE_CLOUD_PROJECT", "WANDB_MODE"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path


# ---------------------------------------------------------------------------
# 合成データ
# ---------------------------------------------------------------------------


def test_synthetic_data_shape_and_determinism(data: G0Data):
    assert data.class_order == ["food", "drink", "menu", "inside", "outside"]
    for name in BLOCK_NAMES:
        b = data.block(name)
        assert b.n > 0, name
        assert tuple(b.meta.columns) == META_COLUMNS
        assert b.probs is not None and b.probs.shape == (b.n, 5)
    other = make_synthetic_g0_data(seed=0)
    np.testing.assert_array_equal(data.yelp_eval.X, other.yelp_eval.X)
    assert set(data.ext_eval.col("role")) == {"eval"}
    assert set(data.ext_tune.col("role")) == {"tune"}
    assert data.review is not None and data.review.height == sum(
        data.block(f"yelp_{r}").n for r in ("tune_fit", "tune_cal", "work", "eval")
    )


def test_specs_main_eight_and_work_only_dropped_in_eval(cfg: dict):
    work = build_specs(cfg, "e1", "work")
    ev = build_specs(cfg, "e1", "eval")
    assert [s.key for s in work if s.main] == [
        f"e1/{m}/{mode}"
        for m in ("msp", "knn", "mahalanobis_pp", "c3_binary")
        for mode in ("P1", "P2")
    ]
    assert any(s.work_only for s in work)
    assert not any(s.work_only for s in ev)
    assert {s.family for s in ev} == {"msp", "distance", "c_series"}


# ---------------------------------------------------------------------------
# run_g0
# ---------------------------------------------------------------------------


def test_eval_stage_is_evaluable_and_decision_table_has_eight_rows(result_eval: G0Result):
    assert result_eval.evaluability["pooled"] is True
    assert result_eval.evaluability["n_before"] >= 40
    t = result_eval.decision_table
    assert t.height == 8
    assert set(t["outcome"].to_list()) <= set(OUTCOMES)
    assert t.filter(pl.col("test_method") == "primary")["config"].to_list() == ["synthetic/knn/P1"]
    assert (t.filter(pl.col("test_method") != "primary")["test_method"] == "holm(m=7)").all()
    # 少なくとも 1 型は n_before < 10 で評価しない（合成データの vehicle_street）
    assert False in result_eval.evaluability["types"].values()


def test_work_stage_small_n_before_is_inevaluable(result_work: G0Result):
    assert result_work.evaluability["pooled"] is False
    assert result_work.decision_table["outcome"].to_list() == ["評価不能"] * 8
    assert any("評価不能" in n for n in result_work.notes)


def test_output_tables_match_design_schema(result_eval: G0Result, isolated_data: Path):
    for name, df in (
        ("ood_scores", result_eval.ood_scores),
        ("gate_decisions", result_eval.gate_decisions),
    ):
        assert list(df.columns) == list(tables.SCHEMAS[name].names)
        path = tables.write_table(name, df, "marts")
        back = tables.read_table(name, "marts")
        assert back.height == df.height and path.exists()
    sc = result_eval.ood_scores
    assert set(sc["mode"].to_list()) == {"P1", "P2"}
    assert set(sc["fold_axis"].to_list()) == {"none", "loto", "loso"}
    assert (sc["run_id"] == result_eval.run_id).all()
    gd = result_eval.gate_decisions
    assert set(gd["mode"].to_list()) == {"P0", "P1", "P2"}
    assert set(gd.filter(pl.col("mode") == "P0")["decision"].to_list()) == {"pass"}


def test_p1_only_rejects_p0_food_and_p2_rejects_anything(result_eval: G0Result):
    gd = result_eval.gate_decisions
    p1 = gd.filter(pl.col("mode") == "P1")
    assert (p1.filter(pl.col("decision") == "reject")["stage1_pred"] == "food").all()
    assert (p1.filter(pl.col("decision") == "bypass")["stage1_pred"] != "food").all()
    assert (p1.filter(pl.col("decision") == "reject")["final_label"] == "rejected").all()
    p2 = gd.filter(pl.col("mode") == "P2")
    assert set(p2["decision"].to_list()) <= {"pass", "reject"}
    assert "bypass" not in set(p2["decision"].to_list())


def test_threshold_fit_only_on_tune_cal_correct_food(result_eval: G0Result, data: G0Data):
    cal = data.yelp_tune_cal
    pred = np.asarray(data.class_order)[cal.probs.argmax(1)]
    n_correct_food = int(((cal.col("label") == "food") & (pred == "food")).sum())
    for key, per_loss in result_eval.thresholds.items():
        for lk, thr in per_loss.items():
            assert thr["n"] == n_correct_food, (key, lk)
            assert thr["achieved_loss"] <= thr["loss"] + 1e-12
            assert thr["k_index"] == int(np.floor(thr["loss"] * n_correct_food)) + 1
    assert set(result_eval.thresholds["synthetic/knn/P1"]) == {"loss0.01", "loss0.005", "loss0.02"}


def test_metrics_structure_and_consistency(result_eval: G0Result):
    for s in result_eval.main_specs():
        m = result_eval.metrics_at(s.key)
        assert m["n_before"] == result_eval.evaluability["n_before"]
        assert 0 <= m["r"] <= m["n_before"]
        assert m["reduction"]["p"] == pytest.approx(m["r"] / m["n_before"])
        assert m["reduction"]["lower95"] <= m["reduction"]["p"] <= m["reduction"]["upper95"]
        assert m["food_loss_nominal"]["n"] > 0
        assert m["food_loss_confirmed"] is not None and m["food_loss_confirmed"]["n_q"] > 0
        assert m["false_acceptance"]["gate"]["k"] == m["n_before"] - m["r"]
        assert m["auroc"] is not None and 0 <= m["auroc"] <= 1
        assert m["mcnemar"] is not None and {"b", "c"} <= set(m["mcnemar"])
        if s.mode == "P2":
            assert (
                m["other_frr"] is not None
                and m["layer_b_pass"] is not None
                and m["layer_b_removal"] is None
            )
            assert set(m["other_frr_by_class"]) == {"drink", "menu", "inside", "outside"}
        else:
            assert (
                m["other_frr"] is None
                and m["layer_b_pass"] is None
                and m["layer_b_removal"] is not None
            )
        if s.family == "c_series":
            ho = m["heldout"]
            assert ho["loto"]["n_folds"] >= 1 and ho["loso"]["n_folds"] >= 1
            assert ho["loto"]["lower95"] is not None
        else:
            assert m["heldout"] is None
    assert (
        result_eval.heldout.filter(pl.col("config") == "synthetic/c3_binary/P1").height >= 6
    )  # 6 型 + 3 源のうち存在分
    assert (result_eval.heldout.filter(~pl.col("evaluable"))["n_before"] < 10).all()


def test_p1_sufficiency_one_row_per_main_detector(result_eval: G0Result):
    df = result_eval.p1_sufficiency
    assert df["method"].to_list() == ["msp", "knn", "mahalanobis_pp", "c3_binary"]
    assert set(df["verdict"].to_list()) <= {"全画像ゲート不要", "P1 では不足", "評価不能"}
    assert df["mcnemar_p"].drop_nulls().len() == 4


def test_descriptive_tables(result_eval: G0Result):
    d = result_eval.descriptive
    assert set(d["bins"]) == {"knn", "mahalanobis_pp"}
    hist = d["histograms"]
    assert {"group", "bin_index", "count", "method"} <= set(hist.columns)
    assert "food_eval" in set(hist["group"].to_list())
    sep = d["separation"]
    assert set(sep["verdict"].to_list()) <= {"遠く分離できる", "料理側に入り込む"}
    assert sep.filter(pl.col("group").str.starts_with("layer_a:")).height >= 4
    q = d["quadrants"]
    assert q.height > 0
    for row in q.to_dicts():
        assert row["high_far"] + row["high_near"] + row["low_far"] + row["low_near"] == row["n"]
    # P0 で food に誤分類された層 A は全て高スコア側
    p0 = q.filter(pl.col("group") == "layer_a_p0_food")
    assert (p0["low_far"] + p0["low_near"] == 0).all()


def test_secondary_and_sizes(result_eval: G0Result):
    sec = result_eval.secondary
    assert sec.height == len(result_eval.specs)
    assert sec.filter(pl.col("main")).height == 8
    assert sec["yelp_vs_ext_auroc"].drop_nulls().len() == sec.height
    assert set(result_eval.sizes["set"].to_list()) >= {
        "yelp_tune_fit",
        "yelp_tune_cal",
        "layer_a_eval_p0_food",
    }


def test_to_dict_is_json_serialisable(result_eval: G0Result):
    text = json.dumps(result_eval.to_dict(), ensure_ascii=False, default=str)
    d = json.loads(text)
    assert d["stage"] == "eval" and len(d["decision_table"]) == 8 and d["n_ood_scores"] > 0


def test_recorder_receives_flat_metrics(cfg: dict, data: G0Data, isolated_data: Path):
    from cascade.tracking.adapter import RunRecorder

    rec = RunRecorder(
        {**cfg, "tracking": []},
        taxonomy_version="0.1.0",
        rulebook_version="未導入",
        eval_set_id="synthetic-eval",
    )
    res = run_g0(cfg, stage="work", data=data, recorder=rec)
    assert res.run_id == rec.run_id
    assert "synthetic_knn_P1/reduction_p" in rec.metrics
    fin = rec.finish()
    runs = tables.read_table("runs", "marts")
    assert runs.filter(pl.col("run_id") == fin["run_id"]).height == 1


# ---------------------------------------------------------------------------
# レポート
# ---------------------------------------------------------------------------


def test_report_renders_sections_and_footer(result_eval: G0Result, tmp_path: Path):
    stamps = {
        "run_id": result_eval.run_id,
        "git_sha": "abc123",
        "taxonomy_version": "0.1.0",
        "rulebook_version": "未導入",
        "eval_set_id": "synthetic-eval",
        "api_cost": "0 USD",
        "wandb_url": None,
        "yelp_terms_version": None,
        "yelp_download_date": None,
    }
    md = gate_report.render(result_eval, stamps)
    for h in (
        "## 結論",
        "## 13.1 記述分析の要約",
        "## 13.2 主 8 構成の主指標",
        "## 13.3 帰結表",
        "## 13.4 P1 で足りるか",
        "## 13.5 副指標",
    ):
        assert h in md
    assert md.index("## 結論") < md.index("## 13.1")
    for line in (
        f"- run_id: {result_eval.run_id}",
        "- git sha: abc123",
        "- taxonomy_version: 0.1.0",
        "- rulebook_version: 未導入",
        "- eval_set_id: synthetic-eval",
        "- 概算 API コスト: 0 USD",
        "- W&B run URL: なし",
        "- Yelp 規約の版: 未記入／DL 日: 未記入",
    ):
        assert line in md, line
    assert "採用候補集合" in md and "SYNTHETIC × kNN × P1" in md
    # 画像・ID を載せない
    assert "yeval" not in md and "photo_id" not in md
    out = gate_report.write_report(tmp_path / "r.md", result_eval, stamps)
    assert out.read_text(encoding="utf-8") == md


def test_report_work_stage_marks_inevaluable(result_work: G0Result):
    md = gate_report.render(result_work, {"rulebook_version": "未導入"})
    assert "主比較は評価不能" in md and "work 集合" in md


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_synthetic_work_runs_end_to_end(isolated_data: Path, tmp_path: Path):
    from cascade import g0_nonfood_gate as cli

    out = tmp_path / "report.md"
    res = CliRunner().invoke(
        cli.app, ["--synthetic", "--stage", "work", "--config", str(CONFIG), "--out", str(out)]
    )
    assert res.exit_code == 0, res.output
    summary = json.loads(res.output)
    assert summary["synthetic"] is True and summary["stage"] == "work"
    assert len(summary["decision"]) == 8
    assert out.exists() and "rulebook_version: 未導入" in out.read_text(encoding="utf-8")
    assert Path(summary["result_json"]).exists()
    assert "ood_scores_path" not in summary  # synthetic はテーブルを書かない
    runs = tables.read_table("runs", "marts")
    assert runs.filter(pl.col("run_id") == summary["run_id"]).height == 1
    assert json.loads(runs["metrics_json"][0])["stage"] == "work"


def test_cli_eval_requires_confirm(isolated_data: Path, tmp_path: Path):
    from cascade import g0_nonfood_gate as cli

    res = CliRunner().invoke(
        cli.app,
        [
            "--synthetic",
            "--stage",
            "eval",
            "--config",
            str(CONFIG),
            "--out",
            str(tmp_path / "r.md"),
        ],
    )
    assert res.exit_code != 0
    assert "--confirm-eval" in res.output
    assert not (tmp_path / "r.md").exists()


def test_cli_eval_with_confirm_writes_tables_and_sql_reproduces_metrics(
    isolated_data: Path, tmp_path: Path, data: G0Data
):
    from cascade import g0_nonfood_gate as cli

    res = CliRunner().invoke(
        cli.app,
        [
            "--synthetic",
            "--stage",
            "eval",
            "--confirm-eval",
            "--write-tables",
            "--config",
            str(CONFIG),
            "--out",
            str(tmp_path / "r.md"),
        ],
    )
    assert res.exit_code == 0, res.output
    summary = json.loads(res.output)
    assert summary["evaluable"] is True
    run_id = summary["run_id"]
    sc = tables.read_table("ood_scores", "marts")
    gd = tables.read_table("gate_decisions", "marts")
    assert sc.height > 0 and gd.height > 0
    assert set(sc["run_id"].to_list()) == {run_id}

    # 分析 SQL（DuckDB / BigQuery 両対応）がパイプラインの数値を再現する
    now = datetime.now(UTC)
    labels = data.yelp_eval.meta.select(
        pl.col("photo_id"),
        pl.lit("yelp").alias("label_source"),
        pl.col("label"),
        pl.lit("synthetic").alias("provenance"),
        pl.lit(now).alias("ts"),
    )
    tables.write_table("labels", labels, "core")
    scope_rows = []
    for b, content in ((data.ext_eval, "non_food"), (data.layer_b, "non_food")):
        for r in b.meta.to_dicts():
            scope_rows.append(
                {
                    "photo_id": r["photo_id"],
                    "source": r["source"],
                    "content": content,
                    "operational_truth": r["label"],
                    "subject_type": r["subject_type"],
                    "boundary_flag": False,
                    "quality": "ok",
                    "role": r["role"],
                    "ts": now,
                }
            )
    tables.append_rows("scope_annotations", scope_rows, "core")
    conn = tables.duckdb_connect({})
    params = {
        "gate_decisions": "gate_decisions",
        "scope_annotations": "scope_annotations",
        "labels": "labels",
    }
    literals = {"run_id": run_id}
    result = json.loads(Path(summary["result_json"]).read_text(encoding="utf-8"))
    m = result["metrics"]["synthetic/knn/P1"]["loss0.01"]

    red = tables.run_sql(conn, paths.REPO_ROOT / "analysis" / "d14_reduction.sql", params, literals)
    row = red.filter(
        (pl.col("mode") == "P1")
        & (pl.col("method_id") == "knn")
        & (pl.col("threshold_id") == "loss0.01")
        & (pl.col("subject_type") == "ALL")
    ).to_dicts()[0]
    assert (row["n_before"], row["r"]) == (m["n_before"], m["r"])
    by_type = red.filter(
        (pl.col("mode") == "P1")
        & (pl.col("method_id") == "knn")
        & (pl.col("threshold_id") == "loss0.01")
        & (pl.col("subject_type") != "ALL")
    )
    for r in by_type.to_dicts():
        assert r["n_before"] == m["by_type"][r["subject_type"]]["n_before"]

    loss = tables.run_sql(
        conn, paths.REPO_ROOT / "analysis" / "d14_food_loss.sql", params, literals
    )
    row = loss.filter(
        (pl.col("mode") == "P1")
        & (pl.col("method_id") == "knn")
        & (pl.col("threshold_id") == "loss0.01")
    ).to_dicts()[0]
    assert (row["n_correct_food"], row["n_rejected"]) == (
        m["food_loss_nominal"]["n"],
        m["food_loss_nominal"]["k"],
    )

    lb = tables.run_sql(conn, paths.REPO_ROOT / "analysis" / "d14_layer_b.sql", params, literals)
    p2 = result["metrics"]["synthetic/knn/P2"]["loss0.01"]
    row = lb.filter(
        (pl.col("mode") == "P2")
        & (pl.col("method_id") == "knn")
        & (pl.col("threshold_id") == "loss0.01")
    ).to_dicts()[0]
    assert row["n_layer_b"] == p2["layer_b_pass"]["n"]
    assert row["p2_pass_rate"] == pytest.approx(p2["layer_b_pass"]["p"])


def test_report_path_convention():
    from cascade.g0_nonfood_gate import report_path

    assert report_path("eval", False).name == "g0_nonfood_gate.md"
    assert report_path("work", False).name == "g0_nonfood_gate_work.md"
    assert report_path("work", True).name == "g0_nonfood_gate_work_synthetic.md"


# ---------------------------------------------------------------------------
# load_g0_data（埋め込みキャッシュ＋core テーブル、FakeEncoder）
# ---------------------------------------------------------------------------


def test_load_g0_data_from_cache_and_tables(isolated_data: Path, make_image, cfg: dict):
    from cascade.embed.cache import EmbeddingCache
    from cascade.embed.fake import FakeEncoder
    from cascade.gate.data_loader import load_g0_data

    enc = FakeEncoder(dim=8, model_key="fake00000000")
    cache = EmbeddingCache(enc.model_key, root=paths.EMB)
    now = datetime.now(UTC)

    # Yelp: 4 役割 × 5 クラス
    roles = ["tune_fit", "tune_cal", "work", "eval"]
    classes = ["food", "drink", "menu", "inside", "outside"]
    y_items, labels_rows, split_rows = [], [], []
    i = 0
    for role in roles:
        for c in classes:
            for _ in range(3):
                pid = f"y{i:04d}"
                y_items.append((pid, make_image(f"{pid}.jpg", seed=i)))
                labels_rows.append(
                    {
                        "photo_id": pid,
                        "label_source": "yelp",
                        "label": c,
                        "provenance": "photos.json",
                        "ts": now,
                    }
                )
                split_rows.append(
                    {
                        "photo_id": pid,
                        "role": role,
                        "component_id": f"comp{i % 7}",
                        "eval_balanced": False,
                    }
                )
                i += 1
    cache.encode_dataset("yelp", y_items, enc, batch_size=8, shard_size=16)
    tables.write_table("labels", pl.DataFrame(labels_rows), "core")
    paths.MARTS.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(split_rows).write_parquet(paths.MARTS / "splits.parquet")

    # 外部: 層 A 2 型 × 3 役割、対照正例、補助 far（cord）
    x_items, scope_rows = [], []
    j = 0

    def scope(pid: str, source: str, content: str, truth: str, st: str, role: str) -> dict:
        return {
            "photo_id": pid,
            "source": source,
            "content": content,
            "operational_truth": truth,
            "subject_type": st,
            "boundary_flag": False,
            "quality": "ok",
            "dup_group_id": None,
            "role": role,
            "sampling_prob": 1.0,
            "license_short": "CC BY 4.0",
            "attribution": "x",
            "source_url": "https://example.invalid",
            "sha256": "0" * 64,
            "annotator": "a1",
            "guideline_version": "g1",
            "ts": now,
        }

    for st in ("doll_character_statue", "person_animal"):
        for role in ("tune", "work", "eval"):
            for _ in range(2):
                pid = f"x{j:04d}"
                x_items.append((pid, make_image(f"{pid}.jpg", seed=1000 + j)))
                scope_rows.append(scope(pid, "open_images", "non_food", "out_of_scope", st, role))
                j += 1
    for _ in range(2):
        pid = f"x{j:04d}"
        x_items.append((pid, make_image(f"{pid}.jpg", seed=1000 + j)))
        scope_rows.append(scope(pid, "open_images", "food", "food", "food_contrast", "eval"))
        j += 1
    # 目視で unjudgeable → 除外される
    pid = f"x{j:04d}"
    x_items.append((pid, make_image(f"{pid}.jpg", seed=1000 + j)))
    scope_rows.append(
        {
            **scope(pid, "open_images", "unjudgeable", "out_of_scope", "other", "eval"),
            "quality": "unjudgeable",
        }
    )
    j += 1
    cache.encode_dataset("open_images", x_items, enc, batch_size=8, shard_size=16)
    cord_items = []
    for _ in range(3):
        pid = f"c{j:04d}"
        cord_items.append((pid, make_image(f"{pid}.jpg", seed=2000 + j)))
        scope_rows.append(
            scope(pid, "cord", "non_food", "out_of_scope", "texture_document_screen", "eval")
        )
        j += 1
    cache.encode_dataset("cord", cord_items, enc, batch_size=8, shard_size=16)
    # 層 B（Yelp 由来: 目視で inside、content non_food）
    scope_rows.append(scope("y0003", "yelp", "non_food", "inside", "doll_character_statue", "eval"))
    tables.append_rows("scope_annotations", scope_rows, "core")
    # 目視（jsonl）
    (isolated_data / "review").mkdir()
    (isolated_data / "review" / "food_loss_review.jsonl").write_text(
        "\n".join(json.dumps({"photo_id": f"y{k:04d}", "content": "food"}) for k in range(3))
        + "\n",
        encoding="utf-8",
    )

    data = load_g0_data(cfg, "fake", encoder=enc, cache_root=paths.EMB)
    assert data.encoder_name == "fake" and data.dim == 8
    for role in roles:
        assert data.block(f"yelp_{role}").n == 15
    assert set(data.yelp_eval.col("label")) == set(classes)
    assert data.ext_tune.n == 4 and data.ext_work.n == 4 and data.ext_eval.n == 4
    assert set(data.ext_eval.col("subject_type")) == {"doll_character_statue", "person_animal"}
    assert data.food_contrast.n == 2 and data.aux_far.n == 3
    assert (
        data.layer_b.n == 1
        and data.layer_b.col("label")[0] == "inside"
        and data.layer_b.col("role")[0] == "tune_fit"
    )
    assert data.review is not None and data.review.height == 3
    assert data.yelp_tune_fit.probs is not None and data.yelp_tune_fit.probs.shape == (15, 5)
    assert data.yelp_eval.col("business_id")[0].startswith("comp")
    assert "cord" in data.extra["ext_dataset_ids"]
    assert not os.environ.get("WANDB_API_KEY")
