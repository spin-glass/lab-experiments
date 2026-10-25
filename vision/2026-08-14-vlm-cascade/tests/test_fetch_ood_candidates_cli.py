"""scripts/fetch_ood_candidates.py のオフラインテスト。

``--dry-run`` はネットワークにもファイルにも触れずに取得計画を出す（クライアント生成・ディレクトリ作成を
呼んだら失敗させる）。``--select-only`` は候補選択にネットワークを使うので連絡先未設定なら FetchError。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cascade import paths
from cascade.data import fetch_common as fc

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "ood_sources.yaml"
sys.path.insert(0, str(REPO / "scripts"))

import fetch_ood_candidates as foc  # noqa: E402


@pytest.fixture
def no_side_effects(monkeypatch):
    def _boom(*a, **k):  # pragma: no cover - 呼ばれたら失敗
        raise AssertionError("dry-run はネットワーク・ファイルに触れてはならない")

    monkeypatch.setattr(fc, "make_client", _boom)
    monkeypatch.setattr(fc, "make_session", _boom)
    monkeypatch.setattr(paths, "ensure_dirs", _boom)
    monkeypatch.delenv("CASCADE_CONTACT", raising=False)


def test_dry_run_prints_plan_without_network(no_side_effects):
    result = CliRunner().invoke(foc.app, ["--config", str(CONFIG), "--dry-run"])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["mode"] == "dry_run"
    assert plan["contact_configured"] is False
    assert [s["source"] for s in plan["sources"]] == list(foc.SOURCES)
    by_src = {s["source"]: s for s in plan["sources"]}
    # Open Images: MID と予算（target × multiplier）
    oi = by_src["open_images"]["by_type"]
    assert oi["doll_character_statue"]["mids"]["/m/0167gd"] == "Doll"
    assert oi["doll_character_statue"]["budget"] == 120 * 2
    assert oi["near_food_nonfood"]["budget"] == 80 * 2
    assert set(by_src["open_images"]["splits"]) == {"validation", "test"}
    # Commons / COCO / HF
    assert "Food samples" in by_src["commons"]["by_type"]["near_food_nonfood"]["categories"]
    assert by_src["coco"]["by_type"]["person_animal"]["categories"]["1"] == "person"
    hf = {d["key"]: d for d in by_src["hf"]["datasets"]}
    assert hf["cord"]["revision"] == "7f0115a4b758a71d6473b8d085751692da2fef98"
    assert hf["cord"]["n_take"] == 200 and hf["screenspot"]["enabled"] is False
    # 出力先は data/ext/<source>/images
    assert by_src["commons"]["images_dir"] == str(paths.EXT / "commons" / "images")


def test_dry_run_respects_source_and_max_per_type(no_side_effects):
    result = CliRunner().invoke(
        foc.app,
        ["--config", str(CONFIG), "--dry-run", "--source", "coco,hf", "--max-per-type", "7"],
    )
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert [s["source"] for s in plan["sources"]] == ["coco", "hf"]
    assert all(v == 7 for v in plan["budget_by_type"].values())
    assert plan["sources"][1]["datasets"][0]["n_take"] == 7


def test_unknown_source_is_rejected(no_side_effects):
    result = CliRunner().invoke(foc.app, ["--config", str(CONFIG), "--dry-run", "--source", "nope"])
    assert result.exit_code != 0


def test_select_only_requires_contact(monkeypatch):
    """select-only はクライアントを作る（= ネットワーク前提）ので連絡先未設定なら止まる。"""
    monkeypatch.delenv("CASCADE_CONTACT", raising=False)
    monkeypatch.setattr(paths, "ensure_dirs", lambda: None)
    result = CliRunner().invoke(
        foc.app, ["--config", str(CONFIG), "--select-only", "--source", "open_images"]
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, fc.FetchError)


def test_build_plan_is_pure(no_side_effects):
    from cascade.config import load_config

    cfg = load_config(CONFIG)
    plan = foc.build_plan(cfg, ("open_images",), None, config_path=CONFIG)
    assert plan["config"] == str(CONFIG)
    assert plan["sources"][0]["food_mid"] == "/m/02wbm"
