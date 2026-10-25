"""``python -m cascade.m0_setup`` のオフラインテスト（design.md §6 M0、計画 Phase B-a の ``--check-env``）。

モデルは読まない（``--skip-model``）。torch の import はするがダウンロードは無い。
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from cascade import m0_setup

REQUIRED_TORCH_KEYS = {"torch", "mps_built", "mps_available", "device", "python"}


def test_check_env_subcommand_skip_model_prints_json():
    result = CliRunner().invoke(m0_setup.app, ["check-env", "--skip-model"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert REQUIRED_TORCH_KEYS <= set(data["torch"])
    assert isinstance(data["torch"]["mps_built"], bool)
    assert isinstance(data["torch"]["mps_available"], bool)
    assert data["model"] == {"skipped": "--skip-model"}


def test_check_env_flag_form_maps_to_subcommand(capsys):
    """計画・CLAUDE.md の表記 ``--check-env --skip-model`` も同じ JSON を出す。"""
    with pytest.raises(SystemExit) as ex:
        m0_setup.main(["--check-env", "--skip-model"])
    assert ex.value.code == 0
    data = json.loads(capsys.readouterr().out)
    assert REQUIRED_TORCH_KEYS <= set(data["torch"])
    assert data["model"] == {"skipped": "--skip-model"}


def test_check_env_writes_out_file(tmp_path):
    out = tmp_path / "env.json"
    result = CliRunner().invoke(m0_setup.app, ["check-env", "--skip-model", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text(encoding="utf-8"))["model"] == {"skipped": "--skip-model"}
