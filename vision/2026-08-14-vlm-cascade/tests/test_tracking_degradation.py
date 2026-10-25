"""Tracker fan-out の縮退テスト（CLAUDE.md 絶対制約、design.md §3・M0）。

tracking = [] / [wandb] / [vertex] / [wandb, vertex] のすべてで、
- 認証情報なし → skipped
- 認証情報ありだがバックエンドが start / log / finish で例外 → failed
のいずれでも finish() が返り、runs parquet に行があり、例外が伝播しないことを確認する。
ネットワーク・SDK 初期化は行わない（モンキーパッチ）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl
import pytest

from cascade import paths
from cascade.io import tables
from cascade.tracking import adapter, backends
from cascade.tracking.adapter import RunRecorder, get_git_sha, record_run

TRACKING_SETS = [[], ["wandb"], ["vertex"], ["wandb", "vertex"]]
ENV_KEYS = ("WANDB_API_KEY", "GOOGLE_CLOUD_PROJECT", "WANDB_MODE")


@pytest.fixture
def layers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    d = {k: tmp_path / k for k in ("raw", "core", "marts")}
    monkeypatch.setattr(paths, "LAYERS", d)
    return d


@pytest.fixture
def no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def _config(tracking: list[str]) -> dict:
    return {
        "tracking": tracking,
        "wandb": {"project": "lab-vlm-cascade", "entity": None},
        "vertex": {"location": "us-central1", "experiment": "lab-vlm-cascade"},
        "model": {"hf_id": "x/y", "revision": "deadbeef"},
    }


def _assert_run_written(run_id: str, metrics: dict) -> None:
    df = tables.read_table("runs", "marts")
    rows = df.filter(pl.col("run_id") == run_id)
    assert rows.height == 1
    row = rows.to_dicts()[0]
    assert json.loads(row["metrics_json"]) == metrics
    assert row["taxonomy_version"] == "0.1.0"
    assert row["rulebook_version"] == "未導入"
    assert row["eval_set_id"] == "eval-v1"
    assert row["git_sha"] == "abc123"


# ---------------------------------------------------------------------------
# 認証情報なし → skipped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tracking", TRACKING_SETS, ids=lambda t: "+".join(t) or "none")
def test_skipped_without_credentials(
    tracking: list[str], layers: dict[str, Path], no_env: None
) -> None:
    rec = RunRecorder(
        _config(tracking),
        git_sha="abc123",
        taxonomy_version="0.1.0",
        rulebook_version="未導入",
        eval_set_id="eval-v1",
        model_id="ViT-B-16",
    )
    rec.log_metrics({"acc": 0.9})
    rec.log_metrics({"n": 10})
    rec.log_table("confusion", pl.DataFrame({"a": [1], "b": [2]}))
    out = rec.finish()

    assert out["run_id"] == rec.run_id
    assert Path(out["runs_parquet_path"]).exists()
    assert set(out["backends"]) == set(tracking)
    for name in tracking:
        assert out["backends"][name].startswith("skipped:"), out
    if "wandb" in tracking:
        assert out["backends"]["wandb"] == "skipped:no_api_key"
    if "vertex" in tracking:
        assert out["backends"]["vertex"] == "skipped:no_project"
    assert out["wandb_url"] is None
    _assert_run_written(rec.run_id, {"acc": 0.9, "n": 10})


# ---------------------------------------------------------------------------
# 認証情報あり・バックエンドが各段で例外 → failed、それでも完走
# ---------------------------------------------------------------------------


class _Boom(RuntimeError):
    pass


def _patch_failing(monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    def boom(self, *a, **k):  # noqa: ANN001
        raise _Boom(f"{stage} failed")

    for cls in (backends.WandbBackend, backends.VertexBackend):
        # 例外を投げる段以外は no-op に差し替え、SDK を一切呼ばない
        monkeypatch.setattr(
            cls, "start", boom if stage == "start" else (lambda self, *a, **k: None)
        )
        monkeypatch.setattr(
            cls, "log_metrics", boom if stage == "log" else (lambda self, *a, **k: None)
        )
        monkeypatch.setattr(cls, "log_table", lambda self, *a, **k: None)
        monkeypatch.setattr(cls, "finish", boom if stage == "finish" else (lambda self: None))


@pytest.mark.parametrize("stage", ["start", "log", "finish"])
@pytest.mark.parametrize("tracking", TRACKING_SETS, ids=lambda t: "+".join(t) or "none")
def test_failed_backend_does_not_propagate(
    tracking: list[str], stage: str, layers: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "dummy-project")
    _patch_failing(monkeypatch, stage)

    out = record_run(
        _config(tracking),
        stamps={
            "git_sha": "abc123",
            "taxonomy_version": "0.1.0",
            "rulebook_version": "未導入",
            "eval_set_id": "eval-v1",
        },
        metrics={"acc": 0.5},
    )
    assert set(out["backends"]) == set(tracking)
    for name in tracking:
        assert out["backends"][name] == "failed:_Boom", out
    _assert_run_written(out["run_id"], {"acc": 0.5})


def test_runs_row_written_before_fan_out(
    layers: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """バックエンド start 時点で runs 行がすでに存在すること（正本が先）。"""
    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    seen: dict[str, int] = {}

    def start(self, run_id, stamps):  # noqa: ANN001
        seen["rows_at_start"] = (
            tables.read_table("runs", "marts").filter(pl.col("run_id") == run_id).height
        )
        raise _Boom("after checking")

    monkeypatch.setattr(backends.WandbBackend, "start", start)
    out = record_run(_config(["wandb"]), stamps={"git_sha": "abc123"}, metrics={})
    assert seen["rows_at_start"] == 1
    assert out["backends"] == {"wandb": "failed:_Boom"}


def test_successful_fake_backend_records_wandb_url(
    layers: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "dummy")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    calls: list[tuple] = []

    class FakeWandb:
        name = "wandb"

        def __init__(self, cfg):  # noqa: ANN001
            calls.append(("init", cfg))

        def start(self, run_id, stamps):  # noqa: ANN001
            calls.append(("start", run_id, dict(stamps)))

        def log_metrics(self, m):  # noqa: ANN001
            calls.append(("metrics", dict(m)))

        def log_table(self, name, df):  # noqa: ANN001
            calls.append(("table", name, df.height))

        def finish(self):
            calls.append(("finish",))
            return "https://wandb.ai/x/lab-vlm-cascade/runs/r"

    monkeypatch.setitem(backends.BACKENDS, "wandb", FakeWandb)
    rec = RunRecorder(_config(["wandb", "vertex"]), run_id="run-fixed", git_sha="abc123")
    rec.log_metrics({"f1": 0.7})
    rec.log_table("t", pl.DataFrame({"x": [1, 2]}))
    out = rec.finish()

    assert out["backends"] == {"wandb": "ok", "vertex": "skipped:no_project"}
    assert out["wandb_url"].startswith("https://wandb.ai/")
    assert [c[0] for c in calls] == ["init", "start", "metrics", "table", "finish"]
    assert calls[1][1] == "run-fixed"
    assert calls[1][2]["git_sha"] == "abc123"
    assert "config_json" in calls[1][2]
    row = tables.read_table("runs", "marts").filter(pl.col("run_id") == "run-fixed").to_dicts()[0]
    assert row["wandb_url"] == out["wandb_url"]
    # finish は冪等
    assert rec.finish() is out


def test_unknown_backend_is_failed_not_raised(layers: dict[str, Path], no_env: None) -> None:
    out = record_run({"tracking": ["mlflow"]}, stamps={"git_sha": "abc123"}, metrics={})
    assert out["backends"] == {"mlflow": "failed:UnknownBackend"}
    assert tables.read_table("runs", "marts").height == 1


def test_unknown_stamp_key_rejected(layers: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="刻印"):
        RunRecorder({"tracking": []}, bogus="x")


def test_default_stamps_fill_git_sha_and_config_json(layers: dict[str, Path], no_env: None) -> None:
    cfg = {"tracking": [], "k": 1}
    rec = RunRecorder(cfg)
    assert rec.stamps["git_sha"]  # "unknown" でも空ではない
    assert json.loads(rec.stamps["config_json"]) == cfg


def test_get_git_sha_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*a, **k):  # noqa: ANN001
        raise OSError("no git")

    monkeypatch.setattr(adapter.subprocess, "run", fail)
    assert get_git_sha() == "unknown"


def test_wandb_backend_skips_without_key_and_never_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    b = backends.WandbBackend({"project": "p"})
    with pytest.raises(backends.SkipBackend, match="no_api_key"):
        b.start("r", {})


def test_vertex_backend_skips_without_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    b = backends.VertexBackend({})
    with pytest.raises(backends.SkipBackend, match="no_project"):
        b.start("r", {})


def test_vertex_disables_tensorboard(monkeypatch: pytest.MonkeyPatch) -> None:
    """aiplatform.init に experiment_tensorboard=False を必ず渡す（design.md §3）。"""
    import types

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    seen: dict = {}
    fake = types.SimpleNamespace(
        init=lambda **kw: seen.update(kw),
        start_run=lambda run, **kw: seen.update(run=run),
        log_params=lambda p: seen.setdefault("params", []).append(p),
        log_metrics=lambda m: seen.update(metrics=m),
        end_run=lambda: seen.update(ended=True),
    )
    import sys

    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", fake)
    monkeypatch.setitem(sys.modules, "google.cloud", types.SimpleNamespace(aiplatform=fake))
    b = backends.VertexBackend({"location": "us-central1", "experiment": "exp"})
    b.start("Run_ID/1", {"git_sha": "abc", "config_json": "{...}", "model_id": None})
    b.log_metrics({"acc": 0.5, "name": "skip-me", "flag": True})
    b.log_table("t", pl.DataFrame({"x": [1]}))
    b.finish()
    assert seen["experiment_tensorboard"] is False
    assert seen["project"] == "proj" and seen["experiment"] == "exp"
    assert seen["run"] == "run-id-1"
    assert seen["metrics"] == {"acc": 0.5, "flag": 1}
    assert "config_json" not in seen["params"][0]
    assert seen["ended"] is True


def test_small_table_guard() -> None:
    df = pl.DataFrame({"x": list(range(5))})
    with pytest.raises(ValueError, match="上限"):
        backends._check_small(df, 3, "big")


@pytest.mark.skipif(not os.environ.get("CASCADE_NETWORK_TESTS"), reason="network")
def test_real_wandb_offline_roundtrip(
    layers: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_MODE", "offline")
    out = record_run(_config(["wandb"]), stamps={"git_sha": "abc123"}, metrics={"acc": 1.0})
    assert out["backends"]["wandb"] in {"ok", "skipped:no_api_key"}
