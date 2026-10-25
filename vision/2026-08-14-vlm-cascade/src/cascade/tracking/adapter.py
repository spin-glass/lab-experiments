"""Tracker アダプタ（design.md §3「runs 正本パターン（fan-out）」、M0 完了条件）。

- 実験記録の正本は ``marts/runs.parquet``。``finish()`` はまず runs 行を追記し、その後に
  ``config["tracking"]``（``[wandb, vertex]`` の任意の組合せ・空も可）へ二重送出する
- どのバックエンドが不通でもパイプラインは完走する。各段の例外は
  ``failed:<ExceptionName>``、認証情報なしは ``skipped:<reason>`` として status に残し、決して再送出しない
- metrics / tables は ``finish()`` までバッファし、runs 行の ``metrics_json`` にも刻む
- 刻印（stamps）: git_sha / taxonomy_version / rulebook_version / eval_set_id / model_id / config_json
"""

from __future__ import annotations

import json
import logging
import subprocess
import uuid
from datetime import UTC, datetime
from typing import Any

import polars as pl

from cascade import paths
from cascade.io import tables
from cascade.tracking import backends as _backends
from cascade.tracking.backends import SkipBackend, to_builtin

log = logging.getLogger(__name__)

STAMP_KEYS = (
    "git_sha",
    "taxonomy_version",
    "rulebook_version",
    "eval_set_id",
    "model_id",
    "config_json",
)


def _git(args: list[str]) -> str | None:
    """リポジトリルート（paths.REPO_ROOT）で git を実行し stdout を返す。失敗は None。"""
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
            cwd=paths.REPO_ROOT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def get_git_sha(short: bool = False) -> str:
    """HEAD の sha（cwd に依らず paths.REPO_ROOT で解決）。追跡ファイルに未コミット変更があれば
    ``-dirty`` を付ける。git が無い／リポジトリ外なら "unknown"。"""
    out = _git(["rev-parse", "--short" if short else "--verify", "HEAD"])
    sha = (out or "").strip()
    if not sha:
        return "unknown"
    status = _git(["status", "--porcelain", "--untracked-files=no"])
    if status is not None and status.strip():
        sha += "-dirty"
    return sha


def new_run_id(now: datetime | None = None) -> str:
    """時刻＋乱数の run_id（小文字英数とハイフン。Vertex の run 名制約と両立）。"""
    now = now or datetime.now(UTC)
    return f"{now:%Y%m%dt%H%M%S}-{uuid.uuid4().hex[:8]}"


def _validate_tracking(value: Any) -> list[str]:
    """``config["tracking"]`` は None / str / list[str]（tuple 可）のみ。それ以外は早期に ValueError。"""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(n, str) for n in value):
        return list(value)
    raise ValueError(
        f"config['tracking'] はバックエンド名の list（例 [wandb, vertex]）か str が必要: {value!r}"
    )


def _json_default(v: Any) -> Any:
    """metrics_json の直列化: numpy は数値のまま（文字列化しない）、それ以外は str。"""
    b = to_builtin(v)
    if b is v:
        return str(v)
    return b


class RunRecorder:
    """1 run 分の記録。``log_metrics`` / ``log_table`` をバッファし ``finish`` で書き出す。"""

    def __init__(self, config: dict[str, Any], run_id: str | None = None, **stamps: Any) -> None:
        unknown = sorted(set(stamps) - set(STAMP_KEYS))
        if unknown:
            raise ValueError(f"未知の刻印キー {unknown}（許容: {STAMP_KEYS}）")
        _validate_tracking(config.get("tracking"))
        self.config = config
        self.run_id = run_id or new_run_id()
        self.stamps: dict[str, Any] = {k: stamps.get(k) for k in STAMP_KEYS}
        if self.stamps["git_sha"] is None:
            self.stamps["git_sha"] = get_git_sha()
        if self.stamps["config_json"] is None:
            self.stamps["config_json"] = json.dumps(config, sort_keys=True, default=str)
        self.metrics: dict[str, Any] = {}
        self.tables: list[tuple[str, pl.DataFrame]] = []
        self._finished: dict[str, Any] | None = None

    # -- バッファ --------------------------------------------------------------
    def log_metrics(self, metrics: dict[str, Any]) -> None:
        self.metrics.update(metrics)

    def log_table(self, name: str, df: pl.DataFrame) -> None:
        self.tables.append((name, df))

    # -- 書き出し --------------------------------------------------------------
    def _runs_row(self, wandb_url: str | None) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            **{k: (None if v is None else str(v)) for k, v in self.stamps.items()},
            "metrics_json": json.dumps(self.metrics, sort_keys=True, default=_json_default),
            "wandb_url": wandb_url,
            "ts": datetime.now(UTC),
        }

    def _backend_names(self) -> list[str]:
        return _validate_tracking(self.config.get("tracking"))

    def _fan_out(self) -> tuple[dict[str, str], str | None]:
        status: dict[str, str] = {}
        wandb_url: str | None = None
        for name in self._backend_names():
            cls = _backends.BACKENDS.get(name)
            if cls is None:
                status[name] = "failed:UnknownBackend"
                log.warning("tracking backend %r は未知。skip", name)
                continue
            backend: Any = None
            try:
                backend = cls(self.config.get(name, {}))
                backend.start(self.run_id, self.stamps)
                backend.log_metrics(self.metrics)
                for tname, df in self.tables:
                    backend.log_table(tname, df)
                url = backend.finish()
                status[name] = "ok"
                if name == "wandb" and url:
                    wandb_url = str(url)
            except SkipBackend as e:
                status[name] = f"skipped:{e}"
            except Exception as e:  # noqa: BLE001 — 縮退が契約。どんな例外でも完走する
                status[name] = f"failed:{type(e).__name__}"
                log.warning("tracking backend %r failed: %s: %s", name, type(e).__name__, e)
                if backend is not None:
                    try:
                        backend.finish()
                    except Exception:  # noqa: BLE001
                        pass
        return status, wandb_url

    def finish(self) -> dict[str, Any]:
        """runs 行を書いてから fan-out。戻り値に各バックエンドの status を含む。冪等。"""
        if self._finished is not None:
            return self._finished
        # 1) 正本を先に書く（バックエンドの成否に依存しない）
        runs_path = tables.append_rows("runs", [self._runs_row(None)], "marts")
        # 2) fan-out（例外は status に落ちる）
        status, wandb_url = self._fan_out()
        # 3) 得られた URL を正本へ反映（read-modify-write はロック内。失敗しても run は完走）
        if wandb_url:
            try:
                with tables.table_lock("runs", "marts"):
                    df = tables.read_table("runs", "marts")
                    df = df.with_columns(
                        pl.when(pl.col("run_id") == self.run_id)
                        .then(pl.lit(wandb_url))
                        .otherwise(pl.col("wandb_url"))
                        .alias("wandb_url")
                    )
                    tables.write_table("runs", df, "marts")
            except Exception as e:  # noqa: BLE001
                log.warning("runs.wandb_url の更新に失敗: %s", e)
        self._finished = {
            "run_id": self.run_id,
            "runs_parquet_path": runs_path,
            "backends": status,
            "wandb_url": wandb_url,
        }
        return self._finished


def record_run(
    config: dict[str, Any],
    stamps: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """1 回で runs 行＋fan-out を行う簡易 API。"""
    rec = RunRecorder(config, run_id=run_id, **(stamps or {}))
    if metrics:
        rec.log_metrics(metrics)
    return rec.finish()
