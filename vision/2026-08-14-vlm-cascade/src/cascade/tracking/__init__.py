"""Tracker fan-out（design.md §3「runs 正本パターン」）。正本はローカル parquet の runs テーブル。"""

from cascade.tracking.adapter import RunRecorder, get_git_sha, record_run

__all__ = ["RunRecorder", "get_git_sha", "record_run"]
