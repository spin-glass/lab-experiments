"""Shared pytest helpers for synthetic EdNet-KT1 fixtures."""

from __future__ import annotations

from pathlib import Path


def write_kt1(raw_dir: Path, users: dict[int, list[tuple]]) -> Path:
    """Write synthetic per-user CSVs under ``<raw_dir>/KT1/``.

    ``users`` maps ``user_id`` -> list of rows, each row
    ``(timestamp, solving_id, question_id, user_answer, elapsed_time)``.
    """
    kt1 = raw_dir / "KT1"
    kt1.mkdir(parents=True, exist_ok=True)
    header = "timestamp,solving_id,question_id,user_answer,elapsed_time\n"
    for uid, rows in users.items():
        lines = [header] + [f"{t},{s},{q},{a},{e}\n" for (t, s, q, a, e) in rows]
        (kt1 / f"u{uid}.csv").write_text("".join(lines), encoding="utf-8")
    return kt1


def write_questions(raw_dir: Path, rows: list[tuple]) -> Path:
    """Write a synthetic ``questions.csv``.

    Rows: ``(qid, bid, eid, correct, part, tags, deployed_at)``.
    """
    p = raw_dir / "questions.csv"
    header = "question_id,bundle_id,explanation_id,correct_answer,part,tags,deployed_at\n"
    lines = [header]
    for qid, bid, eid, correct, part, tags, deployed in rows:
        lines.append(f"{qid},{bid},{eid},{correct},{part},{tags},{deployed}\n")
    p.write_text("".join(lines), encoding="utf-8")
    return p
