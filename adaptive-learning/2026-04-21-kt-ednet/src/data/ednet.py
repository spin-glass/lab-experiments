"""EdNet-KT1 loader.

Reads per-user CSVs under ``<raw_dir>/KT1/u{user_id}.csv`` and the
``questions.csv`` metadata table from the separately-distributed ``Contents``
archive, then produces a single normalized Polars DataFrame suitable for EDA
and downstream sequence models.

Normalized schema (output of :func:`load_interactions`)::

    user_id         UInt32      # parsed from filename ("u100000" -> 100000)
    timestamp       Int64       # milliseconds since epoch (shifted, see EdNet README)
    solving_id      Int32
    question_id     UInt32      # "q2319" -> 2319
    user_answer     Utf8        # 'a'..'d'
    elapsed_time    Int32       # ms
    bundle_id       UInt32      # from questions.csv, "b1707" -> 1707
    correct_answer  Utf8        # from questions.csv
    correct         UInt8       # 1 if user_answer == correct_answer else 0
    part            UInt8       # TOEIC part 1..7
    tags            List[UInt16]  # from questions.csv "179;53;183" -> [179,53,183]

See:
- https://github.com/riiid/ednet (README with full column spec)
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import polars as pl

__all__ = [
    "KT1_DIR_NAME",
    "QUESTION_CSV_NAME",
    "list_user_files",
    "load_questions",
    "load_interactions",
    "user_id_from_path",
]

KT1_DIR_NAME = "KT1"
QUESTION_CSV_NAME = "questions.csv"

_USER_FILE_RE = re.compile(r"^u(\d+)\.csv$")
_Q_ID_RE = re.compile(r"^q(\d+)$")
_B_ID_RE = re.compile(r"^b(\d+)$")


def user_id_from_path(path: Path) -> int:
    """Extract the integer user id from a ``u{n}.csv`` filename.

    Raises
    ------
    ValueError
        If ``path.name`` does not match the expected pattern.
    """
    m = _USER_FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"not an EdNet-KT1 user file: {path}")
    return int(m.group(1))


def list_user_files(raw_dir: str | Path) -> list[Path]:
    """Return all ``u*.csv`` paths under ``<raw_dir>/KT1/``, sorted by user id."""
    kt1 = Path(raw_dir).expanduser().resolve() / KT1_DIR_NAME
    if not kt1.is_dir():
        raise FileNotFoundError(f"KT1 directory not found: {kt1}")
    files = [p for p in kt1.iterdir() if _USER_FILE_RE.match(p.name)]
    files.sort(key=user_id_from_path)
    return files


def _strip_prefix_to_uint(col: str, pattern: re.Pattern[str]) -> pl.Expr:
    """Return an expression that strips a ``q``/``b`` prefix and casts to UInt32."""
    return pl.col(col).str.extract(pattern.pattern, 1).cast(pl.UInt32, strict=False).alias(col)


def load_questions(questions_csv: str | Path) -> pl.DataFrame:
    """Load and normalize the EdNet ``questions.csv`` metadata table.

    Parameters
    ----------
    questions_csv
        Path to ``questions.csv`` (from the Contents archive, ``bit.ly/ednet-content``).

    Returns
    -------
    polars.DataFrame
        Columns: ``question_id`` (UInt32), ``bundle_id`` (UInt32),
        ``correct_answer`` (Utf8), ``part`` (UInt8),
        ``tags`` (List[UInt16]).
    """
    path = Path(questions_csv).expanduser().resolve()
    df = pl.read_csv(path)
    required = {"question_id", "bundle_id", "correct_answer", "part", "tags"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"questions.csv missing columns: {sorted(missing)}")

    tags_expr = (
        pl.col("tags")
        .cast(pl.Utf8)
        .fill_null("")
        .str.split(";")
        .list.eval(pl.element().cast(pl.UInt16, strict=False))
        .alias("tags")
    )

    return df.select(
        _strip_prefix_to_uint("question_id", _Q_ID_RE),
        _strip_prefix_to_uint("bundle_id", _B_ID_RE),
        pl.col("correct_answer").cast(pl.Utf8),
        pl.col("part").cast(pl.UInt8),
        tags_expr,
    )


def _read_one_user(path: Path) -> pl.DataFrame:
    """Read a single user CSV and tag it with its ``user_id``."""
    uid = user_id_from_path(path)
    df = pl.read_csv(
        path,
        schema_overrides={
            "timestamp": pl.Int64,
            "solving_id": pl.Int32,
            "question_id": pl.Utf8,
            "user_answer": pl.Utf8,
            "elapsed_time": pl.Int32,
        },
    )
    return df.with_columns(pl.lit(uid, dtype=pl.UInt32).alias("user_id"))


def load_interactions(
    raw_dir: str | Path,
    *,
    questions: pl.DataFrame | str | Path | None = None,
    user_files: Sequence[Path] | None = None,
    sort: bool = True,
) -> pl.DataFrame:
    """Load per-user interaction CSVs and JOIN with question metadata.

    Parameters
    ----------
    raw_dir
        Root containing ``KT1/`` (and optionally ``questions.csv``).
    questions
        Either a preloaded DataFrame, a path to ``questions.csv``, or ``None``
        to auto-detect at ``<raw_dir>/questions.csv``.
    user_files
        Optional explicit list of user CSV paths (skips directory scan).
        Useful for sampling — see :mod:`src.data.sample`.
    sort
        If True, final rows are sorted by (``user_id``, ``timestamp``).

    Returns
    -------
    polars.DataFrame
        Normalized interactions with ``correct`` label joined in.
    """
    raw_dir = Path(raw_dir).expanduser().resolve()
    files = list(user_files) if user_files is not None else list_user_files(raw_dir)
    if not files:
        raise ValueError(f"no user CSVs found under {raw_dir}")

    frames: Iterable[pl.DataFrame] = (_read_one_user(p) for p in files)
    interactions = pl.concat(frames, how="vertical_relaxed")
    interactions = interactions.with_columns(
        _strip_prefix_to_uint("question_id", _Q_ID_RE),
    )

    if questions is None:
        default_q = raw_dir / QUESTION_CSV_NAME
        q_df = load_questions(default_q) if default_q.exists() else None
    elif isinstance(questions, pl.DataFrame):
        q_df = questions
    else:
        q_df = load_questions(questions)

    if q_df is not None:
        interactions = interactions.join(
            q_df.select("question_id", "bundle_id", "correct_answer", "part", "tags"),
            on="question_id",
            how="left",
        ).with_columns(
            (pl.col("user_answer") == pl.col("correct_answer")).cast(pl.UInt8).alias("correct"),
        )

    if sort:
        interactions = interactions.sort(["user_id", "timestamp"])
    return interactions
