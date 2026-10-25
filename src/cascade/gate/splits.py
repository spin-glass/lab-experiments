"""役割（tune_fit / tune_cal / work / eval）のガードと交差検証 fold（g0_nonfood_plan.md §3.1、§3.8、§5.4）。

- 役割: Yelp は ``tune_fit``（バンク・共分散・ヘッド学習）／``tune_cal``（動作点）／``work``（開発）／
  ``eval``（最終報告のみ）。外部集合は ``tune`` / ``work`` / ``eval``（§3.8）
- **eval を fit・閾値決定に渡してはならない**。:class:`SplitGuard` と :func:`assert_no_eval` は
  eval 行が fit 側の経路に入った時点で :class:`SplitLeakError` を送出し、eval の参照回数を記録する
  （計画「検証」: eval を fit・閾値・手法選定に使っていないことをテストとログで確認）
- LOTO（leave-one-type-out）: subject_type ごとに held-out。held-out 型は tune / work / eval の
  全体で評価し、学習側は他の型の fit 役割のみ。``food_contrast``（対照正例）・
  ``texture_document_screen``（補助 far）・``other`` は fold 軸から外す
- LOSO（leave-one-source-out）: source ∈ {open_images, commons, coco} の 3 fold、型は pooled
- fold は ``n_before < 10`` なら評価不能（:func:`fold_evaluable`）。型 × 源の交絡表を併記する
  （:func:`type_source_crosstab`）
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

__all__ = [
    "EVAL_ROLE",
    "FIT_ROLES",
    "FOLD_MIN_N",
    "LOSO_SOURCES",
    "LOTO_EXCLUDE_TYPES",
    "ROLES",
    "ROLES_EXTERNAL",
    "Fold",
    "RoleTable",
    "SplitGuard",
    "SplitLeakError",
    "assert_no_eval",
    "fold_evaluable",
    "loso_folds",
    "loto_folds",
    "summarize_roles",
    "type_source_crosstab",
]

EVAL_ROLE = "eval"
ROLES: tuple[str, ...] = ("tune_fit", "tune_cal", "work", "eval")
"""Yelp 側の役割（§3.1）。"""
ROLES_EXTERNAL: tuple[str, ...] = ("tune", "work", "eval")
"""外部集合の役割（§3.8）。"""
FIT_ROLES: frozenset[str] = frozenset({"tune_fit", "tune_cal", "work", "tune"})
"""fit・閾値決定・発動判断に渡してよい役割。eval は含まない。"""
_KNOWN_ROLES: frozenset[str] = frozenset(ROLES) | frozenset(ROLES_EXTERNAL)

LOTO_EXCLUDE_TYPES: tuple[str, ...] = ("food_contrast", "texture_document_screen", "other")
LOSO_SOURCES: tuple[str, ...] = ("open_images", "commons", "coco")
FOLD_MIN_N = 10

_REQUIRED_COLUMNS: tuple[str, ...] = ("photo_id", "role", "source", "subject_type")


class SplitLeakError(RuntimeError):
    """eval の行が fit・閾値決定・手法選定の経路に渡された。"""


# ---------------------------------------------------------------------------
# ガード
# ---------------------------------------------------------------------------


def assert_no_eval(roles: Iterable[str] | np.ndarray | pl.Series, *, context: str = "fit") -> None:
    """役割列に ``eval`` が含まれていれば :class:`SplitLeakError`。未知の役割は ``ValueError``。

    numpy 配列・polars Series・任意の iterable を受け付ける。fit / 閾値決定の関数は入口で
    これを呼ぶ。
    """
    if isinstance(roles, pl.Series):
        values = roles.to_list()
    else:
        values = [str(r) for r in np.asarray(list(roles), dtype=object).reshape(-1)]
    n_eval = sum(1 for r in values if r == EVAL_ROLE)
    if n_eval:
        raise SplitLeakError(f"{context}: eval の行が {n_eval} 件混入している")
    unknown = sorted({r for r in values if r not in _KNOWN_ROLES})
    if unknown:
        raise ValueError(f"{context}: 未知の役割 {unknown}")


@dataclass(frozen=True)
class RoleTable:
    """photo_id / role / source / subject_type（任意で label 等）を持つ表。

    構築時に必須列と役割の値域を検証する。行順は保持し、埋め込み行列と行で対応させる。
    """

    df: pl.DataFrame

    def __post_init__(self) -> None:
        missing = [c for c in _REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"RoleTable に必須列が無い: {missing}")
        if self.df["photo_id"].n_unique() != self.df.height:
            raise ValueError("photo_id が重複している")
        unknown = sorted(set(self.df["role"].unique().to_list()) - _KNOWN_ROLES)
        if unknown:
            raise ValueError(f"未知の役割 {unknown}（許容: {sorted(_KNOWN_ROLES)}）")

    @property
    def height(self) -> int:
        return self.df.height

    @property
    def roles(self) -> np.ndarray:
        return self.df["role"].to_numpy().astype(str)

    def mask(self, role: str) -> np.ndarray:
        """``role`` の行を True にする bool マスク（行順に対応）。"""
        return (self.df["role"] == role).to_numpy()

    def indices(self, role: str) -> np.ndarray:
        """``role`` の行番号（埋め込み行列の行選択に使う）。"""
        return np.flatnonzero(self.mask(role))

    def rows(self, role: str) -> pl.DataFrame:
        return self.df.filter(pl.col("role") == role)

    def counts(self) -> pl.DataFrame:
        return summarize_roles(self.df)


@dataclass
class SplitGuard:
    """役割ごとの参照を仕切り、eval の fit 側への流入を止める。

    - :meth:`fit_view`  ``role ∈ FIT_ROLES`` の行（と行番号）を返す。``eval`` を要求すると
      :class:`SplitLeakError`
    - :meth:`eval_view` eval の行を返し、参照回数を :attr:`eval_access_count` に記録する。
      最終報告の 1 回だけ呼ばれるべきで、回数はログとレポートで確認する
    - :meth:`fit_indices` / :meth:`eval_indices` は numpy 行選択用
    """

    table: RoleTable
    eval_access_count: int = 0
    fit_access_log: list[str] = field(default_factory=list)

    @classmethod
    def from_frame(cls, df: pl.DataFrame) -> SplitGuard:
        return cls(RoleTable(df))

    def _check_fit_role(self, role: str) -> None:
        if role == EVAL_ROLE:
            raise SplitLeakError("eval を fit / 閾値決定 / 手法選定に渡そうとした")
        if role not in FIT_ROLES:
            raise ValueError(f"fit に使えない役割 {role!r}（許容: {sorted(FIT_ROLES)}）")

    def fit_view(self, role: str) -> pl.DataFrame:
        """fit 側の役割の行。``eval`` は :class:`SplitLeakError`。"""
        self._check_fit_role(role)
        self.fit_access_log.append(role)
        out = self.table.rows(role)
        assert_no_eval(out["role"], context=f"fit_view({role})")
        return out

    def fit_indices(self, role: str) -> np.ndarray:
        self._check_fit_role(role)
        self.fit_access_log.append(role)
        return self.table.indices(role)

    def eval_view(self) -> pl.DataFrame:
        """eval の行（最終報告専用）。呼ぶたびに参照回数を加算する。"""
        self.eval_access_count += 1
        return self.table.rows(EVAL_ROLE)

    def eval_indices(self) -> np.ndarray:
        self.eval_access_count += 1
        return self.table.indices(EVAL_ROLE)

    def guard_arrays(self, roles: Iterable[str] | np.ndarray | pl.Series, *, context: str) -> None:
        """任意の役割配列を fit 側に渡す前の検査（eval が混じっていれば raise）。"""
        assert_no_eval(roles, context=context)

    def summary(self) -> dict[str, Any]:
        return {
            "eval_access_count": self.eval_access_count,
            "fit_access_log": list(self.fit_access_log),
            "n_by_role": {
                r: int(self.table.mask(r).sum()) for r in _KNOWN_ROLES if self.table.mask(r).any()
            },
        }


# ---------------------------------------------------------------------------
# fold
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fold:
    """held-out の単位（型または源）と、行順に対応する train / test の bool マスク。

    ``(held_out, train_mask, test_mask)`` のタプルとしても展開できる。
    """

    held_out: str
    train_mask: np.ndarray
    test_mask: np.ndarray

    def __iter__(self):
        yield self.held_out
        yield self.train_mask
        yield self.test_mask

    def __getitem__(self, i: int):
        return (self.held_out, self.train_mask, self.test_mask)[i]

    @property
    def n_train(self) -> int:
        return int(self.train_mask.sum())

    @property
    def n_test(self) -> int:
        return int(self.test_mask.sum())


def _fit_role_mask(items: pl.DataFrame, role_col: str | None) -> np.ndarray:
    """役割列があれば fit 役割の行、無ければ全行。eval は学習側に入れない。"""
    if role_col is None or role_col not in items.columns:
        return np.ones(items.height, dtype=bool)
    return items[role_col].is_in(sorted(FIT_ROLES)).to_numpy()


def _col_str(items: pl.DataFrame, col: str) -> np.ndarray:
    if col not in items.columns:
        raise ValueError(f"列 {col!r} が無い: {items.columns}")
    return items[col].fill_null("").to_numpy().astype(str)


def loto_folds(
    items: pl.DataFrame,
    type_col: str = "subject_type",
    exclude_types: Sequence[str] = LOTO_EXCLUDE_TYPES,
    *,
    role_col: str | None = "role",
    types: Sequence[str] | None = None,
) -> list[Fold]:
    """leave-one-type-out の fold を返す（§5.4）。

    - held-out 型ごとに 1 fold。型の並びは ``types``（省略時は ``exclude_types`` を除いた
      出現型のソート順）
    - ``test_mask``: held-out 型の全行（tune / work / eval を含む）
    - ``train_mask``: held-out 型でも除外型でもなく、``role_col`` があれば fit 役割の行のみ
      （eval は学習側に入らない）
    - held-out 型は train_mask に現れない（テストで保証）
    """
    t = _col_str(items, type_col)
    excluded = set(exclude_types)
    fit_ok = _fit_role_mask(items, role_col)
    if types is None:
        types = sorted(set(t.tolist()) - excluded - {""})
    else:
        bad = sorted(set(types) & excluded)
        if bad:
            raise ValueError(f"除外型を held-out に指定した: {bad}")
    not_excluded = ~np.isin(t, sorted(excluded))
    folds: list[Fold] = []
    for held in types:
        test_mask = t == held
        train_mask = not_excluded & ~test_mask & fit_ok
        folds.append(Fold(held, train_mask, test_mask))
    return folds


def loso_folds(
    items: pl.DataFrame,
    source_col: str = "source",
    sources: Sequence[str] = LOSO_SOURCES,
    *,
    role_col: str | None = "role",
    type_col: str | None = "subject_type",
    exclude_types: Sequence[str] = LOTO_EXCLUDE_TYPES,
) -> list[Fold]:
    """leave-one-source-out の fold を返す（§5.4）。型は pooled。

    - ``sources`` の各源を held-out にする 3 fold。``test_mask`` はその源の全行
    - ``train_mask`` は ``sources`` の他の源の行（列挙外の源は学習に入れない）。``role_col`` が
      あれば fit 役割のみ、``type_col`` があれば ``exclude_types`` を除く（対照正例・補助 far は
      負例学習に混ぜない）
    """
    s = _col_str(items, source_col)
    src_list = list(sources)
    in_sources = np.isin(s, src_list)
    fit_ok = _fit_role_mask(items, role_col)
    if type_col is not None and type_col in items.columns:
        t = _col_str(items, type_col)
        type_ok = ~np.isin(t, sorted(set(exclude_types)))
    else:
        type_ok = np.ones(items.height, dtype=bool)
    folds: list[Fold] = []
    for held in src_list:
        test_mask = s == held
        train_mask = in_sources & ~test_mask & fit_ok & type_ok
        folds.append(Fold(held, train_mask, test_mask))
    return folds


def fold_evaluable(n_before: int, min_n: int = FOLD_MIN_N) -> bool:
    """fold の ``n_before``（P0 が food 確定した held-out 枚数）が評価可能か（§5.4、§8.1）。"""
    if n_before < 0:
        raise ValueError(f"n_before が負: {n_before}")
    return int(n_before) >= int(min_n)


# ---------------------------------------------------------------------------
# 集計（レポート用）
# ---------------------------------------------------------------------------

_SUMMARY_SCHEMA = {"role": pl.Utf8, "source": pl.Utf8, "subject_type": pl.Utf8, "n": pl.Int64}


def summarize_roles(df: pl.DataFrame) -> pl.DataFrame:
    """role × source × subject_type の枚数（列: role, source, subject_type, n）。

    ``source`` / ``subject_type`` が無ければ null 列として扱う。role の並びは
    tune_fit → tune_cal → tune → work → eval、他はその後にソート。
    """
    if "role" not in df.columns:
        raise ValueError("role 列が無い")
    cols = ["role", "source", "subject_type"]
    work = df
    for c in cols[1:]:
        if c not in work.columns:
            work = work.with_columns(pl.lit(None, dtype=pl.Utf8).alias(c))
    out = work.group_by(cols).agg(pl.len().cast(pl.Int64).alias("n"))
    order = {r: i for i, r in enumerate(("tune_fit", "tune_cal", "tune", "work", "eval"))}
    out = (
        out.with_columns(
            pl.col("role")
            .replace_strict(order, default=len(order), return_dtype=pl.Int64)
            .alias("_o")
        )
        .sort(["_o", "role", "source", "subject_type"], nulls_last=True)
        .drop("_o")
        .select(cols + ["n"])
        .cast(_SUMMARY_SCHEMA)
    )
    return out


def type_source_crosstab(
    df: pl.DataFrame, type_col: str = "subject_type", source_col: str = "source"
) -> pl.DataFrame:
    """型 × 源の交絡表（行: subject_type、列: 各 source の枚数、末尾 total）。§5.4 の併記用。"""
    for c in (type_col, source_col):
        if c not in df.columns:
            raise ValueError(f"列 {c!r} が無い")
    counts = df.group_by([type_col, source_col]).agg(pl.len().cast(pl.Int64).alias("n"))
    wide = counts.pivot(on=source_col, index=type_col, values="n").fill_null(0)
    src_cols = sorted(c for c in wide.columns if c != type_col)
    wide = wide.select([type_col, *src_cols])
    if src_cols:
        wide = wide.with_columns(pl.sum_horizontal(src_cols).cast(pl.Int64).alias("total"))
    else:
        wide = wide.with_columns(pl.lit(0, dtype=pl.Int64).alias("total"))
    return wide.sort(type_col)
