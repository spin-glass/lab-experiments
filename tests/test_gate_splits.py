"""gate.splits: 役割ガード（eval 漏洩で raise）、LOTO / LOSO fold、評価可否、集計（§3.1、§3.8、§5.4）。"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cascade.gate.splits import (
    FIT_ROLES,
    LOSO_SOURCES,
    LOTO_EXCLUDE_TYPES,
    Fold,
    RoleTable,
    SplitGuard,
    SplitLeakError,
    assert_no_eval,
    fold_evaluable,
    loso_folds,
    loto_folds,
    summarize_roles,
    type_source_crosstab,
)
from cascade.gate.threshold import fit_threshold

TYPES_A = (
    "doll_character_statue",
    "amusement_tourist",
    "signage_decor_goods",
    "person_animal",
    "vehicle_street",
    "near_food_nonfood",
)


def _yelp_table(n: int = 40) -> pl.DataFrame:
    roles = ["tune_fit", "tune_cal", "work", "eval"]
    labels = ["food", "drink", "menu", "inside", "outside"]
    return pl.DataFrame(
        {
            "photo_id": [f"y{i}" for i in range(n)],
            "role": [roles[i % 4] for i in range(n)],
            "source": ["yelp"] * n,
            "subject_type": [None] * n,
            "label": [labels[i % 5] for i in range(n)],
        },
        schema_overrides={"subject_type": pl.Utf8},
    )


def _external_table(seed: int = 0) -> pl.DataFrame:
    """型 × 源 × 役割の全組合せを 1 行ずつ＋対照正例・補助 far・other。"""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    i = 0
    for t in TYPES_A:
        for s in LOSO_SOURCES:
            for r in ("tune", "work", "eval"):
                for _ in range(int(rng.integers(1, 4))):
                    rows.append({"photo_id": f"x{i}", "role": r, "source": s, "subject_type": t})
                    i += 1
    for t, s in (
        ("food_contrast", "open_images"),
        ("food_contrast", "commons"),
        ("texture_document_screen", "cord"),
        ("other", "commons"),
    ):
        for r in ("tune", "eval"):
            rows.append({"photo_id": f"x{i}", "role": r, "source": s, "subject_type": t})
            i += 1
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# RoleTable
# ---------------------------------------------------------------------------


def test_role_table_validates_columns_roles_and_ids():
    with pytest.raises(ValueError, match="必須列"):
        RoleTable(pl.DataFrame({"photo_id": ["a"], "role": ["eval"]}))
    with pytest.raises(ValueError, match="未知の役割"):
        RoleTable(_yelp_table().with_columns(pl.lit("test").alias("role")))
    dup = pl.concat([_yelp_table(4), _yelp_table(4)])
    with pytest.raises(ValueError, match="重複"):
        RoleTable(dup)
    tbl = RoleTable(_yelp_table(8))
    assert tbl.height == 8
    assert tbl.indices("eval").tolist() == [3, 7]
    assert tbl.mask("tune_fit").tolist() == [True, False, False, False] * 2
    assert tbl.rows("work")["photo_id"].to_list() == ["y2", "y6"]


# ---------------------------------------------------------------------------
# ガード
# ---------------------------------------------------------------------------


def test_fit_view_rejects_eval_and_accepts_fit_roles():
    g = SplitGuard.from_frame(_yelp_table())
    for r in ("tune_fit", "tune_cal", "work"):
        view = g.fit_view(r)
        assert set(view["role"].to_list()) == {r}
    with pytest.raises(SplitLeakError):
        g.fit_view("eval")
    with pytest.raises(SplitLeakError):
        g.fit_indices("eval")
    with pytest.raises(ValueError):
        g.fit_view("holdout")
    assert g.eval_access_count == 0
    assert g.fit_access_log == ["tune_fit", "tune_cal", "work"]


def test_external_tune_role_is_fit_side():
    g = SplitGuard.from_frame(_external_table())
    assert "tune" in FIT_ROLES
    assert set(g.fit_view("tune")["role"].to_list()) == {"tune"}


def test_eval_view_counts_accesses():
    g = SplitGuard.from_frame(_yelp_table())
    ev = g.eval_view()
    assert set(ev["role"].to_list()) == {"eval"}
    g.eval_indices()
    assert g.eval_access_count == 2
    s = g.summary()
    assert s["eval_access_count"] == 2 and s["n_by_role"]["eval"] == 10


def test_assert_no_eval_on_arrays_series_and_lists():
    assert_no_eval(np.array(["tune_fit", "work"]))
    assert_no_eval(pl.Series(["tune", "tune_cal"]))
    assert_no_eval(["work"])
    with pytest.raises(SplitLeakError, match="eval の行が 2 件"):
        assert_no_eval(["work", "eval", "eval"], context="knn.fit")
    with pytest.raises(ValueError, match="未知の役割"):
        assert_no_eval(["train"])


def test_eval_rows_cannot_reach_threshold_fit():
    """閾値決定の直前に役割配列をガードすると eval 行の混入で止まる。"""
    df = _yelp_table(400)
    tbl = RoleTable(df)
    scores = np.linspace(-1, 1, df.height)
    cal = tbl.indices("tune_cal")
    assert_no_eval(tbl.roles[cal], context="threshold")
    res = fit_threshold(scores[cal], 0.01)
    assert res.n == 100
    leaked = np.concatenate([cal, tbl.indices("eval")])
    with pytest.raises(SplitLeakError):
        assert_no_eval(tbl.roles[leaked], context="threshold")


# ---------------------------------------------------------------------------
# LOTO
# ---------------------------------------------------------------------------


def test_loto_folds_exclude_types_and_hold_out_absent_from_train():
    df = _external_table()
    folds = loto_folds(df)
    assert [f.held_out for f in folds] == sorted(TYPES_A)
    t = df["subject_type"].to_numpy()
    r = df["role"].to_numpy()
    for held, train, test in folds:
        assert isinstance(train, np.ndarray) and train.dtype == bool
        assert train.shape == (df.height,) and test.shape == (df.height,)
        assert held not in set(t[train])
        assert set(t[test]) == {held}
        assert not (train & test).any()
        # 除外型は学習にも held-out にも出ない
        assert not (set(t[train]) & set(LOTO_EXCLUDE_TYPES))
        # held-out は tune / work / eval 全体
        assert set(r[test]) == {"tune", "work", "eval"}
        # 学習側は fit 役割のみ（eval を含まない）
        assert "eval" not in set(r[train])
        assert set(r[train]) <= FIT_ROLES
        assert train.sum() > 0 and test.sum() > 0


def test_loto_folds_tuple_unpack_and_indexing():
    folds = loto_folds(_external_table())
    f = folds[0]
    assert isinstance(f, Fold)
    held, tr, te = f
    assert held == f[0] and (tr == f[1]).all() and (te == f[2]).all()
    assert f.n_train == int(tr.sum()) and f.n_test == int(te.sum())


def test_loto_without_role_column_uses_all_rows_for_train():
    df = _external_table().drop("role")
    folds = loto_folds(df)
    t = df["subject_type"].to_numpy()
    for held, train, _ in folds:
        others = set(TYPES_A) - {held}
        assert set(t[train]) == others


def test_loto_explicit_types_and_bad_types():
    df = _external_table()
    folds = loto_folds(df, types=["person_animal", "vehicle_street"])
    assert [f.held_out for f in folds] == ["person_animal", "vehicle_street"]
    with pytest.raises(ValueError, match="除外型"):
        loto_folds(df, types=["food_contrast"])
    with pytest.raises(ValueError, match="列"):
        loto_folds(df, type_col="kind")


# ---------------------------------------------------------------------------
# LOSO
# ---------------------------------------------------------------------------


def test_loso_folds_three_sources_pooled_types():
    df = _external_table()
    folds = loso_folds(df)
    assert [f.held_out for f in folds] == list(LOSO_SOURCES)
    s = df["source"].to_numpy()
    t = df["subject_type"].to_numpy()
    r = df["role"].to_numpy()
    for held, train, test in folds:
        assert set(s[test]) == {held}
        assert held not in set(s[train])
        assert set(s[train]) == set(LOSO_SOURCES) - {held}
        # 型は pooled（層 A 6 型が全て学習に入る）
        assert set(t[train]) == set(TYPES_A)
        assert "eval" not in set(r[train])
        assert not (train & test).any()
    # 列挙外の源（cord）はどの fold でも held-out にならない
    assert "cord" not in {f.held_out for f in folds}


def test_loso_train_excludes_food_contrast_and_far():
    df = _external_table()
    t = df["subject_type"].to_numpy()
    for _, train, _ in loso_folds(df):
        assert not (set(t[train]) & set(LOTO_EXCLUDE_TYPES))
    # type_col を無効化すると対照正例も学習に混ざる（呼び出し側の明示が必要）
    for _, train, _ in loso_folds(df, type_col=None):
        assert "food_contrast" in set(t[train])


# ---------------------------------------------------------------------------
# 評価可否・集計
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "ok"), [(0, False), (9, False), (10, True), (40, True)])
def test_fold_evaluable(n, ok):
    assert fold_evaluable(n) is ok


def test_fold_evaluable_custom_min_and_negative():
    assert fold_evaluable(5, min_n=5)
    with pytest.raises(ValueError):
        fold_evaluable(-1)


def test_summarize_roles_counts_and_order():
    df = pl.concat([_yelp_table(8), _external_table()], how="diagonal")
    out = summarize_roles(df)
    assert out.columns == ["role", "source", "subject_type", "n"]
    assert out["n"].sum() == df.height
    roles = out["role"].to_list()
    first_eval = roles.index("eval")
    assert all(r != "eval" for r in roles[:first_eval])
    assert roles[0] == "tune_fit"
    yelp = out.filter(pl.col("source") == "yelp")
    assert yelp["subject_type"].null_count() == yelp.height
    assert yelp.filter(pl.col("role") == "eval")["n"].item() == 2
    ext = out.filter(
        (pl.col("source") == "coco")
        & (pl.col("subject_type") == "person_animal")
        & (pl.col("role") == "eval")
    )
    expected = df.filter(
        (pl.col("source") == "coco")
        & (pl.col("subject_type") == "person_animal")
        & (pl.col("role") == "eval")
    ).height
    assert ext["n"].item() == expected


def test_summarize_roles_without_optional_columns():
    out = summarize_roles(pl.DataFrame({"role": ["eval", "work", "work"]}))
    assert out["role"].to_list() == ["work", "eval"]
    assert out["n"].to_list() == [2, 1]
    with pytest.raises(ValueError):
        summarize_roles(pl.DataFrame({"x": [1]}))


def test_type_source_crosstab():
    df = _external_table()
    ct = type_source_crosstab(df)
    assert ct.columns[0] == "subject_type" and ct.columns[-1] == "total"
    assert ct["total"].sum() == df.height
    row = ct.filter(pl.col("subject_type") == "texture_document_screen")
    assert row["cord"].item() == 2 and row["coco"].item() == 0
    for c in ct.columns[1:]:
        assert ct[c].dtype == pl.Int64
    with pytest.raises(ValueError):
        type_source_crosstab(df.drop("source"))
