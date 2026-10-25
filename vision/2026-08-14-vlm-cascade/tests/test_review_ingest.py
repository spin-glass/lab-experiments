"""review_queue（盲検キュー）と review_ingest（jsonl → scope_annotations、κ）のオフラインテスト。"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from cascade.data import review_ingest as ri
from cascade.data import review_queue as rq

SEED = "g0-review-seed-1"


def _items(n: int = 12) -> list[dict]:
    src = ["open_images", "commons", "yelp"]
    return [
        {
            "photo_id": f"p{i:03d}",
            "source": src[i % 3],
            "path": f"ext/{src[i % 3]}/images/p{i:03d}.jpg",
            "candidate_label": "doll",
            "score": 0.9,
            "split_hint": "eval",
            "license_short": "CC BY 4.0",
            "sha256": "ab" * 32,
        }
        for i in range(n)
    ]


# --- queue ---------------------------------------------------------------------------------


def test_queue_is_blind_and_deterministic(tmp_path: Path):
    q1 = rq.build_queue(_items(), 1, SEED, tmp_path / "a.jsonl")
    q2 = rq.build_queue(list(reversed(_items())), 1, SEED, tmp_path / "b.jsonl")
    assert q1 == q2, "入力順に依存せずシードで決まる"
    assert q1 != rq.build_queue(_items(), 1, "other-seed", tmp_path / "c.jsonl")

    rows = [json.loads(line) for line in (tmp_path / "a.jsonl").read_text().splitlines()]
    assert len(rows) == 12
    for r in rows:
        assert set(r) == {"item_id", "path", "pass"}, (
            "盲検: source / label / score / split を載せない"
        )
        assert r["pass"] == 1
        assert "open_images" not in r["item_id"] and "p0" not in r["item_id"]
    assert [r["item_id"] for r in rows] != sorted(r["item_id"] for r in rows), (
        "シャッフルされている"
    )

    m = rq.load_item_map(tmp_path / "a.map.jsonl")
    assert set(m) == {r["item_id"] for r in rows}
    assert m[rows[0]["item_id"]]["source"] in {"open_images", "commons", "yelp"}
    assert rq.load_queue(tmp_path / "a.jsonl") == rows


def test_load_queue_rejects_forbidden_keys(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"item_id": "x", "path": "a.jpg", "pass": 1, "source": "yelp"}) + "\n")
    with pytest.raises(ValueError, match="盲検"):
        rq.load_queue(p)


def test_build_queue_rejects_path_outside_data(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rq.paths, "DATA", tmp_path / "data")
    with pytest.raises(ValueError):
        rq.build_queue([{"photo_id": "a", "path": "/etc/passwd"}], 1, SEED, tmp_path / "q.jsonl")
    q = rq.build_queue(
        [{"photo_id": "a", "path": str(tmp_path / "data" / "ext" / "a.jpg")}],
        1,
        SEED,
        tmp_path / "q.jsonl",
    )
    assert q[0]["path"] == "ext/a.jpg"


def test_pass2_queue_keeps_only_keep_and_boundary(tmp_path: Path):
    q1 = rq.build_queue(_items(6), 1, SEED, tmp_path / "q1.jsonl")
    ids = [r["item_id"] for r in q1]
    labels = [
        {"item_id": ids[0], "pass": 1, "decision": "keep", "ts": "2026-09-07T00:00:00Z"},
        {"item_id": ids[1], "pass": 1, "decision": "drop", "ts": "2026-09-07T00:00:01Z"},
        {"item_id": ids[2], "pass": 1, "decision": "boundary", "ts": "2026-09-07T00:00:02Z"},
        {"item_id": ids[3], "pass": 1, "decision": "keep", "ts": "2026-09-07T00:00:03Z"},
        {
            "item_id": ids[3],
            "pass": 1,
            "decision": "drop",
            "ts": "2026-09-07T00:00:04Z",
        },  # 最新が勝つ
        {
            "item_id": ids[4],
            "pass": 2,
            "decision": "keep",
            "ts": "2026-09-07T00:00:05Z",
        },  # pass 違い
    ]
    lp = tmp_path / "labels1.jsonl"
    lp.write_text("".join(json.dumps(r) + "\n" for r in labels))
    q2 = rq.build_pass2_queue(
        lp, SEED, tmp_path / "q2.jsonl", tmp_path / "q1.jsonl", tmp_path / "q2.map.jsonl"
    )
    assert {r["item_id"] for r in q2} == {ids[0], ids[2]}
    assert all(set(r) == {"item_id", "path", "pass"} and r["pass"] == 2 for r in q2)
    assert set(rq.load_item_map(tmp_path / "q2.map.jsonl")) == {ids[0], ids[2]}


# --- ingest --------------------------------------------------------------------------------


def _p2(item_id: str, ts: str, **kw) -> dict:
    row = {
        "item_id": item_id,
        "pass": 2,
        "content": "non_food",
        "operational_truth": "out_of_scope",
        "subject_type": "doll_character_statue",
        "boundary_flag": False,
        "quality": "ok",
        "annotator": "a01",
        "guideline_version": "g1",
        "ts": ts,
    }
    row.update(kw)
    return row


def test_ingest_latest_wins_and_columns(tmp_path: Path):
    lookup = {
        "i1": {
            "photo_id": "p1",
            "source": "open_images",
            "license_short": "CC BY 4.0",
            "sha256": "ff",
            "role": "eval",
        },
        "i2": {"photo_id": "p2", "source": "yelp", "dup_group_id": "g7"},
        "i3": {"photo_id": "p3", "source": "commons"},
    }
    rows = [
        _p2("i1", "2026-09-07T01:00:00Z", operational_truth="inside"),
        _p2("i1", "2026-09-07T02:00:00Z", operational_truth="outside", boundary_flag=True),
        _p2(
            "i2",
            "2026-09-07T01:00:00Z",
            content="food",
            operational_truth="food",
            subject_type="food_contrast",
        ),
        {
            "item_id": "i3",
            "pass": 1,
            "decision": "drop",
            "annotator": "a01",
            "guideline_version": "g1",
            "ts": "t",
        },
        {
            "item_id": "i1",
            "pass": 1,
            "decision": "keep",
            "annotator": "a01",
            "guideline_version": "g1",
            "ts": "t",
        },
        _p2("i3", "2026-09-07T01:00:00Z", guideline_version="pilot"),  # 版違いは無視
    ]
    scope, p1 = ri.ingest(rows, lookup, "g1")
    assert list(scope.columns) == list(ri.SCOPE_COLUMNS)
    assert scope.height == 2
    r1 = scope.filter(pl.col("photo_id") == "p1").row(0, named=True)
    assert r1["operational_truth"] == "outside" and r1["boundary_flag"] is True
    assert (
        r1["source"] == "open_images"
        and r1["license_short"] == "CC BY 4.0"
        and r1["role"] == "eval"
    )
    assert scope.schema["boundary_flag"] == pl.Boolean
    assert scope.filter(pl.col("photo_id") == "p2").row(0, named=True)["dup_group_id"] == "g7"
    assert p1.height == 2 and set(p1["decision"].to_list()) == {"drop", "keep"}
    assert list(p1.columns) == list(ri.PASS1_COLUMNS)


def test_ingest_from_jsonl_file_and_missing_lookup(tmp_path: Path):
    lp = tmp_path / "l.jsonl"
    lp.write_text(json.dumps(_p2("zz", "2026-09-07T01:00:00Z")) + "\n")
    with pytest.raises(ValueError, match="manifest_lookup"):
        ri.ingest(lp, {}, "g1")
    scope, _ = ri.ingest(lp, {"zz": {"photo_id": "p", "source": "coco"}}, "g1")
    assert scope.height == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"content": "foods"},
        {"operational_truth": "other"},
        {"subject_type": "dolls"},
        {"quality": "bad"},
        {"boundary_flag": "yes"},
    ],
)
def test_ingest_validates_enums(bad: dict):
    with pytest.raises(ValueError):
        ri.ingest(
            [_p2("i1", "2026-09-07T01:00:00Z", **bad)],
            {"i1": {"photo_id": "p", "source": "coco"}},
            "g1",
        )


def test_ingest_validates_pass1_decision():
    with pytest.raises(ValueError, match="decision"):
        ri.ingest(
            [
                {
                    "item_id": "i",
                    "pass": 1,
                    "decision": "maybe",
                    "guideline_version": "g1",
                    "ts": "t",
                }
            ],
            {},
            "g1",
        )


def test_ingest_empty_gives_empty_frames_with_schema():
    scope, p1 = ri.ingest([], {}, "g1")
    assert scope.height == 0 and list(scope.columns) == list(ri.SCOPE_COLUMNS)
    assert p1.height == 0


def test_write_scope_annotations_without_io_module(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("cascade.io"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    scope, _ = ri.ingest([], {}, "g1")
    assert ri.write_scope_annotations(scope) is None


# --- kappa ---------------------------------------------------------------------------------


def test_kappa_known_value():
    # 2x2: a=20 (yes/yes), b=5, c=10, d=15 → po=0.7, pe=0.5 → κ=0.4
    pairs = (
        [("food", "food")] * 20
        + [("food", "inside")] * 5
        + [("inside", "food")] * 10
        + [("inside", "inside")] * 15
    )
    assert ri.cohen_kappa(pairs) == pytest.approx(0.4)
    assert ri.cohen_kappa([("a", "a"), ("b", "b")]) == pytest.approx(1.0)
    assert ri.cohen_kappa([("a", "a")]) == pytest.approx(1.0)
    assert ri.cohen_kappa([("a", "b")]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        ri.cohen_kappa([])


def test_kappa_self_agreement_on_labels(tmp_path: Path):
    a = [
        _p2(f"i{i}", "2026-09-07T01:00:00Z", operational_truth=v)
        for i, v in enumerate(["food", "food", "inside", "menu"])
    ]
    b = [
        _p2(f"i{i}", "2026-09-08T01:00:00Z", operational_truth=v, boundary_flag=True)
        for i, v in enumerate(["food", "inside", "inside", "menu"])
    ]
    b.append(_p2("i99", "2026-09-08T01:00:00Z"))  # 片方だけの項目は無視
    pa, pb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    pa.write_text("".join(json.dumps(r) + "\n" for r in a))
    pb.write_text("".join(json.dumps(r) + "\n" for r in b))
    # po=3/4, pe = (2*1 + 1*2 + 1*1)/16 = 5/16 → κ = (0.75-0.3125)/(0.6875)
    expect = (0.75 - 5 / 16) / (1 - 5 / 16)
    assert ri.kappa_self_agreement(pa, pb) == pytest.approx(expect)
    rep = ri.self_agreement_report(a, b)
    assert rep["n"] == 4 and rep["agreement"] == pytest.approx(0.75)
    assert rep["boundary_rate_a"] == 0.0 and rep["boundary_rate_b"] == 1.0
