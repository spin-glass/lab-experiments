"""dedup: 近重複群・決定的役割割当・役割非跨ぎ（design.md §2）。"""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from cascade.data.dedup import (
    UnionFind,
    assign_roles,
    assign_roles_stratified,
    build_groups,
    check_no_crossing,
    hamming64,
    phash64,
    sha256_file,
    split_by_components,
)


def test_sha256_file_matches_hashlib(make_image):
    p = make_image("a.jpg", seed=1)
    assert sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_phash64_is_64bit_and_stable(make_image):
    p = make_image("a.jpg", seed=1)
    h1, h2 = phash64(p), phash64(p)
    assert h1 == h2
    assert 0 <= h1 < (1 << 64)
    assert hamming64(h1, h1) == 0
    assert hamming64(0, (1 << 64) - 1) == 64


def test_union_find_components():
    uf = UnionFind(["a", "b", "c", "d"])
    uf.union("a", "b")
    uf.union("c", "b")
    comps = uf.components()
    assert sorted(map(sorted, comps.values())) == [["a", "b", "c"], ["d"]]


def test_near_duplicates_group_and_distinct_does_not(
    make_image, make_near_duplicate, make_far_image
):
    a = make_image("a.jpg", seed=42)
    a_dup = make_near_duplicate(a, "a_dup.jpg")
    far = make_far_image("far.jpg")
    assert sha256_file(a) != sha256_file(a_dup)  # 再エンコードで完全一致ではない
    assert hamming64(phash64(a), phash64(a_dup)) <= 8
    assert hamming64(phash64(a), phash64(far)) > 8

    items = [
        {"photo_id": "a", "path": a},
        {"photo_id": "a_dup", "path": a_dup},
        {"photo_id": "far", "path": far},
    ]
    g = build_groups(items, phash_max=8, cos_min=0.95)
    assert g["a"] == g["a_dup"]
    assert g["far"] != g["a"]
    assert all(v.startswith("g") and len(v) == 13 for v in g.values())


def test_group_id_is_stable_over_order(make_image, make_near_duplicate):
    a = make_image("a.jpg", seed=3)
    b = make_near_duplicate(a, "b.jpg")
    items = [{"photo_id": "a", "path": a}, {"photo_id": "b", "path": b}]
    g1 = build_groups(items)
    g2 = build_groups(list(reversed(items)))
    assert g1 == g2


def test_sha256_exact_duplicate(make_image, tmp_path: Path):
    a = make_image("a.jpg", seed=5)
    copy = tmp_path / "copy.jpg"
    shutil.copy(a, copy)
    # phash を無効化（None を明示）しても sha256 で結合される
    items = [
        {"photo_id": "a", "sha256": sha256_file(a), "phash": None},
        {"photo_id": "c", "sha256": sha256_file(copy), "phash": None},
        {"photo_id": "z", "sha256": "0" * 64, "phash": None},
    ]
    g = build_groups(items)
    assert g["a"] == g["c"] != g["z"]


def test_shared_group_key_merges():
    items = [
        {"photo_id": "p1", "sha256": "1" * 64, "group_key": "biz_A"},
        {"photo_id": "p2", "sha256": "2" * 64, "group_key": "biz_A"},
        {"photo_id": "p3", "sha256": "3" * 64, "group_key": "biz_B"},
        {"photo_id": "p4", "sha256": "4" * 64},
    ]
    g = build_groups(items)
    assert g["p1"] == g["p2"]
    assert len({g["p1"], g["p3"], g["p4"]}) == 3
    g_off = build_groups(items, use_group_key=False)
    assert len(set(g_off.values())) == 4


def test_embedding_cosine_merges():
    rng = np.random.default_rng(0)
    base = rng.normal(size=16).astype(np.float32)
    near = base + 0.01 * rng.normal(size=16).astype(np.float32)
    far = rng.normal(size=16).astype(np.float32)
    items = [
        {"photo_id": "b", "sha256": "b" * 64, "emb": base},
        {"photo_id": "n", "sha256": "c" * 64, "emb": near},
        {"photo_id": "f", "sha256": "d" * 64, "emb": far},
    ]
    g = build_groups(items, cos_min=0.95)
    assert g["b"] == g["n"] != g["f"]


def test_build_groups_rejects_duplicate_photo_id():
    with pytest.raises(ValueError):
        build_groups([{"photo_id": "x", "sha256": "1" * 64}, {"photo_id": "x", "sha256": "2" * 64}])


def test_assign_roles_deterministic_and_respects_ratios():
    keys = [f"key{i}" for i in range(5000)]
    ratios = {"tune": 0.3, "work": 0.1, "eval": 0.6}
    r1 = assign_roles(keys, ratios, seed="s1")
    r2 = assign_roles(list(reversed(keys)), ratios, seed="s1")
    assert r1 == r2
    c = Counter(r1.values())
    for role, target in ratios.items():
        assert abs(c[role] / 5000 - target) <= 0.03, (role, c[role])
    # seed を変えると割当が変わる
    r3 = assign_roles(keys, ratios, seed="s2")
    assert r3 != r1
    # 比率が 1 でない場合は正規化
    r4 = assign_roles(keys, {"a": 3, "b": 1}, seed="s1")
    assert set(r4.values()) <= {"a", "b"}
    assert abs(Counter(r4.values())["a"] / 5000 - 0.75) <= 0.03


def test_assign_roles_invalid_ratios():
    with pytest.raises(ValueError):
        assign_roles(["k"], {}, "s")
    with pytest.raises(ValueError):
        assign_roles(["k"], {"a": -1, "b": 2}, "s")


def test_assign_roles_stratified_reports_per_stratum():
    keys = [f"k{i}" for i in range(2000)]
    strata = {k: ("food" if i % 2 == 0 else "menu") for i, k in enumerate(keys)}
    ratios = {"tune": 0.3, "work": 0.1, "eval": 0.6}
    roles, df = assign_roles_stratified(keys, strata, ratios, seed="s")
    assert roles == assign_roles(keys, ratios, "s")
    assert set(df.columns) == {"stratum", "role", "n", "share"}
    assert df["n"].sum() == 2000
    for stratum in ("food", "menu"):
        sub = df.filter(df["stratum"] == stratum)
        assert abs(sub["share"].sum() - 1.0) < 1e-9
        ev = sub.filter(sub["role"] == "eval")["share"][0]
        assert abs(ev - 0.6) <= 0.05


def test_check_no_crossing_detects_violation():
    groups = {"a": "g1", "b": "g1", "c": "g2", "d": "g2", "e": "g3"}
    ok = {"a": "tune", "b": "tune", "c": "eval", "d": "eval", "e": "work"}
    assert check_no_crossing(ok, groups) == []
    bad = {**ok, "b": "eval", "d": "work"}
    assert check_no_crossing(bad, groups) == ["g1", "g2"]
    # group 情報のない item は無視
    assert check_no_crossing({"zz": "tune"}, groups) == []


def test_split_by_components_never_crosses():
    rng = np.random.default_rng(1)
    items = []
    for i in range(3000):
        biz = f"biz{rng.integers(0, 600)}"
        items.append(
            {
                "photo_id": f"p{i}",
                "sha256": hashlib.sha256(f"p{i}".encode()).hexdigest(),
                "group_key": biz,
                "label": rng.choice(["food", "drink", "menu", "inside", "outside"]),
            }
        )
    # 若干の完全重複（別店舗間）を混ぜる
    for i in range(0, 100, 2):
        items[i + 1]["sha256"] = items[i]["sha256"]
    ratios = {"tune": 0.3, "work": 0.1, "eval": 0.6}
    role_of_item, summary = split_by_components(items, ratios, seed="g0")
    assert set(role_of_item) == {it["photo_id"] for it in items}
    groups = build_groups(items)
    assert check_no_crossing(role_of_item, groups) == []
    biz_of = {it["photo_id"]: it["group_key"] for it in items}
    assert check_no_crossing(role_of_item, biz_of) == []
    assert set(summary.columns) == {"role", "n_components", "n_items", "label", "n"}
    assert summary["n"].sum() == 3000
    assert summary.filter(summary["role"] == "eval")["n_items"][0] > 0
    # 決定性
    role2, _ = split_by_components(list(reversed(items)), ratios, seed="g0")
    assert role2 == role_of_item


def test_split_by_components_with_precomputed_groups():
    items = [
        {"photo_id": "a", "group_key": "b1"},
        {"photo_id": "b", "group_key": "b2"},
        {"photo_id": "c", "group_key": "b3"},
    ]
    groups = {"a": "gX", "b": "gX", "c": "gY"}
    role_of_item, summary = split_by_components(
        items, {"tune": 0.5, "eval": 0.5}, "s", groups=groups
    )
    assert role_of_item["a"] == role_of_item["b"]
    assert summary["label"].null_count() == summary.height


def test_phash_blocked_path_matches_exhaustive():
    from cascade.data.dedup import _phash_pairs

    rng = np.random.default_rng(7)
    n = 400
    hashes = [int(x) for x in rng.integers(0, 1 << 63, size=n, dtype=np.int64)]
    # 近傍を意図的に作る: 1〜8 bit 反転
    for i in range(0, 200, 2):
        h = hashes[i]
        for b in rng.choice(64, size=int(rng.integers(1, 9)), replace=False):
            h ^= 1 << int(b)
        hashes[i + 1] = h
    ids = [f"h{i}" for i in range(n)]
    full = {tuple(sorted(p)) for p in _phash_pairs(ids, hashes, 8, full_pairs_max=10_000)}
    blocked = {tuple(sorted(p)) for p in _phash_pairs(ids, hashes, 8, full_pairs_max=0)}
    assert len(full) >= 100
    assert full == blocked
