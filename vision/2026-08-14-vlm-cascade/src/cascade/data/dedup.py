"""近重複群の構築・決定的な役割割当・役割非跨ぎ検証（design.md §2、計画「評価集合」）。

分割キーは ``business_id`` と近重複群（sha256 完全一致 → pHash Hamming ≤ 8 → 埋め込み余弦 ≥ 0.95）の
連結成分。同一店舗・同一群は tune / work / eval を跨がない。

- :func:`build_groups`      union-find で dup 群を作り photo_id → dup_group_id を返す。
                            **dup 群は sha256 / pHash / 余弦のみ**（既定 ``use_group_key=False``）。
                            business_id は「分割キー」であり dup 群には混ぜない
- :func:`assign_roles`      sha1(seed ␀ key) から一様乱数を作り、累積比率で役割を決める（決定的）
- :func:`assign_roles_stratified` 層ごとに決定的に並べ替え、累積比率の分位で切る層化割当
- :func:`report_role_shares` 層 × 役割の実現比率を報告する
- :func:`check_no_crossing` 群が役割を跨いでいないかを検証する
- :func:`split_by_components` business_id ∪ dup 群の連結成分単位で役割を割り当てる

pHash は 64 bit 符号なしで扱う。parquet / BigQuery の int64 へ永続化するときは
:func:`to_int64` / :func:`from_int64`（2 の補数）を使い、``build_groups`` は負値入力も正規化する。

画像・生データはリポジトリに置かず、ハッシュと seed のみを記録する（CLAUDE.md 絶対制約）。
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
import polars as pl
from PIL import Image

__all__ = [
    "UnionFind",
    "assign_roles",
    "assign_roles_stratified",
    "build_groups",
    "check_no_crossing",
    "from_int64",
    "hamming64",
    "phash64",
    "report_role_shares",
    "sha256_file",
    "split_by_components",
    "to_int64",
]

ROLES_DEFAULT: dict[str, float] = {"tune": 0.3, "work": 0.1, "eval": 0.6}

_MASK64 = 0xFFFF_FFFF_FFFF_FFFF


# ---------------------------------------------------------------------------
# ハッシュ
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """ファイルの sha256（16 進 64 桁）。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def phash64(path: str | Path | Image.Image) -> int:
    """imagehash.phash(hash_size=8) を 64 bit 符号なし整数にする。"""
    if isinstance(path, Image.Image):
        im = path
        h = imagehash.phash(im, hash_size=8)
    else:
        with Image.open(path) as im:
            h = imagehash.phash(im, hash_size=8)
    bits = np.asarray(h.hash, dtype=np.uint8).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def to_int64(h: int) -> int:
    """64 bit 符号なし → 符号付き int64（2 の補数）。parquet / BigQuery へ永続化するときに使う。"""
    h = int(h) & _MASK64
    return h - (1 << 64) if h >= (1 << 63) else h


def from_int64(v: int) -> int:
    """符号付き int64（2 の補数）→ 64 bit 符号なし。負値・符号なし値のどちらを渡しても同じ値になる。"""
    return int(v) & _MASK64


def hamming64(a: int, b: int) -> int:
    """64 bit 整数同士の Hamming 距離（符号は無視して下位 64 bit を比較）。"""
    return ((int(a) ^ int(b)) & _MASK64).bit_count()


# ---------------------------------------------------------------------------
# union-find
# ---------------------------------------------------------------------------


class UnionFind:
    """文字列キーの union-find（経路圧縮＋ランク）。"""

    def __init__(self, keys: Iterable[str] = ()) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}
        for k in keys:
            self.add(k)

    def add(self, key: str) -> None:
        if key not in self._parent:
            self._parent[key] = key
            self._rank[key] = 0

    def find(self, key: str) -> str:
        self.add(key)
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[key] != root:
            self._parent[key], key = root, self._parent[key]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1

    def components(self) -> dict[str, list[str]]:
        """root → 所属キー（ソート済み）。"""
        comp: dict[str, list[str]] = defaultdict(list)
        for k in self._parent:
            comp[self.find(k)].append(k)
        return {r: sorted(v) for r, v in comp.items()}


def _group_id(members: Sequence[str]) -> str:
    """群 ID = "g" + sha1(ソート済み photo_id を "\\n" 連結)[:12]。メンバー集合に対して安定。"""
    joined = "\n".join(sorted(members)).encode("utf-8")
    return "g" + hashlib.sha1(joined).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 近重複群
# ---------------------------------------------------------------------------


def _popcount64(x: np.ndarray) -> np.ndarray:
    return np.bitwise_count(x.astype(np.uint64, copy=False))


def _pairs_all(arr: np.ndarray, idx: np.ndarray, phash_max: int) -> Iterable[tuple[int, int]]:
    """idx（元配列の添字）内の全対を XOR + popcount で検査し、(i, j) i<j を返す。"""
    sub = arr[idx]
    m = len(idx)
    for a in range(m - 1):
        cnt = _popcount64(sub[a] ^ sub[a + 1 :])
        for b in np.flatnonzero(cnt <= phash_max):
            yield int(idx[a]), int(idx[a + 1 + int(b)])


def _block_bounds(n_bits: int, k: int) -> list[tuple[int, int]]:
    """n_bits を k 個の連続ブロックに正確に分割する（境界 b_j = n_bits * j // k）。"""
    return [(n_bits * j // k, n_bits * (j + 1) // k) for j in range(k)]


def _pairs_blocked(
    arr: np.ndarray,
    idx: np.ndarray,
    free_bits: Sequence[int],
    phash_max: int,
    bucket_max: int,
    depth: int,
) -> Iterable[tuple[int, int]]:
    """multi-index hashing（段数を ``depth`` で制限）。

    ``free_bits``（まだ鍵に使っていないビット位置）を k = phash_max + 1 個のブロックに正確に分割し、
    ブロックの値ごとにバケットへ分ける。鳩の巣原理により、残りビット上で Hamming ≤ phash_max の
    ペアは少なくとも 1 ブロックが一致するので、バケット内の全対検査で取りこぼさない。
    バケットが ``bucket_max`` 以下、段数を使い切った、または残りビット数が phash_max 以下で
    鳩の巣が効かなくなったら全対比較へ落ちる。同じペアが複数ブロックから出ることはあるが、
    union は冪等なので問題ない。
    """
    k = phash_max + 1
    if depth <= 0 or len(idx) <= bucket_max or len(free_bits) <= phash_max:
        yield from _pairs_all(arr, idx, phash_max)
        return
    sub = arr[idx]
    for lo, hi in _block_bounds(len(free_bits), k):
        block = free_bits[lo:hi]
        if not block:
            continue
        mask = np.uint64(sum(1 << b for b in block))
        keys = sub & mask
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(order)]))
        block_set = set(block)
        rest = [b for b in free_bits if b not in block_set]
        for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
            if e - s < 2:
                continue
            yield from _pairs_blocked(arr, idx[order[s:e]], rest, phash_max, bucket_max, depth - 1)


def _phash_pairs(
    ids: list[str],
    hashes: list[int],
    phash_max: int,
    *,
    full_pairs_max: int = 20_000,
    bucket_max: int = 2_000,
    max_depth: int = 2,
) -> Iterable[tuple[str, str]]:
    """Hamming ≤ phash_max のペアを列挙する。

    まず完全一致（距離 0）の hash を代表 1 件に畳み（同一 hash の id は代表と直接結ぶ）、相異なる
    hash だけを探索する。n ≤ full_pairs_max は全対比較（ベクトル化した XOR + popcount）。それ以上は
    multi-index hashing: 64 bit を (phash_max + 1) 個のブロックに **正確に** 分ける（64 = 8 ブロック
    × 8 bit ではなく、k = 9 なら幅 7/7/7/7/7/7/7/8）ので、鳩の巣原理により Hamming ≤ phash_max の
    ペアは少なくとも 1 ブロックが一致し、閾値ちょうどの距離でも再現率は落ちない。ブロック一致の
    候補は正確な Hamming 距離で確定する。巨大なバケットは残りビットで最大 ``max_depth`` 段まで
    再分割する。
    """
    n = len(ids)
    if n < 2:
        return
    arr = np.fromiter((int(h) & _MASK64 for h in hashes), dtype=np.uint64, count=n)
    uniq, first, inverse = np.unique(arr, return_index=True, return_inverse=True)
    # 完全一致は代表（最初の id）と直接結ぶ
    for i, u in enumerate(inverse.tolist()):
        rep = int(first[u])
        if rep != i:
            yield ids[rep], ids[i]
    m = len(uniq)
    if m < 2:
        return
    pos = np.arange(m)
    if m <= full_pairs_max:
        pairs: Iterable[tuple[int, int]] = _pairs_all(uniq, pos, phash_max)
    else:
        pairs = _pairs_blocked(uniq, pos, list(range(64)), phash_max, bucket_max, max_depth)
    for i, j in pairs:
        yield ids[int(first[i])], ids[int(first[j])]


def _cosine_pairs(
    ids: list[str], embs: list[np.ndarray], cos_min: float, block: int = 512
) -> Iterable[tuple[str, str]]:
    """L2 正規化後の内積 ≥ cos_min のペアを列挙する（上三角のみのブロック行列積）。

    ブロック b × 残り n' の float32 行列を作るので、メモリは block × n × 4 byte 程度。
    """
    n = len(ids)
    if n < 2:
        return
    m = np.stack([np.asarray(e, dtype=np.float32).reshape(-1) for e in embs])
    norm = np.linalg.norm(m, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    m = m / norm
    for start in range(0, n - 1, block):
        q = m[start : start + block]
        sims = q @ m[start:].T  # [b, n - start]（列 c は global start + c）
        rows, cols = np.nonzero(sims >= cos_min)
        keep = cols > rows  # 上三角のみ（自己・重複を除く）
        for r, c in zip(rows[keep].tolist(), cols[keep].tolist(), strict=True):
            yield ids[start + r], ids[start + c]


def build_groups(
    items: Sequence[Mapping[str, Any]],
    phash_max: int = 8,
    cos_min: float = 0.95,
    *,
    use_group_key: bool = False,
) -> dict[str, str]:
    """photo_id → dup_group_id を返す。

    ``items`` の各要素は ``photo_id``、``path``（sha256 / pHash を計算する。``sha256`` / ``phash``
    が与えられていれば再計算しない）、任意の ``emb``。``phash`` は符号なし・符号付き（int64 の
    2 の補数）どちらでもよい。結合条件は sha256 完全一致 / pHash Hamming ≤ phash_max /
    余弦 ≥ cos_min の和。``use_group_key=True`` は ``group_key``（business_id 等）の共有も結合するが、
    それは dup 群ではなく分割キーの合成であり、通常は :func:`split_by_components` に任せる。
    """
    uf = UnionFind()
    ids: list[str] = []
    for it in items:
        pid = str(it["photo_id"])
        if pid in uf._parent:
            raise ValueError(f"photo_id が重複している: {pid}")
        uf.add(pid)
        ids.append(pid)

    # sha256 完全一致
    by_sha: dict[str, str] = {}
    for it in items:
        pid = str(it["photo_id"])
        sha = it.get("sha256")
        if sha is None and it.get("path") is not None:
            sha = sha256_file(it["path"])
        if sha is None:
            continue
        if sha in by_sha:
            uf.union(by_sha[sha], pid)
        else:
            by_sha[sha] = pid

    # pHash
    ph_ids: list[str] = []
    ph_vals: list[int] = []
    for it in items:
        ph = it.get("phash")
        if ph is None and it.get("path") is not None:
            ph = phash64(it["path"])
        if ph is None:
            continue
        ph_ids.append(str(it["photo_id"]))
        ph_vals.append(from_int64(ph))
    for a, b in _phash_pairs(ph_ids, ph_vals, phash_max):
        uf.union(a, b)

    # 埋め込み余弦
    em_ids = [str(it["photo_id"]) for it in items if it.get("emb") is not None]
    em_vals = [it["emb"] for it in items if it.get("emb") is not None]
    for a, b in _cosine_pairs(em_ids, em_vals, cos_min):
        uf.union(a, b)

    # group_key 共有（任意）
    if use_group_key:
        first_of_key: dict[str, str] = {}
        for it in items:
            gk = it.get("group_key")
            if gk is None or gk == "":
                continue
            gk = str(gk)
            pid = str(it["photo_id"])
            if gk in first_of_key:
                uf.union(first_of_key[gk], pid)
            else:
                first_of_key[gk] = pid

    out: dict[str, str] = {}
    for members in uf.components().values():
        gid = _group_id(members)
        for pid in members:
            out[pid] = gid
    return {pid: out[pid] for pid in ids}


# ---------------------------------------------------------------------------
# 役割割当
# ---------------------------------------------------------------------------


def _uniform_from_key(seed: str, key: str) -> float:
    """sha1(seed ␀ key) の先頭 8 バイトを [0, 1) に写す（区切りを入れ (seed, key) の衝突を防ぐ）。"""
    digest = hashlib.sha1(f"{seed}\x00{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _cumulative(ratios: Mapping[str, float]) -> list[tuple[str, float]]:
    """役割順を **名前のソート順** に固定した累積比率（dict の挿入順に依存しない）。"""
    if not ratios:
        raise ValueError("ratios が空")
    total = float(sum(ratios.values()))
    if total <= 0 or any(v < 0 for v in ratios.values()):
        raise ValueError(f"ratios が不正: {dict(ratios)}")
    acc = 0.0
    cum: list[tuple[str, float]] = []
    for role in sorted(ratios):
        acc += ratios[role] / total
        cum.append((role, acc))
    cum[-1] = (cum[-1][0], 1.0)  # 丸め誤差で取りこぼさない
    return cum


def assign_roles(keys: Sequence[str], ratios: Mapping[str, float], seed: str) -> dict[str, str]:
    """各 key に決定的に役割を割り当てる。

    u = sha1(seed ␀ key) → [0, 1)。役割を名前順に並べた累積比率 [0, r1), [r1, r1+r2), ... の区間で
    決める。ratios の合計が 1 でない場合は正規化する。
    """
    cum = _cumulative(ratios)
    out: dict[str, str] = {}
    for key in keys:
        u = _uniform_from_key(seed, str(key))
        for role, upper in cum:
            if u < upper:
                out[str(key)] = role
                break
    return out


def report_role_shares(roles: Mapping[str, str], strata: Mapping[str, str]) -> pl.DataFrame:
    """層（strata[key]）× 役割の実現比率。列: stratum, role, n, share。"""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    stratum_total: dict[str, int] = defaultdict(int)
    for key, role in roles.items():
        s = str(strata.get(str(key), "_unknown"))
        counts[(s, role)] += 1
        stratum_total[s] += 1
    rows = [
        {"stratum": s, "role": r, "n": n, "share": n / stratum_total[s]}
        for (s, r), n in sorted(counts.items())
    ]
    schema = {"stratum": pl.Utf8, "role": pl.Utf8, "n": pl.Int64, "share": pl.Float64}
    return pl.DataFrame(rows, schema=schema)


def assign_roles_stratified(
    keys: Sequence[str],
    strata: Mapping[str, str],
    ratios: Mapping[str, float],
    seed: str,
) -> tuple[dict[str, str], pl.DataFrame]:
    """層化割当（決定的・丸め以内で正確）。

    層ごとに key を u = sha1(seed ␀ key) で並べ替え、累積比率の分位（round(m · cum)）で切る。
    各層で役割比率は ±1 件の丸め以内で ratios に一致する。戻り値は
    (key → role, :func:`report_role_shares` の DataFrame)。
    """
    cum = _cumulative(ratios)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_stratum[str(strata.get(str(key), "_unknown"))].append(str(key))
    roles: dict[str, str] = {}
    for _, members in sorted(by_stratum.items()):
        ordered = sorted(members, key=lambda k: (_uniform_from_key(seed, k), k))
        m = len(ordered)
        bounds = [round(m * upper) for _, upper in cum]
        bounds[-1] = m
        lo = 0
        for (role, _), hi in zip(cum, bounds, strict=True):
            for k in ordered[lo:hi]:
                roles[k] = role
            lo = max(lo, hi)
    roles = {str(k): roles[str(k)] for k in keys}
    return roles, report_role_shares(roles, strata)


def check_no_crossing(
    role_of_item: Mapping[str, str], group_of_item: Mapping[str, str]
) -> list[str]:
    """役割を跨いでいる群 ID をソートして返す（空なら OK）。

    group_of_item に無い item は単独群として扱う（跨ぎようがない）。
    """
    roles_by_group: dict[str, set[str]] = defaultdict(set)
    for item, role in role_of_item.items():
        gid = group_of_item.get(item)
        if gid is None:
            continue
        roles_by_group[gid].add(role)
    return sorted(g for g, rs in roles_by_group.items() if len(rs) > 1)


def _compose_components(
    items: Sequence[Mapping[str, Any]], groups: Mapping[str, str]
) -> dict[str, str]:
    """dup 群（photo_id → dup_group_id）と group_key（business_id）の連結成分 ID を返す。"""
    uf = UnionFind()
    first_of_key: dict[str, str] = {}
    first_of_grp: dict[str, str] = {}
    for it in items:
        pid = str(it["photo_id"])
        uf.add(pid)
        gk = it.get("group_key")
        if gk not in (None, ""):
            gk = str(gk)
            if gk in first_of_key:
                uf.union(first_of_key[gk], pid)
            else:
                first_of_key[gk] = pid
        g = groups.get(pid)
        if g is not None:
            if g in first_of_grp:
                uf.union(first_of_grp[g], pid)
            else:
                first_of_grp[g] = pid
    merged: dict[str, str] = {}
    for members in uf.components().values():
        cid = _group_id(members)
        for pid in members:
            merged[pid] = cid
    return merged


def split_by_components(
    items: Sequence[Mapping[str, Any]],
    ratios: Mapping[str, float] = ROLES_DEFAULT,
    seed: str = "g0",
    *,
    phash_max: int = 8,
    cos_min: float = 0.95,
    groups: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], pl.DataFrame]:
    """business_id ∪ dup 群の連結成分単位で役割を割り当て、(photo_id → role, summary) を返す。

    ``items`` は build_groups と同じ形（``group_key`` が business_id）。``groups`` を渡すと
    dup 群の再計算を省く（photo_id → dup_group_id。``build_groups(use_group_key=False)`` の結果）。
    dup 群と business_id はここで合成する（dup 群自体には business を混ぜない）。
    成分 ID はメンバー photo_id のハッシュで、役割は成分 ID から決定的に決まる。summary は
    role 別の成分数・画像数、``label`` キーがあればクラス別の枚数も含む
    （列: role, n_components, n_items, label, n。label 欠損は null）。
    """
    if groups is None:
        groups = build_groups(items, phash_max=phash_max, cos_min=cos_min, use_group_key=False)
    components = _compose_components(items, groups)

    component_ids = sorted(set(components.values()))
    role_of_component = assign_roles(component_ids, ratios, seed)
    role_of_item = {pid: role_of_component[components[pid]] for pid in components}

    crossing = check_no_crossing(role_of_item, components)
    if crossing:  # 構造上起き得ないが、契約として明示する
        raise RuntimeError(f"成分が役割を跨いだ: {crossing[:5]}")

    # summary
    roles_order = [r for r, _ in _cumulative(ratios)]
    comp_per_role: dict[str, int] = defaultdict(int)
    for r in role_of_component.values():
        comp_per_role[r] += 1
    items_per_role: dict[str, int] = defaultdict(int)
    label_counts: dict[tuple[str, str | None], int] = defaultdict(int)
    has_label = any("label" in it for it in items)
    for it in items:
        pid = str(it["photo_id"])
        r = role_of_item[pid]
        items_per_role[r] += 1
        if has_label:
            lb = it.get("label")
            label_counts[(r, None if lb is None else str(lb))] += 1
    rows: list[dict[str, Any]] = []
    for r in roles_order:
        base = {
            "role": r,
            "n_components": comp_per_role.get(r, 0),
            "n_items": items_per_role.get(r, 0),
        }
        if has_label:
            labels = sorted(
                {lb for (rr, lb) in label_counts if rr == r}, key=lambda x: (x is None, x or "")
            )
            if not labels:
                rows.append({**base, "label": None, "n": 0})
            for lb in labels:
                rows.append({**base, "label": lb, "n": label_counts[(r, lb)]})
        else:
            rows.append({**base, "label": None, "n": items_per_role.get(r, 0)})
    schema = {
        "role": pl.Utf8,
        "n_components": pl.Int64,
        "n_items": pl.Int64,
        "label": pl.Utf8,
        "n": pl.Int64,
    }
    return role_of_item, pl.DataFrame(rows, schema=schema)
