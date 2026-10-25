"""G0 の入力データ束（g0_nonfood_plan.md §2.3・§3、design.md §2・§5）。

役割ごとに「埋め込み（numpy）＋メタ（polars）」を束ねた :class:`G0Data` を、
埋め込みキャッシュと core テーブルから組み立てる（:func:`load_g0_data`）。
テスト・スモーク用に、同じ形の合成データを決定的に生成する
（:func:`make_synthetic_g0_data`。ネットワーク・モデル不要）。

役割（§3.1 Yelp / §3.8 外部）:

- Yelp: ``tune_fit``（バンク・共分散・ヘッド学習）／``tune_cal``（動作点）／``work``／``eval``
- 外部（層 A・near-food・補助 far・ソース対照）: ``tune`` 30% / ``work`` 10% / ``eval`` 60%
- 層 B は Yelp 由来なので Yelp の分割に従う（``role`` 列は Yelp の役割名）

メタ列は全ブロック共通: ``photo_id, source, label, subject_type, role, content, business_id``。
``label`` は Yelp ラベル（Yelp）または ``operational_truth``（外部・層 B）。第 1 階層は持たない。
P0（Stage1 argmax、τ = 1）の確率 ``probs`` は :func:`attach_p0` が全ブロックに付ける。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from cascade import paths
from cascade.embed.cache import EmbeddingCache
from cascade.embed.encoder import Encoder, find_spec, load_encoder, model_key_for
from cascade.embed.text import embed_prompts
from cascade.io import tables
from cascade.stage1.zeroshot import ZeroShot

__all__ = [
    "BLOCK_NAMES",
    "META_COLUMNS",
    "Block",
    "G0Data",
    "attach_p0",
    "load_g0_data",
    "make_synthetic_g0_data",
]

META_COLUMNS: tuple[str, ...] = (
    "photo_id",
    "source",
    "label",
    "subject_type",
    "role",
    "content",
    "business_id",
)
META_SCHEMA: dict[str, type[pl.DataType]] = dict.fromkeys(META_COLUMNS, pl.Utf8)

BLOCK_NAMES: tuple[str, ...] = (
    "yelp_tune_fit",
    "yelp_tune_cal",
    "yelp_work",
    "yelp_eval",
    "ext_tune",
    "ext_work",
    "ext_eval",
    "layer_b",
    "food_contrast",
    "aux_far",
)

LAYER_A_TYPES: tuple[str, ...] = (
    "doll_character_statue",
    "amusement_tourist",
    "signage_decor_goods",
    "person_animal",
    "vehicle_street",
)
NEAR_FOOD_TYPE = "near_food_nonfood"
AUX_FAR_TYPE = "texture_document_screen"
CONTRAST_TYPE = "food_contrast"
OUT_OF_SCOPE = "out_of_scope"
YELP_ROLES: tuple[str, ...] = ("tune_fit", "tune_cal", "work", "eval")
EXT_ROLES: tuple[str, ...] = ("tune", "work", "eval")


# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------


@dataclass
class Block:
    """1 役割分の埋め込みとメタ。``X[i]`` は ``meta[i]`` に対応する。"""

    X: np.ndarray
    meta: pl.DataFrame
    probs: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float32)
        if self.X.ndim != 2:
            raise ValueError(f"X は [n, dim] が必要: {self.X.shape}")
        missing = [c for c in META_COLUMNS if c not in self.meta.columns]
        if missing:
            raise ValueError(f"meta に列が無い: {missing}")
        if self.meta.height != self.X.shape[0]:
            raise ValueError(
                f"X の行数 {self.X.shape[0]} と meta の行数 {self.meta.height} が不一致"
            )
        self.meta = self.meta.select(list(META_COLUMNS)).cast(META_SCHEMA)  # type: ignore[arg-type]

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @classmethod
    def empty(cls, dim: int) -> Block:
        return cls(np.zeros((0, dim), dtype=np.float32), pl.DataFrame(schema=META_SCHEMA))

    def subset(self, mask: np.ndarray) -> Block:
        m = np.asarray(mask, dtype=bool)
        return Block(
            self.X[m], self.meta.filter(pl.Series(m)), None if self.probs is None else self.probs[m]
        )

    def col(self, name: str) -> np.ndarray:
        return self.meta[name].to_numpy()


def concat_blocks(blocks: list[Block], dim: int) -> Block:
    """複数ブロックを縦に連結する（P0 確率も揃っていれば連結）。"""
    blocks = [b for b in blocks if b.n > 0]
    if not blocks:
        return Block.empty(dim)
    X = np.concatenate([b.X for b in blocks], axis=0)
    meta = pl.concat([b.meta for b in blocks], how="vertical")
    probs = None
    if all(b.probs is not None for b in blocks):
        probs = np.concatenate([b.probs for b in blocks], axis=0)  # type: ignore[misc]
    return Block(X, meta, probs)


@dataclass
class G0Data:
    """G0 の入力束。``text_emb`` は class_order 順のテキスト埋め込み [C, dim]（P0 と MSP が使う）。"""

    encoder_name: str
    class_order: list[str]
    text_emb: np.ndarray
    yelp_tune_fit: Block
    yelp_tune_cal: Block
    yelp_work: Block
    yelp_eval: Block
    ext_tune: Block
    ext_work: Block
    ext_eval: Block
    layer_b: Block
    food_contrast: Block
    aux_far: Block
    review: pl.DataFrame | None = None
    """目視結果（photo_id, content）。棄却された名目 food 全数＋通過側無作為（§3.1）。無ければ None。"""
    eval_set_id: str | None = None
    prompts: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.text_emb.shape[1])

    def block(self, name: str) -> Block:
        if name not in BLOCK_NAMES:
            raise KeyError(name)
        return getattr(self, name)

    def blocks(self) -> dict[str, Block]:
        return {n: getattr(self, n) for n in BLOCK_NAMES}

    def summary(self) -> pl.DataFrame:
        rows = [{"block": n, "n": b.n} for n, b in self.blocks().items()]
        return pl.DataFrame(rows, schema={"block": pl.Utf8, "n": pl.Int64})

    def ext_all(self, roles: tuple[str, ...] = EXT_ROLES) -> Block:
        """外部集合を役割横断で連結する（LOTO / LOSO の評価に使う）。"""
        names = {"tune": self.ext_tune, "work": self.ext_work, "eval": self.ext_eval}
        return concat_blocks([names[r] for r in roles], self.dim)


def attach_p0(data: G0Data, tau: float = 1.0, force: bool = False) -> G0Data:
    """全ブロックに P0（Stage1 zero-shot、τ = 1）の確率を付ける。"""
    zs = ZeroShot(
        data.prompts or {"class_order": data.class_order}, data.text_emb, data.class_order
    )
    for b in data.blocks().values():
        if b.probs is None or force:
            b.probs = (
                zs.probs(b.X, tau=tau) if b.n else np.zeros((0, len(data.class_order)), np.float32)
            )
    return data


# ---------------------------------------------------------------------------
# 合成データ（テスト・スモーク）
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _meta(
    n: int,
    prefix: str,
    source: str,
    label: list[str] | str,
    subject_type: list[str] | str,
    role: str,
    content: list[str] | str,
    business_id: list[str] | None = None,
) -> pl.DataFrame:
    def _rep(v: list[str] | str) -> list[str]:
        return list(v) if isinstance(v, list) else [v] * n

    return pl.DataFrame(
        {
            "photo_id": [f"{prefix}{i:06d}" for i in range(n)],
            "source": [source] * n,
            "label": _rep(label),
            "subject_type": _rep(subject_type),
            "role": [role] * n,
            "content": _rep(content),
            "business_id": business_id if business_id is not None else [None] * n,
        },
        schema=META_SCHEMA,
    )


def make_synthetic_g0_data(
    seed: int = 0,
    n_per_role: int | dict[str, int] | None = None,
    dim: int = 32,
    class_order: tuple[str, ...] = ("food", "drink", "menu", "inside", "outside"),
    label_noise: float = 0.03,
) -> G0Data:
    """事前登録の集合構造を持つ合成データ（決定的）。

    - Yelp: クラス平均＋等方雑音。ラベルは ``label_noise`` の割合で誤り（q̂ < 1 を再現）。
      ``review`` に全 Yelp 画像の真の content を入れる（目視の代わり）
    - 層 A: 5 型。型ごとに food 方向への引き込み強さを変え、P0 が food と誤る画像が一定数出る
    - near-food / 補助 far / ソース対照（food 平均＋源シフト）/ 層 B（inside・outside 寄り）
    - 外部は subject_type × source で tune 30 / work 10 / eval 60 に配分

    ``n_per_role`` は Yelp の役割別 food 枚数（int なら全役割同じ）。既定 {tune_fit: 600, tune_cal: 400, work: 300, eval: 400}。
    """
    rng = np.random.default_rng(seed)
    classes = list(class_order)
    C = len(classes)
    if n_per_role is None:
        n_food = {"tune_fit": 600, "tune_cal": 400, "work": 300, "eval": 400}
    elif isinstance(n_per_role, int):
        n_food = dict.fromkeys(YELP_ROLES, int(n_per_role))
    else:
        n_food = {r: int(n_per_role.get(r, 300)) for r in YELP_ROLES}

    text = _unit(rng.standard_normal((C, dim)))
    # クラス平均はテキスト方向を主成分にした少しずれた方向
    class_mean = _unit(0.85 * text + 0.15 * rng.standard_normal((C, dim)))
    sigma = 0.45
    fi = classes.index("food")

    def sample(mean: np.ndarray, n: int, s: float = sigma) -> np.ndarray:
        return (mean[None, :] + s * rng.standard_normal((n, dim)) / np.sqrt(dim)).astype(np.float32)

    # --- Yelp ---------------------------------------------------------------
    yelp_blocks: dict[str, Block] = {}
    review_rows: list[dict[str, str]] = []
    for role in YELP_ROLES:
        nf = n_food[role]
        counts = {c: (nf if c == "food" else max(nf // 3, 30)) for c in classes}
        Xs, labels, contents, biz = [], [], [], []
        for c in classes:
            n = counts[c]
            Xs.append(sample(class_mean[classes.index(c)], n))
            true = [c] * n
            noisy = list(true)
            flip = rng.random(n) < label_noise
            for i in np.flatnonzero(flip):
                noisy[i] = classes[(classes.index(c) + int(rng.integers(1, C))) % C]
            labels.extend(noisy)
            contents.extend(
                "food" if t == "food" else ("drink" if t == "drink" else "non_food") for t in true
            )
            biz.extend(f"biz{role}{int(rng.integers(0, max(n // 5, 1))):04d}" for _ in range(n))
        X = np.concatenate(Xs)
        meta = _meta(len(X), f"y{role}", "yelp", labels, "yelp", role, contents, biz)
        yelp_blocks[role] = Block(X, meta)
        review_rows.extend(
            {"photo_id": p, "content": ct} for p, ct in zip(meta["photo_id"], contents, strict=True)
        )

    # --- 外部 -------------------------------------------------------------------
    pull = {  # food 方向への引き込み（大きいほど P0 が food に誤りやすい）
        "doll_character_statue": 0.55,
        "amusement_tourist": 0.25,
        "signage_decor_goods": 0.35,
        "person_animal": 0.45,
        "vehicle_street": 0.15,
        NEAR_FOOD_TYPE: 0.75,
    }
    type_dir = {t: _unit(rng.standard_normal(dim)) for t in pull}
    sources = ("open_images", "commons", "coco")
    ext_parts: dict[str, list[Block]] = {r: [] for r in EXT_ROLES}
    n_type = 150
    for t, a in pull.items():
        for s_i, src in enumerate(sources):
            n = n_type // len(sources) + (1 if s_i < n_type % len(sources) else 0)
            mean = _unit(
                a * class_mean[fi] + (1 - a) * type_dir[t] + 0.1 * rng.standard_normal(dim)
            )
            X = sample(mean, n, 0.6)
            meta = _meta(n, f"x{t[:4]}{src[:3]}", src, OUT_OF_SCOPE, t, "", "non_food")
            roles = _split_roles(n, rng)
            for r in EXT_ROLES:
                m = roles == r
                if m.any():
                    ext_parts[r].append(
                        Block(X[m], meta.filter(pl.Series(m)).with_columns(pl.lit(r).alias("role")))
                    )
    ext_blocks = {r: concat_blocks(bs, dim) for r, bs in ext_parts.items()}

    # ソース対照正例: food 平均 ＋ 源シフト
    n_c = 120
    shift = 0.2 * _unit(rng.standard_normal(dim))
    Xc = sample(_unit(class_mean[fi] + shift), n_c, sigma)
    roles_c = _split_roles(n_c, rng)
    contrast = Block(
        Xc,
        _meta(n_c, "xcontrast", "open_images", "food", CONTRAST_TYPE, "", "food").with_columns(
            pl.Series("role", roles_c.tolist())
        ),
    )
    # 補助 far: 無相関
    n_far = 200
    Xf = _unit(rng.standard_normal((n_far, dim))).astype(np.float32)
    roles_f = _split_roles(n_far, rng)
    aux_far = Block(
        Xf,
        _meta(n_far, "xcord", "cord", OUT_OF_SCOPE, AUX_FAR_TYPE, "", "non_food").with_columns(
            pl.Series("role", roles_f.tolist())
        ),
    )
    # 層 B: inside / outside / menu 寄り、少し food 方向。Yelp の役割名
    n_b = 240
    truths = [
        ("inside", "doll_character_statue"),
        ("outside", "signage_decor_goods"),
        ("menu", "person_animal"),
    ]
    Xb, lab_b, st_b, role_b = [], [], [], []
    for truth, st in truths:
        n = n_b // len(truths)
        mean = _unit(0.6 * class_mean[classes.index(truth)] + 0.4 * class_mean[fi])
        Xb.append(sample(mean, n, 0.5))
        lab_b += [truth] * n
        st_b += [st] * n
        role_b += rng.choice(list(YELP_ROLES), size=n, p=[0.25, 0.25, 0.2, 0.3]).tolist()
    layer_b = Block(
        np.concatenate(Xb),
        _meta(len(lab_b), "ylayerb", "yelp", lab_b, st_b, "eval", "non_food").with_columns(
            pl.Series("role", role_b)
        ),
    )

    data = G0Data(
        encoder_name="synthetic",
        class_order=classes,
        text_emb=text.astype(np.float32),
        yelp_tune_fit=yelp_blocks["tune_fit"],
        yelp_tune_cal=yelp_blocks["tune_cal"],
        yelp_work=yelp_blocks["work"],
        yelp_eval=yelp_blocks["eval"],
        ext_tune=ext_blocks["tune"],
        ext_work=ext_blocks["work"],
        ext_eval=ext_blocks["eval"],
        layer_b=layer_b,
        food_contrast=contrast,
        aux_far=aux_far,
        review=pl.DataFrame(review_rows, schema={"photo_id": pl.Utf8, "content": pl.Utf8}),
        eval_set_id="synthetic-eval",
        prompts={"class_order": classes},
        extra={"seed": seed, "synthetic": True},
    )
    return attach_p0(data)


def _split_roles(n: int, rng: np.random.Generator) -> np.ndarray:
    """外部集合の役割配分 tune 30 / work 10 / eval 60（§3.8）を決定的に割る。"""
    roles = np.array(["tune"] * n, dtype=object)
    perm = rng.permutation(n)
    n_tune = int(round(0.3 * n))
    n_work = int(round(0.1 * n))
    roles[perm[n_tune : n_tune + n_work]] = "work"
    roles[perm[n_tune + n_work :]] = "eval"
    return roles


# ---------------------------------------------------------------------------
# 実データの読み込み
# ---------------------------------------------------------------------------


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else paths.DATA / path


def _read_review(path: Path | None) -> pl.DataFrame | None:
    if path is None or not path.exists():
        return None
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        return None
    df = pl.DataFrame(rows)
    if "content" not in df.columns or "photo_id" not in df.columns:
        raise ValueError(f"review には photo_id と content が必要: {path}")
    # 同一 photo_id が複数回ある場合は最後の判定を採る（中断再開・再判定）
    return (
        df.select(pl.col("photo_id").cast(pl.Utf8), pl.col("content").cast(pl.Utf8))
        .group_by("photo_id", maintain_order=True)
        .agg(pl.col("content").last())
    )


def _cache_block(cache: EmbeddingCache, dataset_id: str) -> tuple[np.ndarray, pl.DataFrame]:
    emb, idx = cache.get(dataset_id)
    return emb, idx.select("photo_id")


def load_g0_data(
    config: dict[str, Any],
    encoder_name: str,
    *,
    encoder: Encoder | None = None,
    cache_root: Path | None = None,
) -> G0Data:
    """埋め込みキャッシュ＋core テーブルから :class:`G0Data` を組み立てる。

    入力（config["data"]）:

    - ``yelp_dataset_id``: Yelp の埋め込み dataset_id。ラベルは core/labels（``labels_source``）、
      役割は ``yelp_splits_path``（marts/splits.parquet: photo_id, role, component_id）
    - ``ext_dataset_ids``: 外部源の dataset_id。メタは core/scope_annotations（目視確定分のみ）
    - 層 B は scope_annotations の ``source == yelp`` 行（operational_truth ∈ inside/outside/menu）
    - ``review_path``: 目視（jsonl: photo_id, content）

    ``encoder`` を渡すとテキスト埋め込みにそれを使う（テストは FakeEncoder）。渡さなければ
    ``encoders_config`` の spec から遅延ロードする（テキスト埋め込みがキャッシュ済みならモデルは読まない）。
    """
    dcfg = dict(config.get("data", {}))
    class_order = list(config.get("class_order", ("food", "drink", "menu", "inside", "outside")))
    root = Path(cache_root) if cache_root is not None else paths.EMB

    if encoder is None:
        from cascade.config import load_config

        enc_cfg = load_config(
            paths.REPO_ROOT / str(config.get("encoders_config", "configs/encoders.yaml"))
        )
        spec = find_spec(enc_cfg, encoder_name)
        encoder = load_encoder(spec)
        model_key = model_key_for(spec)
    else:
        model_key = encoder.model_key
    cache = EmbeddingCache(model_key, root=root)

    prompts_path = paths.REPO_ROOT / str(config.get("prompts_json", "taxonomy/prompts.json"))
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    text_emb, order = embed_prompts(encoder, prompts, root=root)
    if order != class_order:
        raise ValueError(f"prompts.json の class_order {order} と config の {class_order} が不一致")
    dim = int(text_emb.shape[1])

    # --- Yelp -------------------------------------------------------------------
    emb_y, idx_y = _cache_block(cache, str(dcfg.get("yelp_dataset_id", "yelp")))
    labels = tables.read_table("labels", "core")
    src = str(dcfg.get("labels_source", "yelp"))
    labels = (
        labels.filter(pl.col("label_source") == src)
        .select("photo_id", "label")
        .unique("photo_id", keep="last")
    )
    splits_path = _resolve(str(dcfg.get("yelp_splits_path", "core/yelp_splits.parquet")))
    if not splits_path.exists():
        raise FileNotFoundError(f"Yelp の分割が無い（B-b で生成）: {splits_path}")
    splits = pl.read_parquet(splits_path)
    if "business_id" not in splits.columns:
        # marts/splits.parquet は component_id（business_id ∪ dup 群の連結成分）を持つ。店舗単位
        # ブートストラップのクラスタにはこれを使う（店舗を跨がないのでクラスタとして妥当）
        cluster = (
            pl.col("component_id")
            if "component_id" in splits.columns
            else pl.lit(None, dtype=pl.Utf8)
        )
        splits = splits.with_columns(cluster.cast(pl.Utf8).alias("business_id"))
    ymeta = (
        idx_y.with_row_index("_i")
        .join(labels, on="photo_id", how="left")
        .join(splits.select("photo_id", "role", "business_id"), on="photo_id", how="left")
        .with_columns(
            pl.lit("yelp").alias("source"),
            pl.lit("yelp").alias("subject_type"),
            pl.lit(None, dtype=pl.Utf8).alias("content"),
        )
    )
    yelp_blocks: dict[str, Block] = {}
    for role in YELP_ROLES:
        sel = ymeta.filter(pl.col("role") == role)
        ii = sel["_i"].to_numpy()
        yelp_blocks[role] = Block(emb_y[ii], sel.drop("_i")) if len(ii) else Block.empty(dim)

    # --- 外部 + 層 B（scope_annotations） -------------------------------------------
    scope = tables.read_table("scope_annotations", "core")
    scope = scope.filter(pl.col("quality") != "unjudgeable")
    ext_ids = [str(x) for x in dcfg.get("ext_dataset_ids", [])]
    parts: list[tuple[np.ndarray, pl.DataFrame]] = []
    for ds in ext_ids:
        if not cache.exists(ds):
            continue
        emb, idx = _cache_block(cache, ds)
        parts.append((emb, idx.with_columns(pl.lit(ds).alias("_dataset"))))
    if parts:
        emb_x = np.concatenate([p[0] for p in parts])
        idx_x = pl.concat([p[1] for p in parts], how="vertical")
    else:
        emb_x = np.zeros((0, dim), np.float32)
        idx_x = pl.DataFrame(schema={"photo_id": pl.Utf8, "_dataset": pl.Utf8})
    xmeta = (
        idx_x.with_row_index("_i")
        .join(scope.filter(pl.col("source") != "yelp"), on="photo_id", how="inner")
        .with_columns(
            pl.col("operational_truth").alias("label"),
            pl.lit(None, dtype=pl.Utf8).alias("business_id"),
        )
    )
    sets = dict(config.get("sets", {}))
    contrast_t = str(sets.get("contrast_type", CONTRAST_TYPE))
    aux_t = str(sets.get("aux_far_type", AUX_FAR_TYPE))

    def _blk(df: pl.DataFrame) -> Block:
        ii = df["_i"].to_numpy()
        return Block(emb_x[ii], df.drop("_i")) if len(ii) else Block.empty(dim)

    is_contrast = pl.col("subject_type") == contrast_t
    is_aux = pl.col("subject_type") == aux_t
    layer_a = xmeta.filter(~is_contrast & ~is_aux & (pl.col("operational_truth") == OUT_OF_SCOPE))
    ext_blocks = {r: _blk(layer_a.filter(pl.col("role") == r)) for r in EXT_ROLES}
    contrast = _blk(xmeta.filter(is_contrast & (pl.col("content") == "food")))
    aux_far = _blk(xmeta.filter(is_aux))

    # 層 B: Yelp 由来（scope_annotations.source == yelp）＋ 外部候補で inside/outside/menu と判定されたもの
    b_truths = [str(x) for x in sets.get("layer_b_truths", ("inside", "outside", "menu"))]
    scope_b = scope.filter(
        pl.col("operational_truth").is_in(b_truths) & pl.col("content").is_in(["non_food", "mixed"])
    )
    b_yelp = (
        idx_y.with_row_index("_i")
        .join(scope_b.filter(pl.col("source") == "yelp").drop("role"), on="photo_id", how="inner")
        .join(splits.select("photo_id", "role", "business_id"), on="photo_id", how="left")
        .with_columns(pl.col("operational_truth").alias("label"))
    )
    b_ext = xmeta.join(scope_b.select("photo_id"), on="photo_id", how="inner")
    lb_parts = []
    if b_yelp.height:
        ii = b_yelp["_i"].to_numpy()
        lb_parts.append(Block(emb_y[ii], b_yelp.drop("_i")))
    if b_ext.height:
        lb_parts.append(_blk(b_ext))
    layer_b = concat_blocks(lb_parts, dim)

    review = _read_review(_resolve(str(dcfg["review_path"])) if dcfg.get("review_path") else None)
    data = G0Data(
        encoder_name=encoder_name,
        class_order=class_order,
        text_emb=np.asarray(text_emb, np.float32),
        yelp_tune_fit=yelp_blocks["tune_fit"],
        yelp_tune_cal=yelp_blocks["tune_cal"],
        yelp_work=yelp_blocks["work"],
        yelp_eval=yelp_blocks["eval"],
        ext_tune=ext_blocks["tune"],
        ext_work=ext_blocks["work"],
        ext_eval=ext_blocks["eval"],
        layer_b=layer_b,
        food_contrast=contrast,
        aux_far=aux_far,
        review=review,
        eval_set_id=dcfg.get("eval_set_id"),
        prompts=prompts,
        extra={"model_key": model_key, "ext_dataset_ids": ext_ids},
    )
    return attach_p0(data)


def block_fields() -> list[str]:
    return [f.name for f in fields(G0Data) if f.name in BLOCK_NAMES]
