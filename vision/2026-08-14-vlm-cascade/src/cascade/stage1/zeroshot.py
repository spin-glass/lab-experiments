"""Stage1 zero-shot 分類（design.md §1 Stage1、§6 G0 の P0 / MSP、taxonomy.md §5）。

- logits = cos(img_norm, text_norm) / tau、probs = softmax(logits)
- ``tau=1.0``: MCM（Ming et al. 2022）の既定。余弦をそのまま softmax に入れるため確率が平坦で、
  MSP（max softmax）を OOD スコアとして使うときの主設定（計画「MSP（MCM）: max softmax、τ=1」）
- ``tau=0.01``: CLIP の学習時 logit_scale（≈100）相当。確率が尖り、分類の argmax は tau に依らないが
  MSP の分布は変わる（計画: τ=0.01 はアブレーション）
- 葉スコア→ブランチ内 max の階層方式（D11）は M1 で追加する。ここは第2階層フラット
- 画像埋め込みは正規化前のものを渡してよい（内部で L2 正規化する）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cascade.embed.encoder import Encoder
from cascade.embed.text import embed_prompts, parse_prompts


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class ZeroShot:
    """クラス別テキスト埋め込みによる zero-shot 分類器。"""

    def __init__(
        self, prompts: dict[str, Any], text_emb: np.ndarray, class_order: list[str]
    ) -> None:
        self.prompts = prompts
        self.class_order = list(class_order)
        t = np.asarray(text_emb, dtype=np.float32)
        if t.ndim != 2 or t.shape[0] != len(self.class_order):
            raise ValueError(
                f"text_emb は [n_classes={len(self.class_order)}, dim] が必要: {t.shape}"
            )
        self.text_emb = _l2(t)

    @property
    def n_classes(self) -> int:
        return len(self.class_order)

    def scores(self, img_emb: np.ndarray) -> np.ndarray:
        """余弦類似度 [n, C]（tau を掛ける前）。"""
        x = np.asarray(img_emb, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        return _l2(x) @ self.text_emb.T

    def probs(self, img_emb: np.ndarray, tau: float = 1.0) -> np.ndarray:
        if tau <= 0:
            raise ValueError("tau は正")
        return _softmax(self.scores(img_emb) / tau).astype(np.float32)

    def predict(self, img_emb: np.ndarray, tau: float = 1.0) -> list[str]:
        idx = self.probs(img_emb, tau).argmax(axis=1)
        return [self.class_order[i] for i in idx]

    def margin(self, img_emb: np.ndarray, tau: float = 1.0) -> np.ndarray:
        """top1 − top2 の確率差（design.md §5 predictions.margin）。"""
        p = np.sort(self.probs(img_emb, tau), axis=1)
        return (p[:, -1] - p[:, -2]).astype(np.float32) if p.shape[1] >= 2 else p[:, -1]

    def prob_of(self, img_emb: np.ndarray, class_id: str, tau: float = 1.0) -> np.ndarray:
        """特定クラスの確率（P1 用 p(food) など）。"""
        return self.probs(img_emb, tau)[:, self.class_order.index(class_id)]


def build_zeroshot(
    encoder: Encoder, prompts_json_path: Path | dict[str, Any], **embed_kwargs: Any
) -> ZeroShot:
    """prompts.json を埋め込んで ZeroShot を作る（テキスト埋め込みはキャッシュされる）。"""
    import json

    data = (
        prompts_json_path
        if isinstance(prompts_json_path, dict)
        else json.loads(Path(prompts_json_path).read_text(encoding="utf-8"))
    )
    parse_prompts(data)  # 早期に形式検証
    text_emb, order = embed_prompts(encoder, prompts_json_path, **embed_kwargs)
    return ZeroShot(data, text_emb, order)
