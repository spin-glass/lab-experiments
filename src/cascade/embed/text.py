"""prompts.json → クラス別テキスト埋め込み（design.md §1 Stage1、taxonomy.md §5「プロンプト埋め込み」）。

- クラス埋め込み = そのクラスのプロンプト埋め込みを各々 L2 正規化 → 平均 → 再正規化
- キャッシュ: ``<root>/<model_key>/text/<sha1(prompts.json 内容)>.npy``（＋ ``.json`` に class_order とプロンプト数）
- prompts.json（taxonomy_build.py の生成物）の形は L1 レーンが確定する。ここでは以下を寛容に受ける::

    {"classes": {"food": {"prompts": [{"text": "...", "source": "..."}, ...]}, ...}}
    {"classes": [{"id": "food", "prompts": [...]}, ...]}
    {"food": ["a photo of food", ...], ...}

  ``class_order`` キーがあればそれを採用し、無ければ既定順（food, drink, menu, inside, outside）→ 残りは出現順。
  ``class_order`` と prompts のクラス集合は一致しなければならない（欠落・余剰とも ValueError）。
  フラット形式では値が list / dict でないトップキー（``scheme_version`` などのメタ）はクラスとして扱わない。
- キャッシュ鍵 ``prompts_sha1`` は Path / dict どちらでも正規化 JSON（sort_keys、区切り最小、ensure_ascii=False）の
  sha1。同一内容なら鍵が一致し、run 間で記録値が揃う。
- 葉（第3階層）スコアのブランチ内 max 集約（D11 / M1）はこの関数の対象外。ここは第2階層クラス単位のフラット平均。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from cascade import paths
from cascade.embed.encoder import Encoder

DEFAULT_CLASS_ORDER = ("food", "drink", "menu", "inside", "outside")
_SKIP_KEYS = {
    "class_order",
    "taxonomy_version",
    "scheme_version",
    "version",
    "scheme",
    "generated_at",
    "meta",
}


def _texts(value: Any) -> list[str]:
    """prompts の値（list[str] / list[dict] / {"prompts": ...}）から文字列列を取り出す。"""
    if isinstance(value, dict):
        if "prompts" in value:
            return _texts(value["prompts"])
        if "text" in value:
            return [str(value["text"])]
        if "prompt" in value:
            return [str(value["prompt"])]
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for v in value:
            out.extend(_texts(v))
        return out
    return []


def parse_prompts(prompts: dict[str, Any]) -> tuple[dict[str, list[str]], list[str]]:
    """prompts.json の dict → ({class_id: [prompt, ...]}, class_order)。空クラスは ValueError。"""
    body: Any = prompts.get("classes", prompts)
    per_class: dict[str, list[str]] = {}
    if isinstance(body, list):
        for entry in body:
            cid = entry.get("id") or entry.get("node_id") or entry.get("class_id")
            if not cid:
                raise ValueError(f"classes 要素に id が無い: {entry}")
            per_class[str(cid)] = _texts(entry)
    elif isinstance(body, dict):
        flat = "classes" not in prompts
        for cid, v in body.items():
            if cid in _SKIP_KEYS:
                continue
            if flat and not isinstance(v, (list, dict)):
                # フラット形式のメタ値（"0.1.0" 等）はクラスではない
                continue
            per_class[str(cid)] = _texts(v)
    else:
        raise ValueError("prompts.json の形式を解釈できない")

    explicit = prompts.get("class_order")
    if explicit:
        order = [str(c) for c in explicit]
        missing = [c for c in order if c not in per_class]
        if missing:
            raise ValueError(f"class_order にあるがプロンプトが無いクラス: {missing}")
        extra = [c for c in per_class if c not in order]
        if extra:
            raise ValueError(f"プロンプトにあるが class_order に無いクラス: {extra}")
        if len(set(order)) != len(order):
            raise ValueError(f"class_order に重複がある: {order}")
    else:
        order = [c for c in DEFAULT_CLASS_ORDER if c in per_class]
        order += [c for c in per_class if c not in order]
    empty = [c for c in order if not per_class[c]]
    if empty:
        raise ValueError(f"プロンプトが空のクラス: {empty}")
    return {c: per_class[c] for c in order}, order


def canonical_json(data: Any) -> bytes:
    """正規化 JSON（sort_keys、最小区切り、ensure_ascii=False）。Path / dict のどちらでも同じ鍵になる。"""
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def prompts_sha1(prompts_json: Path | dict[str, Any]) -> str:
    data = (
        prompts_json
        if isinstance(prompts_json, dict)
        else json.loads(Path(prompts_json).read_text(encoding="utf-8"))
    )
    return hashlib.sha1(canonical_json(data)).hexdigest()


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def embed_prompts(
    encoder: Encoder,
    prompts_json: Path | dict[str, Any],
    root: Path = paths.EMB,
    use_cache: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """クラス別テキスト埋め込み [C, dim]（L2 正規化済み）と class_order を返す。"""
    data = (
        prompts_json
        if isinstance(prompts_json, dict)
        else json.loads(Path(prompts_json).read_text(encoding="utf-8"))
    )
    per_class, order = parse_prompts(data)
    key = prompts_sha1(prompts_json)
    cdir = Path(root) / encoder.model_key / "text"
    npy = cdir / f"{key}.npy"
    meta = cdir / f"{key}.json"
    if use_cache and npy.exists() and meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        if m.get("class_order") == order and m.get("model_key") == encoder.model_key:
            return np.load(npy).astype(np.float32), order

    flat = [t for c in order for t in per_class[c]]
    emb = _l2(np.asarray(encoder.encode_texts(flat), dtype=np.float32))
    rows = []
    i = 0
    for c in order:
        k = len(per_class[c])
        rows.append(_l2(emb[i : i + k].mean(axis=0)))
        i += k
    out = np.stack(rows).astype(np.float32)

    if use_cache:
        cdir.mkdir(parents=True, exist_ok=True)
        np.save(npy, out)
        meta.write_text(
            json.dumps(
                {
                    "model_key": encoder.model_key,
                    "encoder_name": getattr(encoder, "name", ""),
                    "prompts_sha1": key,
                    "class_order": order,
                    "n_prompts": {c: len(per_class[c]) for c in order},
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return out, order
