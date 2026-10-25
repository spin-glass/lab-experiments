"""YAML config の読み込みと、モデル・データ ID のピン留め検証。

CLAUDE.md: モデル ID・taxonomy_version・rulebook_version は config にピン留めし、"latest" 系エイリアスを禁止する。

- ``revision`` は HF Hub の **完全なコミットハッシュ**（``^[0-9a-f]{40}$``）のみ。未指定は "main"
  (= latest) と同義、ブランチ名・タグ・短縮ハッシュ・None も拒否する
- ピン留め対象キーの値は文字列のみ（None / 数値 / list は PinError）
- ``*_version`` は semver（``X.Y.Z``）。例外は ``rulebook_version`` の ``"未導入"``（CLAUDE.md
  完了条件のフッタ表記。rulebook は M2 で導入）
- ``check_taxonomy_version`` は config の ``taxonomy_version`` と ``taxonomy/taxonomy.yaml`` の
  ``scheme.version`` の一致を検証する（乖離した版が runs に刻印されないようにする）
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

FORBIDDEN_ALIASES = frozenset(
    {
        "latest",
        "main",
        "master",
        "newest",
        "current",
        "head",
        "dev",
        "stable",
        "release",
        "nightly",
        "",
    }
)
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
#: rulebook 未導入の表記（configs/m0.yaml・reports フッタ）。semver 以外で唯一許容する
RULEBOOK_NOT_INTRODUCED = "未導入"

PINNED_KEYS: tuple[str, ...] = (
    "revision",
    "model_id",
    "hf_id",
    "pretrained",
    "dataset_id",
    "taxonomy_version",
    "rulebook_version",
)


class PinError(ValueError):
    """ピン留めされていない ID / revision を検出した。"""


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config のトップレベルは mapping である必要がある: {path}")
    return data


def _check_value(key_path: str, key: str, value: Any) -> None:
    if not isinstance(value, str):
        raise PinError(
            f"{key_path} = {value!r} は文字列でピン留めする必要がある（None / 数値は不可）"
        )
    if value.strip().lower() in FORBIDDEN_ALIASES:
        raise PinError(f"{key_path} = {value!r} はピン留めされていない（latest 系エイリアス禁止）")
    if key == "revision" and not REVISION_RE.match(value):
        raise PinError(
            f"{key_path} = {value!r} は完全なコミットハッシュ（40 桁 16 進）ではない"
            "（ブランチ・タグ・短縮ハッシュ不可）"
        )
    if key.endswith("_version"):
        if key == "rulebook_version" and value == RULEBOOK_NOT_INTRODUCED:
            return
        if not SEMVER_RE.match(value):
            raise PinError(f"{key_path} = {value!r} は semver（X.Y.Z）ではない")


def validate_pins(cfg: dict[str, Any], *, keys: tuple[str, ...] = PINNED_KEYS) -> None:
    """config 木を再帰的に走査し、ピン留め対象キーの値が不正なら PinError を投げる。

    ``revision`` は存在すること自体も要求する（``hf_id`` / ``dataset_id`` を持つ mapping には ``revision`` 必須）。
    """

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if ("hf_id" in node or "dataset_id" in node) and "revision" not in node:
                raise PinError(
                    f"{path}: hf_id / dataset_id には revision（コミットハッシュ）が必須"
                )
            for k, v in node.items():
                kp = f"{path}.{k}" if path else str(k)
                if k in keys:
                    _check_value(kp, str(k), v)
                walk(v, kp)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(cfg, "")


def load_config(path: str | Path, *, validate: bool = True) -> dict[str, Any]:
    cfg = load_yaml(path)
    if validate:
        validate_pins(cfg)
    return cfg


def taxonomy_yaml_version(taxonomy_yaml_path: str | Path) -> str:
    """``taxonomy/taxonomy.yaml`` の ``scheme.version`` を読む（taxonomy_build に依存しない軽量版）。"""
    data = load_yaml(taxonomy_yaml_path)
    scheme = data.get("scheme")
    if not isinstance(scheme, dict) or not isinstance(scheme.get("version"), str):
        raise PinError(f"{taxonomy_yaml_path}: scheme.version が読めない")
    return str(scheme["version"]).strip()


def check_taxonomy_version(cfg: dict[str, Any], taxonomy_yaml_path: str | Path) -> str:
    """config の ``taxonomy_version`` と taxonomy.yaml の ``scheme.version`` の一致を検証し、版を返す。

    未設定・不一致は PinError（古い版が runs に刻印されるのを防ぐ）。
    """
    pinned = cfg.get("taxonomy_version")
    if not isinstance(pinned, str) or not pinned.strip():
        raise PinError("config に taxonomy_version（semver）が無い")
    actual = taxonomy_yaml_version(taxonomy_yaml_path)
    if pinned.strip() != actual:
        raise PinError(
            f"taxonomy_version の不一致: config={pinned!r} taxonomy.yaml(scheme.version)={actual!r}"
        )
    return actual
