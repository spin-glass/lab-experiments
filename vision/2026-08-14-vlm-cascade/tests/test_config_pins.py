"""config のピン留め検証（CLAUDE.md: latest 系エイリアス禁止、revision は完全なコミットハッシュ）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from cascade import paths
from cascade.config import (
    FORBIDDEN_ALIASES,
    PinError,
    check_taxonomy_version,
    load_config,
    validate_pins,
)

SHA = "d110532e8d4ff91c574ee60a342323f28468b287"
TAXONOMY_YAML = paths.REPO_ROOT / "taxonomy" / "taxonomy.yaml"


def test_rejects_latest_alias():
    with pytest.raises(PinError):
        validate_pins({"encoders": [{"hf_id": "x/y", "revision": "latest"}]})


def test_rejects_main_and_missing_revision():
    with pytest.raises(PinError):
        validate_pins({"e": {"hf_id": "x/y", "revision": "main"}})
    with pytest.raises(PinError):
        validate_pins({"e": {"hf_id": "x/y"}})


def test_accepts_pinned_hash():
    validate_pins(
        {
            "e": {"hf_id": "x/y", "revision": SHA},
            "taxonomy_version": "0.1.0",
            "rulebook_version": "未導入",
            "model_id": "ViT-B-16",
        }
    )


@pytest.mark.parametrize(
    "revision",
    [
        None,  # `revision:`（空）
        "",
        "v1",  # タグ
        "HEAD",
        "release",
        "d110532",  # 短縮ハッシュ
        SHA[:-1] + "g",  # 16 進でない
        SHA.upper(),  # 大文字は HF の正規形ではない
        12345,  # 非文字列
        [SHA],
    ],
)
def test_revision_must_be_full_commit_hash(revision):
    with pytest.raises(PinError):
        validate_pins({"e": {"hf_id": "x/y", "revision": revision}})


@pytest.mark.parametrize("alias", sorted(FORBIDDEN_ALIASES - {""}))
def test_extended_alias_list_rejected(alias):
    with pytest.raises(PinError):
        validate_pins({"model_id": alias})
    with pytest.raises(PinError):
        validate_pins({"model_id": alias.upper()})


@pytest.mark.parametrize("value", [None, 1, 1.5, True, ["ViT-B-16"], {"id": "x"}])
def test_pinned_keys_must_be_strings(value):
    with pytest.raises(PinError):
        validate_pins({"model_id": value})
    with pytest.raises(PinError):
        validate_pins({"taxonomy_version": value})


@pytest.mark.parametrize("version", ["v1", "1.0", "latest", "0.1", "0.1.0-rc1", ""])
def test_version_keys_require_semver(version):
    with pytest.raises(PinError):
        validate_pins({"taxonomy_version": version})
    with pytest.raises(PinError):
        validate_pins({"rulebook_version": version})


def test_rulebook_version_allows_not_introduced_sentinel_only():
    validate_pins({"rulebook_version": "未導入"})
    validate_pins({"rulebook_version": "1.2.3"})
    with pytest.raises(PinError):
        validate_pins({"rulebook_version": "draft"})
    with pytest.raises(PinError):
        validate_pins({"taxonomy_version": "未導入"})  # taxonomy には sentinel を認めない


def test_repo_configs_still_pass():
    for name in ("encoders.yaml", "m0.yaml", "ood_sources.yaml", "g0_nonfood_gate.yaml"):
        p = paths.REPO_ROOT / "configs" / name
        if p.exists():
            load_config(p)


def test_check_taxonomy_version_matches_yaml():
    assert check_taxonomy_version({"taxonomy_version": "0.1.0"}, TAXONOMY_YAML) == "0.1.0"
    with pytest.raises(PinError, match="不一致"):
        check_taxonomy_version({"taxonomy_version": "0.2.0"}, TAXONOMY_YAML)
    with pytest.raises(PinError):
        check_taxonomy_version({}, TAXONOMY_YAML)
    with pytest.raises(PinError):
        check_taxonomy_version({"taxonomy_version": None}, TAXONOMY_YAML)


def test_check_taxonomy_version_with_broken_yaml(tmp_path: Path):
    bad = tmp_path / "taxonomy.yaml"
    bad.write_text("scheme: {id: x}\n", encoding="utf-8")
    with pytest.raises(PinError):
        check_taxonomy_version({"taxonomy_version": "0.1.0"}, bad)
