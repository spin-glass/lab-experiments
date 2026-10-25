"""ZeroShot / embed_prompts のオフラインテスト。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cascade.embed.fake import FakeEncoder
from cascade.embed.text import DEFAULT_CLASS_ORDER, embed_prompts, parse_prompts
from cascade.stage1.zeroshot import ZeroShot, build_zeroshot

CLASSES = list(DEFAULT_CLASS_ORDER)


def _onehot_text_emb(dim: int = 8) -> np.ndarray:
    t = np.zeros((5, dim), dtype=np.float32)
    for i in range(5):
        t[i, i] = 1.0
    return t


def test_probs_sum_to_one_and_argmax_matches_constructed_case() -> None:
    zs = ZeroShot({}, _onehot_text_emb(), CLASSES)
    img = np.zeros((3, 8), dtype=np.float32)
    img[0, 0] = 5.0  # food 軸
    img[1, 2] = 1.0  # menu 軸（正規化されるので大きさは無関係）
    img[2, 4] = 0.3
    img[2, 1] = 0.2  # outside > drink
    for tau in (1.0, 0.01):
        p = zs.probs(img, tau=tau)
        assert p.shape == (3, 5)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)
        assert (p >= 0).all()
        assert zs.predict(img, tau=tau) == ["food", "menu", "outside"]
    # tau=0.01 は確率が尖る（MSP が上がる）、tau=1 は平坦
    p1, p001 = zs.probs(img, 1.0), zs.probs(img, 0.01)
    assert p001.max(axis=1).min() > p1.max(axis=1).max()
    assert p001[0, 0] > 0.99
    # margin と prob_of
    assert zs.margin(img)[0] == pytest.approx(p1[0].max() - np.sort(p1[0])[-2])
    np.testing.assert_allclose(zs.prob_of(img, "food"), p1[:, 0])


def test_1d_input_and_bad_shapes() -> None:
    zs = ZeroShot({}, _onehot_text_emb(), CLASSES)
    assert zs.probs(np.ones(8, dtype=np.float32)).shape == (1, 5)
    with pytest.raises(ValueError):
        ZeroShot({}, _onehot_text_emb()[:4], CLASSES)
    with pytest.raises(ValueError):
        zs.probs(np.ones((1, 8)), tau=0.0)


def _prompts_nested() -> dict:
    return {
        "taxonomy_version": "0.1.0",
        "classes": {
            "food": {
                "prompts": [
                    {"text": "a photo of food", "source": "food"},
                    {"text": "a dish", "source": "food"},
                ]
            },
            "drink": {"prompts": [{"text": "a photo of a drink", "source": "drink"}]},
            "menu": {"prompts": [{"text": "a photo of a menu", "source": "menu"}]},
            "inside": {
                "prompts": [{"text": "a photo of a restaurant interior", "source": "inside"}]
            },
            "outside": {"prompts": [{"text": "a photo of a storefront", "source": "outside"}]},
        },
    }


def test_parse_prompts_shapes() -> None:
    per, order = parse_prompts(_prompts_nested())
    assert order == CLASSES and per["food"] == ["a photo of food", "a dish"]
    per2, order2 = parse_prompts({"outside": ["x"], "food": ["y"], "extra": ["z"]})
    assert order2 == ["food", "outside", "extra"]
    per3, order3 = parse_prompts(
        {
            "classes": [{"id": "b", "prompts": ["p"]}, {"id": "a", "prompts": ["q"]}],
            "class_order": ["a", "b"],
        }
    )
    assert order3 == ["a", "b"]
    with pytest.raises(ValueError):
        parse_prompts({"food": []})
    with pytest.raises(ValueError):
        parse_prompts({"food": ["x"], "class_order": ["food", "drink"]})


def test_embed_prompts_mean_of_normalised_and_cache(tmp_path: Path) -> None:
    # food の 2 プロンプトを固定ベクトルにし、平均→再正規化を検算する
    v1 = np.array([2, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    v2 = np.array([0, 3, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    enc = FakeEncoder(text_vectors={"a photo of food": v1, "a dish": v2})
    pj = tmp_path / "prompts.json"
    pj.write_text(json.dumps(_prompts_nested()))
    root = tmp_path / "emb"

    emb, order = embed_prompts(enc, pj, root=root)
    assert order == CLASSES and emb.shape == (5, 8)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-6)
    expect = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32) / np.sqrt(2)
    np.testing.assert_allclose(emb[0], expect, atol=1e-6)

    cached = list((root / enc.model_key / "text").glob("*.npy"))
    assert len(cached) == 1
    # キャッシュ命中: エンコーダを差し替えても同じ値が返る
    other = FakeEncoder(text_vectors={"a photo of food": v2, "a dish": v2})
    emb2, _ = embed_prompts(other, pj, root=root)
    np.testing.assert_allclose(emb2, emb)
    # 内容が変わればキャッシュキーも変わる
    d = _prompts_nested()
    d["classes"]["food"]["prompts"].append({"text": "another"})
    pj.write_text(json.dumps(d))
    embed_prompts(enc, pj, root=root)
    assert len(list((root / enc.model_key / "text").glob("*.npy"))) == 2


def test_build_zeroshot_end_to_end(tmp_path: Path) -> None:
    enc = FakeEncoder()
    pj = tmp_path / "prompts.json"
    pj.write_text(json.dumps(_prompts_nested()))
    zs = build_zeroshot(enc, pj, root=tmp_path / "emb")
    assert zs.class_order == CLASSES
    # 各クラスのテキスト埋め込みそのものを画像として入れれば当該クラスが argmax
    assert zs.predict(zs.text_emb) == CLASSES
    p = zs.probs(zs.text_emb, tau=0.01)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)


# --- レビュー指摘の回帰テスト（text.py:28 / :69 / :84） --------------------------------------


def test_parse_prompts_ignores_scheme_version_meta() -> None:
    # 実 prompts.json のトップキーは scheme_version。フラット形式でも文字列メタをクラスにしない
    per, order = parse_prompts({"scheme_version": "0.1.0", "food": ["x"], "drink": ["y"]})
    assert order == ["food", "drink"] and "0.1.0" not in per
    per2, order2 = parse_prompts({"scheme_version": "0.1.0", "generated_at": 1, "outside": ["z"]})
    assert order2 == ["outside"]
    nested = _prompts_nested()
    nested["scheme_version"] = "0.1.0"
    del nested["taxonomy_version"]
    assert parse_prompts(nested)[1] == CLASSES


def test_parse_prompts_rejects_classes_missing_from_class_order() -> None:
    with pytest.raises(ValueError, match="class_order に無い"):
        parse_prompts({"food": ["x"], "drink": ["y"], "class_order": ["food"]})
    with pytest.raises(ValueError, match="重複"):
        parse_prompts({"food": ["x"], "class_order": ["food", "food"]})


def test_real_prompts_json_parses_with_class_order() -> None:
    from cascade import paths

    pj = paths.REPO_ROOT / "taxonomy" / "prompts.json"
    if not pj.exists():
        pytest.skip("taxonomy/prompts.json が未生成")
    data = json.loads(pj.read_text(encoding="utf-8"))
    per, order = parse_prompts(data)
    assert order == list(data["class_order"]) and set(per) == set(order)
    assert all(per[c] for c in order)


def test_prompts_sha1_is_normalised_across_path_and_dict(tmp_path: Path) -> None:
    from cascade.embed.text import prompts_sha1

    d = _prompts_nested()
    compact = tmp_path / "compact.json"
    pretty = tmp_path / "pretty.json"
    compact.write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")
    pretty.write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")
    assert prompts_sha1(d) == prompts_sha1(compact) == prompts_sha1(pretty)
    d2 = json.loads(json.dumps(d))
    d2["classes"]["food"]["prompts"].append({"text": "another"})
    assert prompts_sha1(d2) != prompts_sha1(d)

    # Path / dict で同じテキストキャッシュが使われる（二重化しない）
    enc = FakeEncoder()
    root = tmp_path / "emb"
    e1, _ = embed_prompts(enc, pretty, root=root)
    e2, _ = embed_prompts(enc, d, root=root)
    np.testing.assert_allclose(e1, e2)
    assert len(list((root / enc.model_key / "text").glob("*.npy"))) == 1
