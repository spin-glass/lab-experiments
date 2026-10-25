"""configs/encoders.yaml のピン留め検証と encoder spec の検査（オフライン。モデルは読まない）。"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

from cascade import paths
from cascade.config import PinError, load_config, validate_pins
from cascade.embed.encoder import (
    HF_SIGLIP_REQUIRED_FILES,
    OPEN_CLIP_CONFIG_FILE,
    SIGLIP_TEXT_MAX_LENGTH,
    HfSiglipEncoder,
    OpenClipEncoder,
    PreprocessMismatch,
    find_spec,
    load_encoder,
    model_key_for,
    resolve_device,
    validate_spec,
)

CONFIG = paths.REPO_ROOT / "configs" / "encoders.yaml"


def test_encoders_yaml_passes_pin_validation() -> None:
    cfg = load_config(CONFIG)  # validate_pins 込み
    names = [e["name"] for e in cfg["encoders"]]
    assert names == ["e1", "e2"]
    for e in cfg["encoders"]:
        assert len(e["revision"]) == 40 and all(c in "0123456789abcdef" for c in e["revision"])
        assert e["dtype"] == "float32" and e["image_size"] == 224
    assert cfg["taxonomy_version"] == "0.1.0"


def test_mutated_revision_main_rejected() -> None:
    cfg = load_config(CONFIG)
    bad = copy.deepcopy(cfg)
    bad["encoders"][0]["revision"] = "main"
    with pytest.raises(PinError):
        validate_pins(bad)
    with pytest.raises(PinError):
        validate_spec(bad["encoders"][0])
    bad2 = copy.deepcopy(cfg)
    bad2["encoders"][1]["revision"] = "latest"
    with pytest.raises(PinError):
        load_encoder(bad2["encoders"][1])
    bad3 = copy.deepcopy(cfg)
    del bad3["encoders"][0]["revision"]
    with pytest.raises(PinError):
        validate_pins(bad3)


def test_model_key_is_stable_and_sensitive_to_pins() -> None:
    cfg = load_config(CONFIG)
    e1 = cfg["encoders"][0]
    k = model_key_for(e1)
    assert len(k) == 12 and k == model_key_for(dict(e1))
    assert k != model_key_for(dict(e1, revision="0" * 40))
    assert k != model_key_for(dict(e1, image_size=256))
    assert k != model_key_for(cfg["encoders"][1])
    # device は model_key に影響しない（CPU/MPS で同じキャッシュを共有する）
    assert k == model_key_for(dict(e1, device="cpu"))


def test_load_encoder_is_lazy_and_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(CONFIG)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    e1 = load_encoder(dict(cfg["encoders"][0], device="cpu"))
    e2 = load_encoder(dict(cfg["encoders"][1], device="cpu"))
    assert isinstance(e1, OpenClipEncoder) and isinstance(e2, HfSiglipEncoder)
    assert e1.device == "cpu" and e1.name == "e1" and e2.name == "e2"
    assert e1._loaded is False and e2._loaded is False  # 構築時にモデルを読まない
    assert find_spec(cfg, "e2")["name"] == "e2"
    with pytest.raises(KeyError):
        find_spec(cfg, "e9")
    with pytest.raises(ValueError):
        load_encoder(dict(cfg["encoders"][0], backend="nope"))
    with pytest.raises(ValueError):
        load_encoder(dict(cfg["encoders"][0], dtype="float16"))


def test_resolve_device_auto() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("mps", "cpu")


@pytest.mark.skipif(not os.environ.get("CASCADE_NETWORK_TESTS"), reason="network")
def test_real_encoders_encode(tmp_path) -> None:  # pragma: no cover
    import numpy as np
    from PIL import Image

    cfg = load_config(CONFIG)
    for spec in cfg["encoders"]:
        enc = load_encoder(spec)
        img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        v = enc.encode_images([img, img])
        t = enc.encode_texts(["a photo of food"])
        assert v.shape == (2, enc.dim) and t.shape == (1, enc.dim) and v.dtype == np.float32
        np.testing.assert_allclose(v[0], v[1], atol=1e-5)


# --- レビュー指摘の回帰テスト（encoder.py:33 / :170 / :230 / :246 / :254） ---------------------


def test_local_files_only_from_spec_and_argument() -> None:
    cfg = load_config(CONFIG)
    e1 = dict(cfg["encoders"][0], device="cpu")
    assert load_encoder(e1).local_files_only is False
    assert load_encoder(dict(e1, local_files_only=True)).local_files_only is True
    assert (
        load_encoder(dict(e1, local_files_only=True), local_files_only=False).local_files_only
        is False
    )
    assert load_encoder(e1, local_files_only=True).local_files_only is True
    # model_key は local_files_only に影響されない
    assert model_key_for(dict(e1, local_files_only=True)) == model_key_for(e1)
    with pytest.raises(ValueError):
        validate_spec(dict(e1, local_files_only="yes"))
    e2 = dict(cfg["encoders"][1], device="cpu")
    assert load_encoder(e2, local_files_only=True).local_files_only is True


def test_hf_siglip_required_files_include_tokenizer() -> None:
    for fn in ("tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"):
        assert fn in HF_SIGLIP_REQUIRED_FILES


def test_open_clip_download_uses_cache_markers_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """キャッシュ済み → 即返す。.no_exist → HEAD を出さず FileNotFoundError。不明 → local_files_only を伝える。"""
    import huggingface_hub

    cfg = load_config(CONFIG)
    spec = dict(cfg["encoders"][0], device="cpu")
    cached_cfg = tmp_path / "snap" / OPEN_CLIP_CONFIG_FILE
    cached_cfg.parent.mkdir()
    cached_cfg.write_text("{}")
    cached_bin = cached_cfg.parent / "open_clip_pytorch_model.bin"
    cached_bin.write_bytes(b"")

    def fake_cache(repo_id, filename, revision=None, **_):
        assert repo_id == spec["hf_id"] and revision == spec["revision"]
        return {
            OPEN_CLIP_CONFIG_FILE: str(cached_cfg),
            "open_clip_model.safetensors": huggingface_hub._CACHED_NO_EXIST,
            "open_clip_pytorch_model.bin": str(cached_bin),
        }.get(filename)

    calls: list[dict] = []

    def fake_download(**kw):
        calls.append(kw)
        raise AssertionError("network")

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", fake_cache)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    enc = OpenClipEncoder(spec)
    assert enc._download(OPEN_CLIP_CONFIG_FILE) == cached_cfg
    with pytest.raises(FileNotFoundError):
        enc._download("open_clip_model.safetensors")
    assert enc._weights_path() == cached_bin
    assert enc.snapshot_dir() == cached_cfg.parent
    assert calls == []  # ネットワーク（hf_hub_download）は一度も呼ばれない

    # キャッシュ状態が不明なファイルだけ hf_hub_download に行き、local_files_only が伝わる
    enc_off = OpenClipEncoder(spec, local_files_only=True)
    with pytest.raises(AssertionError, match="network"):
        enc_off._download("unknown.json")
    assert calls[-1]["local_files_only"] is True and calls[-1]["revision"] == spec["revision"]


def test_preprocess_mismatch_detected_after_load() -> None:
    cfg = load_config(CONFIG)
    spec = dict(cfg["encoders"][1], device="cpu")

    class Stub(HfSiglipEncoder):
        actual = "hf_siglip/resize_square-bilinear-256-mean0.5-std0.5"

        def _load(self) -> None:
            self._dim = 768

        def _actual_preprocess_desc(self) -> str:
            return self.actual

    bad = Stub(spec)
    with pytest.raises(PreprocessMismatch):
        bad.ensure_loaded()
    assert bad._loaded is False
    good = Stub(spec)
    good.actual = good.preprocess_desc
    good.ensure_loaded()
    assert good.loaded_preprocess_desc == good.preprocess_desc


def test_actual_preprocess_desc_from_loaded_transforms() -> None:
    """spec の image_size と processor / preprocess_cfg の size が食い違えば検出する。"""
    from types import SimpleNamespace

    cfg = load_config(CONFIG)
    e2 = HfSiglipEncoder(dict(cfg["encoders"][1], device="cpu"))
    ip = SimpleNamespace(
        size={"height": 224, "width": 224}, resample=2, image_mean=[0.5] * 3, image_std=[0.5] * 3
    )
    e2._processor = SimpleNamespace(image_processor=ip)
    assert e2._actual_preprocess_desc() == e2.preprocess_desc
    ip.size = {
        "height": 256,
        "width": 256,
    }  # preprocessor_config.json の値のまま、spec だけ変えた状況
    with pytest.raises(PreprocessMismatch):
        e2._actual_preprocess_desc()
    ip.size = {"height": 224, "width": 224}
    ip.resample = 3  # bicubic
    assert e2._actual_preprocess_desc() != e2.preprocess_desc

    e1 = OpenClipEncoder(dict(cfg["encoders"][0], device="cpu"))
    e1._model = SimpleNamespace(
        visual=SimpleNamespace(
            preprocess_cfg={
                "size": (224, 224),
                "resize_mode": "shortest",
                "interpolation": "bicubic",
            }
        )
    )
    assert e1._actual_preprocess_desc() == e1.preprocess_desc
    e1._model.visual.preprocess_cfg["interpolation"] = "bilinear"
    assert e1._actual_preprocess_desc() != e1.preprocess_desc
    e1._model.visual.preprocess_cfg["size"] = (224, 256)
    with pytest.raises(PreprocessMismatch):
        e1._actual_preprocess_desc()


def test_hf_siglip_tokenize_is_fixed_64(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config(CONFIG)
    e2 = HfSiglipEncoder(dict(cfg["encoders"][1], device="cpu"))
    seen: dict = {}

    def proc(**kw):
        seen.update(kw)
        return {"input_ids": None}

    e2._processor = proc
    e2._tokenize(["a"])
    assert seen["max_length"] == SIGLIP_TEXT_MAX_LENGTH == 64
    assert seen["padding"] == "max_length" and seen["truncation"] is True
