"""HF キャッシュ済みの実モデルをオフライン（HF_HUB_OFFLINE=1）で読む回帰テスト。

ピン留め revision のファイルが HF キャッシュに無ければ skip する（ネットワークには出ない）。
検証: (a) E1 の埋め込みが local-dir スキーマの直接ロードと一致、(b) パラメータが float32、
(c) processor / transform の size が spec の image_size と一致、(d) HF Hub 到達不能でも 60 秒以内に読める。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cascade import paths
from cascade.config import load_config
from cascade.embed.encoder import (
    OPEN_CLIP_CONFIG_FILE,
    HfSiglipEncoder,
    OpenClipEncoder,
    find_spec,
    is_cached_locally,
    load_encoder,
)

CONFIG = paths.REPO_ROOT / "configs" / "encoders.yaml"
_CFG = load_config(CONFIG)
E1 = dict(find_spec(_CFG, "e1"), device="cpu")
E2 = dict(find_spec(_CFG, "e2"), device="cpu")
E1_CACHED = is_cached_locally(E1)
E2_CACHED = is_cached_locally(E2)
needs_e1 = pytest.mark.skipif(not E1_CACHED, reason="E1 がピン留め revision で HF キャッシュに無い")
needs_e2 = pytest.mark.skipif(not E2_CACHED, reason="E2 がピン留め revision で HF キャッシュに無い")


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub.constants as c

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(c, "HF_HUB_OFFLINE", True, raising=False)


def _images() -> list[Image.Image]:
    rng = np.random.default_rng(0)
    a = Image.fromarray(rng.integers(0, 256, (300, 200, 3), dtype=np.uint8))
    b = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
    return [a, b]


@needs_e1
def test_e1_matches_direct_local_dir_load() -> None:
    import json

    import open_clip
    import torch
    from huggingface_hub import try_to_load_from_cache

    enc = load_encoder(E1, local_files_only=True)
    assert isinstance(enc, OpenClipEncoder)
    imgs = _images()
    v = enc.encode_images(imgs)
    t = enc.encode_texts(["a photo of food", "a photo of a menu"])
    assert v.shape == (2, 512) and t.shape == (2, 512) and v.dtype == np.float32

    snap = Path(
        try_to_load_from_cache(E1["hf_id"], OPEN_CLIP_CONFIG_FILE, revision=E1["revision"])
    ).parent
    assert enc.snapshot_dir() == snap
    model, _, pp = open_clip.create_model_and_transforms(
        f"local-dir:{snap}", precision="fp32", device="cpu"
    )
    model.eval()
    tok = open_clip.get_tokenizer(f"local-dir:{snap}")
    with torch.no_grad():
        v_ref = model.encode_image(torch.stack([pp(im) for im in imgs])).numpy()
        t_ref = model.encode_text(tok(["a photo of food", "a photo of a menu"])).numpy()
    np.testing.assert_allclose(v, v_ref, atol=1e-5)
    np.testing.assert_allclose(t, t_ref, atol=1e-5)

    # ピン留めした open_clip_config.json の preprocess_cfg が実際に使われている
    cfg = json.loads((snap / OPEN_CLIP_CONFIG_FILE).read_text(encoding="utf-8"))
    pc = enc._model.visual.preprocess_cfg
    assert list(pc["mean"]) == cfg["preprocess_cfg"]["mean"]
    assert list(pc["std"]) == cfg["preprocess_cfg"]["std"]
    assert enc.loaded_preprocess_desc == enc.preprocess_desc
    assert enc.dim == cfg["model_cfg"]["embed_dim"] == 512


@needs_e1
def test_e1_loads_without_hub_download_when_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """キャッシュ済みなら local_files_only=False でも hf_hub_download（HEAD 要求）を呼ばない。"""
    import huggingface_hub

    def boom(**_):
        raise AssertionError("hf_hub_download が呼ばれた")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    enc = load_encoder(E1, local_files_only=False)
    enc.ensure_loaded()
    assert enc._loaded


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(E1, id="e1", marks=needs_e1),
        pytest.param(E2, id="e2", marks=needs_e2),
    ],
)
def test_parameters_are_float32(spec: dict) -> None:
    import torch

    enc = load_encoder(spec, local_files_only=True)
    enc.ensure_loaded()
    dtypes = {p.dtype for p in enc._model.parameters()}
    assert dtypes == {torch.float32}


@needs_e1
def test_e1_transform_size_matches_spec() -> None:
    enc = load_encoder(E1, local_files_only=True)
    enc.ensure_loaded()
    pc = enc._model.visual.preprocess_cfg
    assert tuple(pc["size"]) == (E1["image_size"], E1["image_size"])
    assert pc["resize_mode"] == "shortest" and pc["interpolation"] == "bicubic"
    x = enc._preprocess(_images()[0])
    assert tuple(x.shape) == (3, E1["image_size"], E1["image_size"])


@needs_e2
def test_e2_processor_size_follows_spec_image_size() -> None:
    enc = load_encoder(E2, local_files_only=True)
    assert isinstance(enc, HfSiglipEncoder)
    enc.ensure_loaded()
    ip = enc._processor.image_processor
    assert (int(ip.size["height"]), int(ip.size["width"])) == (224, 224)
    assert enc.loaded_preprocess_desc == enc.preprocess_desc
    px = enc._processor(images=_images(), return_tensors="pt")["pixel_values"]
    assert tuple(px.shape) == (2, 3, 224, 224)
    v = enc.encode_images(_images())
    t = enc.encode_texts(["a photo of food", "word " * 200])  # 長文は 64 トークンに切り詰め
    assert v.shape == (2, enc.dim) and t.shape == (2, enc.dim) and enc.dim == 768
    ids = enc._tokenize(["a", "word " * 200])["input_ids"]
    assert tuple(ids.shape) == (2, 64)

    # image_size を変えると processor の size も変わる（model_key と前処理が食い違わない）
    enc256 = load_encoder(dict(E2, image_size=256), local_files_only=True)
    assert enc256.model_key != enc.model_key
    enc256.ensure_loaded()
    ip256 = enc256._processor.image_processor
    assert (int(ip256.size["height"]), int(ip256.size["width"])) == (256, 256)
    px256 = enc256._processor(images=_images()[:1], return_tensors="pt")["pixel_values"]
    assert tuple(px256.shape) == (1, 3, 256, 256)
    assert enc256.loaded_preprocess_desc == enc256.preprocess_desc


_SUBPROCESS_SCRIPT = """
import sys, time, json
import numpy as np
from PIL import Image
from cascade.config import load_config
from cascade.embed.encoder import find_spec, load_encoder
cfg = load_config(sys.argv[1])
t0 = time.time()
out = {}
for name in sys.argv[2].split(","):
    spec = dict(find_spec(cfg, name), device="cpu")
    enc = load_encoder(spec, local_files_only=True)
    v = enc.encode_images([Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))])
    out[name] = [int(v.shape[0]), int(v.shape[1])]
out["elapsed"] = time.time() - t0
print(json.dumps(out))
"""


@pytest.mark.skipif(not (E1_CACHED or E2_CACHED), reason="キャッシュ済みモデルが無い")
def test_offline_load_completes_quickly_with_unreachable_hub() -> None:
    """HF_HUB_OFFLINE 未設定・HF Hub 到達不能でも local_files_only=True なら 60 秒以内に読める。"""
    import json

    names = ",".join(n for n, ok in (("e1", E1_CACHED), ("e2", E2_CACHED)) if ok)
    env = dict(os.environ)
    env.pop("HF_HUB_OFFLINE", None)
    env["HF_ENDPOINT"] = "http://127.0.0.1:9"  # 到達不能
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT, str(CONFIG), names],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(paths.REPO_ROOT),
    )
    elapsed = time.time() - t0
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert elapsed < 60, f"オフラインロードに {elapsed:.1f}s（stderr: {proc.stderr[-500:]}）"
    for n in names.split(","):
        assert out[n][0] == 1 and out[n][1] in (512, 768)
