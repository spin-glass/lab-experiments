"""画像・テキストエンコーダ（design.md §1 Stage1、§6 G0「検出器」の E1 / E2）。

- E1: open_clip ``ViT-B-16`` / ``datacomp_xl_s13b_b90k``（HF Hub ``laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K``）
- E2: transformers ``google/siglip2-base-patch16-224``（``model_type: siglip`` → ``SiglipModel``）

方針:
- **revision ピン留めを実効化する**。open_clip の ``hf-hub:`` スキーマは revision を受け取らない
  （``open_clip.factory._get_hf_config`` / ``download_pretrained_from_hf`` が revision=None で呼ばれる）ため、
  設定ファイルと重みを ``huggingface_hub.hf_hub_download(..., revision=<hash>)`` で自前取得し、
  その HF キャッシュの ``snapshots/<revision>/`` を ``create_model_and_transforms("local-dir:<dir>")`` に渡す。
  local-dir 経路はピン留めした ``open_clip_config.json`` の ``model_cfg`` と ``preprocess_cfg`` を両方読む
  （open_clip 3.3.0 ``factory.create_model`` の local-dir 分岐）。組み込み ``ViT-B-16`` 構成は使わない。
  transformers 側は ``from_pretrained(..., revision=<hash>)`` をそのまま使う。
- **オフライン**: ``local_files_only``（コンストラクタ引数、または spec キー ``local_files_only``、既定 False）が
  True なら HF Hub に一切アクセスしない。False でも ``try_to_load_from_cache`` でキャッシュ済み／存在しないことが
  判明しているファイルは HEAD 要求を出さない。
  TODO(m0_setup): ``check_env`` は ``is_cached_locally(spec) and not allow_net`` のとき ``local_files_only=True`` で
  エンコーダを作るべきだが、``src/cascade/m0_setup.py`` は別レーンが編集中のため未配線（``--strict`` の終了コードも同様）。
- 常に float32。device は ``auto`` → MPS があれば mps、なければ cpu。
- **遅延ロード**: モデルは最初の encode で読む。``model_key`` は spec だけから決まり、ダウンロード不要。
- ロード後に実 transform（open_clip: ``model.visual.preprocess_cfg``、HF: processor の size/resample/mean/std）から
  ``loaded_preprocess_desc`` を作り、spec 由来の ``preprocess_desc`` と一致しなければ ``PreprocessMismatch`` を投げる。
  model_key が前処理を偽らないための検証。
- 出力は L2 正規化**しない**（正規化は呼び出し側。kNN / Mahalanobis++ / MCM で正規化の有無を明示するため）。
- ``model_key = sha1(backend|hf_id|revision|image_size|dtype|preprocess_desc)[:12]``（計画: model_key に前処理と dtype を含む）
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image

from cascade.config import PinError, validate_pins

SUPPORTED_BACKENDS = ("open_clip", "hf_siglip")
OPEN_CLIP_WEIGHT_FILES = ("open_clip_model.safetensors", "open_clip_pytorch_model.bin")
OPEN_CLIP_CONFIG_FILE = "open_clip_config.json"
# ピン留め revision 75de2d5 に実在するファイル名。AutoProcessor は tokenizer 側も要求する
HF_SIGLIP_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
)
# SigLIP のテキスト側は 64 トークン固定（tokenizer.json の padding Fixed:64 に依存せず明示する）
SIGLIP_TEXT_MAX_LENGTH = 64
# PIL の resample 整数 → 名前（transformers の SiglipImageProcessor.resample は PIL 定数）
_PIL_RESAMPLE_NAMES = {
    0: "nearest",
    1: "lanczos",
    2: "bilinear",
    3: "bicubic",
    4: "box",
    5: "hamming",
}


class PreprocessMismatch(RuntimeError):
    """ロード後の実 transform が spec 由来の preprocess_desc と一致しない（model_key が前処理を偽る）。"""


@runtime_checkable
class Encoder(Protocol):
    """他レーンが依存するエンコーダ契約。"""

    name: str
    model_key: str
    device: str

    @property
    def dim(self) -> int: ...

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


def resolve_device(device: str | None) -> str:
    """``auto`` → mps（利用可能時）/ cpu。明示指定はそのまま返す。"""
    if device in (None, "", "auto"):
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"
    return str(device)


def _fmt(x: float) -> str:
    return f"{float(x):g}"


def _fmt_stats(values: Sequence[float]) -> str:
    """mean/std の記述。全チャネル同値なら 1 値、そうでなければ ',' 区切り。"""
    vals = [float(v) for v in values]
    if len(set(vals)) == 1:
        return _fmt(vals[0])
    return ",".join(_fmt(v) for v in vals)


def open_clip_preprocess_desc(
    model_id: str, pretrained: str, *, size: int, resize_mode: str, interpolation: str
) -> str:
    return (
        f"open_clip/{model_id}/{pretrained}/resize_{resize_mode}-centercrop-{interpolation}-{size}"
    )


def hf_siglip_preprocess_desc(
    *, size: int, resample: str, mean: Sequence[float], std: Sequence[float]
) -> str:
    return f"hf_siglip/resize_square-{resample}-{size}-mean{_fmt_stats(mean)}-std{_fmt_stats(std)}"


def preprocess_desc(spec: dict[str, Any]) -> str:
    """前処理の決定的な記述文字列（model_key の材料）。ダウンロード不要で spec から決まる。

    ロード後に ``_TorchEncoderBase.loaded_preprocess_desc`` と一致検証される（不一致は PreprocessMismatch）。
    """
    backend = spec["backend"]
    size = int(spec.get("image_size", 224))
    if backend == "open_clip":
        # open_clip 既定: 短辺 resize + center crop、bicubic、pretrained tag の mean/std（OpenAI 値）
        return open_clip_preprocess_desc(
            spec["model_id"],
            spec.get("pretrained", ""),
            size=size,
            resize_mode="shortest",
            interpolation="bicubic",
        )
    if backend == "hf_siglip":
        # SiglipImageProcessor: 正方形 resize（bilinear）、mean=std=0.5
        return hf_siglip_preprocess_desc(
            size=size, resample="bilinear", mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        )
    raise ValueError(f"未対応 backend: {backend!r}（対応: {SUPPORTED_BACKENDS}）")


def model_key_for(spec: dict[str, Any]) -> str:
    validate_spec(spec)
    dtype = spec.get("dtype", "float32")
    raw = f"{spec['backend']}|{spec['hf_id']}|{spec['revision']}|{int(spec.get('image_size', 224))}|{dtype}|{preprocess_desc(spec)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def validate_spec(spec: dict[str, Any]) -> None:
    """必須キーとピン留め（cascade.config.validate_pins）を検査する。"""
    for k in ("name", "backend", "hf_id", "revision"):
        if k not in spec:
            raise PinError(f"encoder spec に {k!r} がない: {spec}")
    if spec["backend"] not in SUPPORTED_BACKENDS:
        raise ValueError(f"未対応 backend: {spec['backend']!r}（対応: {SUPPORTED_BACKENDS}）")
    if spec["backend"] == "open_clip" and "model_id" not in spec:
        raise PinError("open_clip backend には model_id（アーキテクチャ名）が必須")
    if spec.get("dtype", "float32") != "float32":
        raise ValueError("dtype は float32 のみ対応（MPS の fp16 差異を避けるため）")
    if not isinstance(spec.get("local_files_only", False), bool):
        raise ValueError("local_files_only は bool")
    validate_pins(spec)


def _batches(seq: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


class _TorchEncoderBase:
    """遅延ロードと no_grad バッチ推論の共通部分。"""

    def __init__(
        self,
        spec: dict[str, Any],
        batch_size: int = 64,
        local_files_only: bool | None = None,
    ) -> None:
        validate_spec(spec)
        self.spec = dict(spec)
        self.name: str = spec["name"]
        self.backend: str = spec["backend"]
        self.hf_id: str = spec["hf_id"]
        self.revision: str = spec["revision"]
        self.image_size: int = int(spec.get("image_size", 224))
        self.dtype: str = spec.get("dtype", "float32")
        self.device: str = resolve_device(spec.get("device", "auto"))
        self.local_files_only: bool = (
            bool(spec.get("local_files_only", False))
            if local_files_only is None
            else bool(local_files_only)
        )
        self.model_key: str = model_key_for(spec)
        self.preprocess_desc: str = preprocess_desc(spec)
        self.loaded_preprocess_desc: str | None = None
        self.batch_size = batch_size
        self._loaded = False
        self._dim: int | None = None

    # --- 遅延ロード -------------------------------------------------------------
    def _load(self) -> None:  # pragma: no cover - 実モデル
        raise NotImplementedError

    def _actual_preprocess_desc(self) -> str:  # pragma: no cover - 実モデル
        raise NotImplementedError

    def _check_preprocess(self) -> None:
        """ロード後の実 transform と spec 由来の preprocess_desc の一致を検証する。"""
        actual = self._actual_preprocess_desc()
        self.loaded_preprocess_desc = actual
        if actual != self.preprocess_desc:
            raise PreprocessMismatch(
                f"{self.name}: ロード後の前処理 {actual!r} が spec 由来の {self.preprocess_desc!r} と一致しない"
                "（model_key が前処理を偽るため中断）"
            )

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()
            self._check_preprocess()
            self._loaded = True

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.ensure_loaded()
        assert self._dim is not None
        return self._dim

    # --- 推論 ------------------------------------------------------------------
    def _encode_image_batch(self, images: Sequence[Image.Image]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _encode_text_batch(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        self.ensure_loaded()
        if len(images) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        outs = [self._encode_image_batch(b) for b in _batches(list(images), self.batch_size)]
        return np.concatenate(outs, axis=0).astype(np.float32, copy=False)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        self.ensure_loaded()
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        outs = [self._encode_text_batch(b) for b in _batches(list(texts), self.batch_size)]
        return np.concatenate(outs, axis=0).astype(np.float32, copy=False)


def cached_file(repo_id: str, filename: str, revision: str) -> Path | None | bool:
    """HF キャッシュの状態: Path（キャッシュ済み）/ False（その revision に存在しないと判明済み）/ None（不明）。"""
    from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache

    r = try_to_load_from_cache(repo_id, filename, revision=revision)
    if isinstance(r, str):
        return Path(r)
    if r is _CACHED_NO_EXIST:
        return False
    return None


class OpenClipEncoder(_TorchEncoderBase):
    """open_clip エンコーダ。重み・設定は revision 固定で HF Hub から自前取得し、local-dir で組み立てる。"""

    def _download(self, filename: str, *, local_files_only: bool | None = None) -> Path:
        """キャッシュ優先で取得する。存在しないと判明済みのファイルは HEAD を出さず FileNotFoundError。"""
        state = cached_file(self.hf_id, filename, self.revision)
        if isinstance(state, Path):
            return state
        if state is False:
            raise FileNotFoundError(
                f"{self.hf_id}@{self.revision} に {filename} は存在しない（.no_exist）"
            )
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=self.hf_id,
                filename=filename,
                revision=self.revision,
                local_files_only=(
                    self.local_files_only if local_files_only is None else local_files_only
                ),
            )
        )

    def _weights_path(self) -> Path:
        last: Exception | None = None
        for fn in OPEN_CLIP_WEIGHT_FILES:
            try:
                return self._download(fn)
            except Exception as e:  # noqa: BLE001 - 次候補へ
                last = e
        raise FileNotFoundError(
            f"{self.hf_id}@{self.revision} に {OPEN_CLIP_WEIGHT_FILES} が見つからない: {last}"
        )

    def snapshot_dir(self) -> Path:
        """ピン留め revision の HF キャッシュ snapshot ディレクトリ（open_clip_config.json と重みが揃う）。"""
        cfg_path = self._download(OPEN_CLIP_CONFIG_FILE)
        weights = self._weights_path()
        snap = cfg_path.parent
        if weights.parent != snap:
            raise RuntimeError(
                f"open_clip_config.json と重みの snapshot が異なる: {cfg_path} / {weights}"
            )
        return snap

    def _load(self) -> None:  # pragma: no cover - 実モデル（ネットワーク／キャッシュ必須）
        import json

        import open_clip
        import torch

        snap = self.snapshot_dir()
        cfg = json.loads((snap / OPEN_CLIP_CONFIG_FILE).read_text(encoding="utf-8"))
        local = f"local-dir:{snap}"
        # local-dir 経路: ピン留めした open_clip_config.json の model_cfg / preprocess_cfg を両方使う
        model, _, preprocess = open_clip.create_model_and_transforms(
            local,
            precision="fp32",
            device=self.device,
            force_image_size=self.image_size,
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(local)
        self._torch = torch
        self._dim = int(
            cfg.get("model_cfg", {}).get("embed_dim") or model.text_projection.shape[-1]
        )

    def _actual_preprocess_desc(self) -> str:
        pc = dict(self._model.visual.preprocess_cfg)
        size = pc.get("size")
        if isinstance(size, (tuple, list)):
            if len(set(int(s) for s in size)) != 1:
                raise PreprocessMismatch(f"非正方形の前処理サイズ: {size}")
            size = size[0]
        return open_clip_preprocess_desc(
            self.spec["model_id"],
            self.spec.get("pretrained", ""),
            size=int(size),
            resize_mode=str(pc.get("resize_mode")),
            interpolation=str(pc.get("interpolation")),
        )

    def _encode_image_batch(self, images: Sequence[Image.Image]) -> np.ndarray:  # pragma: no cover
        torch = self._torch
        x = torch.stack([self._preprocess(im.convert("RGB")) for im in images]).to(self.device)
        with torch.no_grad():
            f = self._model.encode_image(x)
        return f.float().cpu().numpy()

    def _encode_text_batch(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        torch = self._torch
        tok = self._tokenizer(list(texts)).to(self.device)
        with torch.no_grad():
            f = self._model.encode_text(tok)
        return f.float().cpu().numpy()


class HfSiglipEncoder(_TorchEncoderBase):
    """transformers SigLIP（``SiglipModel.get_image_features / get_text_features``）。

    ピン留めチェックポイントは ``model_type: siglip`` なので ``AutoModel`` は ``SiglipModel`` を返す
    （Siglip2 系ではない）。processor には spec の image_size を明示的に渡し、ロード後に一致を検証する。
    """

    def _load(self) -> None:  # pragma: no cover - 実モデル
        import torch
        from transformers import AutoModel, AutoProcessor

        self._model = AutoModel.from_pretrained(
            self.hf_id,
            revision=self.revision,
            dtype=torch.float32,
            local_files_only=self.local_files_only,
        ).to(self.device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self.hf_id,
            revision=self.revision,
            local_files_only=self.local_files_only,
            size={"height": self.image_size, "width": self.image_size},
        )
        self._torch = torch
        cfg = self._model.config
        self._dim = int(getattr(cfg, "projection_dim", None) or cfg.text_config.hidden_size)
        # テキスト側の長さが固定 64 であることを確認する
        probe = self._tokenize(["a photo", "a " * (SIGLIP_TEXT_MAX_LENGTH * 2)])
        n = int(probe["input_ids"].shape[-1])
        if n != SIGLIP_TEXT_MAX_LENGTH:
            raise RuntimeError(f"SigLIP tokenizer の出力長 {n} != {SIGLIP_TEXT_MAX_LENGTH}")

    @staticmethod
    def _size_of(size: Any) -> tuple[int, int]:
        if isinstance(size, dict):
            return int(size["height"]), int(size["width"])
        h = getattr(size, "height", None)
        w = getattr(size, "width", None)
        if h is None or w is None:
            raise PreprocessMismatch(f"processor の size を解釈できない: {size!r}")
        return int(h), int(w)

    def _actual_preprocess_desc(self) -> str:
        ip = getattr(self._processor, "image_processor", self._processor)
        h, w = self._size_of(ip.size)
        if h != w:
            raise PreprocessMismatch(f"非正方形の前処理サイズ: {(h, w)}")
        if h != self.image_size:
            raise PreprocessMismatch(
                f"processor の size {h} が spec の image_size {self.image_size} と一致しない"
            )
        resample = ip.resample
        rname = (
            _PIL_RESAMPLE_NAMES.get(int(resample), str(resample))
            if isinstance(resample, int)
            else str(getattr(resample, "name", resample)).lower()
        )
        return hf_siglip_preprocess_desc(
            size=h, resample=rname, mean=tuple(ip.image_mean), std=tuple(ip.image_std)
        )

    def _tokenize(self, texts: Sequence[str]) -> Any:
        return self._processor(
            text=list(texts),
            padding="max_length",
            max_length=SIGLIP_TEXT_MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        )

    @staticmethod
    def _pooled(out: Any) -> Any:
        # transformers 5.x は BaseModelOutputWithPooling を返す。旧版は Tensor を返す
        return getattr(out, "pooler_output", out)

    def _encode_image_batch(self, images: Sequence[Image.Image]) -> np.ndarray:  # pragma: no cover
        torch = self._torch
        inputs = self._processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            f = self._pooled(self._model.get_image_features(**inputs))
        return f.float().cpu().numpy()

    def _encode_text_batch(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        torch = self._torch
        inputs = {k: v.to(self.device) for k, v in self._tokenize(texts).items()}
        with torch.no_grad():
            f = self._pooled(self._model.get_text_features(**inputs))
        return f.float().cpu().numpy()


def is_cached_locally(spec: dict[str, Any]) -> bool:
    """ピン留め revision の必要ファイルが HF キャッシュに揃っているか（ネットワーク不要）。"""
    validate_spec(spec)
    repo, rev = spec["hf_id"], spec["revision"]

    def has(fn: str) -> bool:
        return isinstance(cached_file(repo, fn, rev), Path)

    if spec["backend"] == "open_clip":
        return has(OPEN_CLIP_CONFIG_FILE) and any(has(fn) for fn in OPEN_CLIP_WEIGHT_FILES)
    return all(has(fn) for fn in HF_SIGLIP_REQUIRED_FILES)


def load_encoder(
    spec: dict[str, Any], batch_size: int = 64, local_files_only: bool | None = None
) -> Encoder:
    """spec（configs/encoders.yaml の 1 要素）からエンコーダを作る。モデルはまだ読まない。

    ``local_files_only`` は None なら spec の ``local_files_only``（既定 False）に従う。
    """
    validate_spec(spec)
    if spec["backend"] == "open_clip":
        return OpenClipEncoder(spec, batch_size=batch_size, local_files_only=local_files_only)
    return HfSiglipEncoder(spec, batch_size=batch_size, local_files_only=local_files_only)


def find_spec(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    for s in cfg.get("encoders", []):
        if s.get("name") == name:
            return s
    raise KeyError(f"encoders に name={name!r} がない")
