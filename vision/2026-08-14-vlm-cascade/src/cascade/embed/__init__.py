"""cascade.embed — 画像／テキストの凍結埋め込み（design.md §1 Stage1、§6 G0 の E1/E2）。

- ``encoder``: open_clip / HF SigLIP2 のエンコーダ（revision ピン留め、遅延ロード、float32）
- ``cache``: npy シャード＋parquet index の再開可能な埋め込みキャッシュ
- ``text``: prompts.json → クラス別テキスト埋め込み（キャッシュ付き）
- ``fake``: ネットワーク・モデル不要の決定的ダミーエンコーダ（テスト用）
"""

from cascade.embed.cache import EmbeddingCache
from cascade.embed.encoder import Encoder, load_encoder, model_key_for, resolve_device
from cascade.embed.fake import FakeEncoder
from cascade.embed.text import embed_prompts, parse_prompts

__all__ = [
    "EmbeddingCache",
    "Encoder",
    "FakeEncoder",
    "embed_prompts",
    "load_encoder",
    "model_key_for",
    "parse_prompts",
    "resolve_device",
]
