"""データ配置の単一定義。

`data/` は git 管理外（CLAUDE.md 絶対制約）。環境変数 ``CASCADE_DATA_DIR`` で置き場を変えられる。
三層（raw / core / marts）は design.md §3・§5 に対応する。
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("CASCADE_DATA_DIR", REPO_ROOT / "data")).resolve()

RAW = DATA / "raw"
CORE = DATA / "core"
MARTS = DATA / "marts"
EXT = DATA / "ext"  # 外部の公開画像データ（Open Images / Commons / COCO / CORD 等）
EMB = DATA / "embeddings"  # npy シャード＋parquet index
REVIEW = DATA / "review"  # 目視確認のキューとラベル（jsonl）
YELP = DATA / "yelp"  # Yelp Open Dataset（ユーザが手動配置）

LAYERS = {"raw": RAW, "core": CORE, "marts": MARTS}


def ensure_dirs() -> None:
    """三層と作業ディレクトリを作る（冪等）。"""
    for p in (RAW, CORE, MARTS, EXT, EMB, REVIEW, YELP):
        p.mkdir(parents=True, exist_ok=True)
