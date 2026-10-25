"""fetch_openimages のオフラインテスト（MID 抽出・Food 規則・クラス別上限・決定的順序・マニフェスト）。"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from PIL import Image

from cascade import paths
from cascade.data import fetch_common as fc
from cascade.data import fetch_openimages as oi

FOOD = "/m/02wbm"
DOLL = "/m/0167gd"
TEDDY = "/m/0kmg4"
PLATE = "/m/050gv4"
PERSON = "/m/01g317"


def jpeg_bytes(w: int = 300, h: int = 200) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 30, 30)).save(buf, format="JPEG")
    return buf.getvalue()


class FakeResp:
    def __init__(
        self, status: int = 200, text: str = "", content: bytes = b"", headers: dict | None = None
    ):
        self.status_code = status
        self.text = text
        self.content = content
        self.headers = headers or {}

    def json(self) -> Any:
        raise NotImplementedError


class FakeClient:
    def __init__(self, routes: dict[str, FakeResp]):
        self.routes = routes
        self.calls: list[str] = []

    def get(self, url: str, *, params=None, timeout=None) -> FakeResp:
        self.calls.append(url)
        return self.routes.get(url, FakeResp(404))


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(paths, "EXT", tmp_path / "ext")
    return tmp_path / "ext"


LABELS_CSV = "\n".join(
    [
        "ImageID,Source,LabelName,Confidence",
        f"img_b,verification,{DOLL},1",
        f"img_a,verification,{DOLL},1.0",
        f"img_c,verification,{DOLL},1",
        f"img_c,verification,{FOOD},1",  # Food positive → 通常型では除外
        f"img_d,verification,{DOLL},0.0",  # negative なので不採用
        f"img_e,verification,{TEDDY},1",
        f"img_e,verification,{DOLL},1",  # 同一画像が 2 MID: DOLL で先に採られ TEDDY では重複しない
        f"img_p1,verification,{PLATE},1",
        f"img_p1,verification,{FOOD},0",  # near_food: Food 明示 negative → 採用
        f"img_p2,verification,{PLATE},1",  # near_food: Food 記載なし → 不採用
        f"img_p3,verification,{PLATE},1",
        f"img_p3,verification,{FOOD},1",  # near_food: Food positive → 不採用。food_contrast では採用
        f"img_f1,verification,{FOOD},1",
        f"img_x,verification,{PERSON},1",
        f"img_x,verification,{DOLL},1",  # 先勝ち: doll 型に割当、person 型には出ない
    ]
)


def test_parse_image_labels_numeric_confidence():
    labels = oi.parse_image_labels(io.StringIO(LABELS_CSV))
    assert labels["img_a"]["pos"] == {DOLL}
    assert labels["img_d"]["neg"] == {DOLL}
    assert labels["img_p1"]["neg"] == {FOOD}
    assert "ImageID" not in labels


def test_select_candidates_rules_and_order():
    labels = oi.parse_image_labels(io.StringIO(LABELS_CSV))
    type_mids = {
        "doll_character_statue": {DOLL: "Doll", TEDDY: "Teddy bear"},
        "person_animal": {PERSON: "Person"},
        "near_food_nonfood": {PLATE: "Plate"},
        "food_contrast": {FOOD: "Food"},
    }
    picks = oi.select_candidates(
        labels,
        type_mids,
        food_mid=FOOD,
        per_class_cap=40,
        budget_of={t: 100 for t in type_mids},
        split="validation",
    )
    by_type: dict[str, list[str]] = {}
    for p in picks:
        by_type.setdefault(p["subject_type_hint"], []).append(p["ext_id"])
    # ImageID 昇順、Food positive (img_c) 除外、negative (img_d) 除外、img_e は DOLL で 1 回のみ
    assert by_type["doll_character_statue"] == ["img_a", "img_b", "img_e", "img_x"]
    assert "person_animal" not in by_type  # img_x は先勝ちで doll に割当済み
    assert by_type["near_food_nonfood"] == ["img_p1"]
    assert by_type["food_contrast"] == ["img_c", "img_f1", "img_p3"]  # Food positive なら採用
    assert all(p["split_hint"] == "validation" for p in picks)


def test_per_class_cap_and_budget():
    labels = {f"img_{i:03d}": {"pos": {DOLL}, "neg": set()} for i in range(10)}
    labels.update({f"ted_{i:03d}": {"pos": {TEDDY}, "neg": set()} for i in range(10)})
    tm = {"doll_character_statue": {DOLL: "Doll", TEDDY: "Teddy bear"}}
    picks = oi.select_candidates(
        labels,
        tm,
        food_mid=FOOD,
        per_class_cap=3,
        budget_of={"doll_character_statue": 100},
        split="test",
    )
    assert [p["candidate_label"] for p in picks].count(DOLL) == 3
    assert [p["candidate_label"] for p in picks].count(TEDDY) == 3
    picks = oi.select_candidates(
        labels,
        tm,
        food_mid=FOOD,
        per_class_cap=40,
        budget_of={"doll_character_statue": 4},
        split="test",
    )
    assert len(picks) == 4
    assert [p["ext_id"] for p in picks] == ["img_000", "img_001", "img_002", "img_003"]


def test_license_short_from_url():
    assert oi.license_short_from_url("https://creativecommons.org/licenses/by/2.0/") == "CC BY 2.0"
    assert (
        oi.license_short_from_url("https://creativecommons.org/licenses/by-sa/4.0")
        == "CC BY-SA 4.0"
    )
    assert oi.license_short_from_url("") == "unknown"


def _cfg() -> dict[str, Any]:
    return {
        "targets": {"default": 2},
        "candidate_multiplier": 1,
        "per_class_cap": 40,
        "subject_types": {"doll_character_statue": {"open_images": {DOLL: "Doll"}}},
        "open_images": {
            "food_mid": FOOD,
            "splits": {
                "validation": {
                    "labels_url": "http://x/labels.csv",
                    "attribution_url": "http://x/attr.csv",
                    "s3_prefix": "validation",
                }
            },
            "image_url": "http://img/{split}/{image_id}.jpg",
            "download_backend": "https",
        },
    }


ATTR_CSV = "\n".join(
    [
        "ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,Author,Title,OriginalSize,OriginalMD5,Thumbnail300KURL,Rotation",
        "img_a,validation,http://o/a.jpg,http://flickr/a,https://creativecommons.org/licenses/by/2.0/,http://p/a,Alice,Doll on shelf,1,md5a,,0",
        "img_b,validation,http://o/b.jpg,http://flickr/b,https://creativecommons.org/licenses/by-sa/2.0/,http://p/b,Bob,Toy,1,md5b,,0",
        "img_zzz,validation,http://o/z.jpg,http://flickr/z,https://creativecommons.org/licenses/by/2.0/,http://p/z,Zed,Unused,1,md5z,,0",
    ]
)


def test_fetch_end_to_end_and_resume(data_root: Path):
    cfg = _cfg()
    routes = {
        "http://x/labels.csv": FakeResp(text=LABELS_CSV),
        "http://x/attr.csv": FakeResp(text=ATTR_CSV),
        "http://img/validation/img_a.jpg": FakeResp(content=jpeg_bytes(320, 240)),
        "http://img/validation/img_b.jpg": FakeResp(content=jpeg_bytes(300, 300)),
    }
    client = FakeClient(routes)
    dry = oi.fetch(cfg, client, dry_run=True)
    assert dry["n_selected"] == 2 and dry["n_downloaded"] == 0
    assert [c["ext_id"] for c in dry["candidates"]] == ["img_a", "img_b"]
    assert not (data_root / "open_images" / "images").exists()

    res = oi.fetch(cfg, client)
    assert res["n_downloaded"] == 2
    df = fc.read_manifest("open_images")
    assert df.columns == list(fc.MANIFEST_COLUMNS)
    assert df.height == 2
    row = df.filter(pl.col("ext_id") == "img_a").to_dicts()[0]
    assert row["source"] == "open_images"
    assert row["subject_type_hint"] == "doll_character_statue"
    assert row["candidate_label"] == DOLL
    assert row["license_short"] == "CC BY 2.0"
    assert row["license_url"] == "https://creativecommons.org/licenses/by/2.0/"
    assert "Alice" in row["attribution"] and "http://flickr/a" in row["attribution"]
    assert row["source_url"] == "http://flickr/a"
    assert (row["width"], row["height"]) == (320, 240)
    assert row["path"] == "ext/open_images/images/img_a.jpg"
    assert (data_root.parent / row["path"]).exists()
    assert row["sha256"] == fc.sha256_file(data_root.parent / row["path"])

    # 再開: done マーカーで画像を再取得しない
    n_calls = len(client.calls)
    res2 = oi.fetch(cfg, client)
    assert res2["n_skipped"] == 2 and res2["n_downloaded"] == 0
    assert not any(u.startswith("http://img/") for u in client.calls[n_calls:])
    assert fc.read_manifest("open_images").height == 2


def test_fetch_max_per_type_override(data_root: Path):
    cfg = _cfg()
    client = FakeClient(
        {
            "http://x/labels.csv": FakeResp(text=LABELS_CSV),
            "http://x/attr.csv": FakeResp(text=ATTR_CSV),
        }
    )
    dry = oi.fetch(cfg, client, dry_run=True, max_per_type=1)
    assert dry["n_selected"] == 1
