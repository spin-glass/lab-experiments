"""fetch_coco のオフラインテスト（主被写体規則・除外集合・iscrowd・person 上限・1 画像 1 型・マニフェスト）。"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from cascade import paths
from cascade.data import fetch_coco as co
from cascade.data import fetch_common as fc

PERSON, CAR, DOG, PIZZA, DINING = 1, 3, 18, 59, 67
EXCL = {44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 67}


def img(i: int, w=640, h=480, lic=1):
    return {
        "id": i,
        "file_name": f"{i:012d}.jpg",
        "width": w,
        "height": h,
        "license": lic,
        "flickr_url": f"http://flickr/{i}",
        "coco_url": f"http://coco/{i}",
    }


def ann(i: int, cat: int, w: float, h: float, crowd=0):
    return {
        "image_id": i,
        "category_id": cat,
        "bbox": [0, 0, w, h],
        "iscrowd": crowd,
        "area": w * h,
    }


IMAGES = [img(i) for i in range(1, 10)]
ANNS = [
    ann(1, PERSON, 400, 400),  # person 主体 → person_animal
    ann(2, PERSON, 400, 400),
    ann(2, PIZZA, 10, 10),  # 食関連 → 除外
    ann(3, CAR, 400, 400),
    ann(3, PERSON, 50, 50),  # 最大 box は car → vehicle_street
    ann(4, PERSON, 100, 100),  # 面積比 0.033 < 0.2 → 除外
    ann(5, PERSON, 400, 400),
    ann(5, PERSON, 30, 30),
    ann(5, PERSON, 30, 30),
    ann(5, PERSON, 30, 30),  # person 4 人 → 除外
    ann(6, DOG, 400, 400),
    ann(6, PERSON, 10, 10, crowd=1),  # iscrowd → 除外
    ann(7, DOG, 300, 300),  # dog 主体
    ann(8, CAR, 300, 300),
    ann(8, DINING, 5, 5),  # dining table → 除外
    ann(
        9,
        PERSON,
        320,
        480,
    ),
    ann(9, PERSON, 30, 30),
    ann(9, PERSON, 30, 30),  # 3 人 → OK
]
TYPES = {"person_animal": {PERSON: "person", DOG: "dog"}, "vehicle_street": {CAR: "car"}}


def test_main_subject_rule_and_one_type():
    picks = co.select_candidates(
        IMAGES,
        ANNS,
        TYPES,
        exclusion_ids=EXCL,
        min_area_ratio=0.2,
        person_id=PERSON,
        person_max_count=3,
        per_class_cap=40,
        budget_of={"person_animal": 100, "vehicle_street": 100},
    )
    got = {(p["subject_type_hint"], p["ext_id"], p["candidate_label"]) for p in picks}
    assert got == {
        ("person_animal", "000000000001", "person"),
        ("person_animal", "000000000007", "dog"),
        ("person_animal", "000000000009", "person"),
        ("vehicle_street", "000000000003", "car"),
    }
    assert [p["ext_id"] for p in picks if p["subject_type_hint"] == "person_animal"] == [
        "000000000001",
        "000000000007",
        "000000000009",
    ]


def test_one_image_one_type_first_wins_and_caps():
    types = {"vehicle_street": {CAR: "car", PERSON: "person"}, "person_animal": {PERSON: "person"}}
    picks = co.select_candidates(
        IMAGES,
        ANNS,
        types,
        exclusion_ids=EXCL,
        min_area_ratio=0.2,
        person_id=PERSON,
        person_max_count=3,
        per_class_cap=1,
        budget_of={"vehicle_street": 100, "person_animal": 100},
    )
    veh = [p["ext_id"] for p in picks if p["subject_type_hint"] == "vehicle_street"]
    per = [p["ext_id"] for p in picks if p["subject_type_hint"] == "person_animal"]
    assert veh == ["000000000001", "000000000003"]  # person は cap=1 で 1 枚、car 1 枚
    assert per == ["000000000009"]  # 残りの person は後続型へ（1 画像 1 型）
    picks = co.select_candidates(
        IMAGES,
        ANNS,
        TYPES,
        exclusion_ids=EXCL,
        min_area_ratio=0.2,
        person_id=PERSON,
        person_max_count=3,
        per_class_cap=40,
        budget_of={"person_animal": 1, "vehicle_street": 0},
    )
    assert [p["ext_id"] for p in picks] == ["000000000001"]


def test_type_categories_from_config_rejects_unknown_name():
    cfg = {
        "coco": {"categories": {"person": 1}},
        "subject_types": {"person_animal": {"coco": ["person", "unicorn"]}},
    }
    with pytest.raises(ValueError):
        co.type_categories_from_config(cfg)


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(paths, "EXT", tmp_path / "ext")
    return tmp_path / "ext"


def _cfg() -> dict[str, Any]:
    return {
        "targets": {"default": 10},
        "candidate_multiplier": 1,
        "per_class_cap": 40,
        "subject_types": {
            "person_animal": {"coco": ["person", "dog"]},
            "vehicle_street": {"coco": ["car"]},
        },
        "coco": {
            "images_zip_url": "http://x/val2017.zip",
            "annotations_zip_url": "http://x/ann.zip",
            "annotations_member": "annotations/instances_val2017.json",
            "images_member_prefix": "val2017/",
            "categories": {"person": PERSON, "car": CAR, "dog": DOG},
            "exclusion_category_ids": sorted(EXCL),
            "min_area_ratio": 0.2,
            "person_max_count": 3,
            "license_ids_allowed": [1, 2, 3, 4, 5, 6, 7, 8],
        },
    }


def _jpeg(w=640, h=480) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (0, 0, 200)).save(buf, format="JPEG")
    return buf.getvalue()


def test_fetch_from_zip(data_root: Path, tmp_path: Path):
    instances = {
        "images": IMAGES,
        "annotations": ANNS,
        "licenses": [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            }
        ],
    }
    images_zip = tmp_path / "val2017.zip"
    with zipfile.ZipFile(images_zip, "w") as zf:
        for im in IMAGES:
            zf.writestr("val2017/" + im["file_name"], _jpeg())
    ann_zip = tmp_path / "ann.zip"
    with zipfile.ZipFile(ann_zip, "w") as zf:
        zf.writestr("annotations/instances_val2017.json", json.dumps(instances))
    assert co.load_instances(ann_zip, "annotations/instances_val2017.json")["images"] == IMAGES

    class NoNet:
        def get(self, url, **kw):
            raise AssertionError("network must not be used")

    cfg = _cfg()
    dry = co.fetch(cfg, NoNet(), dry_run=True, instances=instances)
    assert dry["n_selected"] == 4 and dry["by_type"] == {"person_animal": 3, "vehicle_street": 1}

    res = co.fetch(cfg, NoNet(), instances=instances, images_zip=images_zip)
    assert res["n_downloaded"] == 4
    df = fc.read_manifest("coco")
    assert df.columns == list(fc.MANIFEST_COLUMNS)
    r = df.filter(df["ext_id"] == "000000000003").to_dicts()[0]
    assert r["subject_type_hint"] == "vehicle_street" and r["candidate_label"] == "car"
    assert r["split_hint"] == "val2017"
    assert r["license_short"] == "Attribution-NonCommercial-ShareAlike License"
    assert r["license_url"] == "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    assert r["source_url"] == "http://flickr/3"
    assert "license_id=1" in r["attribution"]
    assert r["path"] == "ext/coco/images/000000000003.jpg"
    assert (r["width"], r["height"]) == (640, 480)
    # 再開
    res2 = co.fetch(cfg, NoNet(), instances=instances, images_zip=images_zip)
    assert res2["n_skipped"] == 4


def test_ensure_zip_reuses_matching_size(tmp_path: Path):
    dst = tmp_path / "a.zip"
    dst.write_bytes(b"12345")

    class NoNet:
        def get_bytes(self, url):
            raise AssertionError("must reuse existing zip")

    assert co.ensure_zip(NoNet(), "http://x", dst, 5) == dst

    class Net:
        def get_bytes(self, url):
            return b"abc"

    with pytest.raises(fc.FetchError):
        co.ensure_zip(Net(), "http://x", tmp_path / "b.zip", 99)
    assert not (tmp_path / "b.zip.part").exists()
    assert co.ensure_zip(Net(), "http://x", tmp_path / "c.zip", 3).read_bytes() == b"abc"
