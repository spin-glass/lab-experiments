"""configs/ood_sources.yaml の検証（ピン留め・enum 整合・必須キー）、fetch_hf と lock 生成のオフラインテスト。"""

from __future__ import annotations

import copy
import io
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from PIL import Image

from cascade import paths
from cascade.config import PinError, load_config, validate_pins
from cascade.data import fetch_coco, fetch_commons, fetch_hf, fetch_openimages
from cascade.data import fetch_common as fc

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "ood_sources.yaml"
sys.path.insert(0, str(REPO / "scripts"))


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config(CONFIG)


def test_validate_pins_passes(cfg: dict):
    validate_pins(cfg)
    assert cfg["hf"]["cord"]["revision"] == "7f0115a4b758a71d6473b8d085751692da2fef98"
    assert cfg["hf"]["cord"]["hf_id"] == "naver-clova-ix/cord-v2"
    assert cfg["hf"]["screenspot"]["revision"] == "0be08781e2e188582f6131625ae1598d443b4d5d"


def test_validate_pins_rejects_alias(cfg: dict):
    bad = copy.deepcopy(cfg)
    bad["hf"]["cord"]["revision"] = "main"
    with pytest.raises(PinError):
        validate_pins(bad)
    bad = copy.deepcopy(cfg)
    del bad["hf"]["cord"]["revision"]
    with pytest.raises(PinError):
        validate_pins(bad)


def test_required_keys_and_enums(cfg: dict):
    assert "{contact}" in cfg["user_agent"]
    assert cfg["contact"] is None  # ベタ書き禁止: env CASCADE_CONTACT
    assert cfg["throttle"]["min_interval_sec"] >= 0.5  # ≤ 2 req/s
    assert cfg["candidate_multiplier"] == 2
    assert cfg["per_class_cap"] == 40
    assert cfg["min_side"] == 256
    assert set(cfg["subject_types"]) <= set(fc.SUBJECT_TYPES)
    for stype, srcs in cfg["subject_types"].items():
        assert set(srcs) <= {"open_images", "commons", "coco", "hf"}, stype
    t = cfg["targets"]
    assert (
        t["near_food_nonfood"] == 80
        and t["food_contrast"] == 100
        and t["texture_document_screen"] == 200
    )
    assert all(
        t[k] == 120
        for k in (
            "doll_character_statue",
            "amusement_tourist",
            "signage_decor_goods",
            "person_animal",
            "vehicle_street",
        )
    )
    assert cfg["hf"]["cord"]["n_take"] == 200
    assert cfg["open_images"]["food_mid"] == "/m/02wbm"
    assert set(cfg["open_images"]["splits"]) == {"validation", "test"}
    assert cfg["coco"]["exclusion_category_ids"] == [
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        67,
    ]
    assert cfg["commons"]["max_depth"] == 2 and cfg["commons"]["imageinfo_batch"] <= 50
    allow = fetch_commons.compile_allowlist(cfg["license_allowlist"])
    assert fetch_commons.license_ok("CC BY-SA 3.0", allow) and not fetch_commons.license_ok(
        "GFDL", allow
    )


def test_config_resolves_in_fetchers(cfg: dict):
    oi = fetch_openimages.type_mids_from_config(cfg)
    assert "/m/0167gd" in oi["doll_character_statue"] and "/m/02wbm" in oi["food_contrast"]
    assert "/m/050gv4" in oi["near_food_nonfood"]
    cc = fetch_coco.type_categories_from_config(cfg)
    assert cc["person_animal"][1] == "person" and cc["vehicle_street"][3] == "car"
    cm = fetch_commons.type_categories_from_config(cfg)
    assert "Food samples" in cm["near_food_nonfood"] and "Ferris wheels" in cm["amusement_tourist"]
    for spec in cfg["hf"].values():
        assert spec["source_name"] in fc.SOURCES
    assert fc.type_budget(cfg, "near_food_nonfood") == 160
    assert fc.type_budget(cfg, "doll_character_statue", max_per_type=5) == 5


# ---------------- fetch_hf ----------------


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(paths, "EXT", tmp_path / "ext")
    return tmp_path / "ext"


def _jpeg(i: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (300 + i, 400), (i * 10, 0, 0)).save(buf, format="JPEG")
    return buf.getvalue()


def test_fetch_hf_from_snapshot(data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    snap = tmp_path / "snap" / "data"
    snap.mkdir(parents=True)
    for name, ids in (("train-00000", [0, 1, 2]), ("validation-00000", [3, 4])):
        table = pa.table(
            {
                "image": [{"bytes": _jpeg(i), "path": f"{i}.jpg"} for i in ids],
                "ground_truth": ["{}"] * len(ids),
            }
        )
        pq.write_table(table, snap / f"{name}.parquet")
    monkeypatch.setattr(
        fetch_hf,
        "snapshot",
        lambda spec, cache_dir=None: (_ for _ in ()).throw(AssertionError("no network")),
    )
    spec = {
        "hf_id": "naver-clova-ix/cord-v2",
        "revision": "7f0115a4b758a71d6473b8d085751692da2fef98",
        "license_short": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_url": "https://huggingface.co/datasets/naver-clova-ix/cord-v2",
        "image_column": "image",
        "n_take": 4,
        "subject_type": "texture_document_screen",
        "source_name": "cord",
    }
    assert fetch_hf.fetch_one("cord", spec, dry_run=True)["n_downloaded"] == 0
    res = fetch_hf.fetch_one("cord", spec, snapshot_dir=snap.parent)
    assert res["n_downloaded"] == 4
    df = fc.read_manifest("cord")
    assert df.columns == list(fc.MANIFEST_COLUMNS)
    assert df["ext_id"].to_list() == [
        "train-00000:000000",
        "train-00000:000001",
        "train-00000:000002",
        "validation-00000:000000",
    ]
    r = df.to_dicts()[0]
    assert r["source"] == "cord" and r["subject_type_hint"] == "texture_document_screen"
    assert r["license_short"] == "CC BY 4.0" and r["attribution"].endswith(
        "@7f0115a4b758a71d6473b8d085751692da2fef98"
    )
    assert r["split_hint"] == "train" and r["width"] == 300
    assert fetch_hf.fetch_one("cord", spec, snapshot_dir=snap.parent)["n_skipped"] == 4


# ---------------- lock writer ----------------


def test_write_ood_lock(data_root: Path, tmp_path: Path):
    import write_ood_lock as wl

    rows = [
        fc.new_manifest_row(
            source="coco",
            ext_id="000000000002",
            subject_type_hint="person_animal",
            candidate_label="person",
            split_hint="val2017",
            path="ext/coco/images/2.jpg",
            sha256="b" * 64,
            width=1,
            height=1,
            license_short="CC BY 2.0",
            license_url="u",
            attribution="a",
            source_url="http://f/2",
        ),
        fc.new_manifest_row(
            source="coco",
            ext_id="000000000001",
            subject_type_hint="vehicle_street",
            candidate_label="car",
            split_hint="val2017",
            path="ext/coco/images/1.jpg",
            sha256="a" * 64,
            width=1,
            height=1,
            license_short="CC BY 2.0",
            license_url="u",
            attribution="a",
            source_url=None,
        ),
    ]
    fc.write_manifest("coco", rows)
    fc.write_manifest(
        "commons",
        [
            fc.new_manifest_row(
                source="commons",
                ext_id="X.jpg",
                subject_type_hint="doll_character_statue",
                candidate_label="Teddy bears",
                split_hint="",
                path="ext/commons/images/X.jpg.jpg",
                sha256="c" * 64,
                width=1,
                height=1,
                license_short="CC0",
                license_url="",
                attribution="",
                source_url="s",
            )
        ],
    )
    fc.write_manifest("cord", [])  # 空マニフェストは lock に出ない

    manifests = fc.read_all_manifests()
    assert set(manifests) == {"coco", "commons", "cord"}
    lock = wl.build_lock(manifests)
    out = wl.write_lock(lock, tmp_path / "ood_sources.lock")
    loaded = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert set(loaded["sources"]) == {"coco", "commons"}
    coco = loaded["sources"]["coco"]
    assert coco["n"] == 2 and coco["by_type"] == {"person_animal": 1, "vehicle_street": 1}
    assert [r["ext_id"] for r in coco["images"]] == ["000000000001", "000000000002"]  # ext_id 昇順
    assert coco["images"][0] == {
        "ext_id": "000000000001",
        "sha256": "a" * 64,
        "license_short": "CC BY 2.0",
        "source_url": "",
    }
    assert set(coco["images"][1]) == set(wl.LOCK_FIELDS)  # path や画像は含めない
    assert loaded["config"] == "configs/ood_sources.yaml"
