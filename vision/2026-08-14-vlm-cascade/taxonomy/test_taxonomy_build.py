"""taxonomy_build / taxonomy_viz のユニットテスト（オフライン）。

CLAUDE.md: taxonomy 検証は「参照を壊した入力で止まる負のテスト」を含めてユニットテスト必須。
各負のテストは (a) TaxonomyError で止まる (b) 生成物を 1 つも書かない、の両方を確認する。
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from rdflib import RDF, SKOS, Graph

import taxonomy_build as tb
import taxonomy_viz as tv

EXPECTED_ORDER = ["food", "drink", "menu", "inside", "outside"]


@pytest.fixture(scope="module")
def raw_yaml() -> dict[str, Any]:
    with tb.DEFAULT_YAML.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump(data: dict[str, Any], tmp_path: Path) -> Path:
    p = tmp_path / "taxonomy.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def _find(data: dict[str, Any], node_id: str) -> dict[str, Any]:
    return next(c for c in data["concepts"] if c["id"] == node_id)


def _assert_nothing_written(out: Path) -> None:
    assert not out.exists() or not any(out.iterdir()), list(out.iterdir())


# --------------------------------------------------------------------------- 正常系


def test_build_writes_three_files_and_prompts_have_fixed_order(tmp_path: Path) -> None:
    out = tmp_path / "out"
    written = tb.build(tb.DEFAULT_YAML, out)
    assert set(written) == set(tb.OUTPUT_NAMES)
    for p in written.values():
        assert p.exists() and p.stat().st_size > 0

    prompts = json.loads((out / "prompts.json").read_text(encoding="utf-8"))
    assert prompts["scheme_version"] == "0.1.0"
    assert prompts["class_order"] == EXPECTED_ORDER
    assert list(prompts["classes"]) == EXPECTED_ORDER
    for cid, cls in prompts["classes"].items():
        assert 3 <= len(cls["prompts"]) <= 5, cid
        assert all(p["source"] == cid for p in cls["prompts"])  # v0.1: 第3階層なし
        assert cls["branch"] in {"food_branch", "non_food_branch"}
    assert prompts["classes"]["food"]["branch"] == "food_branch"
    assert prompts["classes"]["inside"]["branch"] == "non_food_branch"


def test_label_master_derives_level1_from_broader(tmp_path: Path) -> None:
    tb.build(tb.DEFAULT_YAML, tmp_path)
    with (tmp_path / "label_master.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["node_id"] for r in rows][:2] == ["food_branch", "non_food_branch"]
    by_id = {r["node_id"]: r for r in rows}
    assert by_id["food"]["derived_level1"] == "food_branch"
    assert by_id["food"]["broader"] == "food_branch"
    for cid in ("drink", "menu", "inside", "outside"):
        assert by_id[cid]["derived_level1"] == "non_food_branch"
    assert by_id["food_branch"]["broader"] == ""
    assert by_id["food_branch"]["derived_level1"] == "food_branch"
    assert set(rows[0]) == {
        "node_id",
        "level",
        "prefLabel",
        "definition",
        "broader",
        "derived_level1",
    }


def test_ttl_roundtrip(tmp_path: Path) -> None:
    tb.build(tb.DEFAULT_YAML, tmp_path)
    tx = tb.load_and_validate(tb.DEFAULT_YAML)
    g = Graph()
    g.parse(tmp_path / "taxonomy.ttl", format="turtle")
    concepts = set(g.subjects(RDF.type, SKOS.Concept))
    assert len(concepts) == len(tx.concepts) == 7
    schemes = set(g.subjects(RDF.type, SKOS.ConceptScheme))
    assert len(schemes) == 1
    food = tb.concept_uri(tx, "food")
    assert set(g.objects(food, SKOS.broader)) == {tb.concept_uri(tx, "food_branch")}
    alts = [o for o in g.objects(food, SKOS.altLabel)]
    assert alts and all(o.language == "en" for o in alts)
    pref = next(g.objects(food, SKOS.prefLabel))
    assert pref.language == "ja"
    # 往復チェック自体が壊れた ttl を弾くこと
    with pytest.raises(tb.TaxonomyError):
        tb.roundtrip_check("@prefix skos: <http://www.w3.org/2004/02/skos/core#> .\n", tx)


def test_level3_children_contribute_prompts_with_source(
    raw_yaml: dict[str, Any], tmp_path: Path
) -> None:
    data = copy.deepcopy(raw_yaml)
    data["concepts"].append(
        {
            "id": "menu_with_food_photos",
            "level": 3,
            "broader": "menu",
            "prefLabel": "料理写真入りメニューページ",
            "altLabel": {"en": ["a photo of a menu page with pictures of dishes"]},
            "definition": "料理写真が載ったメニューページ。",
        }
    )
    out = tmp_path / "out"
    tb.build(_dump(data, tmp_path), out)
    prompts = json.loads((out / "prompts.json").read_text(encoding="utf-8"))
    menu = prompts["classes"]["menu"]["prompts"]
    assert menu[-1] == {
        "text": "a photo of a menu page with pictures of dishes",
        "source": "menu_with_food_photos",
    }
    assert all(p["source"] == "menu" for p in menu[:-1])
    with (out / "label_master.csv").open(encoding="utf-8", newline="") as f:
        rows = {r["node_id"]: r for r in csv.DictReader(f)}
    assert rows["menu_with_food_photos"]["derived_level1"] == "non_food_branch"


def test_cli_main_returns_zero(tmp_path: Path) -> None:
    assert tb.main(["--out", str(tmp_path)]) == 0
    assert (tmp_path / "prompts.json").exists()


# --------------------------------------------------------------------------- 負のテスト


def _mutate_broken_broader(d: dict[str, Any]) -> None:
    _find(d, "menu")["broader"] = "no_such_branch"


def _mutate_duplicate_id(d: dict[str, Any]) -> None:
    dup = copy.deepcopy(_find(d, "drink"))
    dup["prefLabel"] = "飲み物（重複）"
    d["concepts"].append(dup)


def _mutate_only_four_level2(d: dict[str, Any]) -> None:
    d["concepts"] = [c for c in d["concepts"] if c["id"] != "outside"]


def _mutate_level2_without_prompts(d: dict[str, Any]) -> None:
    _find(d, "inside")["altLabel"] = {"en": []}


def _mutate_level_rule(d: dict[str, Any]) -> None:
    _find(d, "drink")["broader"] = "food"  # 第2階層の親が第2階層


def _mutate_duplicate_pref_label(d: dict[str, Any]) -> None:
    _find(d, "drink")["prefLabel"] = _find(d, "food")["prefLabel"]


def _mutate_level1_with_parent(d: dict[str, Any]) -> None:
    _find(d, "non_food_branch")["broader"] = "food_branch"


def _mutate_version_alias(d: dict[str, Any]) -> None:
    d["scheme"]["version"] = "latest"


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (_mutate_broken_broader, "存在しない"),
        (_mutate_duplicate_id, "id が重複"),
        (_mutate_only_four_level2, "5 個必要"),
        (_mutate_level2_without_prompts, "プロンプト"),
        (_mutate_level_rule, "第1階層でなければならない"),
        (_mutate_duplicate_pref_label, "prefLabel が重複"),
        (_mutate_level1_with_parent, "第1階層に broader"),
        (_mutate_version_alias, "ピン留め"),
    ],
    ids=[
        "broken_broader",
        "duplicate_id",
        "four_level2",
        "level2_without_prompts",
        "level_rule",
        "duplicate_pref_label",
        "level1_with_parent",
        "version_alias",
    ],
)
def test_invalid_yaml_raises_and_writes_nothing(
    raw_yaml: dict[str, Any], tmp_path: Path, mutate, needle: str
) -> None:
    data = copy.deepcopy(raw_yaml)
    mutate(data)
    yaml_path = _dump(data, tmp_path)
    out = tmp_path / "out"

    with pytest.raises(tb.TaxonomyError, match=needle):
        tb.build(yaml_path, out)
    _assert_nothing_written(out)

    assert tb.main(["--yaml", str(yaml_path), "--out", str(out)]) != 0
    _assert_nothing_written(out)

    with pytest.raises(tb.TaxonomyError):
        tv.viz(yaml_path, out)
    _assert_nothing_written(out)
    assert tv.main(["--yaml", str(yaml_path), "--out", str(out)]) != 0
    _assert_nothing_written(out)


# --------------------------------------------------------------------------- rulebook 被覆


def _write_rulebook(tmp_path: Path, targets: list[str]) -> Path:
    lines = ["# rulebook (test)", ""]
    lines += [f"- rule: R{i} | target: {t} | when: ..." for i, t in enumerate(targets, 1)]
    p = tmp_path / "rulebook.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_rulebook_coverage_ok(tmp_path: Path) -> None:
    rb = _write_rulebook(tmp_path, EXPECTED_ORDER)
    written = tb.build(tb.DEFAULT_YAML, tmp_path / "out", rulebook=rb)
    assert set(written) == set(tb.OUTPUT_NAMES)


@pytest.mark.parametrize(
    ("targets", "needle"),
    [
        (["food", "drink", "menu", "inside"], "outside"),  # 不足
        (EXPECTED_ORDER + ["food"], "food"),  # 重複
        (EXPECTED_ORDER + ["other"], "other"),  # 第2階層外
    ],
)
def test_rulebook_coverage_fails_and_writes_nothing(
    tmp_path: Path, targets: list[str], needle: str
) -> None:
    rb = _write_rulebook(tmp_path, targets)
    out = tmp_path / "out"
    with pytest.raises(tb.TaxonomyError, match=needle):
        tb.build(tb.DEFAULT_YAML, out, rulebook=rb)
    _assert_nothing_written(out)


def test_rulebook_absent_is_skipped_with_notice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    written = tb.build(tb.DEFAULT_YAML, tmp_path / "out", rulebook=tmp_path / "missing_rulebook.md")
    assert set(written) == set(tb.OUTPUT_NAMES)
    assert "notice" in capsys.readouterr().out


# --------------------------------------------------------------------------- viz


def test_viz_writes_tree_md_and_skips_svg_without_dot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr(tv.shutil, "which", lambda _name: None)
    written = tv.viz(tb.DEFAULT_YAML, tmp_path)
    assert set(written) == {"taxonomy_tree.md"}
    assert not (tmp_path / "taxonomy.svg").exists()
    assert "dot" in capsys.readouterr().err

    md = (tmp_path / "taxonomy_tree.md").read_text(encoding="utf-8")
    assert "```mermaid" in md and "flowchart TD" in md
    for cid in EXPECTED_ORDER + ["food_branch", "non_food_branch"]:
        assert f"{cid}" in md
    assert "food_branch --> food" in md
    assert "non_food_branch --> outside" in md
    assert "| `food` | `food` | a photo of food served at a restaurant |" in md


def test_viz_writes_svg_when_dot_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tv.shutil, "which", lambda _name: "/fake/bin/dot")

    class _Res:
        returncode = 0
        stdout = "<svg xmlns='http://www.w3.org/2000/svg'></svg>"
        stderr = ""

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        calls.append(cmd)
        assert "digraph taxonomy" in kwargs["input"]
        return _Res()

    monkeypatch.setattr(tv.subprocess, "run", fake_run)
    written = tv.viz(tb.DEFAULT_YAML, tmp_path)
    assert set(written) == {"taxonomy_tree.md", "taxonomy.svg"}
    assert calls == [["/fake/bin/dot", "-Tsvg"]]
    assert (tmp_path / "taxonomy.svg").read_text(encoding="utf-8").startswith("<svg")


def test_viz_cli_exit_zero_without_dot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tv.shutil, "which", lambda _name: None)
    assert tv.main(["--out", str(tmp_path)]) == 0
    assert (tmp_path / "taxonomy_tree.md").exists()


# --------------------------------------------------------------------------- 生成物が正本と同期していること


def test_committed_artifacts_are_up_to_date(tmp_path: Path) -> None:
    """taxonomy/ にコミットされた生成物が現在の YAML から再生成したものと一致する（手編集検出）。"""
    tb.build(tb.DEFAULT_YAML, tmp_path)
    for name in tb.OUTPUT_NAMES:
        committed = tb.HERE / name
        assert committed.exists(), f"{name} が未生成: make taxonomy を実行する"
        assert committed.read_text(encoding="utf-8") == (tmp_path / name).read_text(
            encoding="utf-8"
        ), f"{name} が taxonomy.yaml と不一致: make taxonomy を実行する"
    committed_md = tb.HERE / "taxonomy_tree.md"
    assert committed_md.exists(), "taxonomy_tree.md が未生成: make taxonomy を実行する"
    assert committed_md.read_text(encoding="utf-8") == tv.render_tree_md(
        tb.load_and_validate(tb.DEFAULT_YAML)
    )
