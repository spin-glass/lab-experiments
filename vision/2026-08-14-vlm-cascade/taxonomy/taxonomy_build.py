"""taxonomy.yaml の検証と生成物（prompts.json / label_master.csv / taxonomy.ttl）の生成。

docs/taxonomy.md §6・design.md §4 の実装。正本は ``taxonomy/taxonomy.yaml`` のみで、
本スクリプトは次を行う。

1. 検証 — id / prefLabel の一意性、broader 参照先の存在、階層規則（第1階層は無親・
   第2階層の親は第1階層・第3階層の親は第2階層）、第2階層がちょうど運用クラス数（5）で
   あること、各運用クラスにプロンプト源（自身または第3階層の子）があること、
   ``--rulebook`` 指定時は優先規則が第2階層を被覆する（各クラスが ≥ 1 規則の target）こと。
2. 生成 — 検証に 1 件でも失敗したら **何も書かずに** 異常終了する（TaxonomyError / exit 1）。
   全出力を先にメモリ上で組み立て、Turtle は rdflib で往復パースして概念・版・broader・
   prefLabel・altLabel（プロンプト）が一致することを確認してから、各ファイルを一時ファイル経由で
   原子的に置換する（途中失敗で新旧が混在しない）。

第1階層（food / non-food）はゴールドに記録せず導出のみ（taxonomy.md T2）。
label_master.csv の ``derived_level1`` 列は ``broader`` を辿って機械的に導出し、手で持たせない。

CLI::

    uv run python taxonomy/taxonomy_build.py [--yaml path] [--out dir] [--rulebook path]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from rdflib import DCTERMS, RDF, SKOS, Graph, Literal, Namespace, URIRef

from cascade.config import PinError, validate_pins

HERE = Path(__file__).resolve().parent
DEFAULT_YAML = HERE / "taxonomy.yaml"
DEFAULT_RULEBOOK = HERE.parent / "rulebook.md"

#: 第2階層（運用クラス）の数。design.md §2 の Yelp 5 クラスに一致させる
OPERATIONAL_CLASS_COUNT = 5
#: プロンプトの言語タグ（altLabel の言語キー）
PROMPT_LANG = "en"
#: rulebook.md の優先規則行の文法（design.md §4(c) の優先規則は曖昧ペアごとなので、同じ第2階層が
#: 複数規則の target になるのが正常）:
#:   ``- rule: <id> | target: <level2_id> | when: ...``  を 1 規則 1 行で書く（大文字小文字は無視）。
#:   target は ``target: food`` / ``target: `food```（バッククォート可）。1 行に複数の target を書いてよい。
RULE_LINE_RE = re.compile(r"^-\s*rule:", re.IGNORECASE)
RULE_TARGET_RE = re.compile(r"target:\s*`?([a-z][a-z0-9_]*)`?(?![a-z0-9_-])", re.IGNORECASE)
NODE_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")

#: Turtle の名前空間。実在ドメインに依存しない URN
TX = Namespace("urn:lab-vlm-cascade:taxonomy:")


class TaxonomyError(ValueError):
    """taxonomy.yaml の検証失敗。1 件でも起きたら生成しない。"""


@dataclass(frozen=True)
class Concept:
    id: str
    level: int
    broader: str | None
    prefLabel: str
    definition: str
    scopeNote: str | None = None
    prompts: tuple[str, ...] = ()


@dataclass
class Taxonomy:
    scheme_id: str
    version: str
    scheme_pref_label: str | None
    scheme_definition: str | None
    change_notes: list[str]
    constraints: dict[str, Any]
    concepts: list[Concept]
    by_id: dict[str, Concept] = field(init=False)

    def __post_init__(self) -> None:
        self.by_id = {c.id: c for c in self.concepts}

    def level(self, n: int) -> list[Concept]:
        """文書順を保った level == n の概念一覧。第2階層の順序が class_order になる。"""
        return [c for c in self.concepts if c.level == n]

    def children(self, node_id: str) -> list[Concept]:
        return [c for c in self.concepts if c.broader == node_id]

    def derived_level1(self, node_id: str) -> str:
        """broader を辿って第1階層 id を導出する（手で持たせない。T2）。"""
        c = self.by_id[node_id]
        while c.broader is not None:
            c = self.by_id[c.broader]
        return c.id

    def prompts_of(self, node_id: str) -> list[dict[str, str]]:
        """自身と第3階層の子のプロンプトを source 付きで集約する（文書順）。"""
        out = [{"text": t, "source": node_id} for t in self.by_id[node_id].prompts]
        for child in self.children(node_id):
            out.extend({"text": t, "source": child.id} for t in child.prompts)
        return out


# --------------------------------------------------------------------------- 読み込み


def _as_str(v: Any, where: str, *, required: bool) -> str | None:
    if v is None:
        if required:
            raise TaxonomyError(f"{where}: 必須項目が未指定")
        return None
    if not isinstance(v, str) or not v.strip():
        raise TaxonomyError(f"{where}: 空でない文字列が必要（got {v!r}）")
    return v.strip()


def _prompts_of(raw: Any, where: str) -> tuple[str, ...]:
    """altLabel から言語 ``en`` のプロンプト列を取り出す。list なら en とみなす。"""
    if raw is None:
        return ()
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        for lang in raw:
            if not isinstance(lang, str):
                raise TaxonomyError(f"{where}.altLabel: 言語キーは文字列（got {lang!r}）")
        items = raw.get(PROMPT_LANG) or []
        if not isinstance(items, list):
            raise TaxonomyError(f"{where}.altLabel.{PROMPT_LANG}: list が必要")
    else:
        raise TaxonomyError(f"{where}.altLabel: list か {{lang: list}} が必要")
    out: list[str] = []
    for i, t in enumerate(items):
        s = _as_str(t, f"{where}.altLabel.{PROMPT_LANG}[{i}]", required=True)
        assert s is not None
        out.append(s)
    return tuple(out)


def load_taxonomy(path: str | Path) -> Taxonomy:
    """YAML を読み、型を整えて Taxonomy にする（構造検証は validate で行う）。"""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TaxonomyError(f"{path}: トップレベルは mapping が必要")

    scheme = data.get("scheme")
    if not isinstance(scheme, dict):
        raise TaxonomyError("scheme: mapping が必要")
    scheme_id = _as_str(scheme.get("id"), "scheme.id", required=True)
    version = _as_str(scheme.get("version"), "scheme.version", required=True)
    assert scheme_id is not None and version is not None
    change_notes = scheme.get("changeNote") or []
    if not isinstance(change_notes, list):
        raise TaxonomyError("scheme.changeNote: list が必要")

    constraints = data.get("constraints") or {}
    if not isinstance(constraints, dict):
        raise TaxonomyError("constraints: mapping が必要")

    raw_concepts = data.get("concepts")
    if not isinstance(raw_concepts, list) or not raw_concepts:
        raise TaxonomyError("concepts: 空でない list が必要")

    concepts: list[Concept] = []
    for i, rc in enumerate(raw_concepts):
        where = f"concepts[{i}]"
        if not isinstance(rc, dict):
            raise TaxonomyError(f"{where}: mapping が必要")
        cid = _as_str(rc.get("id"), f"{where}.id", required=True)
        assert cid is not None
        where = f"concepts[{cid}]"
        level = rc.get("level")
        if not isinstance(level, int) or isinstance(level, bool) or level not in (1, 2, 3):
            raise TaxonomyError(f"{where}.level: 1 | 2 | 3 が必要（got {level!r}）")
        pref = _as_str(rc.get("prefLabel"), f"{where}.prefLabel", required=True)
        definition = _as_str(rc.get("definition"), f"{where}.definition", required=True)
        assert pref is not None and definition is not None
        concepts.append(
            Concept(
                id=cid,
                level=level,
                broader=_as_str(rc.get("broader"), f"{where}.broader", required=False),
                prefLabel=pref,
                definition=definition,
                scopeNote=_as_str(rc.get("scopeNote"), f"{where}.scopeNote", required=False),
                prompts=_prompts_of(rc.get("altLabel"), where),
            )
        )

    return Taxonomy(
        scheme_id=scheme_id,
        version=version,
        scheme_pref_label=_as_str(scheme.get("prefLabel"), "scheme.prefLabel", required=False),
        scheme_definition=_as_str(scheme.get("definition"), "scheme.definition", required=False),
        change_notes=[str(n) for n in change_notes],
        constraints=dict(constraints),
        concepts=concepts,
    )


# --------------------------------------------------------------------------- 検証


def validate(tx: Taxonomy, *, rulebook: str | Path | None = None) -> list[str]:
    """全検証を実行し、失敗を list で返す（空 = 合格）。呼び出し側で TaxonomyError にする。"""
    errors: list[str] = []

    if not SEMVER_RE.match(tx.version):
        errors.append(f"scheme.version = {tx.version!r} は semver（X.Y.Z）ではない")
    try:
        validate_pins({"taxonomy_version": tx.version})
    except PinError as e:
        errors.append(str(e))

    # id / prefLabel の一意性と書式
    for cid, n in Counter(c.id for c in tx.concepts).items():
        if n > 1:
            errors.append(f"id が重複: {cid} ({n} 件)")
    for label, n in Counter(c.prefLabel for c in tx.concepts).items():
        if n > 1:
            errors.append(f"prefLabel が重複: {label!r} ({n} 件)")
    for c in tx.concepts:
        if not NODE_ID_RE.match(c.id):
            errors.append(f"id が snake_case ではない: {c.id!r}")

    # broader の存在と階層規則
    ids = {c.id for c in tx.concepts}
    for c in tx.concepts:
        if c.level == 1:
            if c.broader is not None:
                errors.append(f"{c.id}: 第1階層に broader は置けない（got {c.broader!r}）")
            continue
        if c.broader is None:
            errors.append(f"{c.id}: 第{c.level}階層には broader が必要")
            continue
        if c.broader not in ids:
            errors.append(f"{c.id}: broader = {c.broader!r} が存在しない")
            continue
        if c.broader == c.id:
            errors.append(f"{c.id}: broader が自分自身")
            continue
        parent_level = tx.by_id[c.broader].level
        if parent_level != c.level - 1:
            errors.append(
                f"{c.id}: 第{c.level}階層の親は第{c.level - 1}階層でなければならない"
                f"（{c.broader} は第{parent_level}階層）"
            )

    # 第2階層の数
    level2 = tx.level(2)
    if len(level2) != OPERATIONAL_CLASS_COUNT:
        errors.append(
            f"第2階層（運用クラス）は {OPERATIONAL_CLASS_COUNT} 個必要（got {len(level2)}: "
            f"{[c.id for c in level2]}）"
        )
    if not tx.level(1):
        errors.append("第1階層が存在しない")

    # 各運用クラスにプロンプト源（自身または第3階層の子）があること
    if not errors:  # 参照が壊れていると prompts_of が KeyError になるので健全な場合のみ
        for c in level2:
            if not tx.prompts_of(c.id):
                errors.append(f"{c.id}: プロンプト（altLabel.{PROMPT_LANG}）が自身にも子にもない")
        for c in tx.level(1):
            if c.prompts:
                errors.append(f"{c.id}: 第1階層にプロンプトは置かない（否定は列挙で表す。T1）")

    # 制約ブロック（v0.1 の固定値）
    cons = tx.constraints
    for key in ("exclusive", "exhaustive"):
        if cons.get(key) is not True:
            errors.append(f"constraints.{key} は true が必要（got {cons.get(key)!r}）")
    if cons.get("derive_level1") != "parent(level2)":
        errors.append(
            f"constraints.derive_level1 は 'parent(level2)' が必要（got {cons.get('derive_level1')!r}）"
        )

    # rulebook の被覆（任意）
    if rulebook is not None and not errors:
        errors.extend(check_rulebook_coverage(Path(rulebook), [c.id for c in level2]))

    return errors


def parse_rule_targets(text: str) -> Counter[str]:
    """rulebook 本文から ``- rule:`` 行の target（小文字化）を数える。"""
    targets: Counter[str] = Counter()
    for line in text.splitlines():
        stripped = line.strip()
        if not RULE_LINE_RE.match(stripped):
            continue
        for t in RULE_TARGET_RE.findall(stripped):
            targets[t.lower()] += 1
    return targets


def check_rulebook_coverage(rulebook: Path, level2_ids: list[str]) -> list[str]:
    """rulebook.md の ``- rule: ... target: <id>`` 行が第2階層を被覆するか（各クラス ≥ 1 件）。

    同じクラスが複数規則の target になるのは正常（曖昧ペアごとの規則）。第2階層以外の target は
    エラー。ファイルが無ければ通知だけ出して空（合格）を返す。rulebook は M1 以降で導入される。
    """
    if not rulebook.exists():
        print(f"[taxonomy_build] notice: rulebook が無いので被覆検証をスキップ: {rulebook}")
        return []
    targets = parse_rule_targets(rulebook.read_text(encoding="utf-8"))
    errors: list[str] = []
    for cid in level2_ids:
        n = targets.get(cid, 0)
        if n < 1:
            errors.append(
                f"rulebook: 第2階層 {cid} を target とする優先規則が 0 件（1 件以上必要）"
            )
    for t in targets:
        if t not in level2_ids:
            errors.append(f"rulebook: target = {t!r} は第2階層ではない")
    return errors


def load_and_validate(path: str | Path, *, rulebook: str | Path | None = None) -> Taxonomy:
    tx = load_taxonomy(path)
    errors = validate(tx, rulebook=rulebook)
    if errors:
        raise TaxonomyError("taxonomy.yaml の検証に失敗:\n  - " + "\n  - ".join(errors))
    return tx


# --------------------------------------------------------------------------- 生成


def class_order(tx: Taxonomy) -> list[str]:
    """Stage1 の出力列順。第2階層の文書順で固定する。"""
    return [c.id for c in tx.level(2)]


def render_prompts_json(tx: Taxonomy) -> str:
    classes = {
        c.id: {
            "prefLabel": c.prefLabel,
            "prompts": tx.prompts_of(c.id),
            "branch": tx.derived_level1(c.id),
        }
        for c in tx.level(2)
    }
    payload = {
        "scheme_version": tx.version,
        "classes": classes,
        "class_order": class_order(tx),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def render_label_master_csv(tx: Taxonomy) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["node_id", "level", "prefLabel", "definition", "broader", "derived_level1"])
    for c in tx.concepts:
        w.writerow(
            [c.id, c.level, c.prefLabel, c.definition, c.broader or "", tx.derived_level1(c.id)]
        )
    return buf.getvalue()


def scheme_uri(tx: Taxonomy) -> URIRef:
    return TX[tx.scheme_id]


def concept_uri(tx: Taxonomy, node_id: str) -> URIRef:
    return TX[f"{tx.scheme_id}:{node_id}"]


def build_graph(tx: Taxonomy) -> Graph:
    g = Graph()
    g.bind("skos", SKOS)
    g.bind("dcterms", DCTERMS)
    g.bind("tx", TX)

    s = scheme_uri(tx)
    g.add((s, RDF.type, SKOS.ConceptScheme))
    g.add((s, DCTERMS.identifier, Literal(tx.scheme_id)))
    g.add((s, DCTERMS.hasVersion, Literal(tx.version)))
    if tx.scheme_pref_label:
        g.add((s, SKOS.prefLabel, Literal(tx.scheme_pref_label, lang="ja")))
    if tx.scheme_definition:
        g.add((s, SKOS.definition, Literal(tx.scheme_definition, lang="ja")))
    for note in tx.change_notes:
        g.add((s, SKOS.changeNote, Literal(note, lang="ja")))

    for c in tx.concepts:
        u = concept_uri(tx, c.id)
        g.add((u, RDF.type, SKOS.Concept))
        g.add((u, SKOS.inScheme, s))
        g.add((u, SKOS.notation, Literal(c.id)))
        g.add((u, SKOS.prefLabel, Literal(c.prefLabel, lang="ja")))
        g.add((u, SKOS.definition, Literal(c.definition, lang="ja")))
        if c.scopeNote:
            g.add((u, SKOS.scopeNote, Literal(c.scopeNote, lang="ja")))
        for p in c.prompts:
            g.add((u, SKOS.altLabel, Literal(p, lang=PROMPT_LANG)))
        if c.broader is None:
            g.add((s, SKOS.hasTopConcept, u))
            g.add((u, SKOS.topConceptOf, s))
        else:
            b = concept_uri(tx, c.broader)
            g.add((u, SKOS.broader, b))
            g.add((b, SKOS.narrower, u))
    return g


def render_ttl(tx: Taxonomy) -> str:
    return build_graph(tx).serialize(format="turtle")


def roundtrip_check(ttl_text: str, tx: Taxonomy) -> None:
    """Turtle を再パースして概念数・版・broader・prefLabel・altLabel が一致することを確認する
    （標準形式への出口の保証）。"""
    g = Graph()
    g.parse(data=ttl_text, format="turtle")
    n_concepts = len(set(g.subjects(RDF.type, SKOS.Concept)))
    if n_concepts != len(tx.concepts):
        raise TaxonomyError(
            f"Turtle 往復パースで概念数が不一致: yaml={len(tx.concepts)} ttl={n_concepts}"
        )
    versions = {str(o) for o in g.objects(scheme_uri(tx), DCTERMS.hasVersion)}
    if versions != {tx.version}:
        raise TaxonomyError(f"Turtle 往復パースで版が不一致: yaml={tx.version} ttl={versions}")
    for c in tx.concepts:
        u = concept_uri(tx, c.id)
        if c.broader is not None:
            parent = set(g.objects(u, SKOS.broader))
            if parent != {concept_uri(tx, c.broader)}:
                raise TaxonomyError(f"Turtle 往復パースで broader が不一致: {c.id}")
        pref = {str(o) for o in g.objects(u, SKOS.prefLabel)}
        if pref != {c.prefLabel}:
            raise TaxonomyError(
                f"Turtle 往復パースで prefLabel が不一致: {c.id} yaml={c.prefLabel!r} ttl={pref}"
            )
        alts = {
            str(o)
            for o in g.objects(u, SKOS.altLabel)
            if getattr(o, "language", None) == PROMPT_LANG
        }
        if alts != set(c.prompts):
            raise TaxonomyError(
                f"Turtle 往復パースで altLabel@{PROMPT_LANG} が不一致: {c.id} "
                f"yaml={sorted(c.prompts)} ttl={sorted(alts)}"
            )


OUTPUT_NAMES = ("prompts.json", "label_master.csv", "taxonomy.ttl")


def write_atomic(path: Path, text: str) -> None:
    """``<name>.tmp`` に書いてから ``os.replace`` で置換する（書きかけを残さない）。"""
    write_all_atomic({path: text})


def write_all_atomic(files: dict[Path, str]) -> None:
    """複数ファイルを全件 ``<name>.tmp`` に書き終えてから順に ``os.replace`` する。

    tmp の書き込みで失敗したら tmp を全部消して何も置換しない（新旧混在を作らない）。
    """
    tmps: dict[Path, Path] = {p: p.with_name(p.name + ".tmp") for p in files}
    try:
        for p, text in files.items():
            tmps[p].write_text(text, encoding="utf-8")
    except BaseException:
        for t in tmps.values():
            t.unlink(missing_ok=True)
        raise
    for p, t in tmps.items():
        os.replace(t, p)


def build(
    yaml_path: str | Path = DEFAULT_YAML,
    out_dir: str | Path = HERE,
    rulebook: str | Path | None = None,
) -> dict[str, Path]:
    """検証 → 全生成物をメモリで組み立て → 往復確認 → 原子的に書き出し。検証失敗時は何も書かない。"""
    tx = load_and_validate(yaml_path, rulebook=rulebook)
    outputs = {
        "prompts.json": render_prompts_json(tx),
        "label_master.csv": render_label_master_csv(tx),
        "taxonomy.ttl": render_ttl(tx),
    }
    roundtrip_check(outputs["taxonomy.ttl"], tx)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {name: out / name for name in outputs}
    write_all_atomic({written[name]: text for name, text in outputs.items()})
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--yaml", type=Path, default=DEFAULT_YAML, help="正本 YAML")
    ap.add_argument("--out", type=Path, default=HERE, help="生成物の出力先ディレクトリ")
    ap.add_argument(
        "--rulebook",
        type=Path,
        default=None,
        help=f"rulebook.md（優先規則の被覆検証。省略時は {DEFAULT_RULEBOOK} が存在すれば使う）",
    )
    args = ap.parse_args(argv)
    rulebook = args.rulebook if args.rulebook is not None else DEFAULT_RULEBOOK
    try:
        written = build(args.yaml, args.out, rulebook=rulebook)
    except TaxonomyError as e:
        print(f"[taxonomy_build] FAILED — 生成物は書いていない\n{e}", file=sys.stderr)
        return 1
    for name, p in written.items():
        print(f"[taxonomy_build] wrote {name}: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
