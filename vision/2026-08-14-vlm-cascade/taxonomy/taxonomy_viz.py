"""taxonomy.yaml の検証と可視化（taxonomy_tree.md / taxonomy.svg）。

docs/taxonomy.md §6 の ``taxonomy_viz.py``。``taxonomy_build.py`` と同じ検証を通してから、

- ``taxonomy_tree.md``: Mermaid flowchart TD（全画像 → 第1階層 → 第2階層 → 第3階層）と
  運用クラスごとのプロンプト表。GitHub がそのまま描画する。**常に生成する**
- ``taxonomy.svg``: Graphviz ``dot`` が PATH にある場合のみ生成する。無ければ警告を出して
  終了コード 0（開発機に dot が無くても build/viz の束ねが止まらないようにする）

検証に失敗した場合は何も書かずに終了コード 1。dot の実行を先に済ませてから md / svg を
一時ファイル経由で原子的に書く（dot の OSError で md だけ更新された状態を残さない）。
ノード id は概念 id に ``n_`` を前置する（Mermaid / DOT の予約語 ``end`` / ``class`` 等や
ルートノードとの衝突を避ける）。

CLI::

    uv run python taxonomy/taxonomy_viz.py [--yaml path] [--out dir] [--rulebook path]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from taxonomy_build import (
    DEFAULT_RULEBOOK,
    DEFAULT_YAML,
    HERE,
    Concept,
    Taxonomy,
    TaxonomyError,
    class_order,
    load_and_validate,
    write_all_atomic,
    write_atomic,
)

ROOT_ID = "root"
ROOT_LABEL = "全画像"
LEVEL_NAMES = {
    1: "第1階層（ブランチ・導出のみ）",
    2: "第2階層（運用クラス・ゴールドに記録）",
    3: "第3階層（プロンプト用サブクラス）",
}


def _mermaid_escape(s: str) -> str:
    return s.replace('"', "&quot;")


def node_id(concept_id: str) -> str:
    """図のノード id。予約語・ルートノードと衝突しないよう ``n_`` を前置する。"""
    return f"n_{concept_id}"


def _mermaid_node(c: Concept) -> str:
    label = _mermaid_escape(f"{c.prefLabel}<br/><code>{c.id}</code>")
    nid = node_id(c.id)
    if c.level == 1:
        return f'{nid}(["{label}"])'
    if c.level == 2:
        return f'{nid}["{label}"]'
    return f'{nid}[/"{label}"/]'


def render_mermaid(tx: Taxonomy) -> str:
    lines = ["flowchart TD", f'    {ROOT_ID}(("{ROOT_LABEL}"))']
    for level in (1, 2, 3):
        nodes = tx.level(level)
        if not nodes:
            continue
        lines.append(f'    subgraph L{level}["{LEVEL_NAMES[level]}"]')
        lines.append("        direction LR")
        for c in nodes:
            lines.append(f"        {_mermaid_node(c)}")
        lines.append("    end")
    for c in tx.concepts:
        parent = ROOT_ID if c.broader is None else node_id(c.broader)
        lines.append(f"    {parent} --> {node_id(c.id)}")
    lines.append("    classDef l1 fill:#eeeeee,stroke:#555;")
    lines.append("    classDef food fill:#ffe8cc,stroke:#c60;")
    lines.append("    classDef nonfood fill:#dde8ff,stroke:#36c;")
    l1 = [node_id(c.id) for c in tx.level(1)]
    if l1:
        lines.append(f"    class {','.join(l1)} l1;")
    food_ids = [
        node_id(c.id)
        for c in tx.concepts
        if c.level >= 2 and tx.derived_level1(c.id) == "food_branch"
    ]
    non_ids = [
        node_id(c.id)
        for c in tx.concepts
        if c.level >= 2 and tx.derived_level1(c.id) != "food_branch"
    ]
    if food_ids:
        lines.append(f"    class {','.join(food_ids)} food;")
    if non_ids:
        lines.append(f"    class {','.join(non_ids)} nonfood;")
    return "\n".join(lines) + "\n"


def render_tree_md(tx: Taxonomy) -> str:
    parts = [
        f"# taxonomy tree — {tx.scheme_id} v{tx.version}",
        "",
        "生成物（`taxonomy_viz.py`）。手編集禁止。正本は `taxonomy.yaml`。",
        "",
        "```mermaid",
        render_mermaid(tx).rstrip("\n"),
        "```",
        "",
        "## 概念一覧",
        "",
        "| level | node_id | prefLabel | broader | derived_level1 | definition |",
        "|---|---|---|---|---|---|",
    ]
    for c in tx.concepts:
        definition = c.definition.replace("|", "\\|").replace("\n", " ")
        parts.append(
            f"| {c.level} | `{c.id}` | {c.prefLabel} | {c.broader or '—'} | "
            f"`{tx.derived_level1(c.id)}` | {definition} |"
        )
    parts += [
        "",
        f"## Stage1 プロンプト（class_order: {', '.join(class_order(tx))}）",
        "",
        "| class | source | prompt (en) |",
        "|---|---|---|",
    ]
    for cid in class_order(tx):
        for p in tx.prompts_of(cid):
            parts.append(f"| `{cid}` | `{p['source']}` | {p['text']} |")
    if tx.change_notes:
        parts += ["", "## changeNote", ""]
        parts += [f"- {n}" for n in tx.change_notes]
    return "\n".join(parts) + "\n"


def _dot_escape(s: str) -> str:
    """DOT の二重引用符文字列: ``\\`` と ``"`` をエスケープし、実改行は ``\\n`` にする。"""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_dot(tx: Taxonomy) -> str:
    lines = [
        "digraph taxonomy {",
        "    rankdir=TB;",
        '    node [shape=box, style="rounded,filled", fontname="Helvetica"];',
        f'    {ROOT_ID} [label="{ROOT_LABEL}", shape=doublecircle, fillcolor="#ffffff"];',
    ]
    fill = {1: "#eeeeee", 2: "#ffffff", 3: "#f7f7ff"}
    for c in tx.concepts:
        color = "#ffe8cc" if tx.derived_level1(c.id) == "food_branch" else "#dde8ff"
        if c.level == 1:
            color = fill[1]
        label = _dot_escape(f"{c.prefLabel}\n{c.id}")
        lines.append(f'    {node_id(c.id)} [label="{label}", fillcolor="{color}"];')
    for c in tx.concepts:
        parent = ROOT_ID if c.broader is None else node_id(c.broader)
        lines.append(f"    {parent} -> {node_id(c.id)};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def render_svg(dot_text: str) -> str | None:
    """dot が PATH にあれば SVG 文字列を返す。無い／失敗なら警告して None（ファイルは書かない）。"""
    dot = shutil.which("dot")
    if dot is None:
        print(
            "[taxonomy_viz] warning: Graphviz `dot` が PATH に無いため taxonomy.svg は生成しない"
            "（taxonomy_tree.md の Mermaid 図で代替。brew install graphviz で有効化）",
            file=sys.stderr,
        )
        return None
    try:
        res = subprocess.run(
            [dot, "-Tsvg"], input=dot_text, capture_output=True, text=True, check=False
        )
    except OSError as e:
        print(
            f"[taxonomy_viz] warning: dot を実行できないため taxonomy.svg は生成しない: {e}",
            file=sys.stderr,
        )
        return None
    if res.returncode != 0:
        print(
            f"[taxonomy_viz] warning: dot が失敗したため taxonomy.svg は生成しない\n{res.stderr}",
            file=sys.stderr,
        )
        return None
    return res.stdout


def write_svg(dot_text: str, svg_path: Path) -> bool:
    """dot が PATH にあれば SVG を書いて True、無ければ警告して False。"""
    svg = render_svg(dot_text)
    if svg is None:
        return False
    write_atomic(svg_path, svg)
    return True


def viz(
    yaml_path: str | Path = DEFAULT_YAML,
    out_dir: str | Path = HERE,
    rulebook: str | Path | None = None,
) -> dict[str, Path]:
    """検証 → dot 実行（先）→ taxonomy_tree.md（常に）→ taxonomy.svg（dot がある場合のみ）を原子的に書く。"""
    tx = load_and_validate(yaml_path, rulebook=rulebook)
    md_text = render_tree_md(tx)
    dot_text = render_dot(tx)
    svg_text = render_svg(dot_text)  # 書き込みの前に外部コマンドを済ませる

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {"taxonomy_tree.md": out / "taxonomy_tree.md"}
    files = {written["taxonomy_tree.md"]: md_text}
    if svg_text is not None:
        written["taxonomy.svg"] = out / "taxonomy.svg"
        files[written["taxonomy.svg"]] = svg_text
    write_all_atomic(files)
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
        help="rulebook.md（省略時は既定パスが存在すれば使う）",
    )
    args = ap.parse_args(argv)
    rulebook = args.rulebook if args.rulebook is not None else DEFAULT_RULEBOOK
    try:
        written = viz(args.yaml, args.out, rulebook=rulebook)
    except TaxonomyError as e:
        print(f"[taxonomy_viz] FAILED — 生成物は書いていない\n{e}", file=sys.stderr)
        return 1
    for name, p in written.items():
        print(f"[taxonomy_viz] wrote {name}: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
