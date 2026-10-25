# taxonomy/ — クラス定義の正本と生成物

設計ノートは [docs/taxonomy.md](../docs/taxonomy.md)（§3 階層、§6 データ表現）。

## 手編集するファイル（1 つだけ）

| ファイル | 内容 |
|---|---|
| `taxonomy.yaml` | 正本。SKOS 借用項目（prefLabel / altLabel / definition / scopeNote / broader / changeNote）と構造制約。`scheme.version` が runs / reports に刻む `taxonomy_version` |

変更したら `scheme.version` と `changeNote` を更新し、`make taxonomy` で build と viz を **両方** 実行する。
優先規則・品質規則・運用写像は `rulebook.md`（ポリシー）に書き、ここには置かない。

## 生成物（手編集禁止）

| ファイル | 生成元 | 用途 |
|---|---|---|
| `prompts.json` | `taxonomy_build.py` | Stage1 のクラスプロンプト。第2階層ごとに自身と第3階層の子のプロンプトを `source` 付きで集約。`class_order` は出力列順 |
| `label_master.csv` | `taxonomy_build.py` | アノテーション用マスタ。`derived_level1` は `broader` から導出（第1階層は手で付けない） |
| `taxonomy.ttl` | `taxonomy_build.py` | SKOS/Turtle。rdflib で往復パースし概念数を確認済み |
| `taxonomy_tree.md` | `taxonomy_viz.py` | Mermaid 図＋概念表＋プロンプト表（GitHub がそのまま描画） |
| `taxonomy.svg` | `taxonomy_viz.py` | Graphviz 出力。`dot` が PATH にある場合のみ生成（無ければ警告のみ） |

検証（id / prefLabel 一意、broader 存在、階層規則、第2階層 = 5、プロンプト源、rulebook 被覆）に
1 件でも失敗すると **何も書かずに** 終了コード 1 で止まる。負のテストは `test_taxonomy_build.py`。

```sh
make taxonomy                                   # build + viz
uv run python taxonomy/taxonomy_build.py --help
uv run python taxonomy/taxonomy_viz.py --help
uv run pytest taxonomy -q
```
