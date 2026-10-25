# 図の情報設計ガイドライン

本リポジトリの図（`docs/diagrams/`）を作成・改訂する際の規約と、その根拠となる一次情報。
reports/ に図を載せる場合も本ガイドラインに従う。

## 経緯

現行の4図（architecture / decision-flow / milestones / m7-approval-loop）について「眺めても頭に入らない」との評価があり、説得3枚＋参照4枚への全面再編を一度試みた。ただし**再編案は内容面で現行図に劣ると判断され、現行図に差し戻している**。

本書はその過程で調査した一次情報と規約を、今後の図の改訂に使うために残したもの。**現行の4図は本書の規約を満たしていない**（下表）。規約は「これから作る図が従うべき指針」であって、現行図の現状記述ではない。

実測した症状:

| 図 | ノード | エッジ | 色 | ノード内改行 |
|---|---|---|---|---|
| architecture | 12（4 subgraph） | 15 | 5 | 12 |
| decision-flow | 12 | 11 | 5 | 10 |
| milestones | 9 | 4 | 3 | 25（1ノード約3.8行） |
| m7-approval-loop | 19 | 19 | 5 | 24 |

診断された3つの根本原因:

1. **主張がない** — 全図が「何があるか」の目録で、「だから何が言えるか」がない。README に3つの主張が明記されているのに、そのどれも図になっていない。とくにカスケードの本質である「大多数が安く抜け、少数だけが高い経路に入る」非対称性が、同じ太さの矢印で消えている
2. **抽象度の混在** — 1枚に実行時フロー／データ資産／週次バッチ／ビルド時生成／運用インフラが同居。「実行時に画像が流れる」矢印と「ビルド時にコードを生成する」矢印が同じ記号で描かれ、記号の意味が一意でない
3. **読者の未分化** — 説得用と参照用を1系統で兼ねた結果、説得には細かすぎ、実装には粗すぎる中間点に落ちている

**差し戻しから得た教訓**: 上記の診断は構造面では妥当だが、それに基づく再編案は情報量を削りすぎて内容面で劣化した。規約（ノード5以下等）を機械的に適用すると、図が伝えるべき実質まで削れる。**次に改訂する際は、図をいきなり作らず情報設計だけを先に提示し、合意を取ってから作図する。**

## 規約

### 1. 図を描く前に、主張を完全な1文で書く

書けないなら、それは説得図ではなく参照図。書けた1文を**そのまま図のタイトルにする**（名詞句のタイトルは禁止。「システムアーキテクチャ」ではなく「大多数を安く確定し、曖昧な少数だけに払う」）。

Duarte の Big Idea は3条件すべてを満たすこと: (a) 独自の視点を述べている (b) 何が懸かっているかを伝えている (c) 完全な文である。

### 2. 説得図と参照図を分ける

| | 説得図（A群） | 参照図（B群） |
|---|---|---|
| 読者 | 採用を決める実務側 | 実装者 |
| 正義 | **省略** | **網羅性** |
| ノード数 | 5以下 | 20未満（10前後を目標） |
| 数値 | 必ず1つ以上（未実測なら `X%` プレースホルダ＋確定予定を明記） | 不要 |
| 判断基準 | 3秒で主張が伝わるか | 実装時に参照して迷わないか |

同じ図で両方をやろうとするのが「頭に入らない図」の主因（Diátaxis: reference は地図や辞書のように事実と正確さ、explanation は理解と「なぜ」に奉仕する。役割が違う）。

### 3. 要素数の上限

**20要素で赤信号、12個でも分かりにくければ描き直す**（Simon Brown）。同時に比較させるグループは4前後まで（Cowan 2010: 中心的な作業記憶容量は意味的アイテム3〜5個）。

> ※ **7±2 を根拠に使わない**。Miller 1956 は短期記憶の chunk 数の話であり、図の要素数の規範ではない。Miller 自身 "magical" を修辞として使っており、Laws of UX は "Don't use the 'magical number seven' to justify unnecessary design limitations" と明示的に警告している。

### 4. ノード内は原則1行

削った説明は**消さずにキャプションへ移す**。情報量は減らさない。減らすのは「図の中で同時に処理させる情報量」だけ。ノード内が3行あるなら、それは図ではなく箱に入れた箇条書き。

### 5. 1図1抽象レベル

実行時 / ビルド時 / 運用 を1枚に混ぜない。矢印の意味が一意でない図は読めない。

C4 の階層がこの原則の実例: Context（全員向け）→ Container（技術者向け）→ Component（実装者向け、「価値がある時だけ描く」）。Container 図は deployment を意図的に除外する（環境ごとに変わるため別図にする）。

### 6. 色は1軸のみ

説得図は「安い/高い」、参照図は「自動/人手」など、意味の軸を1つに揃える。**色は最大3系統**。異質なものに同一色を当てない（初版は `store` 灰色を photos・parquet・W&B・Vertex という別物に当てていた）。

### 7. ファイル構成を固定する

```
# 主張文（＝タイトル、完全な1文）

対象読者と scope の1行

```mermaid
...
```

**凡例（色の意味）**

前提と詳細（散文3〜5行。ノードから削った情報の行き先）

正本: design.md §X
```

### 8. その他の記法ルール

- **全エッジにラベルを付ける**。"Uses" のような単語1つは避ける。分岐（菱形）の出口は必ず Yes/No 等のラベル付き
- **タイトルには型と scope を含める**。「Context」だけでは不可
- **凡例を必ず置く**（色・形・線種の意味）
- 単方向の関係を1本の線で表す。無向エッジをフローチャートに混ぜない
- 流量の非対称性は `linkStyle` の `stroke-width` で可視化できる（カスケードの主張はこれだけで言葉なしに伝わる）
- 制御フローでないもの（運用ポリシー、撤退条件など）をノードとして描かない。誤読を招く

### 9. Mermaid 固有の落とし穴

- **subgraph 内の `direction` は、外部リンクがあると無視される**（親グラフの方向を継承）。実アーキ図ではほぼ効かないため、subgraph を重ねてレイアウトを制御しようとしない
- **ノードテキストの小文字 `end` はフローチャートを壊す** → `End` / `END` にする
- ノードIDの先頭が `o` / `x` はエッジと誤解釈される（`dev--- ops` のように空白を入れるか大文字化）
- 特殊文字・Unicode はダブルクォートで囲む
- 方向: パイプライン／段階変換は `LR`、判断分岐や階層は `TD`。1図内で混在させない
- 上限 `maxTextSize` 50000 / `flowchart.maxEdges` 500 は**性能上の上限であり可読性の目安ではない**（可読性の閾値は前述の20要素）
- GitHub は単体の `.mmd` ファイルを描画しない。`.md` 内の ```mermaid フェンスのみ描画される

## 検証手順

図を作ったら、マージ前に以下を通す。

1. **3秒テスト** — 3秒見て目を閉じ、何が残ったかを言う。残った印象が主張文と一致しなければ失敗（説得図のみ）
2. **Big Idea ゲート** — 主張が完全な1文で書けているか
3. **要素数ゲート** — 20未満（説得図は5以下）。ノード内改行は機械的に確認できる:
   ```bash
   grep -o '<br/>' docs/diagrams/*.md | sort | uniq -c
   ```
4. **単一抽象レベル** — 全ノードが同じ型か（実行時／ビルド時／運用が混ざっていないか）
5. **凡例と用語** — 色の意味が書かれているか。design.md の用語をそのまま使っているか（独自の言い換えをしない）

## 出典

一次情報で確認済み（2026-08 時点）。

**C4モデル / 図のレビュー**
- [c4model.com](https://c4model.com/) — 抽象定義、各階層の scope と読者、記法ルール、レビューチェックリスト
- Simon Brown, [Diagramming distributed architectures with the C4 model](https://dev.to/simonbrown/diagramming-distributed-architectures-with-the-c4-model-51cm) — 「20+要素で複雑化」「12個の箱で分かりにくいなら描くな」の出典
- Simon Brown, [How to review a software architecture diagram](https://dev.to/simonbrown/how-to-review-a-software-architecture-diagram-6p0) — 「初期の図の90%以上にタイトルがない」「線の約半分にラベルがない」
- [Misuses and Mistakes of the C4 model](https://www.workingsoftware.dev/misuses-and-mistakes-of-the-c4-model/) — 任意の抽象レベルの発明、決定を図に描く（→ ADR に分離すべき）等

**認知容量**
- Cowan, N. (2010). "The Magical Mystery Four: How Is Working Memory Capacity Limited, and Why?" *Current Directions in Psychological Science*, 19(1), 51–57. [SAGE](https://journals.sagepub.com/doi/abs/10.1177/0963721409359277) / [PubMed 20445769](https://pubmed.ncbi.nlm.nih.gov/20445769/)
- [Laws of UX: Miller's Law](https://lawsofux.com/millers-law/) — 7±2 の誤用に対する警告

**説得と物語**
- Cole Nussbaumer Knaflic, *Storytelling with Data* (Wiley, 2015) — exploratory と explanatory の分離
- [What's the Big Idea?](https://www.storytellingwithdata.com/blog/2014/02/whats-big-idea) — Duarte *Resonate* (2010) の Big Idea 3条件
- [Duarte: The 3-second test](https://www.duarte.com/blog/the-three-second-test/) — glance media
- [Diátaxis](https://diataxis.fr/) — reference と explanation の役割分離

**その他**
- Foote & Yoder, [Big Ball of Mud](https://www.laputan.org/mud/) (PLoP '97) — ※これは**システム構造**のパターンであり「図のアンチパターン」の確立された名称ではない。図に転用するなら比喩と明示する
- Tufte, *The Visual Display of Quantitative Information* (1983) — chartjunk、data-ink ratio。[InfoVis-Wiki](https://infovis-wiki.net/wiki/Data-Ink_Ratio)
- [Mermaid flowchart syntax](https://mermaid.js.org/syntax/flowchart.html) / [config schema](https://mermaid.js.org/config/schema-docs/config.html)
