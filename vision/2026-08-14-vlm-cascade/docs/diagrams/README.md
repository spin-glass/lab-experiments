# docs/diagrams/

設計の可視化。**正本は [docs/design.md](../design.md)** であり、本ディレクトリの図はその可視化にすぎない。design.md を改訂したら、対応する図の Mermaid を更新し、Miro ボードへ再生成する。**注**: design.md の G0（P0/P1/P2 の比較、D14）は現行の図に未反映。情報設計案を提示し合意を得てから作図する（[docs/diagram-guidelines.md](../diagram-guidelines.md)）。

各図は Mermaid フェンス入りの Markdown（GitHub 上でそのまま描画される）。Miro へは同一の Mermaid ソースから MCP（`diagram_create_mermaid`）で生成しており、Miro 上で直接編集しない。

| ファイル | 図 | 正本の節 |
|---|---|---|
| [architecture.md](architecture.md) | システムアーキテクチャ（カスケード＋監査ループ＋taxonomy / rulebook 単一ソース＋tracker fan-out） | design.md §1, §3, §4 |
| [decision-flow.md](decision-flow.md) | 1枚の写真の判定フロー（flag三値の分岐） | design.md §1, §4 |
| [milestones.md](milestones.md) | マイルストーンロードマップ（M0–M7、対応する設計判断ID） | design.md §6, §8 |
| [m7-approval-loop.md](m7-approval-loop.md) | M7 self-improving loop の PR承認ゲート | design.md §6 M7 |

decision matrix（D1–D14）の表は design.md §8 が正本のため、ここには置かない（Miro ボード上にのみ表として展開）。

## Miro ボード

**入口（このリンクから開く）**: <https://miro.com/app/board/uXjVHxs_4FQ=/?moveToWidget=3458764681346279125>

ボードのルートURL（<https://miro.com/app/board/uXjVHxs_4FQ=/>）で開くと、削除できない旧図（ZZ ARCHIVE）が先に見えてしまうため、上の入口リンクを使う。

**「00 START HERE」から、左から右へ 00 → 05 の順に読む**（右下のナビゲーションで Frames 一覧を開ける）。各フレームは見出しと1文の説明を持ち、何を理解すればよいかを明示している。

| フレーム | 中身 |
|---|---|
| 00 START HERE | 何を検証するプロジェクトか、読む順、数値が未記入である理由 |
| 01 全体像 | architecture 図（※ボード上は taxonomy 追加前の旧版。再生成が未実施） |
| 02 進め方 | milestones 図 |
| 03 判定の詳細 | decision-flow 図 |
| 04 ガバナンス | m7-approval-loop 図（M7・任意） |
| 05 判断基準 | decision matrix 表（※ボード上は D1–D10 の旧版。design.md §8 は D1–D14 に拡張済みで、ボードの再生成が未実施） |
| ZZ ARCHIVE | 旧レイアウト。Miro の API では図を削除できないため隔離してある。参照しない |

01→02 だけで全体像は掴める。03 以降は実装・詳細検討向け。
