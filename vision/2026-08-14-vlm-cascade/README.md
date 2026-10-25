# lab-vlm-cascade

曖昧クラスを含む画像分類を「カスケード＋判定層＋監査ループ」で運用する設計の、一般実装と数値検証。Yelp Open Dataset（food / drink / menu / inside / outside の5クラス、約20万枚）を題材に、設計判断をすべて定義済みのデータ分析で下す。

> **現状（2026-09）: M0 実装前の設計リポジトリ。** 存在するのは設計書（`docs/`）・参考文献・Mermaid 図・`CLAUDE.md`・本 README のみで、コード・依存定義（`pyproject.toml`）・`configs/`・`tests/`・`reports/` は未作成。以下の「セットアップ」「リポジトリ構成」は M0 以降で実現する計画であり、現時点では実行できない。設計の正本は [docs/design.md](docs/design.md)。

## 何を実証するか

1. **カスケードの費用対効果** — 確信できる多数を安いエンコーダで確定し、曖昧な少数だけVLMへ委譲すれば、全量VLM比のわずかなコストで同等精度圏に入ること（M3で実測）
2. **既存ラベルは監査できる** — cleanlab＋VLM合議で来歴不明ラベルの疑義を検出し、目視スポットチェックで的中率を測れること（M4）
3. **設計は差し替え可能** — runs正本パターン（記録の正本はローカルparquet、W&B と Vertex AI Experiments へ二重送出）と rulebook 単一ソースにより、トラッカー差し替えとルール変更が再学習なしで完結すること（M5）
4. **非料理画像の除外は料理混入で測る** — 対象外画像の価値は『その他』への回収率ではなく、非料理→food の誤混入をどれだけ減らし、正しい料理を何枚失うかで判定する（G0）

## アーキテクチャ

```
photos ─▶ Stage1: encoder zero-shot ─▶ Stage2: decision layer ─▶ 確定 (clear / rule_resolved)
              │ probs, margin              │ 低margin・指定confusion pairのみ
              ▼                            ▼
        predictions (parquet)        Stage3: VLM escalation ─▶ 確定 or irreducible
                                           │
offline: 監査ループ (cleanlab / VLM合議 / κ定点観測) ◀──┘
single source (語彙):     taxonomy/taxonomy.yaml ─▶ クラス定義 / 階層 / 葉プロンプト
single source (ポリシー): rulebook.md            ─▶ 優先規則 / 判定表 / VLMプロンプト
```

分類体系は2階層タクソノミー（food / non-food → 運用5クラス → プロンプト用サブクラス）。「food でない」という否定を非 food 側の肯定クラスの列挙に置き換え、誤爆しやすい対象を正しい親の下のサブクラスとして命名する。設計ノートは [docs/taxonomy.md](docs/taxonomy.md)。

## 設計判断はデータ分析で下す

カスケード採否・しきい値粒度・較正要否・既存ラベル再利用可否・階層スコアリング採否など14の設計判断（D1–D14）は事前に決め打ちせず、[docs/design.md](docs/design.md) §8 の decision matrix に定義した分析と判断基準で決定し、reports/ に意思決定ログ（基準値・実測値・採否）を残す。分析コードは DuckDB / BigQuery 両対応で書き（`analysis/`、M1 以降で作成）、**公開データで方法論を確立 → 同一スクリプトを自組織のデータに向けて再実行**する二段構えの移植性を持つ。

## マイルストーン

| M | 内容 | 完了条件 |
|---|---|---|
| M0 | セットアップ・eval凍結・トラッカー fan-out・taxonomy 導入 | reports/m0_setup.md |
| G0 | 非料理画像の距離信号による識別（M0 後、M1 と並行可）— 分類スコアでは food に見える非料理を、正しい料理の棄却を同じ水準に揃えたうえで、料理分布からの距離で分類スコアより多く除外できるか。事前登録 docs/g0_nonfood_plan.md | reports/g0_nonfood_gate.md |
| M1 | ゼロショット基線・フラット vs 階層・混同ペアの定量特定 | reports/m1_baseline.md |
| M2 | 判定層・risk–coverage・しきい値探索 | reports/m2_decision_layer.md |
| M3 | カスケード vs 全量VLM の精度・コスト実測比較 | reports/m3_cascade_vs_vlm.md |
| M4 | ラベル監査（cleanlab＋VLM合議、precision@N）・gold パイロットアノテーション | reports/m4_label_audit.md |
| M5 | rulebook改版の即応デモ（再学習なし） | reports/m5_rulebook_change.md |
| M6 | （任意）soft label 蒸留＋WiSE-FT | reports/m6_distill.md |
| M7 | （任意）self-improving rulebook loop — 監査不一致からの自動改訂案生成＋承認ゲート、Stage3サブワークフローの予算制約探索 | reports/m7_self_improving_loop.md |

主要結果の表は各M完了後にここへ追記する。掲載する数値はすべて本リポジトリでの実測値のみ。

## スタック

uv + Python 3.12 ／ OpenCLIP・SigLIP（HF）／ DuckDB＋parquet 三層（raw・core・marts）／ Gemini API（構造化出力・config上限つき）／ W&B Free＋Vertex AI Experiments（fan-out、正本はローカル runs テーブル）／ Optuna＋Vizier（無料枠内のパリティ検証）／ cleanlab。GCP実務への対応表は docs/design.md §3。

## セットアップ（計画。M0 実装後に有効）

```bash
uv sync
export GEMINI_API_KEY=... WANDB_API_KEY=... GOOGLE_CLOUD_PROJECT=...
# データは Yelp 公式ページから各自取得し（同梱規約PDFを確認）、data/ へ配置
uv run python -m cascade.m0_setup --config configs/m0.yaml
```

## リポジトリ構成（計画）

現時点で存在するのは `docs/`・`CLAUDE.md`・本 README のみ。それ以外は各マイルストーンで作成する。

```
taxonomy/            # taxonomy.yaml（クラス定義の正本）＋ build/viz と生成物（prompts.json, label_master.csv, taxonomy.ttl）
rulebook.md          # 分類意図・優先順位ルール・品質規則（ポリシーの単一ソース、semver管理）
configs/             # しきい値・モデルID（ピン留め）・API呼び出し上限
src/cascade/         # stage1_encode / stage2_decide / stage3_escalate / audit / eval
analysis/            # 設計判断 D1–D14 の分析（DuckDB / BigQuery 両対応）
reports/             # 各Mの自動生成レポート（run_id・git sha・taxonomy/rulebook 版・概算コスト記載）
docs/design.md       # 設計の正本
docs/g0_nonfood_plan.md  # G0 事前登録（凍結タグ g0-freeze-v1）
docs/diagrams/       # design.md の可視化（Mermaid、Miro ボードの生成元）
docs/diagram-guidelines.md  # 図の情報設計規約
docs/taxonomy.md     # 分類体系・アノテーション設計ノート
docs/references.md   # 主張→一次文献の対応
CLAUDE.md            # 実装時の制約・規約
```

## 注意事項

- Yelp Open Dataset の画像・生データはリポジトリに含まれない。README・reports にも画像は掲載しない（教育目的での利用、再配布不可）
- Yelp Dataset の規約に従い、photo_id 等の Data はリポジトリに置かず（sha256・config・seed のみ）、reports は集計値のみを載せる。結果の公開はユーザが規約（学術利用・公開前審査）を確認してから行う
- W&B は Free プラン（個人プロジェクト限定）の範囲で使用し、画像はアップロードしない
- 外部API呼び出しは config の上限内で実行し、各レポートに概算コストを記載する
- 特定企業の事例・数値は含まない
