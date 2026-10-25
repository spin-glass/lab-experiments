# CLAUDE.md — lab-vlm-cascade

曖昧クラスを含む画像分類のカスケード実装。設計の正本は `docs/design.md`（分類体系の詳細は `docs/taxonomy.md`、クラス定義の正本は `taxonomy/taxonomy.yaml`）。実装と矛盾したら design.md の改訂を先に提案し、承認後に実装を変える。

## 絶対制約

- `data/` は git 管理外。Yelp の画像・生データをコミットしない。README・reports に Yelp 画像を掲載しない
- 認証は環境変数（`GEMINI_API_KEY`, `WANDB_API_KEY`）と ADC（`GOOGLE_CLOUD_PROJECT`、vertex バックエンド使用時）のみ。コード・config へのベタ書き禁止
- トラッカーへ画像・生データをアップロードしない。scalar / 小テーブル / reference artifact のみ（W&B・Vertex 共通）
- Stage3・監査の API 呼び出しは config の上限（`MAX_ESCALATION` 等）内で実行。managed Vizier は無料枠（月100トライアル）内のみで、超過する探索は Optuna を使う。上限を超える設計変更は先に確認を取る
- runs の正本はローカル parquet。トラッキングは `tracking: [wandb, vertex]` の fan-out で、どのバックエンドが不通でもパイプラインは完走すること（wandb のみ／vertex のみ／none の縮退をテスト）。Vertex AI TensorBoard は有効化しない
- モデルID・taxonomy_version・rulebook_version は config にピン留め。"latest" 系エイリアス禁止
- `taxonomy/` で手編集するのは `taxonomy.yaml` のみ。`prompts.json` / `label_master.csv` / `taxonomy.ttl` / 図は生成物であり手編集禁止。YAML を変えたら build と viz を両方実行する。検証に失敗したら生成しない
- 語彙とポリシーを混ぜない: クラス定義・階層・プロンプトは taxonomy.yaml、優先規則・品質規則・運用写像は rulebook.md。第1階層（food / non-food）はゴールドに記録せず導出のみ
- Yelp の Data（photo_id・business_id・caption・ラベル行）を git に置かない（sha256・config・seed のみ）。reports は集計値のみ
- VLM（Gemini）は G0 では使わない（評価ラベルにも灰色域再判定にも）

## 開発規約

- uv（Python 3.12）。lint/format は ruff、テストは pytest
- Stage2 判定層・rulebook 生成（`gen_from_rulebook.py`）・taxonomy 検証（`taxonomy_build.py`。参照を壊した入力で止まる負のテストを含む）はユニットテスト必須
- 1 マイルストーン = 1 ブランチ = 1 PR。コミットは小さく
- 実行は `uv run python -m cascade.mX_...` の CLI に統一。config は `configs/*.yaml`
- 設計判断（design.md §8 の D1–D14）に関わる分析は `analysis/` に判断ID付きで置く。SQLはDuckDB/BigQuery両対応の書き方に限定し、接続・テーブル名はconfig注入。判断基準・実測値・採否は reports の意思決定ログに記録する
- 外部情報（モデルID、料金、ライブラリAPI）は実行時に確認してから使う。推測で書かない

## 完了条件（全マイルストーン共通）

- `reports/{m,g}X_*.md` を自動生成する。冒頭に結論、フッタに run_id / git sha / taxonomy_version / rulebook_version / eval_set_id / 概算APIコスト / W&B run URL
- design.md の該当「検証命題」に数値で答えているかを自己チェックしてから PR を出す

## 進行

- M0 → M5 の順で進める。各マイルストーン終了時に停止し、reports をレビューに回す。**次のマイルストーンに勝手に進まない**
- レビュー指摘は同一ブランチで修正し、レポートを再生成する
- G0（非料理画像の距離信号による識別）は M0 完了後に M1 と並行可の独立トラック。事前登録は docs/g0_nonfood_plan.md、凍結タグ g0-freeze-v1 以後は逸脱記録なしに固定項目を変えない。eval は最後の 1 回だけ実行する
- M0/G0 の実験前成果物（設計改訂・事前登録・データ調査・凍結・基盤コード）は main に直接コミットして push する（ユーザ指示、2026-09-07）。実験結果の反映は branch → PR
