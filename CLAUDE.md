# lab-experiments の規約

自分で行う調査・実験のコード置き場。分かったことは [lab-research-notes](https://github.com/spin-glass/lab-research-notes) にノートとして書き、このリポジトリにはその再現に必要なコードと小さな結果だけを置く。

## 配置

1実験＝1フォルダ：`<領域>/<YYYY-MM-DD>-<slug>/`

| 領域 | 対象 |
|------|------|
| `recsys/` | 推薦システム |
| `adaptive-learning/` | アダプティブラーニング・知識追跡 |
| `anomaly-detection/` | 異常検知 |
| `vision/` | 画像認識・VLM |
| `misc/` | 上のどれにも入らないもの |

- 日付は実験を始めた日。slug は英小文字・数字・ハイフン。
- 領域のフォルダは、その領域で初めて実験を作るときに作る。空のフォルダを先に作らない。
- 上の5つ以外の領域は作らない。迷ったら `misc/`。

## 実験フォルダの中身

```text
<領域>/<YYYY-MM-DD>-<slug>/
├── README.md       # 問い／結論／再現手順／lab-research-notes のノートへのリンク
├── pyproject.toml  # この実験の依存（uv workspace のメンバー）
├── src/            # 再利用するコード
├── notebooks/      # 分析ノートブック（.qmd / .ipynb）
└── results/        # 小さな表と図だけ
```

README には次の4つを必ず書く。

1. **問い**：何を確かめたいか
2. **結論**：分かったことを短く（詳細はノートへ）
3. **再現手順**：データの取得からコマンドまで
4. **ノート**：lab-research-notes の対応するノートへのリンク（書く前は「未作成」）

## 置かないもの

- データ（生データ・加工済みデータとも）。取得手順を README に書き、`data/` に置く（git 管理外）。
- 案件固有の情報（顧客名・社内の数値・非公開の仕様など）。
- 秘密情報（API キー・トークン・認証情報）。`.env` は git 管理外。
- 大きな生成物（モデルの重み、MLflow の記録、レンダリングした HTML サイト）。`results/` には小さな表（CSV・Markdown）と図（PNG・SVG）だけを置く。

結論はこのリポジトリではなく lab-research-notes にノートとして書く。実験の README からノートへ、ノートから実験フォルダへ互いにリンクする。

## Python 環境

- ルートの `pyproject.toml` が uv workspace。各実験フォルダはそのメンバーになる（`members` の glob で自動的に入る）。
- ルートが持つのは共通の開発ツール（ruff・pytest）だけ。実験の依存は各実験の `pyproject.toml` に書く。
- 依存は `uv add` で足す（pip・poetry は使わない）。版は固定し、公開から14日以上経った版だけを使う。
- ロックファイルはルートの `uv.lock` 1つ。実験フォルダに `uv.lock` を置かない。
- 例外：依存が他の実験と同時に解決できない実験は、ルートの `exclude` に入れて単独の uv プロジェクトとし、その実験フォルダに `uv.lock` を置く。理由は下の「取り込んだ実験」に書く。

```bash
uv sync                                         # 共通の開発ツール
uv sync --package <実験の project.name>          # 実験の依存も入れる
uv run ruff check .                             # lint（ルートから全体）
cd <実験フォルダ> && uv run pytest               # 実験ごとのテスト
```

## 取り込んだ実験

| フォルダ | 元リポジトリ |
|----------|--------------|
| `adaptive-learning/2026-04-21-kt-ednet/` | spin-glass/adaptive-learning-kt（git subtree で履歴ごと） |
| `vision/2026-08-14-vlm-cascade/` | spin-glass/lab-vlm-cascade（git subtree で履歴ごと。wip/2026-10-25 に main の D14 メモをマージして取り込み）。**workspace 外の単独 uv プロジェクト**：Python が 3.12 固定で kt-ednet（3.11 固定）と重ならず、開発ツールも ruff==0.16.6 を固定していてルートの ruff==0.17.0 と同時に解決できないため。この実験は自分のフォルダで `uv sync` する |
