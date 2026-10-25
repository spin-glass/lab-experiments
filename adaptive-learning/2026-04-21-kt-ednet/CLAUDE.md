# Adaptive Learning KT - Project Context

## 目的
「テスト対策アプリはどうすれば学習者の得点を最大化できるか？」を問いの軸に、
EdNet-KT1データセットを用いたアダプティブラーニング・パイプラインを設計・実装する。
Knowledge Tracing → Item Selection のパイプラインを構築。

## ディレクトリの役割
- `notebooks/*.qmd` - 対話的EDA・分析 (Positronで手動実行が原則)
- `src/` - 再利用可能なモジュール (Claude Codeが自由に編集可)
- `tests/` - pytestによる単体テスト
- `data/raw/` - 生データ (git管理外)
- `data/processed/` - Parquetキャッシュ (git管理外)
- `_output/` - Quartoレンダー結果 (gitignore)

## Claude Codeへのルール
- `notebooks/*.qmd` は明示的な指示がない限り編集しない
- `src/` のコードは型ヒント必須、pytestでテスト可能な形で書く
- 依存追加は `uv add <pkg>` を使う (pipやpoetry禁止)
- Python 3.11前提
- PyTorchコードは `device` パラメータを受け取れる形に

## 主要ライブラリ
- pyBKT: Bayesian Knowledge Tracingベースライン
- PyMC: IRT 2PL のベイズ推定
- PyTorch + Lightning: DKT / SAKT / SimpleKT
- MLflow: 実験管理
- polars: 大規模データは pandas より polars を優先

## パイプライン (Notebooks)
1. `notebooks/01-eda.qmd` - EdNet-KT1のEDA
2. `notebooks/02-irt-bkt.qmd` - IRT 2PL + BKT ベースライン
3. `notebooks/03-deep-kt.qmd` - DKT / SAKT / SimpleKT 比較
4. `notebooks/04-item-selection.qmd` - 出題戦略シミュレーション

## src/ モジュール構成
- `src/data/` - ダウンロード・ローダ・サンプリング
- `src/features/` - 前処理・特徴量エンジニアリング
- `src/models/` - IRT / BKT / DKT / SAKT / SimpleKT
- `src/training/` - PyTorch Lightning モジュール
- `src/eval/` - AUC / calibration / metrics
- `src/policy/` - Item Selection (random, difficulty matching, Thompson sampling)

## スタイル
- コメントは日本語可、コード内識別子は英語
- docstringはNumPy形式
