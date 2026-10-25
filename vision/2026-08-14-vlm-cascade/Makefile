# lab-vlm-cascade — 開発用ターゲット（uv 前提）
#   make taxonomy  : taxonomy.yaml を検証して生成物を再生成（build + viz を必ず両方実行）
#   make lint      : ruff check + format --check
#   make format    : ruff format + import 整列
#   make test      : pytest（tests/ と taxonomy/）

UV ?= uv
LINT_PATHS := src tests taxonomy scripts analysis

# 素の `make` で taxonomy/ の生成物を書き換えない（既定はテスト）
.DEFAULT_GOAL := test

.PHONY: taxonomy taxonomy-build taxonomy-viz lint format test

taxonomy: taxonomy-build taxonomy-viz

taxonomy-build:
	$(UV) run python taxonomy/taxonomy_build.py

taxonomy-viz:
	$(UV) run python taxonomy/taxonomy_viz.py

lint:
	$(UV) run ruff check $(LINT_PATHS)
	$(UV) run ruff format --check $(LINT_PATHS)

format:
	$(UV) run ruff check --select I --fix $(LINT_PATHS)
	$(UV) run ruff format $(LINT_PATHS)

test:
	$(UV) run pytest -q
