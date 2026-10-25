.PHONY: help setup download ref-pykt-clone ref-pykt-venv ref-pykt-preprocess \
        ref-pykt-import-processed test lint fmt typecheck check clean clean-ref \
        preview render

SHELL := /bin/bash

# ----- Paths -----
PROJECT_ROOT := $(abspath $(CURDIR))
RAW_DIR      := $(PROJECT_ROOT)/data/raw
PROCESSED_DIR:= $(PROJECT_ROOT)/data/processed

# reference-pykt lives OUTSIDE this project (no dependency, consulted for scripts only)
REF_ROOT       := $(HOME)/reference-pykt
REF_REPO_URL   := https://github.com/pykt-team/pykt-toolkit.git
REF_PY         := 3.9
REF_VENV       := $(REF_ROOT)/.venv
REF_EDNET_IN   := $(REF_ROOT)/data/ednet
REF_EDNET_OUT  := $(REF_ROOT)/data/ednet      # pyKT writes CSVs back alongside inputs

help:
	@echo "Targets:"
	@echo "  setup                 - uv sync (install this project's deps)"
	@echo "  download              - download EdNet-KT1 into data/raw/"
	@echo "  test                  - pytest"
	@echo "  lint / fmt            - ruff check / ruff format"
	@echo "  typecheck             - mypy src tests"
	@echo "  check                 - fmt-check + lint + typecheck + test"
	@echo "  preview FILE=...      - quarto preview (default: notebooks/01-eda.qmd)"
	@echo "  render FILE=...       - quarto render --to html"
	@echo "  clean                 - remove caches"
	@echo ""
	@echo "Notebooks:"
	@echo "  01-eda.qmd            - EDA"
	@echo "  02-irt-bkt.qmd        - IRT + BKT baseline"
	@echo "  03-deep-kt.qmd        - DKT / SAKT / SimpleKT"
	@echo "  04-item-selection.qmd - Item selection policy simulation"
	@echo "  05-item-retrieval.qmd - Two-Tower item retrieval (stretch)"

# ----- Main project -----
setup:
	uv sync

download:
	uv run python -m src.data.download --dest $(RAW_DIR)

test:
	uv run pytest

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

typecheck:
	uv run mypy

check:
	uv run ruff format --check .
	uv run ruff check .
	uv run mypy
	uv run pytest

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ .quarto _output

# ----- Quarto (works around Positron's QUARTO_PYTHON env injection bug) -----
# Positron's Preview button spawns a terminal that does NOT inherit
# `terminal.integrated.env.osx.QUARTO_PYTHON`, so Quarto falls back to the
# system python3 and hangs. Using these targets from a normal shell works.
#
# Usage: `make preview FILE=notebooks/01-eda.qmd`
QUARTO_PYTHON := $(PROJECT_ROOT)/.venv/bin/python
FILE ?= notebooks/01-eda.qmd

preview:
	QUARTO_PYTHON=$(QUARTO_PYTHON) quarto preview $(FILE) --no-watch-inputs

render:
	QUARTO_PYTHON=$(QUARTO_PYTHON) quarto render $(FILE) --to html

# ----- reference-pykt (external, not a dependency) -----
# We clone pyKT once to a sibling location, use its data_preprocess.py and
# split scripts to produce standard CSVs, then copy the results into
# data/processed/ednet/. pyKT itself stays OUT of this project's venv.

ref-pykt-clone:
	@if [ -d "$(REF_ROOT)/.git" ]; then \
	  echo "[skip] $(REF_ROOT) already a git repo"; \
	else \
	  echo "[clone] $(REF_REPO_URL) -> $(REF_ROOT)"; \
	  git clone --depth 1 $(REF_REPO_URL) "$(REF_ROOT)"; \
	fi

ref-pykt-venv: ref-pykt-clone
	@if [ ! -x "$(REF_VENV)/bin/python" ]; then \
	  echo "[venv] creating $(REF_VENV) with python $(REF_PY)"; \
	  cd "$(REF_ROOT)" && uv venv --python $(REF_PY) .venv; \
	fi
	@echo "[install] pyKT editable + deps into $(REF_VENV)"
	cd "$(REF_ROOT)" && uv pip install --python "$(REF_VENV)/bin/python" -e .

# Links raw EdNet-KT1 CSVs into the reference repo's expected location and
# runs its preprocess/split pipeline. pyKT expects data/ednet/ with per-user CSVs.
ref-pykt-preprocess: ref-pykt-venv
	@if [ ! -d "$(RAW_DIR)/KT1" ]; then \
	  echo "error: $(RAW_DIR)/KT1 not found. Run 'make download' first."; exit 1; \
	fi
	mkdir -p "$(REF_EDNET_IN)"
	@echo "[link] $(RAW_DIR)/KT1 -> $(REF_EDNET_IN)/KT1"
	@rm -f "$(REF_EDNET_IN)/KT1"
	ln -s "$(RAW_DIR)/KT1" "$(REF_EDNET_IN)/KT1"
	@echo "[preprocess] running pyKT data_preprocess.py for ednet"
	cd "$(REF_ROOT)/examples" && \
	  "$(REF_VENV)/bin/python" data_preprocess.py -d ednet

ref-pykt-import-processed:
	@if [ ! -d "$(REF_EDNET_OUT)" ]; then \
	  echo "error: $(REF_EDNET_OUT) not found. Run 'make ref-pykt-preprocess' first."; exit 1; \
	fi
	mkdir -p "$(PROCESSED_DIR)/ednet"
	@echo "[copy] pyKT processed artifacts -> $(PROCESSED_DIR)/ednet"
	# Copy everything pyKT produced except the symlinked raw KT1/ dir.
	rsync -av --exclude='KT1' --exclude='*.pkl' "$(REF_EDNET_OUT)/" "$(PROCESSED_DIR)/ednet/"
	@echo "[done] processed files under $(PROCESSED_DIR)/ednet"

clean-ref:
	@read -p "Remove $(REF_ROOT)? [y/N] " ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
	  rm -rf "$(REF_ROOT)"; \
	  echo "removed"; \
	else \
	  echo "aborted"; \
	fi
