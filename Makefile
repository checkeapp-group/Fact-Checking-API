SHELL := /bin/bash
# Ensure UTF-8 locale for tools that rely on locale when invoked via make
export LC_ALL := en_US.UTF-8
export LANG := en_US.UTF-8
export LANGUAGE := en_US:en
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
RUFF := $(VENV)/bin/ruff
NBQA := $(VENV)/bin/nbqa
PRE_COMMIT := $(VENV)/bin/pre-commit

.PHONY: style dev-setup pre-commit-all

dev-setup:
	@test -x $(PY) || python3 -m venv $(VENV)
	@$(PY) -m ensurepip --upgrade
	@$(PY) -m pip install -q -r requirements-dev.txt

# Format and autofix Python and notebooks with Ruff (imports included) and Black via nbQA
style: dev-setup
	@$(RUFF) format .
	@$(RUFF) check . --fix --exit-zero
	@$(NBQA) ruff --fix --exit-zero notebooks || true
	@$(NBQA) black notebooks || true

# Optional: run pre-commit across all files
pre-commit-all: dev-setup
	@$(PRE_COMMIT) install --install-hooks -f
	@$(PRE_COMMIT) run --all-files -a -c .pre-commit-config.yaml || true


