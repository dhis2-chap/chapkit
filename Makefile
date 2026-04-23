.PHONY: help install lint test test-slow test-durations coverage clean docs docs-serve docs-build

# ==============================================================================
# Venv
# ==============================================================================

UV := $(shell command -v uv 2> /dev/null)
VENV_DIR?=.venv
PYTHON := $(VENV_DIR)/bin/python

# ==============================================================================
# Targets
# ==============================================================================

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install      Install dependencies"
	@echo "  lint         Run linter and type checker"
	@echo "  test         Run tests (excludes slow tests)"
	@echo "  test-slow    Run slow CLI scaffolding tests"
	@echo "  test-durations Show 20 slowest tests"
	@echo "  coverage     Run tests with coverage reporting"
	@echo "  migrate      Generate a new migration (use MSG='description')"
	@echo "  upgrade      Apply pending migrations"
	@echo "  downgrade    Revert last migration"
	@echo "  docs-serve   Serve documentation locally with live reload"
	@echo "  docs-build   Build documentation site"
	@echo "  docs         Alias for docs-serve"
	@echo "  clean        Clean up temporary files"
	@echo ""
	@echo "Docker images for 'chapkit run' live in dhis2-chap/chapkit-images;"
	@echo "this repo no longer builds them."

install:
	@echo ">>> Installing dependencies"
	@$(UV) sync --all-extras

lint:
	@echo ">>> Running linter"
	@$(UV) run ruff format .
	@$(UV) run ruff check . --fix
	@echo ">>> Running type checker"
	@$(UV) run mypy --explicit-package-bases src tests examples
	@$(UV) run pyright

test:
	@echo ">>> Running tests (excluding slow)"
	@$(UV) run pytest -q -m "not slow"

test-slow:
	@echo ">>> Running slow CLI scaffolding tests"
	@$(UV) run pytest -v -m slow tests/test_cli_scaffolding.py

test-durations:
	@echo ">>> Running tests and showing 20 slowest"
	@$(UV) run pytest -q -m "not slow" --durations=20

coverage:
	@echo ">>> Running tests with coverage (excluding slow)"
	@$(UV) run coverage run -m pytest -q -m "not slow"
	@$(UV) run coverage report
	@$(UV) run coverage xml

migrate:
	@echo ">>> Generating migration: $(MSG)"
	@$(UV) run alembic revision --autogenerate -m "$(MSG)"
	@echo ">>> Formatting migration file"
	@$(UV) run ruff format src/chapkit/alembic/versions

upgrade:
	@echo ">>> Applying pending migrations"
	@$(UV) run alembic upgrade head

downgrade:
	@echo ">>> Reverting last migration"
	@$(UV) run alembic downgrade -1

docs-serve:
	@echo ">>> Serving documentation at http://127.0.0.1:8000"
	@$(UV) run mkdocs serve

docs-build:
	@echo ">>> Building documentation site"
	@$(UV) run mkdocs build

docs: docs-serve

clean:
	@echo ">>> Cleaning up"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .coverage htmlcov coverage.xml
	@rm -rf .pyright
	@rm -rf dist build *.egg-info
	@find examples -type f -name "uv.lock" -delete
	@find examples -type d -name ".venv" -exec rm -rf {} + 2>/dev/null || true

# ==============================================================================
# Default
# ==============================================================================

.DEFAULT_GOAL := help
