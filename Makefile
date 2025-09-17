# Project settings
APP_NAME := chapkit
DOCKER_TAG := latest
DOCKER_IMAGE := $(APP_NAME):$(DOCKER_TAG)

.PHONY: help run clean test lint

help:
	@echo "Available targets:"
	@echo "  run           - Run the application (requires uvicorn)"
	@echo "  clean         - Remove Python cache and build artifacts"
	@echo "  test          - Run pytest suite"
	@echo "  lint          - Run ruff linter/formatter"

run:
	uv run uvicorn main:app --reload

clean:
	@echo "Cleaning build artifacts..."
	rm -rf .pytest_cache
	rm -rf dist build
	rm -rf target
	find . -type d -name "__pycache__" -exec rm -rf {} +

test: clean
	@echo "Running tests..."
	uv run pytest -q

lint:
	@echo "Linting code..."
	uv run ruff check --fix src tests
	@echo "Formatting code..."
	uv run ruff format src tests
