# Project settings
APP_NAME := chapkit
DOCKER_TAG := latest
DOCKER_IMAGE := $(APP_NAME):$(DOCKER_TAG)

.PHONY: help clean test lint fmt docker-build docker-run docker-shell docker-push

help:
	@echo "Available targets:"
	@echo "  run           - Run the application (requires uvicorn)"
	@echo "  clean         - Remove Python cache and build artifacts"
	@echo "  test          - Run pytest suite"
	@echo "  lint          - Run ruff linter/formatter"
	@echo "  docker-build  - Build Docker image ($(DOCKER_IMAGE))"
	@echo "  docker-run    - Run container with port 8000 exposed"
	@echo "  docker-shell  - Run shell inside built image"

run:
	uvicorn main:app --reload

clean:
	@echo "Cleaning build artifacts..."
	rm -rf .pytest_cache
	rm -rf dist build
	find . -type d -name "__pycache__" -exec rm -rf {} +

test:
	@echo "Running tests..."
	uv run pytest -q

lint:
	@echo "Linting code..."
	uv run ruff check src tests
	@echo "Formatting code..."
	uv run ruff format src tests

docker-build:
	@echo "Building Docker image: $(DOCKER_IMAGE)"
	docker build -t $(DOCKER_IMAGE) .

docker-run:
	@echo "Running Docker container on http://localhost:8000"
	docker run --rm -p 8000:8000 $(DOCKER_IMAGE)

docker-shell:
	@echo "Dropping into a shell inside the container..."
	docker run --rm -it $(DOCKER_IMAGE) /bin/bash
