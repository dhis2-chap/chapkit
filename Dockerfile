# ===== Stage 1: build venv with locked deps (no project install) =====
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project \
    && uv cache clean

# ===== Stage 2: runtime =====
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime
WORKDIR /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src"

COPY --from=builder /app/.venv /app/.venv

# App files
COPY main.py ./main.py
COPY src ./src
COPY templates ./templates
RUN mkdir -p /app/target
COPY target/storage.json /app/target/storage.json
COPY README.md ./README.md

# Non-root
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Run with uvicorn. We use ENTRYPOINT so you can pass extra flags at `docker run`.
ENTRYPOINT ["python", "-m", "uvicorn"]
CMD ["main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
