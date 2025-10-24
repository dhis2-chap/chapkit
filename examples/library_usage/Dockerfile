# Library Usage Demo Dockerfile
FROM ghcr.io/astral-sh/uv:0.9-python3.13-bookworm-slim AS builder

WORKDIR /workspace

ARG USER=chapkit UID=10001
RUN useradd -u ${UID} -m -s /bin/bash ${USER}

# Install git for fetching dependencies from git repositories
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# UV configuration for better build performance
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy entire project structure to maintain relative paths for editable installs
COPY pyproject.toml uv.lock README.md ./
COPY src ./src/
COPY examples/library_usage ./examples/library_usage/

# Install from example directory (which references parent via ../../)
WORKDIR /workspace/examples/library_usage
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev

# Cleanup Python cache files
RUN find .venv -type d -name '__pycache__' -prune -exec rm -rf {} + && \
    find .venv -type f -name '*.py[co]' -delete || true

# ---- runtime ----
FROM python:3.13-slim AS runtime

# OCI labels for container metadata
LABEL org.opencontainers.image.title="Chapkit Library Usage Demo"
LABEL org.opencontainers.image.description="Custom models and CrudRouter demonstration with User management"
LABEL org.opencontainers.image.vendor="Chapkit"
LABEL org.opencontainers.image.source="https://github.com/dhis2-chap/chapkit"

ARG USER=chapkit UID=10001
RUN useradd -u ${UID} -m -s /bin/bash ${USER}

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends ca-certificates tini && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy entire structure to preserve paths for editable chapkit install
COPY --from=builder --chown=${USER}:${USER} /workspace /workspace

WORKDIR /workspace/examples/library_usage

# Set up environment to use the venv from example directory
ENV VIRTUAL_ENV=/workspace/examples/library_usage/.venv
ENV PATH=/workspace/examples/library_usage/.venv/bin:${PATH}
ENV PYTHONPATH=/workspace
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1

# Server configuration
ENV PORT=8000
ENV TIMEOUT=60
ENV GRACEFUL_TIMEOUT=30
ENV KEEPALIVE=5
ENV FORWARDED_ALLOW_IPS="*"

# Worker configuration
ENV MAX_REQUESTS=1000
ENV MAX_REQUESTS_JITTER=200

# Logging configuration
ENV LOG_FORMAT=json
ENV LOG_LEVEL=INFO

USER ${USER}
EXPOSE 8000

# Health check to verify the API is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health').read()" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sh", "-c", "FORWARDED_ALLOW_IPS=${FORWARDED_ALLOW_IPS:-*}; GUNICORN_CONF=$(python -c 'import servicekit, os; print(os.path.join(os.path.dirname(servicekit.__file__), \"gunicorn.conf.py\"))'); exec gunicorn -c \"${GUNICORN_CONF}\" -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:${PORT} --workers 1 --timeout ${TIMEOUT} --graceful-timeout ${GRACEFUL_TIMEOUT} --keep-alive ${KEEPALIVE} --forwarded-allow-ips=${FORWARDED_ALLOW_IPS} --max-requests ${MAX_REQUESTS} --max-requests-jitter ${MAX_REQUESTS_JITTER} --worker-tmp-dir /dev/shm"]
