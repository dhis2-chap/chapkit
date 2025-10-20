# ML Functional Demo Dockerfile
FROM ghcr.io/astral-sh/uv:0.9-python3.13-bookworm-slim AS builder

WORKDIR /app

ARG USER=chapkit UID=10001
RUN useradd -u ${UID} -m -s /bin/bash ${USER}

# UV configuration for better build performance
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Copy and build parent chapkit project
COPY pyproject.toml uv.lock README.md /chapkit/
COPY src /chapkit/src/
WORKDIR /chapkit
RUN uv build

# Install chapkit wheel in app directory
WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv && \
    uv pip install /chapkit/dist/*.whl

# Copy demo files
COPY examples/ml_functional/main.py ./

# Cleanup Python cache files
RUN find /app/.venv -type d -name '__pycache__' -prune -exec rm -rf {} + && \
    find /app/.venv -type f -name '*.py[co]' -delete || true

# ---- runtime ----
FROM python:3.13-slim AS runtime

# OCI labels for container metadata
LABEL org.opencontainers.image.title="Chapkit ML Functional Demo"
LABEL org.opencontainers.image.description="Disease prediction ML service using FunctionalModelRunner with Linear Regression and monitoring"
LABEL org.opencontainers.image.vendor="Chapkit"
LABEL org.opencontainers.image.source="https://github.com/dhis2-chap/chapkit"

WORKDIR /app

ARG USER=chapkit UID=10001
RUN useradd -u ${UID} -m -s /bin/bash ${USER}

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends ca-certificates tini && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy venv and application from builder
COPY --from=builder --chown=${USER}:${USER} /app/.venv /app/.venv
COPY --from=builder --chown=${USER}:${USER} /app/main.py /app/main.py

ENV VIRTUAL_ENV=/app/.venv
ENV PATH=/app/.venv/bin:${PATH}
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
CMD ["sh", "-c", "effective_cpus() { base=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1); if read -r quota period < /sys/fs/cgroup/cpu.max 2>/dev/null; then if [ $quota != max ]; then echo $(( (quota + period - 1) / period )); return; fi; fi; echo $base; }; CPUS=$(effective_cpus); WORKERS=${WORKERS:-$(( CPUS * 2 + 1 ))}; FORWARDED_ALLOW_IPS=${FORWARDED_ALLOW_IPS:-*}; GUNICORN_CONF=$(python -c 'import servicekit, os; print(os.path.join(os.path.dirname(servicekit.__file__), \"gunicorn.conf.py\"))'); exec gunicorn -c \"${GUNICORN_CONF}\" -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:${PORT} --workers ${WORKERS} --timeout ${TIMEOUT} --graceful-timeout ${GRACEFUL_TIMEOUT} --keep-alive ${KEEPALIVE} --forwarded-allow-ips=${FORWARDED_ALLOW_IPS} --max-requests ${MAX_REQUESTS} --max-requests-jitter ${MAX_REQUESTS_JITTER} --worker-tmp-dir /dev/shm"]
