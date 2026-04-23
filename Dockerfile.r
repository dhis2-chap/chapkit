FROM --platform=linux/amd64 ghcr.io/dhis2-chap/docker_r_inla:master

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --chown=root:root ./pyproject.toml ./uv.lock README.md ./
COPY --chown=root:root ./src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.13 && \
    uv sync --frozen --no-dev

WORKDIR /work

EXPOSE 8000

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["chapkit", "run", ".", "--host", "0.0.0.0", "--port", "8000"]
