# chapkit-r-inla: R 4.5 + INLA + spatial/time-series R stack + Python 3.13 + chapkit + uv.
# amd64 only (INLA ships x86_64 Linux binaries only).
#
# Two-stage build:
#   1. inla-builder: full toolchain, compiles INLA + R packages against
#      CRAN and the INLA binary repo, then strips debug symbols and
#      removes help/docs from the compiled site-library.
#   2. runtime:      trixie-slim + r-base + runtime shared libs + the
#      copied site-library + uv-managed Python 3.13 + chapkit venv.
#
# Pre-installed R packages (aligned with chap-core R-model needs):
#   fmesher, INLA, dlnm, yaml, jsonlite, dplyr, readr,
#   sf, spdep, sn, tsModel, xgboost, pak
# tidyverse is intentionally NOT installed (adds ~400 MB); install in a
# downstream layer if you need it.
#
# Security: runs as root. Non-root hardening needs the volume-mapping
# dance from chap-core/compose.yml and is a deferred follow-up.

ARG BASE_PLATFORM=linux/amd64

#############################
# Stage 1: build INLA + R packages
#############################
FROM --platform=${BASE_PLATFORM} debian:trixie-slim AS inla-builder

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        r-base r-base-dev \
        build-essential gfortran cmake \
        libcurl4-openssl-dev libssl-dev libxml2-dev \
        libgdal-dev libgeos-dev libproj-dev libudunits2-dev \
        libgsl-dev libfontconfig1-dev \
    && rm -rf /var/lib/apt/lists/*

# fmesher first (dependency for modern INLA), then INLA itself with
# dep=FALSE (avoid the huge Suggests chain), then the chap-core parity
# R package set. Final inla.prune() drops INLA examples/documentation.
RUN R -q -e "install.packages('fmesher', \
        repos = c('https://cloud.r-project.org', INLA = 'https://inla.r-inla-download.org/R/stable'))" \
    && R -q -e "install.packages('INLA', \
        repos = c('https://cloud.r-project.org', INLA = 'https://inla.r-inla-download.org/R/stable'), \
        dep = FALSE)" \
    && R -q -e "install.packages(c('dlnm','yaml','jsonlite','dplyr','readr','sf','spdep','sn','tsModel','xgboost','pak','renv'), \
        repos='https://cloud.r-project.org')" \
    && R -q -e "library(INLA); INLA::inla.prune()"

# Strip debug symbols + drop help/docs to shrink the site-library.
RUN find /usr/local/lib/R/site-library -name "*.so" -exec strip --strip-debug {} \; 2>/dev/null || true \
    && find /usr/local/lib/R/site-library -type d \( -name help -o -name doc -o -name html \) -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/R/site-library -name "*.pdf" -delete 2>/dev/null || true

#############################
# Stage 2: runtime
#############################
FROM --platform=${BASE_PLATFORM} debian:trixie-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:${PATH}"

# Full R toolchain (so users can install extra CRAN packages or run
# renv::restore() at container runtime) + runtime shared libs for the
# compiled R packages copied from the builder.
#
# r-base-dev + dev headers intentionally present here: avoiding them
# would save ~150 MB but break install.packages()/renv::restore() for
# anything not already in the pre-baked site-library, which defeats a
# lot of the reason for a fat image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        r-base r-base-dev gfortran \
        # Dev headers for the same libs that INLA's package set depends
        # on - lets users install downstream R packages against them.
        libcurl4-openssl-dev libssl-dev libxml2-dev \
        libgdal-dev libgeos-dev libproj-dev libudunits2-dev \
        libgsl-dev libfontconfig1-dev \
        # Build tools for Python ML wheels that fall through to source.
        build-essential pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY --from=inla-builder /usr/local/lib/R/site-library /usr/local/lib/R/site-library
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /usr/local/bin/

WORKDIR /app

COPY ./pyproject.toml ./uv.lock README.md ./
COPY ./src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.13 && \
    uv sync --frozen --no-dev

WORKDIR /work

EXPOSE 8000

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["chapkit", "run", ".", "--host", "0.0.0.0", "--port", "8000"]
