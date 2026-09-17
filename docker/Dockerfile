# syntax=docker/dockerfile:1
#
# Multi-stage build so the final image ships only the resolved virtual
# environment and the application source — no uv binary, no build tools,
# no tests/, no .venv from the host. Build context is the repo root (see
# docker-compose.yml's `build.context: .`), so paths below are repo-relative.

# Pinned by digest: a tag like 3.11-slim is re-pointed on every upstream
# rebuild, so the same Dockerfile would otherwise build a different image
# over time. Shared by both stages. Update tag and digest together.
ARG PYTHON_IMAGE=python:3.11.16-slim-trixie@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

FROM ${PYTHON_IMAGE} AS builder

# Grab the pinned `uv` binary from its own distroless image rather than
# installing it via pip/curl into the build environment.
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /usr/local/bin/

WORKDIR /app

# The base image's system Python already satisfies pyproject.toml's
# requires-python (3.11+); skip uv's own Python download/discovery.
ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Dependencies first (cached independently of source changes) — installs
# everything in uv.lock except the project itself, and none of the `dev`
# extra (pytest/ruff/mypy have no business in a runtime image).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Now the project itself.
COPY README.md ./
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

FROM ${PYTHON_IMAGE} AS runtime

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/src /app/src

ENV PATH="/app/.venv/bin:${PATH}"
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=5 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/openapi.json')"

CMD ["uvicorn", "quant_risk_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
