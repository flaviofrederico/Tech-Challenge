# syntax=docker/dockerfile:1

# ---- Stage 1: build the virtual environment with Poetry ----
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app

# Copy only dependency manifests first to leverage Docker layer caching:
# dependencies only get reinstalled when pyproject.toml/poetry.lock change.
COPY pyproject.toml poetry.lock ./

RUN poetry install --only main --no-root --no-directory

COPY src ./src
COPY README.md ./

RUN poetry install --only main

# ---- Stage 2: slim runtime image ----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY src ./src
COPY configs ./configs

RUN mkdir -p data/raw data/processed models \
    && chown -R appuser:appuser /app

USER appuser

# Default: run the full pipeline (preprocess -> train) end to end.
# Mount ./data and a volume for mlflow.db/mlruns when running in production
# to persist datasets, tracking metadata and model artifacts across runs.
CMD ["sh", "-c", "python -m purchase_intention.pipeline.preprocess --config configs/config.yaml && python -m purchase_intention.pipeline.train --config configs/config.yaml"]