# syntax=docker/dockerfile:1.7

############################
# Builder: export deps & build wheels
############################
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps required for building wheels of some packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Poetry + export plugin (so `poetry export` exists)
RUN python -m pip install --upgrade pip \
    && python -m pip install "poetry==1.8.3" "poetry-plugin-export==1.8.0"

# Copy only dependency files first for better layer caching
COPY pyproject.toml poetry.lock ./

# Export locked deps to requirements.txt
RUN poetry export -f requirements.txt --output /tmp/requirements.txt --without-hashes

# Build wheels for all deps (offline-friendly install in runtime)
RUN python -m pip wheel --wheel-dir /wheels -r /tmp/requirements.txt

# Copy project sources and build project wheel as well
COPY . /app

# Build wheel of the project itself
RUN python -m pip wheel --wheel-dir /wheels .


############################
# Runtime: minimal image, non-root, install from wheels
############################
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # Default config path inside container (you can override via env)
    ANTIFRAUD_CONFIG=/app/configs/dev.yaml

WORKDIR /app

# Create non-root user
RUN addgroup --system app \
    && adduser --system --ingroup app --home /app app

# Copy wheels from builder and install offline
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels /wheels/*.whl \
    && rm -rf /wheels

# Copy only runtime-needed project files (configs, etc.)
# (Package code is already installed from wheel)
COPY --chown=app:app configs /app/configs

USER app

EXPOSE 8000

# Healthcheck using stdlib (no curl)
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()" || exit 1

# Use uvicorn factory so create_app() is called on startup
CMD ["python", "-m", "uvicorn", "app.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
