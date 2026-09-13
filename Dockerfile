# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Dependency builder
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools (not in final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install uv==0.4.30 --no-cache-dir

# Copy only dependency manifest first (layer cache optimization)
COPY pyproject.toml ./

# Install production dependencies into /usr/local
RUN python3 -c "import tomllib; f=open('pyproject.toml', 'rb'); print('\n'.join(tomllib.load(f)['project']['dependencies']))" > /tmp/requirements.txt \
    && uv pip install --system --no-cache -r /tmp/requirements.txt

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Production image
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS production

LABEL org.opencontainers.image.title="SWE Agent API"
LABEL org.opencontainers.image.version="1.0.0"

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    git \
    patch \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup pyproject.toml ./

# Switch to non-root
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# Expose API port
EXPOSE 8000

# Default: run the API server
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--log-config", "/dev/null"]


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: Polyglot Multi-Language Sandbox image (Docker execution)
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS sandbox

LABEL org.opencontainers.image.title="SWE Agent Polyglot Sandbox"

# Install system tools plus Node.js, npm, Go, and Rust/Cargo
RUN apt-get update && apt-get install -y --no-install-recommends \
    patch \
    git \
    curl \
    ca-certificates \
    build-essential \
    cmake \
    nodejs \
    npm \
    golang-go \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Pre-install common Python test dependencies
RUN pip install --no-cache-dir \
    pytest==8.3.0 \
    pytest-asyncio==0.24.0 \
    pytest-cov==6.0.0 \
    pytest-json-report==1.5.0 \
    pytest-timeout==2.3.1

# Sandbox workspace — all code goes here
RUN mkdir -p /workspace
WORKDIR /workspace

# Non-root user in sandbox
RUN groupadd --gid 2001 sandbox && \
    useradd --uid 2001 --gid sandbox --shell /bin/sh sandboxuser
USER sandboxuser

CMD ["sleep", "infinity"]
