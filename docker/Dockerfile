# syntax=docker/dockerfile:1.9
# exo Docker image — multi-stage build with optional CUDA support
# Build args:
#   BASE_IMAGE  — base image (default: python:3.13-slim-bookworm)
#   CUDA_TAG    — set to e.g. "12.8.0-runtime-ubuntu24.04" for GPU support
#   EXO_EXTRA   — uv extra to install (default: mlx-cuda12; use "mlx" for CPU-only)
#
# Usage:
#   docker build -t exo:latest .
#   docker build --build-arg CUDA_TAG=12.8.0-runtime-ubuntu24.04 -t exo:cuda .
#   docker run --rm -p 52415:52415 -v exo-models:/root/.cache/huggingface exo:latest

ARG BASE_IMAGE=python:3.13-slim-bookworm
FROM ${BASE_IMAGE} AS base

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: System dependencies
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS system-deps

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PATH="/root/.local/bin:$PATH"

# Core system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    # Rust build deps (for exo-rs nanobind bindings)
    cmake \
    ninja-build \
    # Node.js for dashboard build
    nodejs \
    npm \
    # Runtime deps
    libffi8 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Dashboard build (SvelteKit → static assets)
# ─────────────────────────────────────────────────────────────────────────────
FROM system-deps AS dashboard-build

WORKDIR /build/dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci --prefer-offline

COPY dashboard/ ./
RUN npm run build

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Rust bindings build (exo_rs → shared library)
# ─────────────────────────────────────────────────────────────────────────────
FROM system-deps AS rust-build

# Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain stable --profile minimal

ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /build/rust
COPY Cargo.toml Cargo.lock ./
COPY rust/ ./rust/

# Build exo_rs cdylib
RUN cargo build --release --package exo_rs && \
    find target/release -name "libexo_rs.so" -o -name "exo_rs*.so" | head -1

# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Python environment + exo install
# ─────────────────────────────────────────────────────────────────────────────
FROM system-deps AS python-env

ARG EXO_EXTRA=mlx-cuda12

WORKDIR /app

# Copy pyproject + workspace config first for layer caching
COPY pyproject.toml uv.lock* ./
COPY rust/exo_rs/Cargo.toml ./rust/exo_rs/
COPY rust/networking/Cargo.toml ./rust/networking/
COPY bench/pyproject.toml ./bench/
COPY tools/pyproject.toml ./tools/

# Install Python dependencies (with optional CUDA extra)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra "${EXO_EXTRA}" --no-dev --frozen || \
    uv sync --extra "${EXO_EXTRA}" --no-dev

# Copy source
COPY src/ ./src/
COPY rust/ ./rust/
COPY Cargo.toml Cargo.lock ./
COPY bench/ ./bench/
COPY tools/ ./tools/
COPY scripts/ ./scripts/

# Copy pre-built Rust bindings
COPY --from=rust-build /build/rust/target/release/*.so ./src/exo_rs/

# Install exo itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps -e .

# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Final runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    # exo config
    EXO_MODELS_DIR="/root/.cache/huggingface" \
    EXO_LOG="/var/log/exo/exo.log" \
    EXO_NODE_ID="" \
    # Network
    EXO_PORT=52415 \
    EXO_API_PORT=52415

# Minimal runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libffi8 \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/log/exo /root/.cache/huggingface

# Copy Python environment from build stage
COPY --from=python-env /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy dashboard static assets
COPY --from=dashboard-build /build/dashboard/build /app/dashboard/build

# Copy application source
COPY --from=python-env /app/src /app/src
COPY --from=python-env /app/rust /app/rust
COPY --from=python-env /app/scripts /app/scripts
COPY --from=python-env /app/pyproject.toml /app/

WORKDIR /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${EXO_API_PORT}/v1/models || exit 1

# Expose API + dashboard port
EXPOSE 52415

# Entrypoint
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["exo"]
