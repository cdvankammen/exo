#!/usr/bin/env bash
# exo Docker entrypoint — handles node identity, model dirs, and graceful shutdown
set -euo pipefail

# ─── Defaults ───────────────────────────────────────────────────────────────
: "${EXO_MODELS_DIR:=/root/.cache/huggingface}"
: "${EXO_LOG:=/var/log/exo/exo.log}"
: "${EXO_PORT:=52415}"
: "${EXO_NODE_ID:=}"
: "${EXO_NODE_NAME:=}"

# Generate a stable node ID from hostname if not set
if [[ -z "$EXO_NODE_ID" ]]; then
  export EXO_NODE_ID="docker-$(hostname | md5sum | head -c 8)"
fi

# Set node name from hostname if not set
if [[ -z "$EXO_NODE_NAME" ]]; then
  export EXO_NODE_NAME="$(hostname)"
fi

# Ensure log directory exists
mkdir -p "$(dirname "$EXO_LOG")" 2>/dev/null || true
mkdir -p "$EXO_MODELS_DIR" 2>/dev/null || true

echo "╔══════════════════════════════════════════════════════╗"
echo "║  exo Docker node starting                           ║"
echo "║  Node ID:   ${EXO_NODE_ID}"
echo "║  Node Name: ${EXO_NODE_NAME}"
echo "║  Models:    ${EXO_MODELS_DIR}"
echo "║  Log:       ${EXO_LOG}"
echo "║  Port:      ${EXO_PORT}"
echo "╚══════════════════════════════════════════════════════╝"

# ─── GPU detection (if NVIDIA runtime) ──────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
  echo "GPU detected:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
  export EXO_BACKEND="mlxcuda"
fi

# ─── Signal handling for graceful shutdown ───────────────────────────────────
EXO_PID=""
cleanup() {
  echo "Received shutdown signal, stopping exo gracefully..."
  if [[ -n "$EXO_PID" ]] && kill -0 "$EXO_PID" 2>/dev/null; then
    kill -TERM "$EXO_PID"
    wait "$EXO_PID" 2>/dev/null || true
  fi
  exit 0
}
trap cleanup SIGTERM SIGINT SIGQUIT

# ─── Start exo ──────────────────────────────────────────────────────────────
# If first arg is "exo" or starts with "--", run exo with it
if [[ "${1:-}" == "exo" || "${1:-}" == --* ]]; then
  # Replace "exo" with the actual command
  if [[ "${1:-}" == "exo" ]]; then
    shift || true
  fi
  exec exo "$@" &
  EXO_PID=$!
else
  # Run arbitrary command
  exec "$@" &
  EXO_PID=$!
fi

wait "$EXO_PID" 2>/dev/null || true
