#!/usr/bin/env bash
# setup_docker.sh — Build and run exo in Docker with optional GPU support
#
# Usage:
#   bash scripts/setup_docker.sh              # build CPU-only image
#   bash scripts/setup_docker.sh --gpu        # build CUDA image
#   bash scripts/setup_docker.sh --run        # build + run single node
#   bash scripts/setup_docker.sh --run --gpu  # build + run with GPU
#   bash scripts/setup_docker.sh --cluster 3  # build + run 3-node cluster
#
# Environment variables:
#   EXO_DOCKER_TAG   — image tag (default: exo:latest)
#   EXO_MODELS_DIR   — host path for model cache (default: ~/.cache/huggingface)
#   EXO_PORT         — host port to expose (default: 52415)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ─── Defaults ───────────────────────────────────────────────────────────────
GPU=0
RUN=0
CLUSTER=0
CLUSTER_SIZE=0
EXO_DOCKER_TAG="${EXO_DOCKER_TAG:-exo:latest}"
EXO_MODELS_DIR="${EXO_MODELS_DIR:-$HOME/.cache/huggingface}"
EXO_PORT="${EXO_PORT:-52415}"

# ─── Parse args ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)    GPU=1; shift ;;
    --run)    RUN=1; shift ;;
    --cluster)
      CLUSTER=1
      CLUSTER_SIZE="${2:-3}"
      [[ "$CLUSTER_SIZE" =~ ^[0-9]+$ ]] || { echo "ERROR: --cluster needs a number"; exit 1; }
      shift 2
      ;;
    --tag)    EXO_DOCKER_TAG="$2"; shift 2 ;;
    --models) EXO_MODELS_DIR="$2"; shift 2 ;;
    --port)   EXO_PORT="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--gpu] [--run] [--cluster N] [--tag TAG] [--models DIR] [--port PORT]"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ─── Colors ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

say()  { printf "${GREEN}[✓]${NC} %s\n" "$1"; }
warn() { printf "${YELLOW}[!]${NC} %s\n" "$1"; }
fail() { printf "${RED}[✗]${NC} %s\n" "$1"; }

# ─── Preflight checks ──────────────────────────────────────────────────────
echo "=== Preflight checks ==="

if ! command -v docker &>/dev/null; then
  fail "Docker not found. Install from https://docs.docker.com/get-docker/"
  exit 1
fi
say "Docker found: $(docker --version)"

if ! docker info &>/dev/null; then
  fail "Docker daemon not running. Start Docker first."
  exit 1
fi
say "Docker daemon running"

if [[ $GPU -eq 1 ]]; then
  if ! command -v nvidia-smi &>/dev/null; then
    fail "nvidia-smi not found — GPU build requires NVIDIA drivers"
    exit 1
  fi
  if ! docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi &>/dev/null; then
    warn "NVIDIA Container Toolkit not detected or not working"
    warn "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    warn "Falling back to CPU build..."
    GPU=0
  else
    say "NVIDIA GPU + Container Toolkit verified"
  fi
fi

# ─── Build ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Building exo Docker image ==="

BUILD_ARGS=()
if [[ $GPU -eq 1 ]]; then
  BUILD_ARGS+=(--build-arg "BASE_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu24.04")
  BUILD_ARGS+=(--build-arg "EXO_EXTRA=mlx-cuda12")
  say "Building CUDA-enabled image..."
else
  BUILD_ARGS+=(--build-arg "BASE_IMAGE=python:3.13-slim-bookworm")
  BUILD_ARGS+=(--build-arg "EXO_EXTRA=mlx")
  say "Building CPU-only image..."
fi

cd "$REPO_ROOT"
docker build "${BUILD_ARGS[@]}" -t "$EXO_DOCKER_TAG" -f docker/Dockerfile .
say "Image built: $EXO_DOCKER_TAG"

# ─── Run single node ────────────────────────────────────────────────────────
if [[ $RUN -eq 1 && $CLUSTER -eq 0 ]]; then
  echo ""
  echo "=== Starting exo node ==="

  RUN_ARGS=(
    --rm
    -it
    -p "${EXO_PORT}:52415"
    -v "${EXO_MODELS_DIR}:/root/.cache/huggingface"
    -v "exo-logs:/var/log/exo"
    --name "exo-node-1"
  )

  if [[ $GPU -eq 1 ]]; then
    RUN_ARGS+=(--gpus all)
  fi

  say "Starting exo on port $EXO_PORT..."
  say "Dashboard: http://localhost:${EXO_PORT}"
  say "API:       http://localhost:${EXO_PORT}/v1"
  say "Models:    $EXO_MODELS_DIR"
  echo ""

  docker run "${RUN_ARGS[@]}" "$EXO_DOCKER_TAG"
fi

# ─── Run cluster ────────────────────────────────────────────────────────────
if [[ $CLUSTER -eq 1 ]]; then
  echo ""
  echo "=== Starting exo cluster ($CLUSTER_SIZE nodes) ==="

  # Create a Docker network for the cluster
  docker network create exo-cluster 2>/dev/null || true
  say "Docker network: exo-cluster"

  # Start each node
  for i in $(seq 1 "$CLUSTER_SIZE"); do
    NODE_NAME="exo-node-$i"
    NODE_PORT=$((EXO_PORT + i - 1))

    RUN_ARGS=(
      -d
      --rm
      --network exo-cluster
      --name "$NODE_NAME"
      -p "${NODE_PORT}:52415"
      -v "${EXO_MODELS_DIR}:/root/.cache/huggingface"
      -v "exo-logs-${i}:/var/log/exo"
      -e "EXO_NODE_NAME=${NODE_NAME}"
    )

    if [[ $GPU -eq 1 ]]; then
      # Distribute GPUs across nodes (if available)
      RUN_ARGS+=(--gpus all)
    fi

    docker run "${RUN_ARGS[@]}" "$EXO_DOCKER_TAG" &
    say "Started $NODE_NAME on port $NODE_PORT"
  done

  wait
  echo ""
  say "Cluster running! Nodes:"
  for i in $(seq 1 "$CLUSTER_SIZE"); do
    echo "  Node $i: http://localhost:$((EXO_PORT + i - 1))"
  done
  echo ""
  echo "To stop the cluster:"
  echo "  docker network rm exo-cluster"
  echo "  docker volume rm exo-logs-{1..$CLUSTER_SIZE}"
fi

# ─── Done ───────────────────────────────────────────────────────────────────
if [[ $RUN -eq 0 && $CLUSTER -eq 0 ]]; then
  echo ""
  say "Build complete! Run with:"
  echo "  docker run --rm -p ${EXO_PORT}:52415 -v ${EXO_MODELS_DIR}:/root/.cache/huggingface $EXO_DOCKER_TAG"
  echo ""
  echo "Or use docker-compose:"
  echo "  docker compose -f docker/docker-compose.yml up"
fi
