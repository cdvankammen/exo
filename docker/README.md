# exo Docker Setup

Run exo in Docker containers for isolated, reproducible deployments with optional GPU acceleration.

## Quick Start

### Build the image

```bash
# CPU-only (default)
docker build -t exo:latest -f docker/Dockerfile .

# With CUDA GPU support
docker build --build-arg BASE_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu24.04 \
             --build-arg EXO_EXTRA=mlx-cuda12 \
             -t exo:cuda -f docker/Dockerfile .
```

### Run a single node

```bash
docker run --rm -it \
  -p 52415:52415 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v exo-logs:/var/log/exo \
  exo:latest
```

Access the dashboard at http://localhost:52415

### Run with GPU

```bash
docker run --rm -it \
  --gpus all \
  -p 52415:52415 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  exo:cuda
```

## Using the Setup Script

The `scripts/setup_docker.sh` script automates the build and run process:

```bash
# Build CPU-only image
bash scripts/setup_docker.sh

# Build and run with GPU
bash scripts/setup_docker.sh --gpu --run

# Run a 3-node cluster
bash scripts/setup_docker.sh --cluster 3

# Custom model directory
bash scripts/setup_docker.sh --run --models /path/to/models
```

## Docker Compose

### Single node

```bash
docker compose -f docker/docker-compose.yml up
```

### GPU-enabled node

```bash
docker compose -f docker/docker-compose.yml --profile gpu up
```

### Multi-node cluster

```bash
docker compose -f docker/docker-compose.yml --profile cluster up
```

This starts 3 nodes on ports 52415, 52416, 52417. Nodes automatically discover each other via the Docker network.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EXO_PORT` | `52415` | Host port to expose the API/dashboard |
| `EXO_MODELS_DIR` | `~/.cache/huggingface` | Host path for model cache |
| `EXO_NODE_ID` | auto-generated | Unique node identifier |
| `EXO_NODE_NAME` | hostname | Human-readable node name |
| `EXO_LOG` | `/var/log/exo/exo.log` | Log file path inside container |

## Build Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `BASE_IMAGE` | `python:3.13-slim-bookworm` | Base Docker image |
| `EXO_EXTRA` | `mlx-cuda12` | Python extra to install (`mlx`, `mlx-cuda12`, `mlx-cuda13`) |
| `CUDA_TAG` | (none) | CUDA version tag for GPU builds |

## Multi-Stage Build

The Dockerfile uses a multi-stage build to keep the final image small:

1. **system-deps** — Install system packages, uv, Node.js
2. **dashboard-build** — Build SvelteKit dashboard → static assets
3. **rust-build** — Compile exo_rs Rust bindings
4. **python-env** — Install Python dependencies + exo source
5. **runtime** — Minimal final image with only runtime dependencies

## GPU Requirements

For CUDA support:

1. **NVIDIA drivers** installed on the host
2. **NVIDIA Container Toolkit** installed:
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

3. Verify with: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`

## Model Persistence

Models are downloaded to `/root/.cache/huggingface` inside the container. Mount a host directory to persist them:

```bash
-v ~/.cache/huggingface:/root/.cache/huggingface
```

This avoids re-downloading models on every container restart.

## Troubleshooting

### Dashboard not loading

Check that port 52415 is not already in use:
```bash
lsof -i :52415
```

### GPU not detected

1. Verify NVIDIA Container Toolkit: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`
2. Check container logs: `docker logs exo-node`
3. Ensure you built with the CUDA base image

### Cluster nodes can't discover each other

Ensure all nodes are on the same Docker network:
```bash
docker network inspect exo-cluster
```

### Out of memory

Increase Docker's memory limit or reduce model size. Check available memory:
```bash
docker stats exo-node
```

## Development

To build with local changes:

```bash
docker build --no-cache -t exo:dev -f docker/Dockerfile .
```

To run tests inside the container:

```bash
docker run --rm exo:latest pytest
```

## Architecture

```
┌─────────────────────────────────────────┐
│  Docker Container                       │
│                                         │
│  ┌──────────────┐  ┌────────────────┐  │
│  │  exo API     │  │  Dashboard     │  │
│  │  :52415      │  │  (SvelteKit)   │  │
│  └──────┬───────┘  └────────────────┘  │
│         │                               │
│  ┌──────▼───────────────────────────┐  │
│  │  Python Runtime (exo)            │  │
│  │  - MLX backend (CPU/CUDA)        │  │
│  │  - Rust bindings (exo_rs)        │  │
│  │  - Zenoh networking              │  │
│  └──────────────────────────────────┘  │
│                                         │
│  Volumes:                               │
│  - /root/.cache/huggingface (models)   │
│  - /var/log/exo (logs)                 │
└─────────────────────────────────────────┘
```

## License

Apache 2.0 — same as exo itself.
