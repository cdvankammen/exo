# Local Setup: Linux + NVIDIA GPU (exo)

> **Use this guide when setting up exo on any new Linux box with NVIDIA GPUs.**
> Covers driver mismatches, MLX two-package trap, Blackwell CUDA requirements, and torch uv.lock bugs.
> Source: battle-tested on 2026-08-06 deploying to a Proxmox VM (Ubuntu 24.04) with 2x RTX 5060 Ti.

---

## Prerequisites

- Ubuntu 24.04+ (or similar Debian-based Linux)
- NVIDIA GPU(s) — identify your architecture first:
  - Ada (RTX 40xx) = sm_89 → needs CUDA 12.4+
  - Hopper (H100) = sm_90 → needs CUDA 12.4+
  - Blackwell (RTX 50xx) = sm_120 → **needs CUDA 12.8+**
- `uv` installed
- exo repo cloned

```bash
# Identify your hardware
lspci | grep -i nvidia
cat /proc/driver/nvidia/version          # exact kernel module version
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

---

## Step 1: Fix Driver/Library Version Mismatch

**Symptom:** `nvidia-smi: Failed to initialize NVML: Driver/library version mismatch`

The kernel module (loaded at boot) and userspace libraries (what nvidia-smi/CUDA load) must be the exact same version. apt often installs a different version than what the kernel loaded.

```bash
# 1. Find the EXACT kernel module version
cat /proc/driver/nvidia/version
# Example output: NVRM version: NVIDIA UNIX Open Kernel Module ... 595.58.03

# 2. Remove WRONG userspace packages
apt-get remove -y libnvidia-compute-535 nvidia-utils-535   # adjust to whatever mismatch exists

# 3. Download NVIDIA's official .run installer for the EXACT version from step 1
DRIVER_VERSION="595.58.03"  # CHANGE THIS to match your kernel module
curl -fsSL -o /tmp/nvidia-driver.run \
  "https://download.nvidia.com/XFree86/Linux-x86_64/${DRIVER_VERSION}/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"

# 4. Extract WITHOUT installing (we only want userspace libs, not the kernel module)
cd /tmp && sh nvidia-driver.run -x

# 5. Copy libs into system library dir
SYS=/usr/lib/x86_64-linux-gnu
cd /tmp/NVIDIA-Linux-x86_64-${DRIVER_VERSION}
cp libcuda.so.${DRIVER_VERSION} \
   libnvidia-ml.so.${DRIVER_VERSION} \
   libnvidia-ptxjitcompiler.so.${DRIVER_VERSION} \
   libnvidia-tls.so.${DRIVER_VERSION} \
   libnvidia-gpucomp.so.${DRIVER_VERSION} "$SYS/"
cp nvidia-smi /usr/bin/nvidia-smi-${DRIVER_VERSION} && chmod 755 /usr/bin/nvidia-smi-${DRIVER_VERSION}

# 6. Point generic symlinks at our exact version
ln -sf libcuda.so.${DRIVER_VERSION}                  "$SYS/libcuda.so.1"
ln -sf libcuda.so.1                                  "$SYS/libcuda.so"
ln -sf libnvidia-ml.so.${DRIVER_VERSION}             "$SYS/libnvidia-ml.so.1"
ln -sf libnvidia-ptxjitcompiler.so.${DRIVER_VERSION} "$SYS/libnvidia-ptxjitcompiler.so.1"
ln -sf libnvidia-tls.so.${DRIVER_VERSION}            "$SYS/libnvidia-tls.so.1"
ln -sf libnvidia-gpucomp.so.${DRIVER_VERSION}        "$SYS/libnvidia-gpucomp.so.1"

# 7. Verify
nvidia-smi-${DRIVER_VERSION}
```

**Warning:** Any future `apt install` of nvidia packages will clobber these symlinks. Re-run steps 5-6 after any such install.

---

## Step 2: Install CUDA Toolchain for Your GPU Architecture

**Symptom:** `nvrtc: error: invalid value for --gpu-architecture (-arch)` during model warmup.

MLX JIT-compiles GPU kernels at runtime via nvrtc. If your CUDA version is older than your GPU architecture, compilation fails.

Know your GPU's compute capability:
- Ada Lovelace (RTX 40xx) = sm_89 → needs CUDA 12.4+
- Hopper (H100) = sm_90 → needs CUDA 12.4+
- Blackwell (RTX 50xx) = sm_120 → **needs CUDA 12.8+**

```bash
# 0. Add NVIDIA's CUDA keyring repo FIRST (required before cuda-nvrtc packages are visible)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update

# 1. Install CUDA toolchain for your GPU arch
# For Blackwell (RTX 50xx, sm_120) — MUST be CUDA 12.8+
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  cuda-nvrtc-12-8 cuda-nvvm-12-8 cuda-cudart-12-8 cuda-cudart-dev-12-8

# Symlink into system path
SYS=/usr/lib/x86_64-linux-gnu
NEW=/usr/local/cuda-12.8/targets/x86_64-linux/lib
ln -sfn "$NEW/libnvrtc.so.12.8.93" "$SYS/libnvrtc.so.12"
ln -sfn "$NEW/libnvrtc.so.12"      "$SYS/libnvrtc.so"
ln -sfn /usr/local/cuda-12.8/nvvm/lib64/libnvvm.so.4.0.0 "$SYS/libnvvm.so.4"
ln -sfn "$SYS/libnvvm.so.4" "$SYS/libnvvm.so"

# Set CUDA_HOME (required for kernel header includes)
ln -sfn /usr/local/cuda-12.8 /usr/local/cuda
# Also add to /etc/environment:
#   CUDA_HOME=/usr/local/cuda
#   CUDA_PATH=/usr/local/cuda

# Verify JIT compilation works
cd /path/to/exo
CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda .venv/bin/python -c "
import mlx.core as mx
@mx.compile
def f(x): return mx.tanh(x * 2.0)
a = mx.random.normal((1024, 1024))
b = f(a); mx.eval(b)
print('JIT compile OK', b.shape)
"
```

---

## Step 3: Install MLX Wheels (Two-Package Trap)

**Symptom:** `ImportError: libmlx.so: cannot open shared object file`

On Linux x86_64, MLX is **two** packages: a slim Python glue wheel (`mlx`) and a fat runtime wheel (`mlx_cuda_12`) containing `libmlx.so`. Installing only the slim wheel fails.

```bash
# Download both wheels from the fork release matching your lock file
BASE="https://github.com/rltakashige/mlx-jaccl-fix-small-recv/releases/download/mlx_cuda"
cd /tmp && mkdir -p mlxwheel && cd mlxwheel
curl -fsSL -O "$BASE/mlx-0.32.0-cp313-cp313-manylinux_2_35_x86_64.whl"
curl -fsSL -O "$BASE/mlx_cuda_12-0.32.0-py3-none-manylinux_2_35_x86_64.whl"

# CRITICAL: verify architecture before installing (wrong arch = silent failure)
unzip -p mlx_cuda_12-*.whl mlx/lib/libmlx.so | file -
# Must say: ELF 64-bit LSB shared object, x86-64
# If it says ARM aarch64, you have the wrong wheel.

# Install both wheels
cd /path/to/exo
uv pip install --force-reinstall --no-deps \
  /tmp/mlxwheel/mlx-0.32.0-cp313-cp313-manylinux_2_35_x86_64.whl \
  /tmp/mlxwheel/mlx_cuda_12-0.32.0-py3-none-manylinux_2_35_x86_64.whl

# Verify
uv run python -c "import mlx.core as mx; print(mx.default_device())"
# Expected: Device(gpu, 0)
```

**Warning:** `uv sync` re-runs may revert to the URL wheel (which lacks the runtime). Keep the wheels in `/tmp/mlxwheel` (or a durable location) and re-install after every sync.

---

## Step 4: Fix torch Resolution (uv.lock Marker Bug)

**Symptom:** `ModuleNotFoundError: No module named 'torch'` — vision disabled but text still works.

Root cause: uv 0.12.x mis-evaluates multi-extra marker clauses in uv.lock, dropping the torch edge on Linux even though pyproject.toml declares it correctly.

```bash
# In uv.lock, find the exo package's torch/torchaudio/torchvision linux-mlx edges.
# Change:
#   marker = "sys_platform == 'linux' and extra == 'mlx'
#             and extra != 'mlx-cpu' and extra != 'mlx-cuda12' and extra != 'mlx-cuda13'"
# To:
#   marker = "sys_platform == 'linux' and extra == 'mlx'"

# Then sync
uv sync --extra mlx

# Verify
.venv/bin/python -c "import torch; print(torch.__version__)"
# Expected: 2.10.0+cu130 (or similar)
```

**Critical:** Do NOT blindly re-run `uv lock` or plain `uv sync` on macOS — it regenerates the lock in a format that also fails to resolve torch on Linux. If the lock is regenerated, re-apply this patch and verify with `uv sync --extra mlx --dry-run` on the Linux box.

---

## Step 5: Sanity Checks

Run all three before declaring the box ready:

```bash
# 1. JIT compile test (exercises nvrtc + MLX CUDA backend)
CUDA_HOME=/usr/local/cuda .venv/bin/python -c "
import mlx.core as mx
@mx.compile
def f(x): return mx.tanh(x * 2.0)
a = mx.random.normal((1024, 1024))
b = f(a); mx.eval(b)
print('JIT OK:', b.shape)
"

# 2. Torch import
.venv/bin/python -c "import torch; print('torch OK:', torch.__version__)"

# 3. Place a model (min_nodes=1) via dashboard or API, verify:
#    - Warmup completes without nvrtc errors
#    - GPU memory allocated (check nvidia-smi)
#    - Chat generation works
```

---

## Step 6: Start exo

Always start detached so SSH disconnects don't kill the process:

```bash
cd /path/to/exo
setsid nohup env CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda \
  uv run exo > /tmp/exo.log 2>&1 < /dev/null &
```

---

## Proxmox VM Notes

If running inside a Proxmox VM with PCIe passthrough:

- The kernel module is loaded by the VM's own kernel, not the host's. The host's GPU driver version is irrelevant.
- NICs are virtio paravirtual devices. There is no Thunderbolt/RDMA — this node can only do plain socket (RING) or single-node inference.
- Never install `nvidia-driver-*` metapackages in the VM (they try DKMS builds needing kernel headers Proxmox doesn't provide). Install only userspace pieces.
- After VM reboot: re-verify nvidia-smi, libnvrtc symlinks, and CUDA_HOME before starting exo.
- Store extracted NVIDIA .run dir and MLX wheels somewhere durable (not /tmp) since tmpfs may wipe on reboot.

---

## Multi-Node Notes (Linux Box in Cluster)

The Linux box has virtio NICs (no Thunderbolt/RDMA hardware), so it can **never** join an RDMA (JACCL) ring. Use plain sockets:

- Always place with `instance_meta: MlxRing` for the Linux node.
- Pipeline sharding works: Mac shard 0 (Metal) <-> socket <-> Linux shard 1 (CUDA).
- Expect slower inter-node transfers than Mac↔Mac TB4 (virtio bridge ~1–10 Gbps).

See `promptsANDplans/research/rdma-multi-node-investigation.md` for full topology guards and verification.

---

## Reference: Known Working Configuration

| Component | Version |
|-----------|--------|
| OS | Ubuntu 24.04.4 LTS (kernel 6.17.13) |
| GPUs | 2x NVIDIA RTX 5060 Ti 16GB (Blackwell, sm_120) |
| Kernel module | NVIDIA 595.58.03 |
| Userspace | 595.58.03 (from .run extraction) |
| CUDA runtime | 12.8 (nvrtc 12.8.93) |
| MLX | 0.32.0 (fork) + mlx_cuda_12 0.32.0 |
| torch | 2.10.0+cu130 |

---

## Security: API Bind Host and Authentication

The exo HTTP API server can be restricted to localhost or authenticated with a bearer token.

### `EXO_API_HOST` (default `0.0.0.0`)

Controls which network address the API binds to. The default `0.0.0.0` exposes the API to all network interfaces — suitable for trusted LANs but risky on shared or public networks.

```bash
# Bind to localhost only (safe for single-machine or SSH-tunneled access)
EXO_API_HOST=127.0.0.1 uv run exo

# Explicit LAN bind (default behavior)
EXO_API_HOST=0.0.0.0 uv run exo
```

### `EXO_API_TOKEN` (default: unset)

When set, all API requests must include `Authorization: Bearer <token>`. Combine with `EXO_API_HOST=127.0.0.1` for defense-in-depth on exposed hosts.

```bash
EXO_API_HOST=127.0.0.1 EXO_API_TOKEN=my-secret-token uv run exo
```
| Python | 3.13.14 (uv-managed) |
