# Setup — exo on Linux with NVIDIA GPUs (Local)

> One document. Every command. From zero to a working exo cluster node with
> CUDA/MLX inference on a Linux box with NVIDIA GPUs.
>
> Proven on: **Ubuntu 24.04.4 LTS (Proxmox VM, kernel 6.17.13-2-pve), 2× RTX 5060 Ti 16GB** (2026-08-06).
> Source of truth: `promptsANDplans/research/linux-nvidia-cuda-setup-guide.md`.
> Branch tested on: `fix-memory-error`.

---

## The 4 problems this guide fixes (read this first)

| # | Symptom | Root cause | Fix |
|---|---------|-----------|-----|
| 1 | `nvidia-smi: Driver/library version mismatch` | apt installed userspace ≠ kernel module version | Extract exact-version userspace from NVIDIA `.run`, symlink over apt's |
| 2 | `ImportError: libmlx.so` | mlx on Linux = 2 wheels (slim `mlx` + fat `mlx_cuda_12` runtime); runtime wheel missing | Install the `mlx_cuda_12` wheel (or copy its `libmlx.so`) |
| 3 | `nvrtc: error: invalid value for --gpu-architecture (-arch)` | CUDA 12.0 (apt) can't compile for Blackwell (sm_120); needs CUDA 12.8+ | Install `cuda-nvrtc-12-8` + `cuda-nvvm-12-8` + dev headers, symlink into system |
| 4 | `ModuleNotFoundError: No module named 'torch'` → vision disabled | uv.lock torch edges darwin-gated (uv multi-extra marker bug) | Patch uv.lock markers (commit `f48bdf86`) |

---

## Prerequisites

| Requirement | Version | How to check |
|-------------|---------|--------------|
| Linux (Debian/Ubuntu) | 24.04 tested | `cat /etc/os-release` |
| NVIDIA GPU | any; see arch table below | `lspci \| grep -i nvidia` |
| NVIDIA driver kernel module | loaded at boot | `cat /proc/driver/nvidia/version` |
| uv | recent (0.12.x ok; lock caveat in §5) | `uv --version` |
| exo checkout | `fix-memory-error` branch | `git -C /path/to/exo branch --show-current` |

**GPU compute-capability → minimum CUDA (critical for RTX 50xx):**

| GPU arch | compute cap | Minimum CUDA for nvrtc/MLX JIT |
|----------|-------------|-------------------------------|
| Ada (RTX 40xx) | sm_90 | 12.4+ |
| **Blackwell (RTX 50xx)** | **sm_120** | **12.8+ (MUST)** |

RTX 5060 Ti = **Blackwell** = compute capability **sm_120**. Blackwell support only
exists in **CUDA 12.8+** compilers. Ubuntu 24.04's `nvidia-cuda-toolkit` ships
**CUDA 12.0** (Jan 2023) — 2+ years older than the GPU. That mismatch is problem #3.

### What you do NOT need
- No DKMS/kernel headers for the NVIDIA module (the module is loaded by the host/VM kernel; you never rebuild it)
- No `nvidia-driver-*` metapackage installs inside a Proxmox VM (see §6.3)
- No Thunderbolt / RDMA hardware (see §6.2 — virtio NICs mean socket/RING only)

---

## Step 0 — Identify GPU + driver state

```bash
lspci | grep -i nvidia
cat /proc/driver/nvidia/version          # exact kernel module version, e.g. 595.58.03
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

If `nvidia-smi` prints `Driver/library version mismatch`, do Step 1 before anything else.

---

## Step 1 — Fix driver/library version mismatch (problem #1)

### What's happening
The **kernel module** (in RAM, loaded at boot) and the **userspace libraries**
(what `nvidia-smi` and CUDA apps load from `/usr/lib/x86_64-linux-gnu/`) must be
the **same version**. NVML refuses to talk to a mismatched driver. The kernel
module CANNOT be changed without a reboot, so make the **userspace match the
kernel module exactly**.

### The fix (step by step)

```bash
# 1. Find the EXACT kernel module version
cat /proc/driver/nvidia/version
#   → NVRM version: NVIDIA UNIX Open Kernel Module ... 595.58.03

# 2. Remove the WRONG userspace packages apt installed
apt-get remove -y libnvidia-compute-535 nvidia-utils-535   # whatever mismatch exists

# 3. Download NVIDIA's official .run installer for the EXACT version
curl -fsSL -o /tmp/nvidia-595.run \
  "https://download.nvidia.com/XFree86/Linux-x86_64/595.58.03/NVIDIA-Linux-x86_64-595.58.03.run"

# 4. Extract WITHOUT installing (we only want the userspace libs, not the kernel module)
cd /tmp && sh nvidia-595.run -x
#   → creates NVIDIA-Linux-x86_64-595.58.03/ containing libcuda.so.595.58.03,
#     libnvidia-ml.so.595.58.03, libnvidia-ptxjitcompiler.so.595.58.03,
#     libnvidia-tls.so.595.58.03, libnvidia-gpucomp.so.595.58.03, nvidia-smi

# 5. Copy the libs into the system library dir
SYS=/usr/lib/x86_64-linux-gnu
cd /tmp/NVIDIA-Linux-x86_64-595.58.03
cp libcuda.so.595.58.03 libnvidia-ml.so.595.58.03 libnvidia-ptxjitcompiler.so.595.58.03 \
   libnvidia-tls.so.595.58.03 libnvidia-gpucomp.so.595.58.03 "$SYS/"
cp nvidia-smi /usr/bin/nvidia-smi-595.58 && chmod 755 /usr/bin/nvidia-smi-595.58

# 6. Point the generic symlinks at OUR version (overriding apt's 595.84)
ln -sf libcuda.so.595.58.03                "$SYS/libcuda.so.1"
ln -sf libcuda.so.1                        "$SYS/libcuda.so"
ln -sf libnvidia-ml.so.595.58.03           "$SYS/libnvidia-ml.so.1"
ln -sf libnvidia-ptxjitcompiler.so.595.58.03 "$SYS/libnvidia-ptxjitcompiler.so.1"
ln -sf libnvidia-tls.so.595.58.03          "$SYS/libnvidia-tls.so.1"
ln -sf libnvidia-gpucomp.so.595.58.03      "$SYS/libnvidia-gpucomp.so.1"

# 7. Verify
nvidia-smi-595.58
#   → NVIDIA-SMI 595.58.03  Driver Version: 595.58.03  CUDA Version: 13.2
#   → 2× NVIDIA GeForce RTX 5060 Ti  16311MiB each
```

### ⚠️ Watch-outs (hit twice)
- **apt installs will clobber these symlinks** whenever you install nvidia
  packages (e.g. `nvidia-cuda-toolkit` or `nvidia-utils-*`). Re-run steps 5–6
  after any such install.
- The `.run` installer's extraction (`-x`) is SAFE — it never touches the
  running kernel.
- Save the extracted dir (`/tmp/NVIDIA-Linux-x86_64-595.58.03`) or re-download;
  it may be wiped on reboot (tmpfs or manual cleanup). Prefer a durable path
  like `/root/toolchain/`.

---

## Step 2 — Install mlx wheels (problem #2: `ImportError: libmlx.so`)

### What's happening
**On Linux x86_64, "mlx" is TWO packages, not one:**

```
┌─ mlx (slim wheel, ~0.7 MB) ────────────────┐
│  mlx/core.cpython-313-...so  (Python glue) │
│  NEEDED: libmlx.so                         │
└────────────────────────────────────────────┘
        │  dlopen at import
        ▼
┌─ mlx_cuda_12 (fat wheel, ~337 MB) ─────────┐
│  mlx/lib/libmlx.so        (the real engine)│
│  mlx_cuda_12.libs/        (bundled CUDA)   │
└────────────────────────────────────────────┘
```

`uv sync` only pulls the slim wheel into the venv; the fat runtime wheel ends up
only in uv's cache. Without `libmlx.so`, the glue `.so` can't load.

### Bonus trap: wrong architecture in the cache
The uv cache can hold **4** mlx builds (CUDA12/CUDA13 × x86_64/arm64). Copying
the WRONG one (arm64) produces:
```
$ file libmlx.so → ELF 64-bit LSB shared object, ARM aarch64   ← WRONG for x86_64
$ ldd libmlx.so  → not a dynamic executable                    ← loader gives up
```
Always `file`-check the architecture before copying anything from a cache.

### The fix

```bash
# From the fork release (matches the lock's source):
BASE="https://github.com/rltakashige/mlx-jaccl-fix-small-recv/releases/download/mlx_cuda"
cd /tmp && mkdir mlxwheel && cd mlxwheel
curl -fsSL -O "$BASE/mlx-0.32.0-cp313-cp313-manylinux_2_35_x86_64.whl"
curl -fsSL -O "$BASE/mlx_cuda_12-0.32.0-py3-none-manylinux_2_35_x86_64.whl"

# Verify arch BEFORE installing:
unzip -p mlx_cuda_12-*.whl mlx/lib/libmlx.so | file -   # must say x86-64

cd /root/exo
uv pip install --force-reinstall --no-deps \
  /tmp/mlxwheel/mlx-0.32.0-cp313-cp313-manylinux_2_35_x86_64.whl \
  /tmp/mlxwheel/mlx_cuda_12-0.32.0-py3-none-manylinux_2_35_x86_64.whl

# Verify:
uv run python -c "import mlx.core as mx; print(mx.default_device())"
#   → Device(gpu, 0)
```

### ⚠️ Watch-out
`uv sync` re-runs later will **revert** to the URL wheel (which lacks the
runtime). Keep the wheels in `/tmp/mlxwheel` (or durable path) and re-run this
install after every sync.

---

## Step 3 — CUDA 12.8 for Blackwell (problem #3: `nvrtc: invalid value for --gpu-architecture`)

### What's happening — the full chain

```
You place a model
   │
   ▼
Master places it on the Linux box (it reports CUDA VRAM as node memory)
   │
   ▼
Runner loads weights into VRAM           ← OK (2.5 GB fits)
   │
   ▼
Warmup starts → MLX CUDA backend needs a GPU kernel
   │
   ▼
MLX JIT-compiles the kernel at runtime via nvrtc (the CUDA runtime compiler)
   │
   ▼
nvrtc asks the GPU "what arch are you?" → sm_120 (Blackwell)
   │
   ▼
System nvrtc = CUDA 12.0 → only knows sm_50..sm_90 → REJECTS sm_120
   │
   ▼
Runner crashes → "Model failed: <model>"  (dashboard shows "Model failed")
```

**Why Macs are fine:** Metal uses precompiled shaders — there is NO nvrtc, NO
runtime kernel compile. Same model, no crash. This is purely a Linux-environment
problem, NOT a code/PR problem.

### The fix

```bash
# 1. Install the first CUDA version that supports Blackwell (sm_120):
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  cuda-nvrtc-12-8 cuda-nvvm-12-8 cuda-cudart-12-8 cuda-cudart-dev-12-8
# (requires NVIDIA's cuda-keyring added inside the VM — see §6.3)

# 2. Point the system libnvrtc.so.12 (what libmlx.so links) at the 12.8 build:
SYS=/usr/lib/x86_64-linux-gnu
NEW=/usr/local/cuda-12.8/targets/x86_64-linux/lib
ln -sfn "$NEW/libnvrtc.so.12.8.93" "$SYS/libnvrtc.so.12"
ln -sfn "$NEW/libnvrtc.so.12"      "$SYS/libnvrtc.so"

# 3. libnvrtc needs libnvvm.so.4 (the JIT engine) — link it into the system path:
ln -sfn /usr/local/cuda-12.8/nvvm/lib64/libnvvm.so.4.0.0 "$SYS/libnvvm.so.4"
ln -sfn "$SYS/libnvvm.so.4" "$SYS/libnvvm.so"

# 4. CUDA_HOME must point at a place with cuda.h (for kernel header includes):
ln -sfn /usr/local/cuda-12.8 /usr/local/cuda   # or ensure /etc/environment has:
#   CUDA_HOME=/usr/local/cuda
#   CUDA_PATH=/usr/local/cuda

# 5. Verify the JIT path (the exact op that crashed):
cd /root/exo
CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda .venv/bin/python -c "
import mlx.core as mx
@mx.compile
def f(x): return mx.tanh(x * 2.0)
a = mx.random.normal((1024, 1024))
b = f(a); mx.eval(b)
print('JIT compile OK', b.shape)
"
#   → JIT compile OK (1024, 1024)

# 6. Restart exo with the env vars (they must be in the launch environment):
setsid nohup env CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda \
  uv run exo > /tmp/exo.log 2>&1 < /dev/null &
```

### How to confirm the fix end-to-end
Place `mlx-community/Qwen3.5-2B-MLX-8bit` (min_nodes: 1) on the box:
- log: `"warmed up by generating 50 tokens"` ← no nvrtc error
- GPU: `2973MiB on GPU 0 + 154MiB on GPU 1` ← spans both RTX 5060 Tis
- chat: `generation_tps ~17.6` ← real inference works

---

## Step 4 — Fix torch/vision (problem #4: `No module named 'torch'`)

### Symptom
```
Loading vision weights from .../mlx-community--Qwen3.5-2B-MLX-8bit
ERROR: Failed to load vision weights — disabling vision for this runner
ModuleNotFoundError: No module named 'torch'
```
(exo degrades gracefully — text/thinking still work; only image input is lost.)

### What's happening
`pyproject.toml` declares `torch==2.10.0; sys_platform == 'linux'` in the mlx
extra, but the committed `uv.lock`'s torch **package** edge only resolved on
darwin. Root cause: uv encodes "torch for plain-mlx vs torch-for-mlx-cpu/
cuda12/cuda13" with `extra != 'mlx-cpu' ...` marker clauses (because those extras
include `exo[mlx]`), and **uv 0.12.1/0.12.2 mis-evaluates those clauses as
FALSE** — so the whole edge is dropped on Linux.

```
pyproject.toml (declaration):  torch==2.10.0; sys_platform == 'linux'   ✅ correct
uv.lock (resolved markers):    ... and extra != 'mlx-cpu' ...           ❌ uv bug drops edge
scratch project test:          torch==2.10.0; sys_platform == 'linux'   ✅ installs fine
→ proven: the mechanism works; only the multi-extra marker form breaks
```

### The fix (committed as `f48bdf86`)

```bash
# In uv.lock, the exo package's torch/torchaudio/torchvision linux-mlx edges:
#   OLD: marker = "sys_platform == 'linux' and extra == 'mlx'
#                 and extra != 'mlx-cpu' and extra != 'mlx-cuda12' and extra != 'mlx-cuda13'"
#   NEW: marker = "sys_platform == 'linux' and extra == 'mlx'"
uv sync --extra mlx    # now installs torch==2.10.0+cu130, torchaudio, torchvision
```

**Caveat (IMPORTANT, verified twice):** `uv lock` regeneration — or a plain
`uv sync` on the Mac — re-introduces the broken clauses (uv quirk). The Mac's
uv 0.12.1 regenerates the whole lock in a *different format* that ALSO fails to
resolve torch on Linux. Do not blindly re-run `uv lock` or plain `uv sync` on
macOS; if the lock is regenerated, re-apply this 3-line patch and re-verify with
`uv sync --extra mlx --dry-run` on the Linux box.

### Verification
```bash
$ uv sync --extra mlx          → + torch==2.10.0+cu130 + torchvision + triton ...
$ .venv/bin/python -c "import torch; print(torch.__version__)" → 2.10.0+cu130
# Vision path test (the exact code that crashed):
#   _torch_tensor_to_mx(torch.randn(4,4,bf16)) → mlx.core.bfloat16 (4,4)  ✅
```

---

## Step 5 — Quick-start checklist for a NEW Linux/NVIDIA box

```bash
# 0. Identify GPU + driver
lspci | grep -i nvidia
cat /proc/driver/nvidia/version          # exact kernel module version
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# 1. Match userspace to kernel module (§1) — extract .run, symlink, verify

# 2. CUDA toolchain for the GPU's arch (§3)
#    sm_90 (Ada)  → cuda-nvrtc-12-4+ is fine
#    sm_120 (Blackwell) → MUST be cuda-nvrtc-12-8+  (RTX 50xx!)
apt-get install -y cuda-nvrtc-12-8 cuda-nvvm-12-8 cuda-cudart-12-8 cuda-cudart-dev-12-8
# symlink libnvrtc.so.12 + libnvvm.so.4 into /usr/lib/x86_64-linux-gnu
# set CUDA_HOME + CUDA_PATH (put in /etc/environment)

# 3. Install mlx wheels for this arch (§2) — slim wheel + runtime wheel, file-check arch

# 4. uv sync --extra mlx (torch now resolves on Linux — commit f48bdf86)

# 5. Sanity tests
#    a. JIT compile test (§3 step 5)
#    b. import torch
#    c. place a model min_nodes=1 → warmup completes → chat works
```

---

## Proxmox-VM specifics (box-specific bits)

### 6.1 GPU passthrough
- The GPUs are **PCIe passthrough** devices: the host pinned them to the VM
  (`/dev/nvidia0`, `/dev/nvidia1`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, ... all
  present inside the VM).
- The **kernel module** (`nvidia`, 595.58.03) is loaded by the VM's own kernel —
  the Proxmox host's GPU driver is irrelevant. That's why the module version can
  differ from any apt userspace you install (see §1).
- `lspci` inside the VM shows `NVIDIA Corporation Device 2d04` (2×) — the
  passthrough is working when you see both + the `/dev/nvidia*` nodes.

### 6.2 The virtio NICs — why RDMA/Thunderbolt don't exist here
```
Host bridge  ──►  virtio-net (eth0@if25, google@if26)  ──►  10.2.0.16/16
```
- The VM's NICs are **virtio paravirtual devices** bridged to the host network.
- There is **no Thunderbolt controller** (that's physical hardware on Macs), no
  `rdma_en2`-style interfaces, no RDMA verbs hardware.
- Consequence for exo: this node can NEVER be part of an RDMA (JACCL) ring. It
  only does plain socket (RING) or single-node inference. See the companion doc
  `rdma-multi-node-investigation.md` for how the code enforces this.

### 6.3 Driver installs from apt vs the host
- `apt-get update` inside the VM uses the VM's own sources; the NVIDIA repo
  (`cuda-keyring`) must be added **inside the VM** for `cuda-nvrtc-12-8` etc.
- Never install `nvidia-driver-*` metapackages in the VM (they try DKMS builds
  that need kernel headers the Proxmox kernel doesn't provide). Install only the
  **userspace** pieces: `libnvidia-compute-*`, `nvidia-utils-*`, `cuda-nvrtc-*`,
  `cuda-nvvm-*`, `cuda-cudart-dev-*`, and the exact-version `.run` extraction (§1).

### 6.4 Persistence & restarts
- The VM keeps its disk, but `/tmp` may be wiped — keep the extracted NVIDIA
  `.run` dir and the mlx wheels somewhere durable (e.g. `/root/toolchain/`) or
  re-download.
- After any VM reboot: re-verify `nvidia-smi-595.58`, the `libnvrtc.so.12`
  symlink, and `CUDA_HOME` before starting exo.
- Start exo detached so SSH disconnects don't kill it:
  `setsid nohup env CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda uv run exo > /tmp/exo.log 2>&1 < /dev/null &`

---

## Version reference (the exact working set on 10.2.0.16)

| Component | Version |
|---|---|
| OS | Ubuntu 24.04.4 LTS (Proxmox VM, kernel 6.17.13-2-pve) |
| GPUs | 2× NVIDIA GeForce RTX 5060 Ti 16GB (Blackwell, sm_120) |
| Kernel module | NVIDIA 595.58.03 (Open Kernel Module) |
| Userspace | 595.58.03 (from NVIDIA .run extraction, symlinked) |
| CUDA runtime | 12.8 (cuda-nvrtc-12-8 12.8.93 + nvvm + cudart) |
| mlx | 0.32.0 (fork wheel) + mlx_cuda_12 0.32.0 (runtime) |
| torch | 2.10.0+cu130 (via uv.lock fix `f48bdf86`) |
| python | 3.13.14 (uv-managed) |
| exo branch | `fix-memory-error` |
---

## Step 7 — Multi-GPU isolation stack (2+ GPUs on ONE machine, all 6 layers MANDATORY)

MLX's CUDA backend has **no multi-device API** (`mx.default_device()` is always
`Device(gpu, 0)`), so multi-GPU = **one exo process per GPU**, each pinned via
`CUDA_VISIBLE_DEVICES`. Two (or more) exo services on the SAME box MUST be
isolated on SIX independent axes. Missing ANY layer corrupts the cluster:

| # | Layer | Env/Flag | Failure if missing |
|---|-------|----------|--------------------|
| 1 | **namespace** | `--namespace e.g. exo-ct118` (**SAME** on all instances of one cluster) | UDP multicast discovery cross-talk: instances join the wrong cluster or see every cluster on the LAN. Namespace is hashed to 8 bytes (blake3) and each Discovery drops Hellos whose namespace differs (`rust/networking/src/discovery.rs`: `"dropped: different namespace"`) |
| 2 | **XDG_DATA_HOME** | `XDG_DATA_HOME=.../exo-gpu0` vs `.../exo-gpu1` | Event log COLLISION: `EXO_EVENT_LOG_DIR = EXO_DATA_HOME / "event_log"` (constants.py) — shared event log = interleaved/corrupted cluster state |
| 3 | **XDG_CACHE_HOME** | `XDG_CACHE_HOME=.../exo-gpu0` vs `.../exo-gpu1` | `node_zid` COLLISION: `EXO_NODE_ZID = EXO_CACHE_HOME / "node_zid"` (constants.py) — two nodes with the SAME persisted identity = cluster corruption |
| 4 | **zenoh ports** | `--zenoh-port 52422` vs `52424` | Transport: both instances bind the same zenoh listen port → bind failure / cross-wired gossip |
| 5 | **API ports** | `--api-port 52414` vs `52416` | HTTP: both serve dashboard/API on the same port → conflict, wrong node answers `/node_id` |
| 6 | **CUDA_VISIBLE_DEVICES** | `CUDA_VISIBLE_DEVICES=0` vs `1` | GPU pinning: without it both processes use GPU 0 → VRAM exhaustion/OOM (`cudaMallocAsync out of memory`) |

### Canonical systemd units (ct118 — 2× RTX 5060 Ti)

```ini
# /etc/systemd/system/exo-gpu0.service
[Service]
WorkingDirectory=/root/exo
Environment=CUDA_VISIBLE_DEVICES=0
Environment=XDG_CACHE_HOME=/root/.cache/exo-gpu0
Environment=XDG_DATA_HOME=/root/.local/share/exo-gpu0
ExecStart=/root/.local/bin/uv run exo --namespace exo-ct118 --zenoh-port 52422 --api-port 52414
Restart=always

# /etc/systemd/system/exo-gpu1.service
[Service]
WorkingDirectory=/root/exo
Environment=CUDA_VISIBLE_DEVICES=1
Environment=XDG_CACHE_HOME=/root/.cache/exo-gpu1
Environment=XDG_DATA_HOME=/root/.local/share/exo-gpu1
ExecStart=/root/.local/bin/uv run exo --namespace exo-ct118 --zenoh-port 52424 --api-port 52416
Restart=always
```

Key detail: `--namespace` is **shared** (both GPUs belong to ONE logical
cluster); every other axis is **per-instance**.

### Verify a correct multi-GPU install

```bash
# 1. Different node ids:
curl http://<host>:<api0>/node_id ; curl http://<host>:<api1>/node_id
# 2. Distinct node_zid files:
ls /root/.cache/exo-gpu0/node_zid /root/.cache/exo-gpu1/node_zid
# 3. Per-service env matches unit file:
cat /proc/<pid>/environ | tr '\0' '\n' | grep -E 'XDG|CUDA'
# 4. All four ports bound to distinct PIDs:
ss -tlnp | grep -E '52422|52424|52414|52416'
# 5. One exo process per GPU:
nvidia-smi
```

> Source: `promptsANDplans/research/multi-gpu-linux-box-options-2026-08-12.md`,
> `promptsANDplans/notes/2026-09-01.md`, and the live ct118 deployment.
> Exhaustive findings: `findings/t_fe64ae5d-6-layer-isolation-stack.md`.

---

## Step 8 — Securing the API (bind host + auth)

By default exo's HTTP API binds **0.0.0.0** (all interfaces) so every node in a
cluster can reach it. On a machine exposed to untrusted networks that also
exposes your models and a chat UI to anyone who can reach the port.

| Env var | Default | Meaning |
|---------|---------|---------|
| `EXO_API_HOST` | `0.0.0.0` | Bind address. `127.0.0.1` = localhost only (hardened). |
| `EXO_API_ADVERTISE_HOST` | (auto) | Address *advertised to peers* for cluster log/error views. Auto-derived from the node's primary non-loopback IP when `EXO_API_HOST` is `0.0.0.0`/`127.0.0.1`; set explicitly for multi-homed/NAT setups. |
| `EXO_API_TOKEN` | (unset) | Optional bearer token. When set, all API routes except dashboard static assets require `Authorization: Bearer <token>`. |

### Localhost-only (single-user, hardened)

```bash
EXO_API_HOST=127.0.0.1 EXO_API_TOKEN=$(openssl rand -hex 24) \
  uv run exo
```

The API now listens only on loopback. Cluster log/error merging still works
because each node advertises its LAN IP (auto-derived) while binding loopback.

### LAN cluster with auth (multi-node)

```bash
EXO_API_TOKEN=<shared-secret> uv run exo   # every node, same token
```

Bind stays `0.0.0.0` (default) so peers can reach the API; the shared token
prevents anonymous access.

### Behind a reverse proxy

Bind loopback and point the proxy at `127.0.0.1:52415`; set
`EXO_API_ADVERTISE_HOST` to the public/proxy address so cluster views are
correct.

> Source: `src/exo/shared/constants.py` (`EXO_API_HOST`, `EXO_API_ADVERTISE_HOST`),
> `src/exo/api/auth.py` (`EXO_API_TOKEN`).
