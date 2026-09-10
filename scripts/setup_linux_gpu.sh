#!/usr/bin/env bash
# setup_linux_gpu.sh — Verify/repair exo NVIDIA GPU support on Linux
#
# Purpose: exo on Linux with an NVIDIA GPU should advertise MlxCuda and report
# GPU VRAM as node memory. This script checks every link in that chain and
# gives exact fix commands when something is wrong — no more silent CPU-only
# fallback with no explanation.
#
# Lessons encoded (from the 10.2.0.16 deployment saga, 2026-08-06):
#   1. mlx on Linux x86_64 = TWO packages: mlx (bindings) + mlx_cuda_12/13 (runtime).
#      `uv sync --extra mlx-cuda12` (or mlx-cuda13) installs both.
#   2. nvidia-ml-py (pynvml) is in the base `mlx` extra on Linux — it must be
#      importable or the node advertises MlxCpu only.
#   3. The NVIDIA kernel module and userspace (libcuda, libnvidia-ml) MUST match
#      exactly, else `nvidia-smi` fails with "Driver/library version mismatch".
#   4. exo looks up `nvidia-smi` by name; a version-suffixed binary
#      (e.g. nvidia-smi-595.58) needs a symlink to /usr/bin/nvidia-smi.
#      (exo >= 2a9e15a2 also has an NVML fallback, but the symlink is cleanest.)
#   5. CUDA JIT needs headers at RUNTIME: install nvidia-cuda-toolkit and set
#      CUDA_HOME / CUDA_PATH.
#
# Usage:
#   bash scripts/setup_linux_gpu.sh            # check + print fixes (no changes)
#   bash scripts/setup_linux_gpu.sh --fix      # apply the fixes automatically
#   bash scripts/setup_linux_gpu.sh --fix --extra mlx-cuda12   # pick the extra
#
# Exit codes: 0 = all good, 1 = problems found (or fixed with --fix).

set -u

EXTRA="${EXTRA:-mlx-cuda12}"          # or mlx-cuda13 for CUDA 13
APPLY_FIXES=0
[[ "${1:-}" == "--fix" ]] && APPLY_FIXES=1
if [[ "${1:-}" == "--extra" && -n "${2:-}" ]]; then EXTRA="$2"; fi

say()  { printf '\033[1;32m[ok]\033[0m %s\n' "$1"; }
warn() { printf '\033[1;33m[!!]\033[0m %s\n' "$1"; }
fail() { printf '\033[1;31m[XX]\033[0m %s\n' "$1"; }

run_fix() {
  if [[ $APPLY_FIXES -eq 1 ]]; then
    eval "$1" && say "fixed: $2" || fail "fix FAILED: $2"
  else
    warn "would fix (run with --fix): $2"
  fi
}

PROBLEMS=0

echo "=== 1. NVIDIA GPU present? ==="
if [[ -d /proc/driver/nvidia ]] || command -v nvidia-smi >/dev/null 2>&1 || \
   command -v nvidia-smi-* >/dev/null 2>&1; then
  say "NVIDIA driver/kernel module present"
else
  warn "no NVIDIA driver detected (/proc/driver/nvidia or nvidia-smi missing)"
  warn "install the NVIDIA driver for your GPU first, then re-run this script."
  exit 1
fi

echo "=== 2. Driver version match (kernel module vs userspace) ==="
KERNEL_VER=""
if [[ -f /proc/driver/nvidia/version ]]; then
  # Format: "NVRM version: NVIDIA UNIX x86_64 Kernel Module  595.58.03  ..."
  KERNEL_VER=$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' /proc/driver/nvidia/version | head -1)
  say "kernel module: ${KERNEL_VER:-unknown}"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi FAILS — Driver/library version mismatch is the usual cause."
    fail "Match the userspace to the kernel module version exactly."
    run_fix "echo 'Install the exact driver version (e.g. NVIDIA .run) or match apt packages.'" "driver version match"
    PROBLEMS=1
  else
    USERSPACE_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    say "nvidia-smi works (GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), driver: ${USERSPACE_VER:-?})"
    if [[ -n "$KERNEL_VER" && -n "$USERSPACE_VER" && "$KERNEL_VER" != "$USERSPACE_VER" ]]; then
      fail "driver version MISMATCH: kernel module=$KERNEL_VER vs userspace=$USERSPACE_VER"
      PROBLEMS=1
    fi
  fi
else
  fail "nvidia-smi not on PATH"
  if ls nvidia-smi-* 2>/dev/null | head -1 >/dev/null; then
    run_fix "ln -sf /usr/bin/nvidia-smi-$(ls nvidia-smi-* | head -1 | sed 's/.*nvidia-smi-//') /usr/bin/nvidia-smi" "symlink version-suffixed nvidia-smi"
  fi
  PROBLEMS=1
fi

echo "=== 3. pynvml (nvidia-ml-py) importable in the venv? ==="
if [[ -x .venv/bin/python ]]; then
  if .venv/bin/python -c "import pynvml" >/dev/null 2>&1; then
    say "pynvml importable"
  else
    fail "pynvml NOT importable — node will advertise MlxCpu only"
    run_fix "uv sync --extra $EXTRA" "install mlx + nvidia-ml-py via '$EXTRA' extra"
    PROBLEMS=1
  fi
else
  warn "no .venv found — run 'uv sync --extra $EXTRA' first"
  run_fix "uv sync --extra $EXTRA" "create venv + install deps"
  PROBLEMS=1
fi

echo "=== 4. CUDA headers available at runtime (JIT)? ==="
if [[ -n "${CUDA_HOME:-}" || -n "${CUDA_PATH:-}" || -d /usr/local/cuda ]]; then
  say "CUDA_HOME/CUDA_PATH set or /usr/local/cuda exists"
else
  warn "CUDA_HOME/CUDA_PATH not set — runner may crash 'Can not find locations of CUDA headers'"
  run_fix "export CUDA_HOME=/usr/local/cuda && echo 'add CUDA_HOME/CUDA_PATH to /etc/environment'" "set CUDA env vars"
  PROBLEMS=1
fi

echo "=== 5. exo would advertise MlxCuda? ==="
if [[ -x .venv/bin/python ]]; then
  OUT=$(.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
import asyncio
from exo.utils.info_gatherer.info_gatherer import NodeBackends
bs = asyncio.run(NodeBackends.gather())
print(','.join(b.value for b in bs.backends))
" 2>/dev/null)
  if [[ "$OUT" == *"MlxCuda"* ]]; then
    say "node advertises: $OUT (MlxCuda present ✅)"
  else
    fail "node advertises: ${OUT:-'unknown'} — MlxCuda MISSING (still CPU-only)"
    PROBLEMS=1
  fi
fi

echo ""
if [[ $PROBLEMS -eq 0 ]]; then
  echo "✅ All checks passed — this node should join the cluster as a CUDA node."
  echo "   Verify live: start exo and check the log for:"
  echo '     "CUDA MLX backend detected; reporting GPU VRAM as node memory"'
  exit 0
else
  echo "⚠️  Problems found — see above."
  [[ $APPLY_FIXES -eq 1 ]] && echo "   Fixes applied (re-run to confirm)." || echo "   Re-run with --fix to apply, or fix manually."
  exit 1
fi
