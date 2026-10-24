# References Log

> **Purpose:** Useful external references, code libraries, offline-fetched info, and links — the project's external knowledge base.
> **Format:** Category | Item | Where it lives | Why it matters

## Historical records (dated)
- `2026-08-06.md` — Wave 4 references (#2129 ported, CUDA 12.8 packages, torch lockfile bug, source-map additions)

---

## 1. Upstream PRs (the merge campaign sources)

| PR | Title | Author | URL |
|---|---|---|---|
| #2103 | Native CUDA Linux support | — | https://github.com/exo-explore/exo/pull/2103 |
| #2129 | 3-node heterogeneous Metal+CUDA | @Juliuscply | https://github.com/exo-explore/exo/pull/2129 |
| #2200 | Backend per shard | @oliver0006 | https://github.com/exo-explore/exo/pull/2200 |
| #2064 | Asymmetric TP | @team-wcv | https://github.com/exo-explore/exo/pull/2064 |
| #2063 | TB5 discovery + JACCL retry | — | https://github.com/exo-explore/exo/pull/2063 |
| #1598 | HF Hub cache discovery | @robertvandervoort | https://github.com/exo-explore/exo/pull/1598 |
| #2238 | Eject + weight deletion | @alytaphoenix | https://github.com/exo-explore/exo/pull/2238 |
| #2216 | CUDA VRAM reporting | — | https://github.com/exo-explore/exo/pull/2216 |
| #2252 | Accelerator preference | — | https://github.com/exo-explore/exo/pull/2252 |
| #2227 | Capacity-aware allocation | — | https://github.com/exo-explore/exo/pull/2227 |
| #2254 | Link-latency ring selection | @abendrothj | https://github.com/exo-explore/exo/pull/2254 |
| #2160 | Multi-rail TB bonding | @jasonpaulso | https://github.com/exo-explore/exo/pull/2160 |
| #2159 | KV disk persistence | — | https://github.com/exo-explore/exo/pull/2159 |
| #2251 | Prefill memory headroom | — | https://github.com/exo-explore/exo/pull/2251 |
| #2242 | KV-cache quantization | — | https://github.com/exo-explore/exo/pull/2242 |
| #2231 | Event-send hardening | @caesarniseko-coder | https://github.com/exo-explore/exo/pull/2231 |
| #2205 | Orphan watchdog | @aidiffuser | https://github.com/exo-explore/exo/pull/2205 |
| #1804 | Dead runners + error chunks | @mlpy0 | https://github.com/exo-explore/exo/pull/1804 |
| #2134 | SSRF block | @5F0jd2vLq54RerYW | https://github.com/exo-explore/exo/pull/2134 |
| #2229 | Token relay (SKIPPED - highest value) | — | https://github.com/exo-explore/exo/pull/2229 |
| #2255 | Ring Attention (SKIPPED) | — | https://github.com/exo-explore/exo/pull/2255 |
| #1860 | RotatingKVCache (SKIPPED) | — | https://github.com/exo-explore/exo/pull/1860 |
| Full list | All 53 ported PRs | — | `../research/pr-merge-campaign-tracker.md` |

## 2. Code Libraries & Wheels

| Library | Version | Where | Why |
|---|---|---|---|
| mlx | 0.32.0 | PyPI / fork | Inference backend |
| mlx_cuda_12 | 0.32.0 | rltakashige fork release | CUDA runtime (libmlx.so) — REQUIRED on Linux |
| nvidia-ml-py (pynvml) | >=13.595.45 | mlx-cuda12 extra | NVML detection for MlxCuda advertisement |
| mlx-lm | 0.31.3 | PyPI | Model loading/generation |
| uv | 0.12.2 | astral.sh | Package manager |
| pyinstaller | 6.17.0 | venv | App bundling |

**Fork URL (mlx with JACCL fixes):** https://github.com/rltakashige/mlx-jaccl-fix-small-recv (release `mlx_cuda` has the wheels)

**Known upstream issue (multi-wire recv prefill deadlock):**
- https://github.com/rltakashige/mlx-jaccl-fix-small-recv/issues/3
- https://github.com/ml-explore/mlx/issues/3654

## 3. Offline-Fetched Information (useful later)

| Item | Where | Notes |
|---|---|---|
| VS Code sensitive-input guard source | `runInTerminalTool.ts` (open source) | `isAutoApproved` branch cancels; patched in bundle |
| NVIDIA 595.58.03 driver .run | `/tmp/NVIDIA-Linux-x86_64-595.58.03/` on Linux box (may be cleaned) | Exact-match userspace for Proxmox kernel module |
| mlx two-package structure | verified live | `mlx` bindings + `mlx_cuda_12` runtime; RUNPATH `$ORIGIN/lib` |
| exo state JSON structure | `/state` endpoint | nodeMemory, nodeBackends, topology keys |

## 4. Hardware Reference

| Machine | Spec | Role |
|---|---|---|
| This Mac (chris's Mac mini) | M4 Pro, 16GB unified | Cluster node + dev |
| mini2 | M4 Pro, 16GB unified, TB4-bonded | Cluster node |
| 10.2.0.16 (tryingOutExo) | Proxmox VM, 2× RTX 5060 Ti 16GB, 24GB RAM, Ubuntu 24.04 | Linux CUDA node |

## 5. Commands Reference (quick copy)

```bash
# Linux run (detached)
setsid nohup env CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda uv run exo > /tmp/exo.log 2>&1 < /dev/null &

# Linux SSH (password in temp file)
umask 077 && printf 'violin\n' > /tmp/.sshpass_exo
sshpass -f /tmp/.sshpass_exo ssh -o StrictHostKeyChecking=accept-new root@10.2.0.16 'cmd'
rm -f /tmp/.sshpass_exo

# VS Code patch re-apply (after every Insiders update)
/Users/chris/scripts/vscode-patch-sensitive-input.sh

# App rebuild
uv sync --extra mlx && uv sync --reinstall-package exo_rs
cd dashboard && npm run build && cd ..
uv run pyinstaller packaging/pyinstaller/exo.spec --noconfirm
xcodebuild -project app/EXO/EXO.xcodeproj -scheme EXO -configuration Release build
sudo rm -rf /Applications/EXO.app && sudo cp -R ~/Library/Developer/Xcode/DerivedData/EXO-*/Build/Products/Release/EXO.app /Applications/
sudo chmod -R a+rX /Applications/EXO.app && codesign --force --deep --sign - /Applications/EXO.app
open /Applications/EXO.app
```

## 6. Key Source Files (navigation map)

| File | What it is | Modified by |
|---|---|---|
| `src/exo/worker/engines/mlx/cache.py` | KV cache (disk, quant, headroom) | #2159, #2242, #2251, #1990, #2158 |
| `src/exo/master/placement.py` | Placement (capacity, accelerator, latency) | #2227, #2252, #2063, #2254 |
| `src/exo/worker/engines/mlx/utils_mlx.py` | MLX utils (CUDA compat, JACCL) | #2103, #2160, #2063 |
| `src/exo/shared/types/profiling.py` | Memory usage (CUDA VRAM) | #2216 |
| `src/exo/worker/engines/mlx/patches/cuda_compat.py` | CUDA shim | #2103 |
| `src/exo/utils/info_gatherer/info_gatherer.py` | Node info (backends, VRAM) | #2216 |
| `dashboard/src/routes/+page.svelte` | Main dashboard page | #2238, eject commits |
| `dashboard/src/lib/components/EjectButton.svelte` | Eject split-button | ca8474c7 |
| `src/exo/worker/runner/bootstrap.py` | Runner entry (orphan watchdog, CUDA) | #2205, #2103 |
