# Session Tracking — 2026-08-07

## BUG DOCUMENTED: CUDA VRAM reports only GPU 0 (not adaptive to GPU count)

- **Symptom:** Linux box (2× RTX 5060 Ti 16GB) advertises only 16GB in exo node UI, though it has 32GB total VRAM + 34GB system RAM.
- **Root cause:** `_query_cuda_vram_bytes()` (profiling.py, from PR #2216 `ddc34a44`, hardened by `2a9e15a2`) reads ONLY `lines[0]` of `nvidia-smi --query-gpu=memory.total,memory.free` — the first GPU. The NVML fallback similarly calls `nvmlDeviceGetHandleByIndex_v2(0, ...)` — index 0 only.
- **Impact:** Any multi-GPU Linux box under-reports VRAM by (n-1)/n. Placement can't fit models that would span multiple GPUs (MLX CUDA can use all GPUs — proven: Qwen3.5 used GPU0 2,973MiB + GPU1 154MiB). Boxes with more GPUs (DGX etc.) would be even more under-reported.
- **Not a memory cap** — no commit sets a limit; this is the VRAM-as-memory design from #2216 reporting one GPU.
- **Fix direction (implemented next):** SUM all GPUs' VRAM in BOTH paths:
  1. nvidia-smi: iterate ALL lines, sum total + sum free
  2. NVML: iterate `nvmlDeviceGetCount_v2` devices, sum memory.total/free for each handle
  - This is adaptive to any GPU count (1, 2, 4, 8...).

## FIXED: CUDA VRAM adaptive multi-GPU summing — commit `a082b091`

- **Fix:** `_query_cuda_vram_bytes()` + `_query_cuda_vram_bytes_nvml()` now SUM across ALL GPUs (both nvidia-smi lines and NVML device loop). Adaptive to any GPU count.
- **Tests:** 5 new (single/2/4 GPU, NVML multi, NVML bad-device-skip) — 9 total pass.
- **Verified live:** 15.93GB → **31.86GB** on 10.2.0.16.
- **Secondary bug found (machine-level):** stale `/root/exo/exo/` dir shadowed `src/` in sys.path → box imported old code. Moved to `/root/exo-stale-shadow-backup`. ⚠️ Any future `uv sync`/edits must ensure no stray `exo/` dir shadows `src/`.
