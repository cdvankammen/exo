# Deep Research: Heterogeneous Mac+Linux Clusters (#2129 + #2200) & MLX-on-Linux

**Date:** 2026-08-06
**Status:** RESEARCH ONLY — no code changes
**Branch:** `fix-memory-error`

---

## 1. Can MLX models run on Linux? Can we split models between Mac + Linux?

**ANSWER: YES on both counts — and we proved it today.**

### What we proved (this session, 2026-08-06)
- The Linux box (10.2.0.16, Proxmox VM, 2× RTX 5060 Ti 16GB) runs exo's MLX engine via the **CUDA backend** — our #2103 port
- End-to-end inference verified: Llama-3.2-1B (73.5 tok/s), Llama-3.2-3B (29 tok/s) — **single-node on Linux**
- The log line `CUDA MLX backend detected; reporting GPU VRAM as node memory` confirms our #2216 VRAM reporting is live

### How MLX maps to hardware
| Backend | Hardware | Where |
|---|---|---|
| `MlxMetal` | Apple GPU | macOS |
| `MlxCuda` | NVIDIA GPU | Linux (via our #2103 port) |
| `MlxCpu` | Any CPU | Anywhere |

All our model cards declare all three (`backends = ["MlxMetal", "MlxCuda", "MlxCpu"]`), so **the same model files run on both Mac and Linux** — MLX is MLX; only the compute backend differs.

### What's needed for Mac+Linux splitting
1. ✅ Same exo version everywhere (all nodes on `fix-memory-error`)
2. ✅ Model cards list both backends (already do)
3. ✅ Placement understands both backends (#2252 accelerator preference ported)
4. ✅ Node discovery on same subnet (10.2.0.x — works)
5. ✅ CUDA nodes report VRAM (#2216)
6. ⚠️ **The gap:** cross-backend distributed inference (see #2129 below)

---

## 2. PR #2129 — 3-node heterogeneous cluster (Metal + CUDA)

**Author:** @Juliuscply · 7 commits · 9 files · +444/−45 · OPEN (never merged upstream)

### Motivation (translated from the PR)
PR #2103 gave Linux CUDA nodes the ability to join MLX rings. But real 3-node Metal+CUDA tests exposed **three blocking bugs**:
1. **MLX 0.32.0 CUDA `send`/`recv` is broken** — CUDA↔CUDA point-to-point communication **hangs** → tensors can't cross between two CUDA nodes
2. **CUDA peer detection is unreliable** — no explicit mechanism to declare which ranks are CUDA
3. **Stale connections block cluster formation** — half-open connections from CUDA probing linger in the `accept()` queue, blocking a 3rd node from joining

Plus minor compat issues: `new_thread_local_stream` missing on CUDA, `MLX_METAL_FAST_SYNCH` warns on Linux, no `networksetup` on Linux (interface types all "unknown"), wired-limit errors on CUDA.

### The 8 changes

| File | Change | In our branch? |
|---|---|---|
| `auto_parallel.py` | **TcpRelay** (+220 lines) — raw TCP bypass for CUDA↔CUDA links: serializes mx.array → numpy → bytes (float32/16, bf16, int32/64, bool, uint8), 60× connect retries, 120× accept retries, MSG_PEEK stale-connection filtering, port 40000+rank | ❌ NOT ported |
| `utils_mlx.py` | `MLX_HOSTS_JSON` env var (persists hostfile past tempdir cleanup); eager TcpRelay server on CUDA nodes; delayed CUDA-peer probing (2s timeout per peer) → `MLX_CUDA_RANKS`; `set_wired_limit_for_model()` CUDA-aware | ⚠️ Partial — wired-limit CUDA branch ✅ (from #2103); MLX_HOSTS_JSON + probing ❌ |
| `generate.py` | **Prefill barrier** — `mx_barrier(group)` after `set_pipeline_prefill(is_prefill=False)` prevents prefill/generation desync deadlock in all_gather | ❌ NOT ported (we have pre-prefill + warmup barriers, not post-prefill) |
| `placement.py` | `_rotate_metal_to_middle()` — for 3-node Pipeline: Metal node rotated to middle rank so all links are Metal↔CUDA (working) not CUDA↔CUDA (hang); general 5+ node interleave | ❌ NOT ported (we have `_cycle_accelerator_score` from #2252 — different mechanism) |
| `cuda_compat.py` | `mx.new_stream` → `new_thread_local_stream` shim | ✅ ALREADY HAVE (identical, from #2103) |
| `patches/__init__.py` | Registers cuda_compat | ✅ ALREADY HAVE |
| `bootstrap.py` | `MLX_METAL_FAST_SYNCH` only on Darwin | ✅ ALREADY HAVE |
| `system_info.py` | `_linux_interface_types()` — regex infer: `wl*`=wifi, `enp\d+s\d+f\d+np\d+`=ethernet, `en*`=maybe_ethernet, `tb*`=thunderbolt | ❌ NOT ported |
| `placement_utils.py` | **IP priority flip: ethernet 0, thunderbolt 2** (for DGX scenarios) | ⚠️ **CONFLICTS with our #2063 + #2254** — we deliberately prefer TB/RDMA for ring hosts. On our 2× M4 Pro this change would be WRONG. Must be gated. |

### Known limitations (from PR)
- `_rotate_metal_to_middle` handles 3 nodes only; 4+ needs the general interleave (EternaPeptix's fork has a follow-up)
- CUDA peer probing adds 2s×peers startup cost
- TcpRelay lacks cleanup (port may linger after process exit)

### Verdict for our setup
**Currently: 2× identical Mac mini M4 Pro + 1× Linux RTX box. #2129's value = enables the Linux box to join the Macs in ONE ring.**

The critical missing pieces for us:
1. **TcpRelay** (~220 lines, self-contained in auto_parallel.py) — needed only if 2+ CUDA nodes exist. For 1 CUDA + 2 Metal, all links are Metal↔CUDA which WORK natively... **but the placement rotation + peer detection are still needed**
2. **`_rotate_metal_to_middle`** — our 3-node topology (Metal, Metal, CUDA) has ONE CUDA↔CUDA pair only if both Macs are adjacent; rotation matters
3. **Linux interface types** — without it, the Linux box reports `unknown` interfaces, which breaks ring host selection quality (our #2254 latency measurement partially compensates)
4. **Post-prefill barrier** — cheap, low-risk, prevents a real deadlock class

**Do NOT port:** the placement_utils IP priority flip (breaks our TB-preference for Mac↔Mac).

---

## 3. PR #2200 — Select and apply MLX backend per shard

**Author:** @oliver0006 · 1 commit · 8 files · OPEN (no reviews)

### What it does
Addresses issue #958 ("Support all MLX backends"). MLX has Metal/CUDA/CPU backends, and the placement layer already knows each node's backends (`state.node_backends`, `INSTANCE_META_BACKENDS`) — but that decision was **never recorded on the instance**, and runners **never called `mx.set_default_device()`**. Backend selection happened implicitly via whatever the local MLX build defaulted to.

### The changes
| File | Change |
|---|---|
| `ShardMetadata` (types) | `backend: Backend \| None` field (default `None` = today's behavior preserved) |
| `placement_utils.py` | `assign_shard_backends()` — pure function: for every runner, pick the first backend in instance-meta preference order (`Metal > Cuda > Cpu` ∩ model card backends) that the node supports |
| `placement.py` | `place_instance()` calls it right after `get_shard_assignments()` |
| runner `bootstrap.py` | `apply_shard_backend()` — maps backend → MLX device, calls `mx.set_default_device()` **before the engine is built** (`MlxMetal`/`MlxCuda` → `mx.gpu`, `MlxCpu` → `mx.cpu`) |

### Why it matters for us
- Today, our Linux box **works by luck** — mlx-cuda defaults to GPU, Metal defaults to GPU. But a CPU-only Linux node would silently run on whatever mlx picked
- With #2200, the **master decides** and the runner obeys — explicit, deterministic
- **Complements #2129 perfectly**: #2129 fixes the *data path* between backends; #2200 fixes the *device selection* per shard
- The PR author tested on Fedora Linux + Radeon iGPU with `mlx-cpu` — verified `apply_shard_backend(MlxCpu)` flips the default device

### Conflict check with our branch
- `ShardMetadata` lives in `src/exo/shared/types/worker/shards.py` — we haven't touched it (low conflict)
- `placement.py` — we've modified heavily (#2227, #2252, #2063, #2254, #086787dd) — `place_instance()` hook point needs care
- `bootstrap.py` — we've modified (CUDA gating #2103, orphan watchdog #2205) — the apply point is early in runner entrypoint, low conflict
- Test expectations: `test_master` asserts shards carry `backend=MlxMetal` — our master tests will need the same update

### Verdict
**Clean-apply candidate, moderate value.** Low conflict, small footprint, makes heterogeneous clusters deterministic. Recommend porting BEFORE #2129 (it touches fewer of the same lines).

---

## 4. Recommended port order (if we proceed)

1. **#2200** (backend per shard) — small, clean, foundation
2. **#2129 pieces in this order:**
   - `system_info.py` `_linux_interface_types()` — tiny, zero conflict
   - `generate.py` post-prefill barrier — 5 lines, zero conflict, prevents deadlock
   - `_rotate_metal_to_middle` in placement.py — moderate, adapt to our `_cycle_accelerator_score` (#2252)
   - `MLX_HOSTS_JSON` + CUDA rank probing in utils_mlx.py — small
   - `TcpRelay` in auto_parallel.py — the big one (~220 lines), self-contained
3. **SKIP:** placement_utils IP priority flip (ethernet>TB) — breaks our Mac↔Mac TB preference; would need a "only when no TB link" gate

---

## 5. Non-MLX models (GGUF/PyTorch) in the download menu — research

**No upstream PR exists** to add non-MLX models. Reasons:
- Model cards encode MLX-specific metadata (backends, quantization scheme, tokenizer configs)
- exo has exactly one inference engine (MLX); no GGUF/llama.cpp/transformers runner
- Closest PRs: #2200 (per-shard backend, still MLX-only) and #2129 (heterogeneous, still MLX-only)

To support GGUF you'd need: a llama.cpp-binding engine, model-card extension, download path for GGUF repos, and backend routing. That's a new subsystem, not a port.

---

## 6. Key files touched by this research (for the port later)
- `src/exo/worker/engines/mlx/auto_parallel.py` — `_pending_prefill_sends`, `flush_prefill_sends()`, `_RingRecvWrapper`, `_RingSendWrapper` all present (TcpRelay hooks go here)
- `src/exo/worker/engines/mlx/utils_mlx.py` — `mlx_distributed_init()` line 90 sets `MLX_HOSTFILE` (line 116); CUDA-aware `set_wired_limit_for_model` at line ~855
- `src/exo/master/placement.py` — `place_instance()` line 124, `node_backends` param, `_cycle_accelerator_score` line 256
- `src/exo/worker/engines/mlx/generator/generate.py` — barriers at 329/425/445; the missing post-prefill one goes after line 373 (`set_pipeline_prefill(model, is_prefill=False)`)
- `src/exo/shared/types/worker/shards.py` — `PipelineShardMetadata` (line 56) gets `backend` field for #2200
- `src/exo/utils/info_gatherer/system_info.py` — `_get_interface_types_from_networksetup()` line 59 returns `{}` on Linux; #2129 replaces with `_linux_interface_types()`
