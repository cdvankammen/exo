# Decision Support Pack — 2026-08-07

> **Purpose:** Research-only briefs for the tasks awaiting user decisions (T6, T9,
> T10-T14). Produced while another agent executes the "no user input needed" queue.
> No code was changed. Each brief: what / effort / risk / recommendation.

---

## T6 — #2064 Asymmetric Tensor Parallelism (port or wait?)

**Source research:** `remaining-prs/track-a-2063-2064/README.md` (comprehensive).

### What it does
Splits model parameters proportional to per-node memory instead of evenly, so a
>200B model can run on mixed-memory nodes (e.g. Qwen3.5-235B on 128GB + 192GB).

### Facts (verified 08-07)
- **#2063 (its base) is now PORTED** (`868088f7`, Wave 2) — the PR's stated blocker
  ("once #2063 lands, rebase is feasible") is GONE. Rebasing is feasible.
- **Zero value on our current cluster:** 2× identical M4 Pro — auto-upgrade needs
  equal split > 90% of smallest node; manual AsymmetricTensor ≈ 50/50 anyway.
- **Our cluster NOW has asymmetry though:** Linux box (2× RTX 5060 Ti = 31.9GB
  VRAM + 34GB RAM, MlxCuda) vs Macs (Metal 17.2GB). But asymmetric TP is
  **Qwen3.5-family only** (`_supports_asymmetric_tensor_parallel` whitelist), and
  heterogeneous TP Metal+CUDA is a bigger lift than this PR assumes (it targets
  same-backend, memory-differing nodes).
- **Conflict surface:** placement.py refactoring — touches our #2252/#2227/Wave-6
  bandwidth-aware allocation (`e18a8f0e`) + link-speed selection (`60f081e3`).

### Recommendation
**Wait.** Port as infrastructure only if/when you acquire a third node with
different memory OR want Qwen3.5-200B+ split across the two Macs (identical RAM
→ no gain). Cost to port later is unchanged (base is in, conflict surface is
stable). Revisit after T4 #2200 lands (backend-per-shard makes backend-mixed
placement cleaner first).

---

## T9 — #1860 RotatingKVCache wiring (needs your default decision)

### What it does
Wires up mlx's `RotatingKVCache` to fix **unbounded KV-cache memory growth** —
long conversations currently grow cache until OOM. Raises `MAX_KV_SIZE` from
**3200** (current, in `src/exo/worker/engines/mlx/constants.py`) to **131072**.

### The trade-off (this is the decision)
| | MAX_KV_SIZE 3200 (today) | 131072 (PR default) |
|---|---|---|
| Long-context chat | Truncates/degrades past ~3.2K tokens | Full context to 128K |
| Memory per conv | Bounded, small | Can grow large (KV per token ~0.5-2MB at 4-bit for 1-3B models; ~50-200MB for 30B+ at 128K) |
| Eviction | N/A (capped) | Rotating (oldest tokens evicted) |

### Facts
- `RotatingKVCache` is already imported/used in `src/exo/worker/engines/mlx/cache.py`
  (copy/snapshot paths exist) — the plumbing is partially present; #1860 wires the
  generator + raises the cap.
- Related PRs (#1988 `EXO_KV_CACHE_BITS` env, #2242 KV quant opt-in) are additive
  knobs that pair well AFTER this.
- Our cluster: Macs 24GB RAM, Linux 34GB. With 131072 default, a 7B model at 4-bit
  KV can use ~0.5-2GB per long conversation — acceptable on 24GB, but multiple
  parallel chats (our queue T24-26 + backend batch 8!) multiply it.

### Recommendation
**Port with a safer default + env override**: land the wiring but default
`MAX_KV_SIZE` to something like **16384** (4× today, still bounded), expose
`EXO_MAX_KV_SIZE` env var (matches #1988's knob philosophy), and let power users
set 131072. This captures the "no unbounded growth" fix without the memory risk
of 128K default under 8-way parallel batches.

---

## T10-T13 — Tier-4 design briefs (all "opt-in" flavored)

### T10 — #2246 PromoteMaster command
**What:** A command (`POST /v1/commands/promote-master` or CLI flag) letting you
force a specific node to become master.
**Design surface:** election is bully-based; a forced promote needs a
`PromoteMaster` command event + master demotion handshake. Risk: split-brain if
two nodes promote simultaneously — needs the existing election sequencing.
**Effort:** Small-Medium. **Rec:** PORT as opt-in CLI flag; default behavior
unchanged.

### T11 — #2135 `--no-master` flag
**What:** Run a node that never campaigns for master (worker-only).
**Design surface:** skip `_elect_loop` campaign; still must apply indexed events
from master. Clean separation — the node is a pure worker.
**Effort:** Small. **Rec:** PORT — trivial, useful for the Linux box (it's our
most powerful compute but a VM on flaky Proxmox; making it a permanent worker
improves cluster stability). Pair with T10.

### T12 — #2190 Configurable API host binding
**What:** Bind API to 127.0.0.1 vs 0.0.0.0 via config/env.
**Design surface:** `EXO_API_HOST` env var, default stays 0.0.0.0 (cluster
behavior unchanged) or flips to localhost for security. Our Linux box already
serves :52415 on a VM network — binding control is a security win.
**Effort:** Tiny. **Rec:** PORT with env var + docs; keep 0.0.0.0 default
(backward compat) or ask user preference.

### T13 — #1864 Disable server-side tools
**What:** Opt-out switch for server-side tool execution (security: a malicious
chat prompt shouldn't trigger tools on your machine).
**Design surface:** `EXO_ENABLE_TOOLS` (default on for compat) + per-request
opt-out; dashboard toggle.
**Effort:** Small. **Rec:** PORT as env var default-on; document the risk.

### T14 — Non-MLX models (GGUF/PyTorch) — new subsystem, NO PR exists
**What:** Support GGUF (llama.cpp) + PyTorch models in addition to MLX.
**Design surface (sketch):** new `backend` plugin per engine: `EngineRunner` for
GGUF (llama.cpp server mode or ctransformers) + PyTorch (transformers/vLLM —
vLLM already present in node_backends!). Memory accounting must be per-backend;
sharded ring inference is MLX-only today — GGUF/PyTorch would start single-node.
**Effort:** Large (new subsystem, weeks). **Rec:** DEFER until the Linux box
(vLLM path) is a stable member — then port vLLM backend first (most overlap with
existing code: it's already advertised).

---

## Summary decision table

| # | Item | Rec | Why (1 line) |
|---|---|---|---|
| T6 | #2064 asymmetric TP | **WAIT** | No asymmetric-MLX nodes; Qwen3.5-only; port later costs same |
| T9 | #1860 RotatingKVCache | **PORT w/ 16384 default + env override** | Fixes OOM; 131072 default risky under 8-way batch |
| T10 | #2246 PromoteMaster | **PORT (opt-in)** | Manual master control, small |
| T11 | #2135 --no-master | **PORT** | Linux box as permanent worker = stability win |
| T12 | #2190 API host binding | **PORT (env var)** | Security, tiny |
| T13 | #1864 disable tools | **PORT (env var, default-on)** | Security opt-out, small |
| T14 | non-MLX models | **DEFER → vLLM backend first** | New subsystem; leverage existing Vllm advertisement |

## Open questions for user
1. T9: OK with 16384 default + `EXO_MAX_KV_SIZE` override? Or keep PR's 131072?
2. T11: want Linux box to be permanent `--no-master` worker?
3. T12: default binding — keep 0.0.0.0 or switch to localhost with opt-in LAN?
