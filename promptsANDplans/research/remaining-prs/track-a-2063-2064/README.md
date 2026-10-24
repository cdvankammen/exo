# Track A: PR #2063 + #2064 (Asymmetric Tensor Parallelism)

**Branch:** `fix-memory-error/track-a-asymmetric-tp`
**Started:** 2026-08-05
**Hardware:** 2× identical Mac mini M4 Pro, RDMA over Thunderbolt (direct cable), no TB5

## Status: #2063 PORTED ✅ (commit 868088f7)

### What was ported (5 features)
1. **Nested `_items` walk** (`thunderbolt.py`) — DFS through TB hub/dock nesting to find real peer endpoints. RDMA-generic.
2. **RDMA host IP selection** (`placement_utils.py`) — `_address_priority()` prefers RFC1918 LAN over Tailscale CGNAT. **Highly relevant to our setup.**
3. **Socket-reachable rank 0** (`placement.py`) — rotates cycle so rank 0 is the most-reachable node.
4. **JACCL init retry** (`utils_mlx.py`) — 8-attempt backoff loop absorbing boot races. **Highly relevant.**
5. **TB5 dashboard filter** (`+page.svelte`) — only link speed > 40 Gb/s counts as TB5. **Fixes false RDMA prompt on M4 Pro (TB4).**

### Adaptations
- JACCL retry manually merged to wrap the init call AFTER our #2160 MLX_JACCL_RING block
- 3 new placement tests updated to pass `node_backends` (required since #2252)

### Verification
basedpyright 0 errors, ruff pass, 35/35 tests (TB parsing + placement).

## #2064 (Asymmetric TP) — DEFERRED, needs decision

### What it does
Splits model parameters proportional to per-node memory instead of evenly. Unlocks >200B models on heterogeneous clusters (e.g. Qwen3.5-235B on 128GB + 192GB).

### Why deferred
1. **Blocked upstream**: based on #2063 (now ported!) — the PR says "once #2063 lands, this will rebase cleanly". We now HAVE #2063, so rebase is feasible.
2. **Zero value on identical nodes**: With 2× identical M4 Pro, auto-upgrade never triggers (needs equal split > 90% of smallest node's RAM). Manual AsymmetricTensor would produce ~50/50 anyway.
3. **HIGH conflict**: placement.py refactoring touches our #2252/#2227/#086787dd changes.
4. **Qwen3.5-only**: `_supports_asymmetric_tensor_parallel` whitelist is Qwen3.5 family only.

### Decision needed
Port as forward-compatible infrastructure (useful if you ever add a third node with different memory), or skip until needed?
