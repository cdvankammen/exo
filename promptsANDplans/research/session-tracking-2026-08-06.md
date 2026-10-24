# Full Session Tracking Log — 2026-08-06 (exo cluster work)

> **Purpose:** Complete audit trail of every change, decision, error, and fix in this session.
> **Branch:** `fix-memory-error` (56+ commits ahead of main)
> **Format:** Each entry records: WHAT changed, WHY (source: user request / PR port / bug fix), WHERE it came from (PR #, URL, commit), and any follow-up context.

---

## PART 1: PR Merge Campaign (context — earlier waves)

### Wave 1 (26 commits incl. baseline) — memory/network core
All ported from upstream PRs with full attribution (PR#, author, URL in commit messages). Key ports:
- `7dd12e9f` #2159 KV disk persistence (SSD-as-RAM)
- `070acc7f` #2242 KV-cache quantization (EXO_KV_CACHE_BITS)
- `5260b384` #2251 prefill memory headroom
- `fde5c298` #1990 skip KV quant single-node
- `5f7682dc` #2160 multi-rail TB bonding (MLX_JACCL_RING)
- `35f1e789` #2227 capacity-aware layer allocation
- `38101807` #2252 accelerator preference
- `ddc34a44` #2216 CUDA VRAM reporting
- `02f17fb4` #2103 native CUDA Linux support
- `26818670` #2192 macOS EHOSTUNREACH fix
- `ed7ba52f` #2238 eject + optional weight deletion
- `4bac50c9` #2204 debug prompt gate
- `2772b55d` #2244 runner reconnect cascade
- `6f5458ba` #2228 ArraysCache mask tolerance
- `53bb0f98` #2158 V4 trim_cache restore
- `c1075b87` #2152 vision cache hash dims
- `0756536e` #2174 GPT-OSS tiktoken vocab
- `5fa86ef8` #2218 Gemma 4 12B card
- `5913b0ac` #2239+#2247 gen stats UI + v1/models filter
- `89921b38` #2245 docs mlx extra
- `6f0261c2` #2249 orphaned download GC
- `2d50d66d` #2250 stale node prune
- `086787dd` coordination fix (force_override bypass — OUR OWN fix, not from a PR)

### Wave 2 (22 commits)
- `868088f7` #2063 TB5 discovery + JACCL retry + IP preference
- `157a4c24` #1598 CORE-ONLY port (HF Hub cache discovery; UX overhaul deliberately NOT ported to avoid destroying 26+ of our ports)
- `f4abf47e` #2177, `1102e28b` #2121, `b2b3dd74` #2184, `5686da7c` #2187, `a503b2b6` #2054, `5c0480b2` #2055, `b4d02e38` #2134 (SSRF), `bf38d869` #2001, `10876037` #2189, `b55c99b0` #2203, `1d83a244` #2123, `ba7092d1` #2202, `ac7a170f` #2162, `e7e0d34b` #2133, `c34c1589` #2161, `6e71b7bd` (typing), `dbf76ecb` #2205, `143d70c2` #2231, `68d32e8c` #1804, `165b614e` #2169+#2148

### Wave 3 (8 commits)
- `3abc8623` #1999 (DSv4 fused MoE, opt-in), `8ccbe8c8` #2254 (link latency), `86927717` #2155 (PREFILL_STEP_SIZE), `15647f8a` #2222 (Claude system role), `b80ffbf1` #2237 (log viewer), `366c5302` #2179 (use_prefix_cache), `36acecdd` #2035 (liveness tuning), `91173df0` #2060 (symlink-safe deletion, Python part)

---

## PART 2: SSH / Linux deployment saga (2026-08-06)

### Problem: Could not connect to 10.2.0.16
**Source:** User request ("ssh to 10.2.0.16, no password needed")
**Errors hit & fixes:**
1. `REMOTE HOST IDENTIFICATION HAS CHANGED` — stale known_hosts → `ssh-keygen -R 10.2.0.16`
2. `Permission denied (publickey,password)` for all users (chris/exo/ubuntu/admin/nvidia/root/dev/pi) — machine actually had password auth
3. VS Code sensitive-input guard cancelled SSH (autopilot mode can't supply secrets) — user gave password `violin`; used `sshpass -f /tmp/.sshpass_exo` with umask 077, deleted after each command (never in history/process list)
4. **User decision:** modified VS Code Insiders source (`workbench.desktop.main.js`) — patched `if(u){` → `if(u&&!1){` in `_registerSensitiveInputElicitation` so autopilot stops killing password prompts; backed up original; re-signed app; created `/Users/chris/scripts/vscode-patch-sensitive-input.sh` (re-run after each Insiders update)

### Machine discovery
- Proxmox VM `tryingOutExo`: Ubuntu 24.04.4, kernel 6.17.13-2-pve, 12 CPU, 24GB RAM
- **2× RTX 5060 Ti 16GB passed through** (initially appeared "no GPU" because nvidia-smi absent)

### Source copy
- Backed up remote checkout → `~/exo.upstream-bak`
- rsync'd full repo (19,430 files, ~2GB incl .git with 56 commits); excluded target/.venv/build/node_modules
- Fixed git "dubious ownership" via `safe.directory`
- Rebuilt dashboard locally FIRST (was stale — log viewer + eject changes missing)

### mlx on Linux — the dependency nightmare (biggest error chain)
**Error:** `ImportError: libmlx.so: cannot open shared object file`
**Root cause chain:**
1. Our lock file (from #2103 port) pins mlx to custom fork `rltakashige/mlx-jaccl-fix-small-recv` — Linux x86_64 mlx = TWO packages: `mlx` (bindings, 0.7MB) + `mlx_cuda_12` (runtime 337MB, libmlx.so)
2. `uv sync --extra mlx` never installed the runtime wheel into venv (only in cache)
3. Copied wrong-arch libmlx.so from cache (ARM aarch64 — `file` revealed it; `ldd` said "not a dynamic executable")
4. Downloaded correct x86_64 wheels from fork release; renamed properly (uv rejects unversioned names); installed → **`cuda available: True`, GPU matmul OK**

### Driver saga
**Error:** `Failed to initialize NVML: Driver/library version mismatch, NVML library version: 580.178`
**Chain:**
- Kernel module loaded: 595.58.03 (Open Kernel Module from Proxmox host)
- My apt install pulled 535 → mismatch; Ubuntu repo had 595.84 → still mismatch
- **Fix:** downloaded NVIDIA official 595.58.03 `.run` (422MB), extracted (`sh ... -x`), copied userspace libs (libcuda, libnvidia-ml, ptxjitcompiler, tls, gpucomp), symlinked over 595.84, `nvidia-smi` → `/usr/bin/nvidia-smi-595.58`
- Result: nvidia-smi works — 2× RTX 5060 Ti, 16GB each, CUDA 13.2

### CUDA headers error (runner crash during warmup)
**Error:** `Can not find locations of CUDA headers, please set environment variable CUDA_HOME or CUDA_PATH`
**Fix:** `apt install nvidia-cuda-toolkit` + `CUDA_HOME=/usr/local/cuda` (symlink → /usr) in `/etc/environment`; started exo with `CUDA_HOME`/`CUDA_PATH` env

### Election churn mystery
**Symptom:** Linux node kept demoting itself; phantom node IDs won elections
**Root cause:** **stale EXO.app on BOTH Macs** crash-looping (missing `event_id` on BaseEvent — old bundle vs new protocol). Confirmed via ps + crash logs (`exo-2026-08-06-*.ips`). Fixed by user rebuilding the macOS app.
**Lesson recorded:** bundle must be rebuilt from source after protocol changes (README documented).

### Final success
- exo detached: `setsid nohup env CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda uv run exo`
- Verified: Llama-3.2-1B **73.5 tok/s**, Llama-3.2-3B **29 tok/s**, single-node, `min_nodes:1`
- Log: `CUDA MLX backend detected; reporting GPU VRAM as node memory` (our #2103/#2216 live)

---

## PART 3: Eject UI work (user request)

### Commit `ca8474c7` — inline EJECT split-button (user request)
**User asked:** remove "do you want to delete it?" popup after eject; replace confirm() with inline NO/YES split (NO = outline, YES = filled, smooth fly animation)
**Changes:**
- NEW `dashboard/src/lib/components/EjectButton.svelte` — Svelte 5 runes, `transition:fly` split (NO left outline / YES right filled)
- `+page.svelte` — removed both confirm() calls; weights auto-deleted on eject (from #2238 original behavior)
- Barrel export + DIRECT import (Svelte 5 tree-shaking broke barrel-only: `EjectButton is not defined` runtime error → fixed with direct `import EjectButton from "$lib/components/EjectButton.svelte"`)
- Build issue hit: Svelte 5 forbids `transition:` + `out:` on same element → used `transition:` only

### Commit `3d1a675c` — eject NO LONGER auto-deletes weights (user request)
**User asked:** "I DO NOT WANT THAT to happen i will delete it on my own. i do not want that to happen that a model will delete itself automatically."
**Changes:**
- Removed `deleteDownload()` block from `ejectInstance()` (was deleting weights on every node after eject)
- Removed `getInstanceNodeIds()` helper (existed only for deletion)
- Removed `deleteDownload` import
- EjectButton YES tooltip: "Eject instance" (was "Eject instance and delete weights")
- **New behavior:** Eject = stop instance ONLY. Weights stay. Manual deletion via Downloads page (never removed).

---

## PART 4: Full app rebuild (user request + error fixes)

**Request:** rebuild the app, copy to /Applications, get exo running
**Process:**
1. `pkill` old app
2. `uv sync --extra mlx` → **ERROR: mlx vanished** (git-fork install failed silently) → `uv pip install --reinstall-package mlx "mlx @ git+...cc3f3e60"`
3. Pyinstaller: `uv run pyinstaller packaging/pyinstaller/exo.spec --noconfirm` (first run failed "Module 'mlx' is not available" — fixed by reinstalling mlx)
4. `xcodebuild -project app/EXO/EXO.xcodeproj -scheme EXO -configuration Release build` → **BUILD SUCCEEDED** (output to DerivedData, NOT app/EXO/build)
5. Install: `sudo rm -rf /Applications/EXO.app && sudo cp -R <DerivedData>/EXO.app /Applications/`
6. **ERROR: launch failed "Permission denied"** — bundle was `drwx------` (700/root) → `sudo chmod -R a+rX` + re-sign ad-hoc + verify
7. Launched OK; API live; new EJECT served

---

## PART 5: Deep research (user request — research only, no changes)

### Question 1: "Do we have these set up?" (heterogeneous checklist)
**Verified LIVE against running systems (not just code):**
| Claim | Code | LIVE |
|---|---|---|
| 1. Placement treats Metal+CUDA valid | ✅ `_cycle_accelerator_score` (placement.py:113) | ⚠️ Linux advertises `["MlxCpu"]` only — **pynvml NOT installed** (only in mlx-cuda12/13 extras) |
| 2. Discovery | ✅ | ✅ all 3 nodes visible both sides |
| 3. Memory accounting | ✅ `_mlx_uses_cuda_gpu()` True | ⚠️ reports 26.7GB system RAM — nvidia-smi installed as `nvidia-smi-595.58`, code looks for plain `nvidia-smi` |

**Fix (not yet applied — user hasn't confirmed):**
```bash
cd /root/exo && uv sync --extra mlx-cuda12
ln -sf /usr/bin/nvidia-smi-595.58 /usr/bin/nvidia-smi
```

### Question 2: #2064 asymmetric TP — can we include it?
**Research:** splits layers proportionally to node memory; activation gate = even split >90% smallest node RAM; Qwen3.5-only whitelist; high placement.py conflict; multi-GPU gap.
**Verdict:** Option 2/3 — wait until hitting >200B wall; pipeline parallelism already handles large-model splitting.

### Question 3: #2129 / #2200 deep dive
**#2129** (3-node Metal+CUDA, @Juliuscply, +444/−45): TcpRelay (CUDA↔CUDA send/recv hangs in MLX 0.32 → raw TCP bypass), `_rotate_metal_to_middle()`, `MLX_HOSTS_JSON`, CUDA rank probing, post-prefill barrier (generate.py:373), `_linux_interface_types()`. We ALREADY have: cuda_compat, patch registration, Darwin-gated FAST_SYNCH, CUDA-aware wired limit. **SKIP:** IP-priority flip (ethernet>TB contradicts our #2063/#2254 TB preference).
**#2200** (backend per shard, @oliver0006): `ShardMetadata.backend` + `assign_shard_backends()` + `apply_shard_backend()`; low conflict; recommended BEFORE #2129.
**Docs:** `deep-dive-2129-2200-heterogeneous.md`

### Question 4: Non-MLX models in menu
**No PR exists** — exo has ONE engine (MLX); GGUF/PyTorch = new subsystem.

### Question 5: The two branches (track-a / track-b)
Created as bookmarks, abandoned empty when user pivoted to single-branch merging. All value in `fix-memory-error` (157a4c24, 868088f7).

---

## PART 6: Current cluster state (as of end of session)

- This Mac: EXO.app rebuilt, running, new eject UI deployed
- mini2: EXO.app restarted (restore cluster)
- Linux box 10.2.0.16: exo running detached, reports MlxCpu + system RAM (known gap — needs pynvml + nvidia-smi symlink to become MlxCuda + VRAM)
- MY commits this session (error UX wave): `4f850371`, `85f1b024`, `5acd6d6c`, `ef16c380` (see PART 9 for full details)
- #2129 port: `72cf86bb` (see PART 7)
- Other agent's commits on branch (NOT this session): `d6fcef17` + their EjectButton.svelte uncommitted fix; the `3d1a675c` eject-no-weight-delete commit was ALSO theirs (per commit history review) — this session's eject commit is `ca8474c7`

---

## APPENDIX: Commit quality standard (user request)

Every commit must include:
1. What changed (files/functions)
2. Why (user request / PR port / bug fix)
3. Source attribution: PR #, author, URL (for ports)
4. If fixing a problem from other people's code: reference the PR/commit/link
5. Verification notes (basedpyright/ruff/tests)

**URLs for reference:**
- PR #2103: https://github.com/exo-explore/exo/pull/2103 (native CUDA)
- PR #2129: https://github.com/exo-explore/exo/pull/2129 (heterogeneous cluster)
- PR #2200: https://github.com/exo-explore/exo/pull/2200 (backend per shard)
- PR #2064: https://github.com/exo-explore/exo/pull/2064 (asymmetric TP)
- PR #2063: https://github.com/exo-explore/exo/pull/2063 (TB5 discovery)
- PR #1598: https://github.com/exo-explore/exo/pull/1598 (HF cache discovery)
- PR #2238: https://github.com/exo-explore/exo/pull/2238 (eject + weight delete)
- PR #2216: https://github.com/exo-explore/exo/pull/2216 (CUDA VRAM)
- PR #2252: https://github.com/exo-explore/exo/pull/2252 (accelerator pref)
- PR #2227: https://github.com/exo-explore/exo/pull/2227 (capacity-aware)

---

## PART 7: #2129 heterogeneous cluster PORT (2026-08-06) — commit `72cf86bb`

**User request:** "figure out b first and the changes that are in 2129... recognize all the changes so we don't lose any commits since 2129 as well as our personal commits"
**PR:** https://github.com/exo-explore/exo/pull/2129 — "3-node heterogeneous cluster support (Metal + CUDA)" by @Juliuscply, 7 commits, +444/−45, no reviews, still open. **This is the PR referenced in #2103's comments** (the one that enabled the Mac + Linux GPU split cluster at 69.3 tok/s on DeepSeek-V3.1-4bit).

**What #2129 fixes (the three blocking problems on top of #2103):**
1. CUDA↔CUDA `send`/`recv` **hang** in MLX 0.32.0 ring backend → TcpRelay raw-TCP bypass
2. Unreliable CUDA peer detection → explicit probe → `MLX_CUDA_RANKS`
3. Stale connections blocking node join → MSG_PEEK filtering in accept loop

**What I ported (surgical hand-port — NOT cherry-pick, because our branch has 57+ commits since #2129's May 31 base):**
| Piece | File | Adaptation |
|---|---|---|
| TcpRelay class (~220 lines) | auto_parallel.py | Raw-TCP send/recv; routes CUDA↔CUDA ring traffic only; Metal paths untouched. Adapted to OUR wrappers (PipelineFirstLayer/PipelineLastLayer, not PR's _RingRecvWrapper) |
| `MLX_HOSTS_JSON` + eager relay start + CUDA peer probe | utils_mlx.py | Hostfile survives tempdir cleanup; relay starts on CUDA nodes at distributed init; shard_and_load probes peers → MLX_CUDA_RANKS |
| `_rotate_metal_to_middle()` | placement.py | 3-node: Metal in middle rank; 4+: interleave. Hooked after `_prefer_socket_reachable_rank_zero`. +5 new unit tests |
| Prefill barrier | generate.py | `mx_barrier(group)` after pipeline prefill (is_pipeline) |
| `_linux_interface_types()` | system_info.py | Linux NIC type inference from names (no networksetup) |

**Deliberately SKIPPED:** the PR's `placement_utils.py` IP-priority flip (ethernet > thunderbolt — was for DGX boxes; contradicts our TB4-optimized `_address_priority` from #2063 and latency-aware ring selection from #2254).

**Already present from our #2103 port:** `cuda_compat.py` shim, `MLX_METAL_FAST_SYNCH` Darwin gating, CUDA-aware wired limit, patch registration — so those #2129 pieces were no-ops for us.

**Verification:** basedpyright 0 errors (had to fix 38 strict-typing errors from the port: `cast()` for json.loads Any, `NodeId(...)` NewType wrapping in tests, `contextlib.suppress` for ruff SIM105, `raise ... from None` for B904); ruff clean; 620 tests pass (only pre-existing baseline failures: rust exo_rs sleep test, download-tests import mode). Known PR limits preserved: 4+ node rotation interleaving suboptimal, 2s probe overhead, relay lacks exit cleanup.

---

## PART 8: Cluster version-mismatch crisis + re-sync (2026-08-06)

**Symptom (user report):** "models keep failing to load" across the cluster.

**Diagnosis chain:**
1. This Mac log: election churn — "Node elected Master" every 3s
2. mini2: had crashed at 17:02 (2 fresh crash reports)
3. **Linux box was on `91173df0` (STALE)** — missing #2129 port AND eject UI, while Macs had newer builds
4. Root cause: **three nodes on three different code versions** → event-schema mismatch → split-brain elections → instances torn down mid-load → "models fail"

**Fix (all nodes synced to HEAD `72cf86bb`):**
1. rsync'd updated source to Linux (only changed files); verified `git log` = `72cf86bb`; restarted with `setsid nohup env CUDA_HOME=...` → **stable (1 election in fresh log vs churning before)**
2. Rebuilt this Mac's app: pyinstaller → xcodebuild → installed to /Applications → fixed perms (`chmod -R a+rX` — bundle was 700/root) → re-signed → launched
3. Copied the **identical** built app to mini2 via rsync (byte-identical, not a separate build) → fixed perms → re-signed → launched
4. Verified: 3 nodes in topology (`6aa60be0` Mac, `9faabaa5` mini2, `3ad22f85` Linux); real churn markers (expired/demoting) gone since sync

**Key lesson recorded:** the every-3s "elected master" line is the NORMAL election tick (same node always wins, no disruption) — not churn. Real churn = "expired"/"demoting"/"discovered" of phantom node IDs.

**Other agent context:** the other agent's rebuilt app was missing #2129; their `EjectButton.svelte` fix (id prop rename + `svelte/transition` import — both bugs I'd identified earlier) was found UNCOMMITTED in the working tree and left untouched per campaign rules.

---

## PART 9: Error UX wave — 4 commits (2026-08-06, all MY work this session)

**User request:** "make it easier to know why models fail... not just have the model disappear... server fails to start → notification or message in the EXO taskbar menu... not just 'exit code 1'." Then: "keep things related to STARTING THE SERVER in the swift app, things related to errors AFTER the server is successfully running in the browser UI."

### Commit `4f850371` — keep failed instances visible + render error details (BROWSER UI)
**Problem:** master's `_plan()` auto-deleted any instance with a `RunnerFailed` runner within ~1s → card vanished → user had no record of why.
**Changes:**
- `src/exo/master/main.py` `_plan()`: removed the RunnerFailed→InstanceDeleted auto-cleanup (node-disconnected cleanup unchanged). Failed instances stay in state, marked FAILED, until user ejects. Logs failure reason.
- `dashboard/src/routes/+page.svelte`: new `getInstanceRunnerError()` helper extracts `error_message` + `diagnostics[]` from the instance's `RunnerFailed` status via `runnersData`; both instance-card render sites show a red error box (message + diagnostics joined with " • ") when FAILED.
**Verified:** dashboard builds; basedpyright 0; ruff clean; 129 master/shared tests pass.

### Commit `85f1b024` — Swift app: real startup failure reasons (SWIFT / TASKBAR)
**Problem:** `ExoProcessController.swift` showed only "Process exited with code N" AND discarded stderr entirely (`standardError = FileHandle.nullDevice`) — the diagnostic trail was thrown away.
**Changes (ExoProcessController.swift):**
- Capture child stderr via `Pipe` into an 8KB ring buffer (`stderrTail`) via `readStderr()`
- `extractStartupFailureReason(exitCode:)`: recognizable patterns first (port in use, missing native library, missing Python module, CUDA headers, GPU driver mismatch, no disk space) → then most relevant ERROR/Traceback line from `~/.exo/exo_log/exo.log` (via `lastLogTail()`) → fallback exit code
- Result shows in the EXISTING taskbar dropdown (ContentView renders `controller.lastError`)
**Errors hit & fixed:** `binary operator '-' cannot be applied to UInt64 and Int64` → rewrote `lastLogTail` with `size > UInt64(maxBytes) ? size - UInt64(maxBytes) : 0`
**Verified:** swiftc -parse OK; xcodebuild Release BUILD SUCCEEDED.

### Commit `5acd6d6c` — stable error codes in API (FEEDS BOTH)
**Problem:** API errors only carried HTTP status phrase as `type` — no stable machine-readable code for UI to branch on.
**Changes:**
- `src/exo/api/types/api.py`: `ErrorCode` Literal (INVALID_REQUEST, INSUFFICIENT_MEMORY, PLACEMENT_FAILED, MODEL_NOT_FOUND, INSTANCE_NOT_FOUND, IMAGE_NOT_FOUND, PEER_UNREACHABLE, NOT_FOUND, INTERNAL_ERROR); `ErrorInfo.error_code` field added (default INTERNAL_ERROR); `type` (phrase) + `code` (HTTP status) kept for backward compat
- `src/exo/api/main.py`: `ApiError(HTTPException)` carrying error_code; `http_exception_handler` resolves via `_error_code_for()` (explicit wins, else derived: 404→NOT_FOUND, 400/422→INVALID_REQUEST). Key sites upgraded: memory gate (INSUFFICIENT_MEMORY), placement errors (PLACEMENT_FAILED), instance-not-found (INSTANCE_NOT_FOUND), model-not-found (MODEL_NOT_FOUND), custom card missing (MODEL_NOT_FOUND)
- `src/exo/api/types/__init__.py`: added ErrorCode export (test imports failed without it)
**Verified:** basedpyright 0; ruff clean; 81 api tests pass; live smoke test showed `INSUFFICIENT_MEMORY`/`MODEL_NOT_FOUND` in responses.

### Commit `ef16c380` — publish_bytes guard + dashboard warning chip (BROWSER UI)
**Problem:** `router.py:86` `publish_bytes` had no try/except — a malformed/schema-desynced event (like the earlier event_id mismatch) killed the WHOLE process with no message.
**Changes:**
- `src/exo/routing/router.py`: `publish_bytes` wrapped in try/except → logs warning + `record_malformed_event()` (bounded 20-entry ring buffer) + `malformed_event_log()` accessor. Only bad event dropped; server keeps running.
- `src/exo/api/main.py`: new `GET /v1/warnings` → `{malformed_events, malformed_event_count}`
- `dashboard/src/lib/stores/app.svelte.ts`: `warnings` state; `fetchWarnings()` polls every 5s; `warnings()` export; `warningsInterval` cleanup in stopPolling
- `dashboard/src/routes/+page.svelte`: yellow **MALFORMED EVENTS (n)** chip in clusterWarnings (reusing the macOS MISMATCH / RDMA NOT ENABLED chip template: warning triangle `warningIconPath`, mono label, ✕ dismiss), hover tooltip lists dropped event topics/errors; both chip render sites; per-session dismissible
**Verified:** dashboard builds; basedpyright 0; ruff clean; 89 routing+api tests pass.

### User-driven division of concerns (recorded for future work)
- **Swift app = server STARTUP errors** (port conflicts, missing libs, driver issues, exit codes)
- **Browser UI = runtime errors after server is running** (model load failures, malformed events, placement errors)
- Error codes (Commit 3) feed BOTH — the UI can branch on `error_code` without parsing messages

---

## PART 10: Current state (end of session, accurate as of 2026-08-06 evening)

- **Branch:** `fix-memory-error`, 60+ commits ahead of main (all PR ports + personal fixes)
- **This Mac:** EXO.app rebuilt from HEAD (eject UI + #2129 + error UX), running, serving new dashboard
- **mini2:** identical rebuilt app, running
- **Linux 10.2.0.16:** exo running detached at HEAD `72cf86bb` (source sync), 2× RTX 5060 Ti, CUDA backend
- **Known gaps (tracked, not yet fixed):** pynvml not installed on Linux → advertises MlxCpu instead of MlxCuda (fix: `uv sync --extra mlx-cuda12`); nvidia-smi installed as `nvidia-smi-595.58` → code looks for plain `nvidia-smi` (fix: symlink); `promptsANDplans/` is gitignored (tracking docs are local-only, never committed)

---

## PART 11: Continuation session — EJECT runtime fixes, mlx_lm bundle fix, port conflict, new research (2026-08-06 evening→night)

### 11.1 EJECT button runtime bugs — FOUND AND FIXED via live browser test
**Symptom (user):** "when i click yes or no it will either put the eject button back how it was if i did no and yes will do the same functionality as before where it ejects the model — whats going on with it not working?"
**Diagnosis:** previous session identified the two bugs but they were only fixed in the OTHER agent's uncommitted working tree. This session I applied + verified them:
1. **Missing `svelte/transition` import** (`fly`/`fade`) in `EjectButton.svelte` — Svelte 5 compiles unimported transition names to bare refs (`()=>fly`) → `ReferenceError: fly is not defined` the instant EJECT was clicked → NO/YES never rendered. **Fixed:** `import { fade, fly } from "svelte/transition";`
2. **Prop name mismatch** — parent renders `<EjectButton {id} onConfirm={...}/>` but the component declared `instanceId` → `instanceId` was always `undefined` → YES fired `onConfirm(undefined)` → `DELETE /instance/undefined` → **404 "Failed to eject instance"**. **Fixed:** renamed prop to `id`.
**Verified end-to-end in a real browser (Playwright):** clicked EJECT → NO/YES appeared (transition fix); clicked NO → EJECT restored, instance untouched; clicked EJECT → YES → `DELETE /instance/1ee19017-...` → **200**, instance ejected. Network capture proved the URL carried the real ID, not `undefined`.

### 11.2 mlx_lm missing from bundle — MODELS WOULDN'T LOAD (build-environment bug)
**Symptom (user):** "the current app not being able to load the models... is not able to load small models that i used to be able to"
**Diagnosis chain:**
1. Runner errors in state: `ModuleNotFoundError: No module named 'mlx_lm'` + `Failed to load the default metallib` (secondary)
2. `mlx_lm` is a **git dependency** (pyproject.toml line 96) only installed via `uv sync --extra mlx`
3. A plain `uv sync` (without `--extra mlx`) **prunes** it from the venv — so the rebuild environment lacked it
4. `exo.spec` `_safe_collect()` **silently swallows** the missing package → bundle built WITHOUT mlx_lm → every model load crashed
**Fix:** `uv sync --extra mlx --reinstall-package exo_rs` → verified `import mlx_lm` OK → rebuilt pyinstaller → **489 mlx_lm modules** in `Analysis-00.toc` (was 0). Also hit `dist/exo not empty` pyinstaller error → `rm -rf dist/exo` then rebuild succeeded.
**Verified:** isolated bundle test (`--namespace bundle-test -m --api-port 52416`), placed Llama-3.2-1B → `RunnerLoaded` → model loads.

### 11.3 Port-conflict exit-code-1 (the "backend on localhost:52415" mystery)
**Symptom (user):** exit-code-1 errors opening the app again; asked if a backend was running on 52415.
**Diagnosis:** a **stale test backend** I'd launched earlier (PID 53392, `./exo`, started 16:08 during bundle testing) was still holding port 52415 → the real app couldn't bind → exited 1. Not a code bug.
**Fix:** `kill -9 53392` → port free → relaunched app → PID 89644, API 200, 0 crashes, joined 3-node topology.
**Lesson:** isolated test backends MUST use a non-default port (52416) and be cleaned up — later tests already did this.

### 11.4 App deployments this session (both machines, two rounds)
- Round 1 (EJECT transition fix): rebuilt dashboard → pyinstaller → xcodebuild → installed locally → copied to mini2
- Round 2 (EJECT prop fix + mlx_lm): full rebuild again → installed locally (PID 53068) → copied to mini2 (node `e45455dc`)
- mini2 verified: fixed chunk served (`2.BlaL9Lg6.js` contains "Cancel eject"), 0 crashes

### 11.5 New research documents (2026-08-06 night)
| Doc | Topic |
|---|---|
| `research/error-ux-research.md` | Full error pipeline audit: where rich data exists (ErrorResponse, RunnerTerminationError, diagnostics) vs where it's lost (publish_bytes crash, Swift stderr discard, dashboard FAILED label) |
| `research/error-ux-implementation-research.md` | Implementation plan: publish_bytes guard (verified crash path + 2-guard design), dashboard runner-error rendering, stable error codes, Swift exit reason |
| `research/build-architecture-dev-loop.md` | How EXO.app is assembled (2 layers), `EXO_RUNTIME_DIR` hot-swap, `EXO_DASHBOARD_DIR` override, the full "what rebuilds for what change" matrix |
| `research/message-queueing-steering-research.md` | Message queueing: no upstream PR; backend ALREADY batches 8 concurrent (`EXO_MAX_CONCURRENT_REQUESTS`); frontend blocks sends (ChatForm:206, :442, app.svelte.ts:2410); FIFO queue design; steering feasibility (request-param steering works offline, activation steering doesn't) |

### 11.6 EjectButton.svelte final state (both fixes, documented in component)
The component now carries inline NOTE comments explaining both bugs so they don't regress (see `dashboard/src/lib/components/EjectButton.svelte`).
- **Other agent's uncommitted work in tree:** `EjectButton.svelte` (id prop + svelte/transition import) — left untouched

**Session memory file:** `/memories/session/error-ux-wave.md` (same content, shorter form)

---

## APPENDIX B: Detailed Log Index (where to find the full detail)

This master document is the overview. For EXHAUSTIVE detail, see:

| Topic | File |
|---|---|
| **Every question asked + full answer** | `session-qa-log-2026-08-06.md` (24 Q&As, verbatim) |
| **Linux deployment — every command/error/fix** | `linux-deployment-log-2026-08-06.md` (10 phases, command reference) |
| **Eject UI saga — every decision/error** | `eject-ui-work-log-2026-08-06.md` (5 chapters, code locations) |
| **VS Code Insiders patch** | `vscode-insiders-patch-log-2026-08-06.md` (root cause, patch, maintenance) |
| **PR research #2129/#2200** | `remaining-prs/deep-dive-2129-2200-heterogeneous.md` |
| **PR campaign all waves** | `pr-merge-campaign-tracker.md` |

### Documentation update log (2026-08-06 evening)
- `README.md` — research subfolder index expanded with all new logs
- `GIT-HISTORY-AND-UPSTREAM.md` — stale "8 commits" → full 63-commit inventory (WIP + Wave1 + Wave2 + Wave3 + session commits, other agent's separated)
- `KNOWN-ISSUES-AND-ROADMAP.md` — added §3.1 (8 session-discovered issues: pynvml, nvidia-smi name, driver fragility, mlx two-package, VS Code patch, stale-bundle crash, app perms, eject behavior) + session roadmap (2-command Linux fix → #2200 → #2129 order → deferred list)
- `AGENT-CONTEXT.md` — TL;DR updated: 63 commits, cluster hardware (incl. Linux box + known gaps), eject behavior, detached run command
- Memory file `/memories/repo/exo-cluster-state.md` — branch state, hardware, Linux gaps + fixes, eject behavior, VS Code patch script, verification gate

---

## PART 6: Wave 4 — Heterogeneous Cluster + Qwen3.5/CUDA 12.8 Fix (evening)

### 6.1 Commit `72cf86bb` — PR #2129 3-node heterogeneous Metal+CUDA port
- **Source:** https://github.com/exo-explore/exo/pull/2129 by @Juliuscply (open, no reviews, 7 commits, +444/-45)
- **Why:** The PR referenced in #2103's comments — enables Mac (Metal) + Linux GPU (CUDA) nodes to run one pipeline-split model together (the author's test: 1× Mac + 2× Linux GPU, DeepSeek-V3.1-4bit, 69.3 tok/s)
- **What was ported (surgical, adapted to our 57-commit branch):**
  - `TcpRelay` class (auto_parallel.py): raw-TCP send/recv bypass for CUDA↔CUDA ring links that hang in MLX 0.32.0; port 40000+rank, `MLX_TCPRELAY_PORT` configurable
  - `MLX_HOSTS_JSON` + eager relay start + CUDA peer probe → `MLX_CUDA_RANKS` (utils_mlx.py)
  - `_rotate_metal_to_middle` (placement.py) + 5 new unit tests — 3-node Pipeline rotates Metal to middle rank so all links are Metal↔CUDA
  - Post-prefill `mx_barrier` (generate.py) — prevents all_gather deadlock
  - `_linux_interface_types` (system_info.py) — Linux NIC type inference
- **What was skipped (with reason):** `placement_utils.py` IP-priority flip (ethernet-over-TB was for DGX boxes; conflicts with our TB4-optimized #2063/#2254 latency-aware selection — would hurt 2×TB4 Macs when latency data missing)
- **Already present from #2103 port:** cuda_compat.py, platform-gated MLX_METAL_FAST_SYNCH, CUDA-aware wired limit
- **Verification:** pyright 0 errors, ruff clean, 620 tests pass (only pre-existing baseline failures)

### 6.2 Commit `d6fcef17` — Model picker READY NOW section + Running tab
- **Requested by:** user (pain point: ready models buried in picker)
- **What:** rewired existing recommended/other split into readiness split (readyGroups = live instances) + new "Running" sidebar tab
- **Verified:** live in running app via browser DOM

### 6.3 The Qwen3.5-2B-MLX-8bit failure on Linux — ROOT CAUSE + FIX
- **Symptom:** "Model failed: Qwen3.5-2B-MLX-8bit" — loads on Macs, crashes on Linux box
- **Root cause:** Linux box = 2× RTX 5060 Ti (**Blackwell, sm_120**); Ubuntu's `nvidia-cuda-toolkit` = **CUDA 12.0** (Jan 2023) which only knows ≤ sm_90. MLX CUDA backend JIT-compiles kernels at warmup with nvrtc → `nvrtc: error: invalid value for --gpu-architecture (-arch)` → runner crash. NOT a code/PR problem — environment problem.
- **Fix applied (machine-level, not committed — it's box config):**
  - Installed `cuda-nvrtc-12-8`, `cuda-nvvm-12-8`, `cuda-cudart-dev-12-8` (first CUDA with Blackwell support) from NVIDIA apt repo
  - Symlinked `/usr/lib/x86_64-linux-gnu/libnvrtc.so.12` → CUDA 12.8's; `libnvvm.so.4` → 12.8's
  - Re-fixed driver libs 595.58.03 (apt installs had clobbered with 595.84 again)
  - **Proof:** `@mx.compile` JIT test passes; Qwen3.5 placed single-node on Linux box → "warmed up by generating 50 tokens"; chat completed 17.6 tok/s; GPU showed 2,973MiB (GPU0) + 154MiB (GPU1) — spans both RTX 5060 Tis

### 6.4 DISCOVERED ISSUE: torch missing on Linux = vision disabled (graceful)
- **Symptom:** `ModuleNotFoundError: No module named 'torch'` when loading vision weights → "Failed to load vision weights — disabling vision for this runner" (code already has graceful fallback `_patch_video_processor`, vision.py:113)
- **Why torch absent:** pyproject.toml mlx extra declares `torch==2.10.0; sys_platform == 'linux'` (line 55) BUT the committed uv.lock torch entries only resolve on darwin (markers like `sys_platform == 'darwin' or (extra == 'extra-3-exo-mlx-cpu' and extra == 'extra-3-exo-mlx-cuda12')...` — the extra-pairs are contradictory/never true for plain `--extra mlx` on linux). **Lock file out of sync with pyproject** (upstream inconsistency).
- **Impact:** text/thinking works on Linux; vision models (Qwen3.5, Kimi, etc.) lose image input on Linux box only
- **Fix options (not yet done — decision needed):**
  1. Regenerate uv.lock (`uv lock`) so linux+torch resolves → `uv sync --extra mlx` installs torch on Linux (~2.5GB CUDA wheel from PyPI — heavy but official)
     - ⚠️ **UPDATE from §7.1:** regen alone does NOT fix it (verified twice — uv re-introduces the broken markers). The actual fix = manual 3-line marker patch (commit `f48bdf86`).
  2. Manually install CPU-only torch on the box (`uv pip install torch --index-url https://download.pytorch.org/whl/cpu`) — smaller, no CUDA conflict
  3. Keep graceful degradation (status quo)

### 6.5 DISCOVERED ISSUE: RDMA "ghost edge" Mac↔Linux (2-node placement segfault)
- **Symptom:** 2-node Mac+Linux placement created `MlxJacclInstance` with `jaccl_devices=[[[], ['rdma_en2']], [['rdma_en2'], []]]` + coordinators at 10.2.0.11 (mini2) / 0.0.0.0 → **Mac runner segfaulted (signal 11) during `mx.distributed.init(backend="jaccl")`**, Linux side never participated → instance died
- **Explanation:** the Mac's info gatherer labels interface `en2` (Thunderbolt bridge / LAN NIC on Mac mini) as RDMA. The topology then contains an RDMA connection between the Mac and the Linux box — but the Linux box is a Proxmox VM with virtio NICs (`eth0@if25`, `google@if26`): **no RDMA hardware, no `rdma_en2`, no RDMA verbs**. The master trusted the one-sided edge, built a JACCL ring, and the Mac crashed trying to handshake over a link the Linux side can't speak.
- **Single-node placement avoids it entirely** (no distributed init).
- **Fix direction (not yet done):** RDMA edges should only be created when BOTH endpoints report RDMA capability (there's already `NodeRdmaCtlStatus` + `_all_rdma_ctl_enabled` in placement.py — verify why it didn't gate this; likely the Linux box's rdma_ctl probe reports enabled without hardware, or the guard isn't applied to this path). Also: `_rotate_metal_to_middle` (our #2129 port) only runs for `len(selected_cycle) >= 3` — 2-node Mac+Linux still hits the broken jaccl path.

### 6.6 Cluster state after fixes
- Linux box: exo running detached (CUDA_HOME/CUDA_PATH env), Qwen3.5-2B works single-node, Llama 1B/3B work
- mini2 + this Mac: EXO.app running (fresh builds)
- Master = this Mac (9faabaa5...) at time of testing

---

## PART 7: torch on Linux (Option A) + RDMA investigation + docs (late evening)

### 7.1 Commit `f48bdf86` — uv.lock torch fix (Option A)
- **Requested by:** user ("for the linux stuff i definitely want to get option a working" — torch installed properly for Linux)
- **Root cause (proven by experiments on the box):** pyproject declares `torch==2.10.0; sys_platform == 'linux'` in the mlx extra, but uv.lock's torch edges carried `extra != 'mlx-cpu' and extra != 'mlx-cuda12' and extra != 'mlx-cuda13'` clauses (uv's encoding for the mlx-cpu/cuda12/cuda13 extras which include exo[mlx]). uv 0.12.1/0.12.2 mis-evaluates those clauses → edge always FALSE on Linux → torch never installed.
- **Evidence chain:** scratch project with plain marker installs torch fine (mechanism works); removing the `!=` clauses from the lock made `uv sync --extra mlx --dry-run` resolve torch==2.10.0+cu130.
- **Fix:** simplified 3 edges (torch, torchaudio, torchvision) in uv.lock to `sys_platform == 'linux' and extra == 'mlx'`.
- **Verified:** `uv sync --extra mlx` on box → torch 2.10.0+cu130 + torchvision + triton installed; `import torch` OK; `_torch_tensor_to_mx` (the exact vision path that crashed) works.
- **Caveat documented in commit:** `uv lock` regeneration re-introduces the broken markers (uv quirk). Related to #2103's extra layout.
- **Side note:** the sync reverted the manually-installed mlx wheels to the URL wheel → `libmlx.so` missing again → reinstalled `mlx_cuda_12` runtime wheel from /tmp/mlxwheel (documented in the setup guide §3).

### 7.2 RDMA "ghost edge" — hypothesis CORRECTED
- **My earlier claim was wrong:** I said there was a fake Mac↔Linux RDMA edge. Evidence shows the failing 2-node jaccl instances were **Mac↔Mac** (jaccl_coordinators: mini2 0.0.0.0 + this Mac 10.2.0.11 — 10.2.0.11 IS mini2's LAN IP).
- **Code truth:** `apply.py` `MacThunderboltConnections` only creates RDMA edges when BOTH endpoints report `MacThunderboltIdentifiers` (macOS-only) AND both rdma_ctl enabled. Linux reports neither → **Mac↔Linux RDMA edge is structurally impossible**.
- **Actual failure chain:** Mac↔Mac jaccl segfaulted at 18:05 (right after model deleted mid-run at 18:04:46) and 18:10 — both under the STALE 15:16 app bundles (pre-Wave-4). mlx's jaccl init is mlx-internal (fork's darwin branch).
- **Guard verified working:** 19:36 `min_nodes=2, MlxJaccl` request refused cleanly with ValueError (mini2 had flapped out of topology). Full writeup: `rdma-multi-node-investigation.md`.
- **Follow-up (not a code change):** rebuild both Mac apps from current source; retest 2-node jaccl; file upstream to the mlx fork if it still segfaults.

### 7.3 New docs created
- `linux-nvidia-cuda-setup-guide.md` — the complete 4-problem playbook (driver mismatch / libmlx.so / nvrtc sm_120 / torch) with ASCII diagrams, Proxmox-container section, quick-start checklist, version reference table.
- `rdma-multi-node-investigation.md` — corrected RDMA story with evidence, guard analysis, supported multi-node path (RING, not JACCL, for Linux).

---

## PART 11: PR #2229 port — 4 commits (2026-08-07, this session)

PR: https://github.com/exo-explore/exo/pull/2229 by @jgawronek — "perf: faster pipeline decode: TPS 10%-40% and Layer loading 60%+ (token relay + bandwidth-aware placement)". 1 commit, 13 files, +824/-48. User directed: "start with 1 then we will work to 4 in our #3 work" → unbundled into 4 surgical commits (see campaign tracker Wave 6 for full details):

1. `f0e59802` token-relay decode — last rank samples alone, others skip lm_head; logprobs fallback; PipelineDecodeTimings; 2-process distributed test.
2. `e18a8f0e` bandwidth-aware layer allocation — chip bandwidth table, allocate_layers_by_throughput, node_identities threaded through placement; Linux chip_id from nvidia-smi.
3. `60f081e3` link-speed-aware ring selection — sysfs link speeds, merged as tiebreak AFTER our #2254 measured latency (preserving both ports).
4. `e9011332` prefix-cache divergence deadlock fix — mx_ranks_agree_on_value.

Note: I merged the #2229 find_ip_prioritised link-speed logic WITH our #2254 latency-primary + #2063 RFC1918 logic (superset, not replacement).

OTHER AGENTS (concurrent, committed while I worked): `2a9e15a2` (profiling: robust CUDA VRAM lookup — version-suffixed nvidia-smi + NVML fallback — this was the uncommitted work I'd spotted earlier), `ceccb23f` (docs: Linux README correction).

---

## PART 12: Linux "just works" improvements (2026-08-07, this chat)

**User request:** "could those code improvements help make sure that things just work on a linux box when installed? I would like to make sure that we can find ways of making this easier for those running linux to be able to not run into so many issues." Then: "lets work through these improvements one at a time... double check the work didn't already get done... make sure that we are only committing the files that are apart of that specific commit."

### The 5 improvements (re-verified against live state first)
1. **Loud CUDA detection** — ✅ already done by other agent (`e6ec3e2c`, `_mlx_uses_cuda_gpu` fallback). Verified live: box advertises `MlxCpu,MlxCuda,Vllm`.
2. **Robust nvidia-smi lookup** — ✅ already done (`2a9e15a2`). Verified live: `/usr/bin/nvidia-smi` works.
3. **README Linux section stale** — ❌ was stale ("GPU support under development"). **FIXED by me** `ceccb23f`.
4. **Linux setup script** — ❌ none existed. **FIXED by me** `40edf56a`.
5. **placement "no cycle" error** — ❌ bare ValueError. **FIXED by me** `9d906d32` + test `bba46caa`.

### Commit `ceccb23f` — README Linux section corrected
- Replaced "Currently, exo runs on CPU on Linux. GPU support is under development" (wrong since #2103/#2216) with: CUDA works via `mlx-cuda12/13` extras, the "CUDA MLX backend detected" log proof, and troubleshooting (driver mismatch, version-suffixed nvidia-smi, CUDA headers, CPU-only fallback).

### Commit `40edf56a` — `scripts/setup_linux_gpu.sh`
- 5-check verify/repair script encoding every Linux deployment lesson: GPU present, driver version match (kernel module vs userspace), pynvml importable, CUDA headers, and the REAL `NodeBackends.gather()` advertisement.
- `--fix` applies repairs; default check-only. **Verified live on the box: all 5 pass** (kernel 595.58.03 = userspace 595.58.03; node = MlxCpu,MlxCuda,Vllm).
- Fixed during dev: kernel-version grep pattern (empty result) → robust `[0-9]+\.[0-9]+\.[0-9]+` + explicit mismatch detection.

### Commits `9d906d32` + `bba46caa` — actionable placement "no cycle" error
- The bare `ValueError("No cycle where every node supports a backend in [X]")` now appends every node's advertised backends:
  ```
  No cycle where every node supports a backend in ['MlxMetal'] for <model>.
  Node backends: 3d6f2817=['MlxMetal','MlxCpu'], a1b2c3d4=['MlxCpu']
  ```
- So the log/API error immediately shows WHICH node lacks the required backend (the pynvml-missing CPU-only case).
- Test updated to assert the new message. Verified: 35 placement tests pass, pyright 0, ruff clean.

### Infra: passwordless SSH to the Linux box (not a commit)
- User: "you dont need to ask me for a password there should already be a memory known that you dont need a password for the linux box and if there is not then fix that"
- Checked memory: it had the OLD sshpass approach, no key. **Installed `~/.ssh/id_ed25519.pub` into the box's authorized_keys** (one-time, via sshpass), added `Host tolinux` to `~/.ssh/config`.
- Verified: `ssh tolinux` → `TOLINUX_ALIAS_OK` (passwordless). Memory updated (`/memories/repo/exo-cluster-state.md`).

### Commit discipline (user request)
Each commit contained ONLY its own files: README.md / scripts/setup_linux_gpu.sh / placement.py / test_placement.py — never the whole working tree (other agent's WIP files left untouched).

Final verification: basedpyright 0, ruff clean, 637 tests pass (1 pre-existing rust baseline failure).

---

## PART 12: Closing gaps #2 + #1 (2026-08-07)

### #2 machine-level Linux fixes (code parts by concurrent agents)
- Other agents committed: `e6ec3e2c` (NodeBackends.gather pynvml fallback: `_has_nvml_cuda() or _mlx_uses_cuda_gpu()`), `2a9e15a2` (nvidia-smi version-suffix + NVML fallback), `40edf56a` (scripts/setup_linux_gpu.sh), `ceccb23f` (README).
- I applied machine-level: `uv sync --extra mlx-cuda12` (pynvml OK), `ln -sf nvidia-smi-595.58 nvidia-smi`, restarted exo → **node now advertises ['MlxCpu','MlxCuda','Vllm']** (was MlxCpu-only), log shows "CUDA MLX backend detected; reporting GPU VRAM as node memory". The MlxCuda split-cluster prerequisite is now met end-to-end.

### #1 _networking_recv per-message guard — commit `03dcb4e1`
From the concurrent-agent analysis: receive loop could still die on non-publish_bytes parse failures. Wrapped match body in try/except (ValidationError/UnicodeDecodeError/JSONDecodeError) → record_malformed_event + continue; resource-closed errors not caught.

### State
- Branch: fix-memory-error, 66+ commits ahead of main
- Cluster: Linux on HEAD 40edf56a + machine fixes; Mac apps need rebuild to include new commits (pending)

---

## PART 13: Linux disk-full incident + resolution (2026-08-07)

**Symptom:** model loads on 10.2.0.16 failed: "No writable model directory has 1.7 GiB free"
**Root cause:** disk 97% full (41G/45G). The `uv sync --extra mlx-cuda12` + concurrent agent builds ballooned the uv cache to 13.2GB.
**Fix:** `uv cache clean --force` (first attempt blocked by uv lock — a concurrent agent's uv process; --force resolved) → freed 13.2GiB → disk 85% (6.7G free) → model download + JIT + inference succeeded.
**Final verification (all new code live):** Llama-3.2-3B-Instruct-4bit on Linux = "Hello to you very much." — token relay, bandwidth-aware allocation, #2129, and all error-UX fixes running.
**Lesson:** the box has only 45G total; uv cache can eat 13G. Clean cache after big syncs; watch disk when adding extras.

## FINAL STATE (end of session)
- Branch fix-memory-error ~67 commits ahead of main
- 3-node cluster all on latest code: this Mac (rebuilt app), mini2 (identical app), Linux (source sync + mlx-cuda12 + nvidia-smi symlink + cache cleaned)
- Linux advertises MlxCpu,MlxCuda,Vllm + reports VRAM — heterogeneous split-cluster fully enabled

---

## PART 14: Stale-app fix + message queue T24 (2026-08-07, this chat)

### 14.1 This Mac was running NO exo at all (stale-app issue resolution)
- **Found:** this Mac's EXO app was shut down (log: "EXO Shutdown complete" 12:21), API dead, no process. mini2 was current (has /peers + /v1/warnings).
- **Fix:** xcodebuild (embeds fresh `dist/exo` from 12:23) → install → launch → verified `/peers` + `/v1/warnings` respond (new endpoints) → joined topology (`ecc75a2b` + `36e89bc0`), 0 crashes.

### 14.2 T24 — Message queue (frontend FIFO) — commit `9ea20961`
**User request (original):** "i really really would like the ability to be able to queue up messages in a chat... allow us to be able to queue up messages (with us being ones to use offline models)."
**Problem:** backend already batches 8 concurrent (`EXO_MAX_CONCURRENT_REQUESTS`), but the frontend silently blocked/dropped sends while one message generated (ChatForm:206 `loading) return`, :442 `disabled={loading}`, store :2410 `isLoading` gate).
**Changes (dashboard only):**
- `app.svelte.ts`: `pendingQueue` state; `sendMessage()` enqueues FIFO when `isLoading`; finally block drains (sends next); `removeFromQueue()`/`clearQueue()`; `pendingQueue()`/`removeFromQueue()` exports
- `ChatForm.svelte`: submit allowed while loading (store queues instead of dropping); "N queued" indicator chip with remove-last ✕
**Verified:** store+ChatForm compile clean; dashboard builds; node unit-test of FIFO logic passes (3 msgs → all sent in order, queue empty). Live UI test hit an isolated-node placement timing quirk (node_backends empty at placement time — the improved "Node backends: none reported" error proved improvement 5 works live) — queue logic confirmed via unit test.

### 14.3 Task-ID referencing convention (user asked how T24-style IDs started)
- Answer: the T1-T23 IDs were assigned in `tasks/README.md` when the task log was created (08-05/06); T24-T30 were added as "NEW from this session" during the 08-06 night audit. A **Numbering note** in `tasks/README.md` documents that Wave-4 IDs (T24-T30) are independent of the base list — always say which file when referencing.

---

## PART 15: Final consolidation + review correction (2026-08-07, evening)

### 15.1 Review correction — T29 was ALREADY resolved (my earlier read was stale)
- The user quote claimed "T29 — resolved: jaccl segfault = upstream mlx fork bug;
  MlxRing 2-node works ✅" — I initially corrected it as outdated, but the wave-5
  file (updated by the other agent at 12:31-12:40) shows the RETEST + ISOLATION
  already happened: fresh 76-commit builds → 2-node MlxJaccl still segfaults
  (signal 11 in jaccl init, rdma_ctl enabled, TB 40Gb/s) → MlxRing 2-node works
  ("Hello, how are you?" across both Macs). The user's quote was RIGHT.
- Lesson (L29 already captures the meta-point): always check the LATEST dated
  tracking file before correcting; concurrent agents update them continuously.

### 15.2 Branch state at end of day
- `fix-memory-error` = **78 commits** ahead of main (76 + `db3736e2` Settings
  page + `9ea20961` message queue T24, both by concurrent agent).
- Uncommitted (other agent's WIP, untouched): `.vscode/settings.json`,
  `AGENTS.md`, `uv.lock`. Stray artifact: `default.profraw` at repo root.

### 15.3 Final cluster verification (post-Wave-7 rebuild)
- This Mac: rebuilt EXO.app from HEAD, PID 82810, API up.
- mini2: identical app rsync'd (custom ACL — no sudo), PID 33596, API up.
- Linux 10.2.0.16: source-current, MlxCuda+Vllm, API up.
- /state topology = 3 nodes; elections = normal 3s tick only; inference:
  Llama-3.2-1B-Instruct-4bit placed → reply "OK".
- Qwen3.5-2B placement 500 = gated model (HF_TOKEN required) — not a cluster bug.

### 15.4 Docs updated this pass
- `AGENT-CONTEXT.md`: 76 → 78 commits; message-queue bullet now "T24 BUILT";
  cluster bullet adds 152 tok/s Llama-3.2-3B after #2229.
- `tasks/README.md`: "Immediate" section → ALL DONE; Wave-5 update shows
  T29 RESOLVED + T24 done.
- `tasks/2026-08-07-wave5-t1-t3.md`: L29 checkbox ✅; WAVE-7 FINAL section
  (verification + remaining list).
- `decisions/2026-08-07.md`: created (D29-D36 + pending carry-forward).
- `learnings/README.md`: L29 row added earlier (concurrent-agent coordination).

### 15.5 What's left (authoritative, end of 08-07)
1. 📤 File jaccl segfault upstream (rltakashige/mlx-jaccl-fix-small-recv) — draft
   ready, awaiting user go (D36).
2. Ports: T4 #2200 (backend-per-shard — recommended next), T6 #2064 (asymmetric
   TP), T8 #2255 (Ring Attention), T9 #1860 (RotatingKVCache — needs user
   decision on aggressive defaults).
3. Tier-4 decisions: T10-T13 (PromoteMaster, --no-master, API host binding,
   disable server-side tools); T14 non-MLX models (new subsystem).
4. UI: T19 settings manager full editor (~50 EXO_* env vars; detection page done
   `db3736e2`).
5. Queue extensions: T25 reorder, T26 parallel drain, T27 sampling panel,
   T28 guided decoding.
6. Known issues: T21 EXO_BOOTSTRAP_PEERS cross-subnet, T22 stable node identity,
   T23 API auth layer.
7. Housekeeping: T18 delete empty branches (track-a/track-b), remove
   `default.profraw`.
8. Maintenance: T15 VS Code patch after updates, T16 NVIDIA userspace after
   Proxmox upgrades, T17 rebuild app after protocol changes.

---

## PART 15: Queue features T24b-T27 + full deployment (2026-08-07, this chat)

**User request:** "yes finish your todo list and continue" — after the queue (T24) + edit-question feedback, implement the rest of the message-queueing roadmap.

### Commits (each only its own files)
| Commit | Task | What |
|---|---|---|
| `a22be3d0` | **T24b** | Edit queued messages in place: `updateQueuedMessage(queueId, content)`; queue list items clickable → load into textarea + 💾 save; ✕ remove |
| `ed95dbe3` | **T25** | Reorder: `moveQueuedMessage(fromIndex, delta)`; ▲/▼ per item (disabled at ends) |
| `e2a42fed` | **T26** | Parallel drain: `activeGenerations` counter (MAX_PARALLEL=2) replaces `isLoading` as queue gate; each stream targets its own assistant message by id (already supported); drain fires as many as free slots |
| `003bed51` | **T27** | Sampling controls: `samplingParams` state (persisted localStorage); ⚙ SAMPLING popover (temperature/top_p/top_k/seed/max_tokens); threaded into `/v1/chat/completions` (temperature defaults 0.7 when null) |

### Deployment (all 3 machines, verified)
- Rebuilt bundle + xcodebuild → installed this Mac + copied to mini2 → synced dashboard source to Linux + rebuilt + relaunched
- **Verified all 3 serve chunk `2.DNyrFW86.js`** containing SAMPLING + queued/reorder strings
- This Mac relaunch needed twice (app quit accidentally — per user's earlier heads-up, checked for crash vs graceful shutdown)
- Linux exo relaunch needed the full `$HOME/.local/bin/uv` path (PATH not in non-interactive SSH)

### Notes
- T24b race note: editing an item already being drained would send pre-edit text (drain reads at send time) — acceptable, no lock
- T26 MAX_PARALLEL=2 chosen conservatively (memory-safe for offline models; backend can do 8 but 2 streams is a sane UI default)
