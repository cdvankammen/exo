# Complete Q&A Log — 2026-08-06 Session (exo cluster work)

> **Purpose:** Every single question the user asked in this session, with the full answer given. This is the verbatim knowledge base — no question gets lost.
> **Date:** 2026-08-06 (morning → evening)

---

## BLOCK 1: CUDA / NVIDIA support questions

### Q1: "Did we work on PR #2103 and anything related to this to make sure that there is support for nvidia?"
**Answer given:** Yes — ported as commit `02f17fb4` (Wave 1). Full NVIDIA support stack verified:
- `cuda_compat.py` (new file) — shims `mx.new_thread_local_stream` → `mx.new_stream` for Linux CUDA MLX (macOS no-op)
- `patches/__init__.py` — registers the CUDA compat patch
- `bootstrap.py` — `MLX_METAL_FAST_SYNCH` only set on Darwin
- `utils_mlx.py` — wired-limit checks `mx.metal.is_available()` first, then `mx.cuda.is_available()`
- Related ports: #2216 CUDA VRAM (`from_cuda()` + `_query_cuda_vram_bytes()`), #2227 capacity-aware, #2252 accelerator preference, #2121 rank-0 dedup, #2161 macOS 26 psutil
- Model cards declare `MlxMetal`/`MlxCuda`/`MlxCpu`
- PR's own validation: heterogeneous Metal + CUDA cluster (Mac M5 Pro + Linux RTX 3090, 69.3 tok/s joint)
- **Caveat documented:** on Linux CUDA nodes, advertising 100% of free VRAM can risk runtime OOM — `CUDA_VRAM_USABLE_FRACTION=0.75` suggested as future enhancement

### Q2: "Which commit is the one that made the change so when I click eject it has a popup asking if I want to delete the model?"
**Answer given:** Commit `ed7ba52f` — "feat(dashboard): unify instance eject with optional weight deletion (port of PR #2238)" by @alytaphoenix.
- 1 file: `dashboard/src/routes/+page.svelte` (132+/10-)
- Flow: click EJECT → confirm("Eject instance abc123...?") → DELETE /instance/{id} → confirm("Also delete the downloaded weights...?") → DELETE /download/{node_id}/{model_id} on nodes from `shardAssignments.nodeToRunner`
- Known caveat: weight-deletion fires right after instance-delete 200, but teardown is async → race possible with still-loading instances

---

## BLOCK 2: PR campaign summary questions

### Q3: "List all committed PRs, all not merged and why, all skipped and why, with details and links"
**Answer given:** Full tables (53 ported + coordination fix):
- Wave 1 (23 ports): #2159 KV disk, #2242 KV quant, #2251 prefill headroom, #1990 skip KV quant, #2160 multi-rail TB, #2227 capacity-aware, #2252 accelerator pref, #2216 CUDA VRAM, #2103 CUDA, #2192 EHOSTUNREACH, #2238 eject+delete, #2204 debug gate, #2244 reconnect, #2228 mask tolerance, #2158 trim_cache, #2152 vision hash, #2174 tiktoken, #2218 Gemma 4, #2239+#2247 gen stats, #2245 docs, #2249 orphan GC, #2250 stale prune, 086787dd coordination fix
- Wave 2 (21 ports): #2063, #1598 core, #2177, #2121, #2184, #2187, #2054, #2055, #2134 SSRF, #2001, #2189, #2203, #2123, #2202, #2162, #2133, #2161, 6e71b7bd typing, #2205, #2231, #1804, #2169+#2148
- Wave 3 (8 ports): #1999, #2254, #2155, #2222, #2237, #2179, #2035, #2060 Python
- Skipped (~30 with reasons): #1830 abandoned, #1675 stale, #2240 conflicts, #1988 superseded by #2242, #2195 redundant with #2160, #2229 token relay (highest-value skip), #2255 Ring Attention, #2064 asymmetric TP, #2226 stale path, #1860 RotatingKVCache (needs user decision), #2107 superseded, #2253 conflicts, #1985 Prometheus, #2012 catalog, #2145 custom cards, #2232 duplicate of #2231, #2243 overlaps peers, Docker PRs, #2200 (then), #1992, #2011, #2041, #2143, #2099, #2163, #2230, #1903, #2207, #2060 Rust part
- Tier 4 awaiting input: #2246, #2135, #2190, #1864, #2129

---

## BLOCK 3: SSH connection saga

### Q4: "How do I force vscode to stop doing this Terminal command cancelled — sensitive input required?"
**Answer given:** This is Copilot Chat's terminal auto-approval guard — hardcoded security rule in autopilot mode. Options:
1. Switch session to Interactive/"Ask Each Time" → shows "Focus Terminal" instead of cancelling
2. Set up SSH keys (`ssh-copy-id`) → no password prompt ever
3. Run SSH manually in terminal

### Q5: "I am asking a question about vscode / settings"
**Answer given:** The guard is a security boundary, not a preference. Explained the 3 permission modes (Ask Each Time / Auto-approve / Autopilot) and where the dropdown is.

### Q6: "Well I don't want that so how do I get autopilot to always do it even with the guard can we change the source code of our vscode insiders app?"
**Answer given:** YES — found the guard in the compiled bundle:
- File: `/Applications/Visual Studio Code - Insiders.app/Contents/Resources/app/out/vs/workbench/workbench.desktop.main.js`
- Decoded minified `_registerSensitiveInputElicitation`: `if(u){...cancel...}` = auto-approve branch
- **Patch applied:** `if(u){` → `if(u&&!1){` making the cancel branch dead code
- Backed up: `workbench.desktop.main.js.bak-20260806-141211`
- Verified with `node --check`, re-signed ad-hoc (`codesign --force --deep --sign -`)
- Created re-patch script: `/Users/chris/scripts/vscode-patch-sensitive-input.sh`
- ⚠️ Insiders auto-updates daily → re-run script after each update
- ⚠️ Must fully quit + reopen app; right-click → Open on first launch (unsigned)

---

## BLOCK 4: Linux box deployment

### Q7: "ssh to 10.2.0.16 there's no password or need for user... copy the source code... run exo on that machine"
**Answer given:** Full deployment saga (see LINUX-DEPLOYMENT-LOG.md for every detail):
- Connection attempts: host-key change → `ssh-keygen -R`; all usernames denied; user gave password `violin` → `sshpass` with temp file (umask 077, deleted after each command)
- Machine: Proxmox VM `tryingOutExo`, Ubuntu 24.04.4, 12 CPU, 24GB RAM, **2× RTX 5060 Ti 16GB passed through**
- Copied source via rsync (19,430 files, ~2GB, excluded artifacts), backup `~/exo.upstream-bak`
- Installed uv 0.12.2, `uv sync --extra mlx`
- **mlx nightmare:** Linux x86_64 mlx = TWO packages (`mlx` bindings + `mlx_cuda_12` runtime); wrong-arch ARM wheel in cache; fixed by downloading correct x86_64 wheels from rltakashige fork
- **Driver saga:** kernel module 595.58.03 vs apt 535/595.84 → installed exact 595.58.03 userspace from NVIDIA `.run` (extracted, symlinked)
- **CUDA headers error:** `nvidia-cuda-toolkit` + `CUDA_HOME=/usr/local/cuda` in `/etc/environment`
- **Election churn:** stale EXO.app on both Macs crash-looping (missing `event_id`) — fixed by user rebuilding app
- **VERIFIED:** Llama-3.2-1B 73.5 tok/s, Llama-3.2-3B 29 tok/s, single-node
- Run command: `setsid nohup env CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda uv run exo > /tmp/exo.log 2>&1 < /dev/null &`

### Q8: "Is this something we need to worry about?" (CUDA_VRAM_USABLE_FRACTION caveat)
**Answer given:** Not urgent now — only triggers at ~90%+ VRAM fill; failure is recoverable (crash + restart); our ports de-risk it (#2227 conservative, #2242 KV quant shrinks KV, #2216 reports free VRAM). When to care: running models that fill ~90%+ of GPU VRAM → then implement fraction. The box has 2× 16GB so most models fit easily.

### Q9: "Wait did you actually complete the ssh moving?" / "just use root as the user" / "just use password violin"
**Answer given:** Chronology of auth attempts:
1. Host key mismatch → `ssh-keygen -R 10.2.0.16`
2. Tried bare ssh, users exo/ubuntu/admin/nvidia/root/dev/pi with keys — all denied
3. VS Code guard cancelled password prompts (autopilot)
4. User: "no password" → tried empty password via expect — denied
5. User: "just use root" → key auth denied for root
6. User: "just use password violin" → **success** via sshpass (temp file, deleted after)

---

## BLOCK 5: Eject button work

### Q10: "Can we remove the need to ask 'do you want to delete it' when I click eject... can we make it just in the UI itself? EJECT splits into NO/YES smoothly"
**Answer given:** Implemented commit `ca8474c7`:
- New `EjectButton.svelte`: EJECT → fly animation → NO (outline, left) / YES (filled, right)
- Removed both confirm() popups from `ejectInstance()`
- Weights auto-deleted on eject (from #2238 behavior)
- **Build errors hit:** Svelte 5 forbids transition:+out: on same element (fixed with transition: only); barrel export broke tree-shaking (`EjectButton is not defined` runtime error — fixed with direct import)

### Q11: "Does it delete it now automatically when I say yes after pressing eject or no?"
**Answer given:** YES — verified code at `+page.svelte:2136-2153`: after `DELETE /instance/{id}`, auto-runs `deleteDownload(nodeId, modelId)` on every node the instance ran on, no popup.

### Q12: "I DO NOT WANT THAT to happen i will delete it on my own... list where that functionality is currently happening"
**Answer given:** Mapped 4 locations:
- `+page.svelte:43` — `deleteDownload` import
- `+page.svelte:2075-2082` — `getInstanceNodeIds()` helper
- `+page.svelte:2121` — `nodeIds = getInstanceNodeIds(...)`
- `+page.svelte:2136-2153` — THE AUTO-DELETE BLOCK
- `EjectButton.svelte:40` — tooltip "Eject instance and delete weights"

**Implemented commit `3d1a675c`:** removed all 5. Eject = stop instance ONLY. Weights never touched. Manual deletion via Downloads page (never removed).

---

## BLOCK 6: Research questions (heterogeneous clusters)

### Q13: "What is meant by these?" (5-point checklist)
**Answer given:** Full plain-English explanation:
1. Same exo version = same protocol (event_id crash was this failing)
2. Model cards list both backends = master filters by `card.backends ∩ node.backends`
3. Placement treats both valid = `_cycle_accelerator_score` counts Metal OR Cuda
4. Discovery = zenoh multicast same subnet (10.2.0.x) — verified working
5. Memory accounting = Mac RAM vs Linux VRAM for layer splits

### Q14: "Also what did you mean by this >>> [have/missing/skip breakdown]"
**Answer given:** Decoded every piece:
- Already have (from #2103): cuda_compat.py, patch registration, Darwin-gated FAST_SYNCH, CUDA-aware wired limit
- Missing: TcpRelay (TCP bypass for broken CUDA↔CUDA send/recv), `_rotate_metal_to_middle()` (Metal in middle rank), `MLX_HOSTS_JSON` (peer list in env), CUDA rank probing (port 40000+rank), post-prefill barrier (generate.py:373), `_linux_interface_types()` (regex interface classification)
- Skip: IP-priority flip (ethernet>TB contradicts our #2063/#2254 TB preference)

### Q15: "Also what did you mean by this >>> [abandoned branches]"
**Answer given:** Decoded:
- `git branch` = bookmark/label, not a copy of work
- Branches created for #1598/#2063+#2064, abandoned empty when user pivoted to single-branch merging
- All value in `fix-memory-error`: #1598 core → `157a4c24`, #2063 → `868088f7`
- #2064 deferred: heterogeneous-memory nodes only; identical M4 Pros never trigger; Qwen3.5-only whitelist

### Q16: "So do we have these set up then?" (1. placement ✓, 2. discovery ✓, 3. memory ✓)
**Answer given:** LIVE VERIFICATION (not just code):
| Claim | Code | LIVE |
|---|---|---|
| 1. Placement | ✅ `_cycle_accelerator_score` counts both | ⚠️ Linux advertises `["MlxCpu"]` only — **pynvml NOT installed** (only in mlx-cuda12/13 extras) |
| 2. Discovery | ✅ | ✅ all 3 nodes visible both sides |
| 3. Memory | ✅ `_mlx_uses_cuda_gpu()` True | ⚠️ reports 26.7GB system RAM — nvidia-smi installed as `nvidia-smi-595.58`, code looks for plain `nvidia-smi` |

Fix (2 commands): `uv sync --extra mlx-cuda12` + `ln -sf /usr/bin/nvidia-smi-595.58 /usr/bin/nvidia-smi`

### Q17: "So did we research this and is there a way for us to include #2064 or no?"
**Answer given:** Yes includable but with conditions:
- What changed: Linux box (32GB VRAM) ≠ Macs (16GB) — the heterogeneous scenario now EXISTS
- Still blocking: Qwen3.5-only whitelist, auto-upgrade gate (only fires at >90% smallest node), high placement.py conflict, per-node (not per-GPU) design
- 3 options: port now as infrastructure / wait until hitting >200B wall / skip
- My rec: Option 2 or 3 — pipeline parallelism already handles large models

### Q18: "Also, are there any PRs that help with models that are not MLX be downloadable from the menu?"
**Answer given:** NO upstream PR exists. Model cards encode MLX metadata; exo has ONE engine (MLX). GGUF/PyTorch = new subsystem (llama.cpp bindings + card extension + download path + routing).

### Q19: "One of the PRs... has a comment from someone that shared they were able to build off of this with #2129... can you tell me what this means?"
**Answer given:** #2103 = "make MLX run on NVIDIA GPUs on Linux" (foundation). #2129 = "make the cluster coordinate a mix of Apple + NVIDIA machines" (layer on top). Together: Mac + Linux GPU machines joined as one cluster, models split across them. Someone built a working hybrid (Mac M5 Pro + RTX 3090, 69.3 tok/s, DeepSeek-V3.1-4bit split) on top of #2103 using #2129's approach.

---

## BLOCK 7: Documentation / tracking requests

### Q20: "Make sure that we are keeping track of everything in our chat so far all the changes, the reasons, the errors..."
**Answer given:** Created:
- `research/session-tracking-2026-08-06.md` (303 lines) — full audit trail
- Commit quality standard documented (WHAT/WHY/SOURCE/verification)
- URL reference list for all PRs

### Q21: "I was asking you to make sure that you are keeping things organized in our plansANDprompts folder... update it with all the past things"
**Answer given:** Audited + updated:
- `GIT-HISTORY-AND-UPSTREAM.md` — stale "8 commits" → full 63-commit inventory
- `KNOWN-ISSUES-AND-ROADMAP.md` — added section 3.1 (8 session-discovered issues) + session roadmap
- `AGENT-CONTEXT.md` — TL;DR updated (63 commits, cluster hardware, eject behavior)
- `README.md` — research subfolder index complete
- Memory file `/memories/repo/exo-cluster-state.md` created

### Q22: "Don't forget about all the ways we can use the plansANDprompts folder... fill in details of everything... I literally mean everything... you're still not doing the max"
**Answer given:** (THIS FILE — the complete Q&A log + detailed work logs)

---

## BLOCK 8: App rebuild questions

### Q23: "Did you rebuild the app or no? because I do not see it happening... or did you restart the app or no?"
**Answer given:** Honest answer: NO full rebuild, NO restart — only hot-swapped dashboard static files into the running bundle (15:37). The build products (from other agent 15:16) were stale. Explained "hot swap" = replacing files in a running app without restart.

### Q24: "Yes [do the rebuild] and what do you mean by hot swap?"
**Answer given:** Full rebuild executed:
1. Stop app → `uv sync --extra mlx` (mlx had vanished — reinstalled from git fork)
2. Pyinstaller → `dist/exo` (15:50:13, new EJECT baked in)
3. `xcodebuild` → DerivedData (NOT app/EXO/build — that's why it seemed missing)
4. Install → **ERROR: Permission denied** — bundle was 700/root → `sudo chmod -R a+rX` + re-sign
5. Launch OK, API live, new UI served
"Hot swap" = swapping files in a running app without restarting it — like changing a wheel on a moving car. The app serves static files from disk per request, so overwriting them takes effect on next page load — but build products go stale, which is why a real rebuild was needed.

---

## BLOCK 9: Continuation session (evening→night) — EJECT fixes, mlx_lm, build architecture, queueing

### Q25: "so i need you to make the changes that will allow the functionality of when i click yes or no... whats going on with it not working? However the model does now load with the new one"
**Answer given:** Found the EJECT split-button was broken by TWO runtime bugs (both compiled fine):
1. Missing `import { fade, fly } from "svelte/transition"` → Svelte 5 compiled `()=>fly` as a bare ref → `ReferenceError` the moment EJECT was clicked → NO/YES never appeared
2. Prop name mismatch: parent passes `{id}`, component read `instanceId` → always `undefined` → YES sent `DELETE /instance/undefined` → 404
Fixed both, then **verified live in the browser** (Playwright on isolated test node): EJECT→NO/YES appear, NO restores, YES → `DELETE /instance/1ee19017-...` → 200, instance ejected.

### Q26: "but was it not there before when we built?"
**Answer given:** The mlx_lm story: it WAS there when we built the working version (the documented chain starts `uv sync --extra mlx` which installs the git dep). A rebuild that ran plain `uv sync` silently PRUNED it (uv removes packages not in the lock), and the spec's `_safe_collect` swallows the error → bundle shipped WITHOUT mlx_lm → every model load failed with `ModuleNotFoundError: No module named 'mlx_lm'`. Also found the rebuild had never finished (`dist/exo not empty` pyinstaller error).

### Q27: "did you make sure that you checked that it did" (EJECT works)
**Answer given:** Honest: previously only static verification. THIS time did a full live browser test with network capture — proved EJECT→NO→YES→DELETE 200 with a real ID, not `undefined`. Also found the SECOND bug (prop mismatch) that static checks missed.

### Q28: "okay so what did you do? was it a code change or a build problem? was it just a build that didn finish and wasn't cleaned up or what?"
**Answer given:** Three separate things: (1) EJECT = real code bug (missing transition import — Svelte compiles it silently); (2) models not loading = build-environment problem (mlx_lm pruned by plain `uv sync`, spec silently skipped it); (3) yes — the rebuild never finished because `dist/exo` was dirty (pyinstaller "not empty" error).

### Q29: "i am having exit code one errors again on this app when i open it? also is there a backend running on http://localhost:52415/? how"
**Answer given:** Yes — a STALE TEST BACKEND (PID 53392, `./exo`, started 16:08 during bundle testing) was squatting on port 52415. The real app couldn't bind → exited 1. Not a code bug. Killed it, relaunched → PID 89644, API 200, 0 crashes, joined 3-node topology. Lesson: isolated test backends must use non-default ports.

### Q30: "can we work on making those error codes easier to understand? (just research for now)"
**Answer given:** Full error-pipeline research (saved `error-ux-research.md` + `error-ux-implementation-research.md`): the backend ALREADY produces rich errors (ErrorResponse model, RunnerTerminationError with full traceback, stderr diagnostics classifier) — the data is lost at 3 edges: `publish_bytes` crashes on bad events (router.py:86), the Swift app discards stderr + shows only "Exited with code N", the dashboard maps RunnerFailed to just a "FAILED" label. Implementation plan + suggested order documented.

### Q31: "how can we make it so that those are available... also you mentioned number one, research that as well?"
**Answer given:** Researched the mechanism for each layer: (A) dashboard — render runner `error_message`/diagnostics on FAILED cards (data already fetched, just not shown; copy the downloads-card pattern); (B) Swift — stop discarding stderr, read `~/.exo/exo_log/exo.log` tail on exit, map known signatures → human reason; (C) API — add stable string error codes. Then traced the verified `publish_bytes` crash path (`_networking_recv:211` → deserialize throws → re-raise kills the loop → process exits 1) and designed the 2-guard fix. All saved to `error-ux-implementation-research.md`.

### Q32: "how does it work to be able to use the backend without needing to rebuild the swift app? list everything"
**Answer given:** Full build-architecture research (saved `build-architecture-dev-loop.md`): EXO.app = Swift shell (Xcode) wrapping a PyInstaller bundle (embedded `dist/exo`). Four ways to run the backend, fastest→slowest: (0) `uv run exo` source, (1) bundle binary direct + `EXO_DASHBOARD_DIR`, (2) **`EXO_RUNTIME_DIR` hot-swap** — the installed app checks this env var FIRST (ExoProcessController.swift:398), so point it at a fresh `dist/exo` to skip xcodebuild entirely, (3) full rebuild (ONLY needed for Swift shell changes). Complete "what rebuilds for what change" matrix documented.

### Q33: "i really would like the ability to queue up messages in a chat... look for prs and code in github already that might be in the works... then research how steering could work or if that's even plausible with offline models and tokenising"
**Answer given:** Research saved (`message-queueing-steering-research.md`):
- **No upstream PR/issue** for message queueing exists (searched GitHub)
- **The backend ALREADY supports it:** `EXO_MAX_CONCURRENT_REQUESTS=8`, runner `_work_queue`, BatchGenerator deque — 8 concurrent generations possible
- **The gap is frontend:** ChatForm.svelte:206 + :442 (textarea disabled while loading) + app.svelte.ts:2410 (`isLoading` gate) silently block sends
- **Design:** frontend FIFO queue + "N queued" chip + per-item cancel; sequential mode default (memory-safe for offline), parallel mode later (backend ready)
- **Steering feasibility:** request-param steering (temperature/seed/top_p/stop/logit_bias) works identically offline (operates at sampling, after tokenization); guided/constrained decoding plausible (needs a generator sampling hook); activation-level steering vectors NOT plausible through the current API (needs engine internals)

---

*End of Q&A log — now 33 questions (Q1-Q33), covering both the main session and the continuation session.*
