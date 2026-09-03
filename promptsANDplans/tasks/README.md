# Tasks Log

> **Purpose:** Every task found in this project — completed and pending — so nothing is lost. This is the master "what needs doing" list.
> **Format:** Status | Task | Source (user request / PR / research) | Where tracked

## Historical records (dated)
- `2026-08-06.md` — Wave 4 updates (T24-T30: #2129 port done, Qwen3.5 fix done, publish_bytes done, **torch fix done (f48bdf86)**, RDMA pending)
- `2026-08-07-wave5-t1-t3.md` — Wave 5-7 (T1/T2/T3 landed, **T29 jaccl = upstream fork bug + MlxRing workaround**, T24 message queue, final Wave-7 rebuild + verify)

> ⚠️ **Numbering note:** Wave-4 task IDs (T24-T30) live in the dated file and are independent of the base T1-T23 list above (which contains its own T28 "guided decoding"). When referencing, say which file: e.g. "T28 in tasks/2026-08-06.md" vs "T28 in tasks/README.md".

---

## ✅ COMPLETED TASKS

### PR Merge Campaign (2026-08-05 → 08-06)
| Task | Status | Detail |
|---|---|---|
| Port 53 upstream PRs with attribution | ✅ | 3 waves, all verified (basedpyright 0 err, ruff pass, pytest) |
| Create research docs for PR survey | ✅ | memory-bandwidth-ssd-prs.md, triage-report.md, track-a/b |

### Linux Deployment (08-06)
| Task | Status | Detail |
|---|---|---|
| SSH into 10.2.0.16 | ✅ | After host-key fix + password provided |
| Copy full source (56 commits) | ✅ | rsync, 19,430 files |
| Install uv + deps | ✅ | uv 0.12.2, `uv sync --extra mlx` |
| Get mlx working (two-package install) | ✅ | mlx + mlx_cuda_12 wheels from fork |
| Fix driver mismatch (595.58.03) | ✅ | NVIDIA .run extract, symlinks |
| Fix CUDA headers error | ✅ | nvidia-cuda-toolkit + CUDA_HOME |
| Run exo detached | ✅ | setsid + CUDA env vars |
| Verify end-to-end inference | ✅ | Llama 1B 73.5 tok/s, 3B 29 tok/s |

### Eject UI (08-06)
| Task | Status | Detail |
|---|---|---|
| Inline NO/YES split-button | ✅ | ca8474c7 |
| Remove auto-weight-delete | ✅ | 3d1a675c |
| Rebuild + install app | ✅ | pyinstaller → xcodebuild → /Applications |
| Deploy to running app | ✅ | hot-swap + full rebuild |
| **Fix Bug A: missing `svelte/transition` import** (EJECT → NO/YES never appeared) | ✅ | `import { fade, fly } from "svelte/transition"` |
| **Fix Bug B: prop name mismatch** (YES → `DELETE /instance/undefined` → 404) | ✅ | prop renamed `instanceId` → `id` |
| **Verify end-to-end in browser** (click EJECT→NO→EJECT→YES, capture network) | ✅ | Playwright on isolated test node 52416 |

### Build fixes (08-06 continuation)
| Task | Status | Detail |
|---|---|---|
| **Fix mlx_lm missing from bundle** (models wouldn't load) | ✅ | `uv sync --extra mlx` → 489 mlx_lm modules in toc |
| **Fix stale dist/exo blocking pyinstaller** | ✅ | `rm -rf dist/exo` before rebuild |
| **Fix port-conflict exit-code-1** (stale test backend on 52415) | ✅ | killed PID 53392, relaunched |
| Rebuild + redeploy to both machines (2 rounds) | ✅ | local + mini2, chunk `2.BlaL9Lg6.js` verified |

### Research (08-06 night)
| Task | Status | Detail |
|---|---|---|
| Error UX research (full pipeline audit) | ✅ | error-ux-research.md |
| Error UX implementation research (publish_bytes guard design) | ✅ | error-ux-implementation-research.md |
| Build architecture dev-loop research (EXO_RUNTIME_DIR hot-swap) | ✅ | build-architecture-dev-loop.md |
| **Message queueing + steering research** (no upstream PR; backend batches 8; frontend-only FIFO design; steering feasibility) | ✅ | message-queueing-steering-research.md |

### VS Code Insiders Patch (08-06)
| Task | Status | Detail |
|---|---|---|
| Find guard in bundle | ✅ | workbench.desktop.main.js |
| Patch auto-cancel branch | ✅ | `if(u&&!1){` |
| Re-patch script | ✅ | /Users/chris/scripts/vscode-patch-sensitive-input.sh |

### Documentation (08-06)
| Task | Status | Detail |
|---|---|---|
| Session tracking doc | ✅ | session-tracking-2026-08-06.md |
| Q&A log (24 questions) | ✅ | session-qa-log-2026-08-06.md |
| Linux deployment log | ✅ | linux-deployment-log-2026-08-06.md |
| Eject UI work log | ✅ | eject-ui-work-log-2026-08-06.md |
| VS Code patch log | ✅ | vscode-insiders-patch-log-2026-08-06.md |
| Update stale docs | ✅ | GIT-HISTORY (63 commits), KNOWN-ISSUES (§3.1), AGENT-CONTEXT, README |
| This folder structure | ✅ | decisions/, learnings/, tasks/, references/, branches/, notes/, examples/ |

---

## ⏳ PENDING TASKS (not completed)

### Immediate (08-06) — ✅ ALL DONE 08-07
| # | Task | Status |
|---|---|---|
| T1 | `cd /root/exo && uv sync --extra mlx-cuda12` | ✅ **CODE FIX LANDED `e6ec3e2c`** (fallback to `_mlx_uses_cuda_gpu()` + nvidia-ml-py in base mlx extra; fresh installs advertise correctly even without pynvml) + live deploy |
| T2 | `ln -sf /usr/bin/nvidia-smi-595.58 /usr/bin/nvidia-smi` | ✅ **CODE FIX LANDED `2a9e15a2`** (glob for `nvidia-smi*` + NVML ctypes fallback; symlink now optional) + live |
| T3 | Restart exo on Linux box | ✅ **DONE 08-07** (box = MlxCuda+Vllm, 17.1GB GPU0 VRAM instead of MlxCpu + 26.7GB RAM) |

### Port candidates (research done, decision pending)
| # | Task | PR | Value |
|---|---|---|---|
| T4 | #2200 backend-per-shard | https://github.com/exo-explore/exo/pull/2200 | ✅ **DONE `4dc6dc57`** (select + apply MLX backend per shard) |
| ~~T5~~ | ~~#2129 heterogeneous cluster~~ | https://github.com/exo-explore/exo/pull/2129 | ✅ **DONE 08-06 — `72cf86bb`** (TcpRelay, metal-to-middle rotation, MLX_HOSTS_JSON, prefill barrier, linux interface types) |
| T6 | #2064 asymmetric TP | https://github.com/exo-explore/exo/pull/2064 | ⏳ **DEFERRED 08-08** — research complete (`research/t6-qwen3.6-linux-investigation.md`): Qwen3.6-35B-A3B fits **single-node on Linux** (19.5GB < 29.2GB VRAM) so asymmetric TP is NOT needed for that goal; PR is strictly Qwen3.5-whitelisted, 2-node-only, Metal-tested (zero backend checks → Metal↔CUDA untested). Real value = >200B models on mixed-memory pairs (not our cluster). Revisit if a >32GB model is needed |
| ~~T7~~ | ~~#2229 token relay~~ | https://github.com/exo-explore/exo/pull/2229 | ✅ **DONE 08-07 — 4 commits** (`f0e59802` relay decode, `e18a8f0e` bandwidth placement, `60f081e3` link-speed ring, `e9011332` prefix deadlock) |
| T8 | #2255 Ring Attention | https://github.com/exo-explore/exo/pull/2255 | ✅ **DONE 08-07 — 4 commits** (`ddf010d3` types, `25cc7973` engine+generator, `9d991fc8` placement/API/profiling, `6670ad2d` dashboard). 65K+ token contexts via sequence-parallel prefill. Full gate: 754 pass, 1 known rust baseline. |
| T9 | #1860 RotatingKVCache | https://github.com/exo-explore/exo/pull/1860 | ✅ **DONE 08-08 — `2cd77a50`** (wired RotatingKVCache — bounded context instead of OOM) |

### Tier 4 (awaiting user decisions)
| # | Task | PR |
|---|---|---|
| T10 | PromoteMaster command | #2246 | ⏳ **DEFERRED 08-08** — researched (388 lines, 9 files: API + commands + election.py `_force_promote` + 205 test lines). Cluster self-heals without manual promotion; election.py is safety-critical and was just modified by T11. Risk > value — revisit if manual master control is ever needed |
| T11 | --no-master flag | #2135 | ✅ **DONE 08-08 — `0a90992e`** (worker-only nodes never self-elect; `is_candidate` in Election + re-propose-last-known-master guard; run-loop refusal warning; 2 tests) |
| T12 | API host binding config | #2190 | ✅ **DONE 08-08 — `8eb9d1db`** (minimal: `EXO_API_HOST` env, default 0.0.0.0; PR's full config system not needed) |
| T13 | Disable server-side tools | #1864 | ✅ **DONE 08-08 — `64650695`** (minimal: `EXO_ENABLE_SERVERSIDE_TOOLCALLS` env, default ON; gates tools in both chat_completions + claude adapters; PR's Claude-only schema-filter replaced with clean env switch) |
| T14 | Non-MLX models (GGUF/PyTorch) | No PR exists — new subsystem |

### Maintenance
| # | Task | When |
|---|---|---|
| T15 | Re-run VS Code patch script | After EVERY VS Code Insiders update (daily) |
| T16 | Re-apply NVIDIA 595.58.03 userspace | After ANY Proxmox host kernel/driver upgrade |
| T17 | Rebuild EXO.app from source | After ANY protocol change (event_id lesson) |
| T18 | Clean up empty branches (track-a, track-b) | ✅ **DONE 08-07** (only fix-memory-error + main remain; no track branches) |

### From earlier sessions (mined from chat history)
| # | Task | Source |
|---|---|---|
| T19 | Settings manager UI for ~50 EXO_* env vars | ✅ **DONE 08-07** (Swift SettingsView custom-env-var editor + dashboard Settings page read-only reference `db3736e2`) |
| ~~T20~~ | ~~Error surfacing hardening~~ | ✅ **DONE 08-06/07 — 6 commits** (`4f850371` failed-instance visibility, `85f1b024` Swift reasons, `5acd6d6c` error codes, `ef16c380` publish_bytes guard + warnings chip, `03dcb4e1` recv-loop guard) |
| T21 | Fix EXO_BOOTSTRAP_PEERS (cross-subnet clusters) | ✅ **DONE `207f1bcd`** (removed 'temporarily removed' raise; Node.create dials host[:port] peers via connect_peer; verified: unreachable→clean error, mini2→connected) |
| T22 | Stable node identity (keypair persistence) | ✅ **DONE `d2120cb9`** (persist stable node ID across restarts) |
| T23 | Authentication layer for API | ✅ **DONE `32dce75a`** (optional `EXO_API_TOKEN` bearer auth — middleware in `src/exo/api/auth.py`; dashboard static exempt; 5 unit tests + live backend 401/200 verified) |

### NEW from this session (08-06 night, research done — decision pending)
| # | Task | Source | Effort | Status |
|---|---|---|---|---|
| T24 | **Message queue (frontend FIFO)** — stop blocking sends while loading; "N queued" chip; per-item cancel | message-queueing-steering-research.md | Small | ✅ **DONE `9ea20961`** (sendMessage enqueues when busy, drains FIFO in finally, removeFromQueue/clearQueue, "N queued" chip, ChatForm gate removed) |
| T24b | **Edit queued messages** — click a queued item to edit its content in place (before it's sent); race-guard items already being sent | user feedback 08-07 | Small | ✅ **DONE `a22be3d0`** (updateQueuedMessage; queue list: click loads text + 💾 save; ✕ remove) |
| T25 | Queue reorder/priority (drag or ↑↓) | message-queueing-steering-research.md | Small | ✅ **DONE `ed95dbe3`** (moveQueuedMessage; ▲/▼ per item, disabled at ends) |
| T26 | Parallel drain (fire N at once; backend already batches 8) | message-queueing-steering-research.md | Medium | ✅ **DONE `e2a42fed`** (activeGenerations counter; MAX_PARALLEL=2; each stream targets own assistant msg) |
| T27 | Sampling controls panel (temperature/seed/top_p/stop/logit_bias) | message-queueing-steering-research.md | Small-Med | ✅ **DONE `003bed51`** (samplingParams persisted; ⚙ SAMPLING popover: temperature/top_p/top_k/seed/max_tokens; threaded into /v1/chat/completions) |
| T28 | Guided/constrained decoding (generator sampling hook) | message-queueing-steering-research.md | Large | ✅ **DONE 08-08** — 2 commits (`7a5af0bc` wire end-to-end, `7d26a590` tests + ByteLevel-decoder discriminator fix). JSON-Schema → byte-FSM logits mask (`constrained_decoding.py`), `response_format` plumbing, 400 on unsupported schemas, 27 tests. Full gate: 798 pass. **08-09 live-hardening — 3 more commits found via live Linux-box testing:** `f6ae8fc8` (vocab-mismatch: mask resized to logits shape instead of failing open — Qwen3.5-2B's output head differs from tokenizer vocab), `862c0148` (OpenAI `json_object` without schema now compiles as bare object instead of string FSM), `07412547` (prompt-boundary: FSM only walks generated tokens, not prompt text — mlx_lm passes prompt context on the first processor call). Live-verified on 3-node cluster: `json_object` → `{}` (valid JSON), `json_schema` → `{"` schema-valid prefix (1B/2B models EOS early; constraint itself proven). 43 constrained-decoding tests pass, pyright 0, ruff clean. All 3 nodes (this Mac, mini2, Linux) redeployed with the fixes. |
| T29 | Dashboard: render runner error_message + diagnostics on FAILED cards | error-ux-implementation-research.md | Small | ✅ done (other agent 4f850371) |
| T30 | Swift app: read log tail + map signatures → human exit reason | error-ux-implementation-research.md | Medium | ✅ done (other agent 85f1b024) |

### Wave-5 update (2026-08-07)
- T1 ✅ `e6ec3e2c`, T2 ✅ `2a9e15a2`, T3 ✅ (live: box = MlxCuda+Vllm, 17.1GB VRAM)
- T29 ✅ **RESOLVED 08-07** — 2-node `MlxJaccl` segfaults confirmed on FRESH 76-commit builds (signal 11 in `mx.distributed.init(backend="jaccl")`, rdma_ctl enabled + TB 40Gb/s up) = **upstream mlx darwin fork bug** (rltakashige/mlx-jaccl-fix-small-recv), NOT our code. Workaround `MlxRing` 2-node WORKS ("Hello, how are you?" across both Macs). Action: file upstream issue. Full detail: `tasks/2026-08-07-wave5-t1-t3.md`
- T24 ✅ `9ea20961` (message queue FIFO — concurrent agent landed 08-07 evening)
- ⚠️ Concurrent agent: `f0e59802` = PR #2229 token-relay part 1/4; uncommitted placement/api/system_info were their WIP (now committed); never `git add .`

### Linux "just works" improvements (2026-08-07, this chat) — all ✅
| # | Improvement | Commit | Verified |
|---|---|---|---|
| 1 | Loud CUDA detection (`_mlx_uses_cuda_gpu` fallback so CUDA nodes never advertise CPU-only) | `e6ec3e2c` (other agent) | live: box = MlxCuda+Vllm |
| 2 | Robust `nvidia-smi` lookup (version-suffixed + NVML fallback) | `2a9e15a2` (other agent) | live |
| 3 | README Linux section corrected (CUDA works; troubleshooting) | `ceccb23f` | grep no stale text |
| 4 | `scripts/setup_linux_gpu.sh` — 5-check verify/repair script | `40edf56a` | live: all 5 pass on box |
| 5 | Placement "no cycle" error lists each node's backends | `9d906d32` + test `bba46caa` | 35 placement tests pass |
| 6 | **Passwordless SSH** to Linux box (`ssh tolinux` alias, key installed) | infra (not a commit) | `TOLINUX_ALIAS_OK` verified |

### T44/T45 (2026-08-07, this chat) — dev mode + log auto-scroll
| # | Task | Status |
|---|---|---|
| T44 | **Dev mode flag** — reveal advanced features (Add Node panel, Integrations nav, advanced settings) via a persisted toggle | ✅ **DONE `62c5a8b0`** (devMode in store + ChatSidebar toggle + HeaderNav Integrations gating + Add Node gating; verified live in browser) |
| T45 | **Log auto-scroll** — start at bottom, stay pinned while at bottom, pause when scrolled up | ✅ **DONE `c5e0216c`** (logViewerEl + onscroll + stickToBottom; verified live: scroll-down sticks, scroll-up pauses) |

### T46 (2026-08-07, this chat) — Settings nav fix
| # | Task | Status |
|---|---|---|
| T46 | **Settings page dead-end fix** — add HeaderNav so users can navigate back without browser back | ✅ **DONE `4a15c431`** (HeaderNav showHome; verified live in IDE browser: Settings → Home works) |

### T9 (2026-08-07, this chat) — RotatingKVCache wiring
| # | Task | Status |
|---|---|---|
| T9 | #1860 RotatingKVCache — bounded KV context instead of OOM | ✅ **DONE `2cd77a50`** (wired MAX_KV_SIZE=16384 default + EXO_MAX_KV_SIZE/EXO_KEEP_KV_SIZE env overrides into generate.py + batch_generate.py make_kv_cache; RotatingKVCache evicts oldest tokens) |

### T8 (2026-08-08) — Ring Attention #2255 — ✅ VERIFIED DONE (other agent)
| # | Task | Status |
|---|---|---|
| T8 | #2255 Ring Attention (sequence-parallel) | ✅ **DONE** 4 parts: `ddf010d3` (sharding types), `25cc7973` (engine+generator), `9d991fc8` (placement+API+profiling), `6670ad2d` (dashboard UI) |

### Track Research D/E/F (2026-08-08) — remaining-PR merge campaign
| Track | Scope | Result | Doc |
|---|---|---|---|
| D | Tier 2 (22 PRs) | **4 PORT NOW** (#2243, #2143, #2028, #1813), 3 already ported, 1 covered | `research/remaining-prs/track-d-tier2/README.md` |
| E | Tier 3 (~70 PRs) | **7 NEWLY VALUABLE** (#2235, #2013, #2244, #2232, #2128, #2152, #2015), ~40 already ported | `research/remaining-prs/track-e-tier3/README.md` |
| F | Tier 4 (5 PRs) | **1 DONE** (#2129), **4 DESIGNED NOT PORTED** (#2246, #2135, #2190, #1864) | `research/remaining-prs/track-f-tier4/README.md` |

### T10-T14 Research (2026-08-08) — deep-dive docs written
| # | Task | Status |
|---|---|---|
| T10 | #2246 PromoteMaster | **RESEARCHED** → `research/decision-support-t10-promote-master.md` — PORT NOW (opt-in) |
| T11 | #2135 --no-master | **RESEARCHED** → `research/decision-support-t11-no-master.md` — PORT NOW (Linux box permanent worker) |
| T12 | #2190 API host binding | **RESEARCHED** → `research/decision-support-t12-api-host-binding.md` — PORT NOW (env var) |
| T13 | #1864 disable tools | **RESEARCHED** → `research/decision-support-t13-disable-tools.md` — PORT NOW (3 hunks; default direction = user call) |
| T14 | non-MLX models | **RESEARCHED** → `research/decision-support-t14-non-mlx-vllm.md` — DEFER → vLLM first |
| — | All remaining | **RESEARCHED** → `research/decision-support-remaining-items.md` — full queue + execution order |

### T10 (2026-08-08, this chat) — PromoteMaster command ✅ PORTED
| # | Task | Status |
|---|---|---|
| T10 | #2246 PromoteMaster | ✅ **DONE `876fdbc6`** — POST /master/promote/{node_id} (idle-only 409/404), FORCE_MASTER_SENIORITY, _force_promote (max-observed+1), 5 tests. T11 was already ported (`0a90992e`) → **Batch 1 (election pair) complete** |

### Port Campaign Completion (2026-08-08) — ALL decision-queue items ported
| PR | Feature | Commit |
|---|---|---|
| #2246 | PromoteMaster (T10) | `876fdbc6` (this chat) |
| #2135 | --no-master (T11) | `0a90992e` |
| #2190 | EXO_API_HOST (T12) | `8eb9d1db` |
| #1864 | EXO_ENABLE_SERVERSIDE_TOOLCALLS (T13) | `64650695` |
| #2143 | burst balance | `4bbb930a` |
| #2028 | backoff reconcile | `b8c5620e` |
| #2235 | non-200 metadata | `20e9433e` |
| #2243 | EXO_ZENOH_CONNECT unicast | `d300410f` (this chat) |
| #2253 | manual layer allocation | `516ad8dd` (this chat) |
| #2128 | drop bad gossip | `ab065557` (this chat) |
| #2231 | harden events + debounce | `143d70c2` |
| #2244 | runner reconnect cascade | (other agent) |
| #2152 | vision cache hash | `c1075b87` |
| #2015 | WarnExtraModel | (other agent) |
| #1813 | TB4/TB5 | (other agent) |
| #2013 | HF retry | (other agent) |

### Deployment (2026-08-08) — 0.3.70 shipped to all 3 machines ✅
| Machine | Version | Build commit | Status |
|---|---|---|---|
| This Mac | 0.3.70 | `648903c9` | ✅ installed + running (PID 81858) |
| mini2 | 0.3.70 | `648903c9` | ✅ installed + running (PID 56491) — no sudo (admin-writable /Applications) |
| Linux box (tolinux) | source sync | `648903c9` md5 match | ✅ running as master (PID 262579) |

Build chain: dashboard npm build → `pyinstaller -y` → xcodebuild (0.3.70) → deploy.
Full suite: **838 passed** (after fixing stale exo_rs test `648903c9`).
Subagent audit: **16/16 decision-queue ports PRESENT**, 128 targeted tests pass.

### Wave 5 (2026-08-09/10) — GLM fix, error aggregation, sidebar panel, deploy
| # | Task | Status |
|---|---|---|
| W5-1 | mini2 stale version | ✅ **FIXED** — re-deployed 0.3.70 (md5 75d1ae16), restarted |
| W5-2 | GLM 4.7 Flash "member of instance" | ✅ **FIXED `679945b0`** — file_meta followed HTTP 302 redirects (GLM repo 302s on metadata fetch → download failed → confusing load error) |
| W5-3 | Error filter gap | ✅ **FIXED `bcbbd51b`** — /v1/logs/errors now cluster-aware (merges remote nodes' APIs via NodeIdentity api info) |
| W5-4 | Logs page state persistence | ✅ **DONE `3a13f5db`** — section/log/filter/scroll via sessionStorage |
| W5-5 | Sidebar logs/errors panel | ✅ **DONE** — Phase 1 `5969494c` (static), Phase 2 `6a80e918` (collapsible), Phase 3 `0de820e8` (resizable), fix `bc660a3e` (unwrap response) — browser-verified |
| W5-6 | Server cleanup | ✅ **DONE** — killed navtest/t28test instances, kept EXO.app + Hermes |
| W5-7 | Deploy all 3 nodes | ✅ **DONE** — 0.3.70 on this Mac (21955), mini2 (25176), Linux (master, rebuilt exo_rs) |

### TODO.md Triage (2026-08-10) — remaining items resolved/verified
| # | Item | Status |
|---|---|---|
| #4 | Network latency visible | ✅ **DONE `234c322a`+`dc0f3041`** — topology links show live latency+bandwidth (browser-verified "27ms 2.3MB/s") |
| #5 | Per-link bandwidth | ✅ **DONE** (same commit — bandwidth_mbps on links) |
| #8 | Offline model detection | ✅ **DONE** (download_utils.py:429) |
| #13 | Memory pressure | ✅ **DONE** (profiling.py pressure) |
| #14 | Connection type in UI | ✅ **DONE** (interface_type + link_speed) |
| #18 | Network type in tests | ✅ **DONE** (cluster marker thunderbolt param) |
| #16 | Dynamic connection priority | ⏳ **DEFERRED** — needs InstanceReplacedAtomically revival + placement logic (mini-campaign) |
| #17 | Stream models from devices | ⏳ **DEFERRED** — overlaps P2P #1992 (26-file huge port) |

### Wave 6 (2026-08-10) — Sidebar "Main Logs" panel interaction rework
| # | Task | Status |
|---|---|---|
| W6-1 | Header renamed "Main Logs" | ✅ **DONE `f0250d79`** |
| W6-2 | Click header collapses/expands (restores same height) | ✅ **DONE** — browser-verified (collapse → expand restores 386px) |
| W6-3 | Click-and-hold drag header to resize | ✅ **DONE** — window-bound move/up (drag 266→386px, persists) |
| W6-4 | Chat area scrollable when panel expands | ✅ **DONE** — flex-1 overflow-y-auto |

### Wave 7 (2026-09-02/03) — Log rotation, smox diagnosis, download queue stall, network traffic
| # | Task | Status |
|---|---|---|
| T49 | **P0 #34 Runner log rotation** — runner stdout/stderr log rotation to prevent disk-full crashes (Linux log grew to 14GB) | ✅ **DONE `57eab048`** — rotation for runner subprocess logs; complements main exo.log rotation (`009b43c6`) |
| T50 | **Smox node isolation investigation** — smox (10.2.0.76, exo-amd) visible in zenoh discovery but HTTP API unreachable | 🔍 **DIAGNOSED** — API server down on smox container; zenoh gossipsub discovers the node but FastAPI on port 52415 is not running. Needs container restart + code update (still on old `main` at PR #2245) |
| T51 | **Qwen3-30B-A3B download stall investigation** — model download hangs indefinitely despite cluster having capacity | 🔍 **DIAGNOSED** — download queue saturated + bad model card entry blocking the entire queue. Single malformed card stalls all pending downloads (see L54) |
| T52 | **Network traffic display in dashboard** — show live network traffic (bytes in/out per node) on topology or node cards | ✅ **DONE** — backend `67020f5f` + UI `ea766091` — per-node rx/tx in placement panel + topology |

### Session 3 audit (2026-09-03) — TODO.md P0 items found already implemented, new tasks added
| # | Task | Status |
|---|---|---|
| T53 | **Download queue resilience** — per-download timeout + dead-letter handling + "stalled" dashboard warning (prevents single bad card from blocking entire queue) | ⏳ **NEW — needs implementation** (TODO #63 partial mitigation done via `9945231c` card removal; general resilience not yet implemented) |
| T54 | **Wire EXO_ZENOH_NAMESPACE to --namespace CLI** — env var is cosmetic (logged but not wired to argparse default at `main.py:593`); add `default=os.getenv('EXO_ZENOH_NAMESPACE', __version__)` | ⏳ **NEW — needs implementation** (small, env var cosmetic today) |

### Deferred tasks (with evidence)
| # | Task | Evidence |
|---|---|---|
| T6 | #2064 asymmetric TP — DEFERRED | `research/t6-qwen3.6-linux-investigation.md`: Qwen3.6-35B fits single-node Linux (19.5GB < 29.2GB VRAM); real value = >200B models only. Revisit when needed. |
| T14 | Non-MLX models (GGUF/PyTorch) — DEFERRED | `research/decision-support-t14-non-mlx-vllm.md`: needs vLLM integration first. No upstream PR exists. |
| T16 | Dynamic connection priority (#16) — DEFERRED | Requires InstanceReplacedAtomically revival + placement logic rewrite. Mini-campaign scope. |
| T17 | Stream models from devices (#17) — DEFERRED | Overlaps P2P #1992 (26-file huge port). Needs dedicated effort. |
| T50 | Smox node isolation — DEFERRED | Container infra issue: API server not running on exo-amd. Needs container restart + code sync. Not a code bug in this branch. |
| T51 | Download queue stall — PARTIALLY MITIGATED | Bad card removed (`9945231c`); general per-download timeout still needed (see T53). |
| T64 | Election cycling — DEFERRED | Architectural: download/placement state in master in-memory; persistent state needed. P5 roadmap item. |
