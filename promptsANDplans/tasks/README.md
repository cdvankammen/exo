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
| T4 | #2200 backend-per-shard | https://github.com/exo-explore/exo/pull/2200 | Deterministic backend selection; foundation |
| ~~T5~~ | ~~#2129 heterogeneous cluster~~ | https://github.com/exo-explore/exo/pull/2129 | ✅ **DONE 08-06 — `72cf86bb`** (TcpRelay, metal-to-middle rotation, MLX_HOSTS_JSON, prefill barrier, linux interface types) |
| T6 | #2064 asymmetric TP | https://github.com/exo-explore/exo/pull/2064 | >200B models on mixed memory (Qwen3.5 only) |
| ~~T7~~ | ~~#2229 token relay~~ | https://github.com/exo-explore/exo/pull/2229 | ✅ **DONE 08-07 — 4 commits** (`f0e59802` relay decode, `e18a8f0e` bandwidth placement, `60f081e3` link-speed ring, `e9011332` prefix deadlock) |
| T8 | #2255 Ring Attention | https://github.com/exo-explore/exo/pull/2255 | 65K+ token contexts |
| T9 | #1860 RotatingKVCache | https://github.com/exo-explore/exo/pull/1860 | Needs user decision on defaults |

### Tier 4 (awaiting user decisions)
| # | Task | PR |
|---|---|---|
| T10 | PromoteMaster command | #2246 |
| T11 | --no-master flag | #2135 |
| T12 | API host binding config | #2190 |
| T13 | Disable server-side tools | #1864 |
| T14 | Non-MLX models (GGUF/PyTorch) | No PR exists — new subsystem |

### Maintenance
| # | Task | When |
|---|---|---|
| T15 | Re-run VS Code patch script | After EVERY VS Code Insiders update (daily) |
| T16 | Re-apply NVIDIA 595.58.03 userspace | After ANY Proxmox host kernel/driver upgrade |
| T17 | Rebuild EXO.app from source | After ANY protocol change (event_id lesson) |
| T18 | Clean up empty branches (track-a, track-b) | When user confirms |

### From earlier sessions (mined from chat history)
| # | Task | Source |
|---|---|---|
| T19 | Settings manager UI for ~50 EXO_* env vars | settings-manager-design.md |
| ~~T20~~ | ~~Error surfacing hardening~~ | ✅ **DONE 08-06/07 — 6 commits** (`4f850371` failed-instance visibility, `85f1b024` Swift reasons, `5acd6d6c` error codes, `ef16c380` publish_bytes guard + warnings chip, `03dcb4e1` recv-loop guard) |
| T21 | Fix EXO_BOOTSTRAP_PEERS (cross-subnet clusters) | Known issues |
| T22 | Stable node identity (keypair persistence) | Known issues |
| T23 | Authentication layer for API | Known issues |

### NEW from this session (08-06 night, research done — decision pending)
| # | Task | Source | Effort | Status |
|---|---|---|---|---|
| T24 | **Message queue (frontend FIFO)** — stop blocking sends while loading; "N queued" chip; per-item cancel | message-queueing-steering-research.md | Small | ✅ **DONE `9ea20961`** (sendMessage enqueues when busy, drains FIFO in finally, removeFromQueue/clearQueue, "N queued" chip, ChatForm gate removed) |
| T24b | **Edit queued messages** — click a queued item to edit its content in place (before it's sent); race-guard items already being sent | user feedback 08-07 | Small | ✅ **DONE `a22be3d0`** (updateQueuedMessage; queue list: click loads text + 💾 save; ✕ remove) |
| T25 | Queue reorder/priority (drag or ↑↓) | message-queueing-steering-research.md | Small | ✅ **DONE `ed95dbe3`** (moveQueuedMessage; ▲/▼ per item, disabled at ends) |
| T26 | Parallel drain (fire N at once; backend already batches 8) | message-queueing-steering-research.md | Medium | ✅ **DONE `e2a42fed`** (activeGenerations counter; MAX_PARALLEL=2; each stream targets own assistant msg) |
| T27 | Sampling controls panel (temperature/seed/top_p/stop/logit_bias) | message-queueing-steering-research.md | Small-Med | ✅ **DONE `003bed51`** (samplingParams persisted; ⚙ SAMPLING popover: temperature/top_p/top_k/seed/max_tokens; threaded into /v1/chat/completions) |
| T28 | Guided/constrained decoding (generator sampling hook) | message-queueing-steering-research.md | Large | ⏳ pending |
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
