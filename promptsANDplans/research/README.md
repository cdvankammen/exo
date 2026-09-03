# Research Folder Index

> **Purpose:** Index for the `research/` folder — deep dives, PR triage, session logs. The dated pattern applies here too: session logs are the chronological record; deep-dives are topical.

## Session Logs (chronological record)
| File | Contents |
|---|---|
| `session-tracking-2026-08-06.md` | Master session audit: all work (SSH saga, Linux deploy, eject UI, Wave 4/5) + Appendix B doc index. **Parts 14-15 (08-07):** T24 message queue, final Wave-7 rebuild + cluster verification + review correction. **Parts 16-27 (08-08/09/10):** T9 KV cache, T10-T14 research, PR port campaign (16 items), Wave 5 (GLM fix, error aggregation, sidebar panel, deploy), TODO triage (#4/#5/#8/#13/#14/#18), Wave 6 (sidebar "Main Logs" rework) |
| `session-qa-log-2026-08-06.md` | Every question asked (24 Q&As) with full answers |
| `linux-deployment-log-2026-08-06.md` | Exhaustive Linux deployment: every command/error/fix |
| `linux-nvidia-cuda-setup-guide.md` | **NVIDIA/CUDA Linux playbook**: the 4 problems (driver mismatch / libmlx.so / nvrtc sm_120 / torch) with fixes, ASCII diagrams, Proxmox section, quick-start checklist, version table |
| `rdma-multi-node-investigation.md` | **RDMA multi-node investigation**: CORRECTS the ghost-edge hypothesis, documents the real Mac↔Mac jaccl segfaults, verifies the guard, supported Mac↔Linux path (RING not JACCL) |
| `eject-ui-work-log-2026-08-06.md` | Eject saga: #2238 → split-button → no-auto-delete |
| `vscode-insiders-patch-log-2026-08-06.md` | VS Code guard patch + maintenance |

## PR Research & Triage
| File | Contents |
|---|---|
| `remaining-prs/triage-report.md` | All PRs categorized (Tiers 1-4) |
| `remaining-prs/track-a-2063-2064/README.md` | #2063 (ported) + #2064 (deferred) |
| `remaining-prs/track-b-1598/README.md` | #1598 core-only port |
| `remaining-prs/track-c-tier1/README.md` | Wave-3 tier-1 ports (see note below) |
| `remaining-prs/deep-dive-2129-2200-heterogeneous.md` | #2129 + #2200 analysis, have/missing/skip |
| `memory-bandwidth-ssd-prs.md` | Original SSD-as-RAM PR survey |
| `pr-merge-campaign-tracker.md` | All waves, every ported commit with attribution |

## Topical Research
| File | Contents |
|---|---|
| `error-ux-research.md` | Error pipeline mapping (API → runner → routing → app) |
| `error-ux-implementation-research.md` | How to surface RunnerFailed details |
| `settings-manager-design.md` | ~50 EXO_* env vars, no UI |
| `build-architecture-dev-loop.md` | EXO.app assembly + rebuild steps per change type |
| `t28-guided-decoding-research.md` | **T28 constrained decoding**: approach (byte-level FSM logits processor), upstream/provider-error surface, completion |
| `t6-qwen3.6-linux-investigation.md` | **T6 #2064 + Qwen3.6-35B on Linux**: whitelist analysis, live placement/load tests, OOM root cause (single-GPU MLX), error semantics, verdict |
| `qwen3.5-30b-a3b-loading-analysis-2026-08-31.md` | **Qwen3.5/3.6-30B-A3B loading analysis**: why MoE models fail on single GPU, placement guardrails, force_override, CUDA memory limits fix, verified 35B across 2 GPUs |
| `cuda-memory-limit-fix-investigation-2026-09-01.md` | **Full investigation timeline**: ct118 debugging methodology, cluster formation fixes, CUDA memory limit implementation, performance results, systemd config, commands used |
| `blocking-issue-linux-cuda-vram-crash-2026-08-10.md` | Linux node crash on CUDA VRAM query hang — SIGTERM during subprocess.run kills EXO process |
| `fix-plan-bug3-cuda-vram-crash.md` | Fix plan for nvidia-smi hang → graceful signal handling + bounded memory monitor |
| `multi-gpu-linux-box-options-2026-08-12.md` | Multi-GPU options: Option A (multi-instance per GPU) endorsed by upstream, Option B/C deferred |
| `one-instance-per-gpu-t6-analysis.md` | Deep analysis of placement, MLX backends, memory accounting for per-GPU sharding |

## Audit Status (2026-09-03)

**All research tasks in this folder are COMPLETE.** A session-3 audit classified every file:
- 16 HISTORICAL session logs (dated, conclusions recorded)
- 34 COMPLETE topical research docs (decisions made, code committed where applicable)
- 2 execution roadmaps (`implementation-order-plan-2026-09-01.md`, `master-implementation-plan-2026-09-02.md`) — these consolidate completed research into prioritized implementation queues
- `remaining-prs/` — 8 track directories, all COMPLETE with attribution tables
- `upstream-jaccl/` — 1 draft issue, COMPLETE and ready to file

Remaining work from this corpus is **implementation only** (TODO.md P0-P4 items, diagnosed-needs-fix items, dashboard backlog). No open research questions remain.

## How to use
- Looking for "what happened on a date"? → session logs
- Looking for "what did we decide about PR X"? → remaining-prs/ + pr-merge-campaign-tracker
- Looking for "how does error X flow"? → error-ux docs
- New research → add a dated file here + link from the root README
