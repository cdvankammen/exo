# Research Folder Index

> **Purpose:** Index for the `research/` folder — deep dives, PR triage, session logs. The dated pattern applies here too: session logs are the chronological record; deep-dives are topical.

## Session Logs (chronological record)
| File | Contents |
|---|---|
| `session-tracking-2026-08-06.md` | Master session audit: all work (SSH saga, Linux deploy, eject UI, Wave 4/5) + Appendix B doc index. **Parts 14-15 (08-07):** T24 message queue, final Wave-7 rebuild + cluster verification + review correction |
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

## How to use
- Looking for "what happened on a date"? → session logs
- Looking for "what did we decide about PR X"? → remaining-prs/ + pr-merge-campaign-tracker
- Looking for "how does error X flow"? → error-ux docs
- New research → add a dated file here + link from the root README
