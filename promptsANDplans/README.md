# promptsANDplans

This folder is the designated home for **everything that is not code** — supplementary materials that support the project without cluttering the codebase.

## What belongs here

- 📝 **Research docs** — findings, experiments, competitor analysis, deep dives
- 🗺️ **Planning documents** — proposals, roadmaps, architecture notes, design docs
- 📋 **Meeting notes** — agendas, minutes, action items
- 💬 **Prompts** — saved prompt templates, agent instructions, conversation artifacts
- 📎 **Any other non-code material** that needs a permanent home

## Folder Structure (organized by category)

| Folder | Purpose | Start here |
|---|---|---|
| `decisions/` | Every decision + reasoning + alternatives | `decisions/README.md` (D1-D18 + pending P1-P11) + `decisions/2026-08-06.md` (D19-D27) + `decisions/2026-08-07.md` (D29-D36) |
| `learnings/` | Hard-won lessons, prevention steps | `learnings/README.md` (L1-L22) + `learnings/2026-08-06.md` (L23-L28) |
| `tasks/` | All tasks: completed ✅ + pending ⏳ | `tasks/README.md` (T1-T23) + `tasks/2026-08-06.md` (T24-T30) |
| `references/` | PR URLs, code libraries, offline info, commands, source map | `references/README.md` + `references/2026-08-06.md` |
| `branches/` | Every branch, what's on it, status | `branches/README.md` + `branches/2026-08-06.md` |
| `notes/` | Loose observations that don't fit elsewhere | `notes/README.md` (N1-N13) + `notes/2026-08-06.md` (N14-N20) |
| `examples/` | Proven working examples (commands, API calls, patterns) | `examples/README.md` (E1-E10) + `examples/2026-08-06.md` (E11-E14) |
| `prompts/` | Saved prompt templates + agent instructions | `prompts/README.md` (P1-P5) + `prompts/2026-08-06.md` (P6-P8) |
| `research/` | Deep dives, PR triage, session logs | `research/` (see table below) |

> **Pattern:** each category folder has a `README.md` (index + what belongs) AND dated `YYYY-MM-DD.md` history files (the actual chronological record). When something happens, file it in the matching folder's dated history immediately.

## Rules of thumb

- **Code goes in the repo; everything else goes here.**
- Each category folder has its own `README.md` explaining what goes in it.
- If a document starts explaining *why* a decision was made rather than *how* the code works, it likely belongs here.
- When something happens (decision/lesson/task/reference), file it in the matching folder IMMEDIATELY — don't wait for "cleanup time."

---

## Project Documentation Index

Full documentation of the exo codebase (generated 2026-08-03, updated 2026-08-06):

| File | Contents |
|---|---|
| `PROJECT-OVERVIEW.md` | **Start here.** Purpose, main features, how it works, quick start |
| `ARCHITECTURE.md` | Deep dive: every module, class, function, message flow |
| `FILE-INVENTORY.md` | **Exhaustive file-by-file map** of the entire repository (every file, class, function) + **Appendix A**: machine-generated complete symbol index of all 1,245 public Python definitions |
| `GIT-HISTORY-AND-UPSTREAM.md` | **Branch work & upstream sources**: `fix-memory-error` 78 commits (08-07), all ported PRs with GitHub URLs, dependency URLs |
| `API-AND-USAGE.md` | REST API reference, dashboard, CLI, code examples with expected output |
| `DEPENDENCIES.md` | Full dependency inventory (Python, Rust, JS, Nix, tooling) |
| `KNOWN-ISSUES-AND-ROADMAP.md` | Known issues (incl. §3.1 session-discovered), platform limits, future work, testing status |
| `AGENT-CONTEXT.md` | **Agent memory**: fast-loading context for AI agents working in this repo |

**Suggested reading order for new agents:** `AGENT-CONTEXT.md` (30-second TL;DR) → `PROJECT-OVERVIEW.md` → `ARCHITECTURE.md` → `FILE-INVENTORY.md` → `GIT-HISTORY-AND-UPSTREAM.md` → the rest as needed.

## Research Subfolder

| File | Contents |
|---|---|
| `research/memory-bandwidth-ssd-prs.md` | Survey of exo PRs on memory bandwidth & SSD-as-RAM (KV cache offload, multi-rail RDMA, bandwidth-aware placement) — active research, 2026-08-05 |
| `research/pr-merge-campaign-tracker.md` | **Full PR merge campaign tracker**: all waves (1-3), every ported commit with attribution, verification gates, session addendum 2026-08-06 (Linux deploy, eject UI, branch cleanup) |
| `research/session-tracking-2026-08-06.md` | **Master session audit 2026-08-06**: overview of ALL work — SSH saga, Linux deploy, mlx install, driver fix, eject UI, app rebuild, deep research, docs index (now through PART 11: EJECT runtime fixes, mlx_lm bundle fix, port conflict, new research) |
| `research/session-qa-log-2026-08-06.md` | **COMPLETE Q&A LOG**: every question the user asked this session (33 questions) with the full answer given — the verbatim knowledge base |
| `research/linux-deployment-log-2026-08-06.md` | **EXHAUSTIVE Linux deployment log**: every command, error, and fix for 10.2.0.16 — SSH saga, rsync, mlx two-package nightmare, driver 595.58.03, CUDA headers, election churn, verified inference, known gaps, command reference |
| `research/eject-ui-work-log-2026-08-06.md` | **Eject UI saga**: original #2238 port → split-button redesign (ca8474c7) → no-auto-delete reversal (3d1a675c) — every error (Svelte transitions, tree-shaking), code locations, deployment |
| `research/vscode-insiders-patch-log-2026-08-06.md` | **VS Code Insiders patch**: the sensitive-input guard, the minified-bundle patch (`if(u&&!1)`), re-signing, the re-patch script, daily-update maintenance |
| `research/remaining-prs/triage-report.md` | All remaining upstream PRs categorized (Tiers 1-4): what was ported, skipped, deferred, needs-user-input |
| `research/remaining-prs/track-a-2063-2064/README.md` | Track A: #2063 (ported, 5 features) + #2064 asymmetric TP (deferred, full analysis) |
| `research/error-ux-research.md` | **Error pipeline audit**: rich data exists (ErrorResponse, RunnerTerminationError, diagnostics) vs where it's lost (publish_bytes crash, Swift stderr discard, dashboard FAILED label) |
| `research/error-ux-implementation-research.md` | **Error UX implementation plan**: publish_bytes 2-guard design (verified crash path), dashboard runner-error rendering, stable API error codes, Swift exit reason |
| `research/build-architecture-dev-loop.md` | **Build architecture**: how EXO.app is assembled (Xcode shell + PyInstaller bundle), `EXO_RUNTIME_DIR` hot-swap, `EXO_DASHBOARD_DIR` override, the full "what rebuilds for what change" matrix |
| `research/message-queueing-steering-research.md` | **Message queueing + steering**: no upstream PR; backend ALREADY batches 8 concurrent; frontend blocks sends (3 gate sites); FIFO queue design for offline users; steering feasibility (request-param steering works offline, activation steering doesn't) |
| `research/remaining-prs/track-b-1598/README.md` | Track B: #1598 core-only port (what was ported vs the UX overhaul deliberately skipped) |
| `research/remaining-prs/deep-dive-2129-2200-heterogeneous.md` | Deep research: #2129 (3-node Metal+CUDA cluster: TcpRelay, rotation, probing, barrier) + #2200 (backend per shard) — have/missing/skip matrix, recommended port order |
| `research/error-ux-research.md` | Error UX research: full error pipeline (API → runner → routing → app shell), what exists, what's lost |
| `research/error-ux-implementation-research.md` | Error UX implementation plan: how to surface RunnerFailed error_message + diagnostics |
| `research/settings-manager-design.md` | Settings manager design: ~50 EXO_* env vars, none in UI, no Settings page exists |
| `research/build-architecture-dev-loop.md` | (other agent) EXO.app assembly explained: Swift shell + PyInstaller backend, which rebuild steps each change type requires |
| `research/linux-nvidia-cuda-setup-guide.md` | **NVIDIA/CUDA Linux playbook (2026-08-06)**: the 4 problems (driver mismatch / libmlx.so / nvrtc sm_120 / torch) with root causes, step-by-step fixes, ASCII diagrams, Proxmox-container section, quick-start checklist, exact version table — reuse for ANY future Linux/NVIDIA box |
| `research/rdma-multi-node-investigation.md` | **RDMA "ghost edge" investigation**: CORRECTS the earlier Mac↔Linux hypothesis (structurally impossible — both endpoints must be macOS TB-reporting), documents the real Mac↔Mac jaccl segfaults, verifies the placement guard works, and gives the supported Mac↔Linux multi-node path (RING not JACCL) |
