# Notes Log

> **Purpose:** Loose notes, observations, and context that don't fit neatly into decisions/learnings/tasks — the "miscellaneous but important" drawer.

## Historical records (dated)
- `2026-08-06.md` — Wave 4 notes (N14-N20: Qwen3.5 root cause, dual-GPU span, vision degradation, RDMA landmine, error-ux closure)

---

## Notes 2026-08-06

### N1: The cluster is 3-node now (was 2)
- This Mac + mini2 (both M4 Pro 16GB, TB4-bonded) + Linux box (2× RTX 5060 Ti)
- All 3 discover each other (verified via /state/topology)
- BUT the Linux box currently presents as MlxCpu + system RAM — so real cluster splitting isn't active yet

### N2: The "phantom node" behavior is explained
Election churn on Linux was NOT a Linux problem — it was stale EXO.app crash-looping on both Macs (missing `event_id`). Once apps were rebuilt, cluster stabilized. Lesson: check ALL nodes when one misbehaves.

### N3: What "CUDA MLX backend detected; reporting GPU VRAM as node memory" means
That log line = our #2103 + #2216 ports working on the Linux box. It's the proof the CUDA path is live.

### N4: The dashboard currently serves from inside the app bundle
`/Applications/EXO.app/Contents/Resources/exo/_internal/dashboard/` — static files served per-request. Hot-swap works for UI, but full rebuild needed for permanence.

### N5: exo has ~50 EXO_* env vars, none in UI
Settings manager is a designed-but-unbuilt feature (see `../research/settings-manager-design.md`).

### N6: The backend ALREADY queues/batches — the gap is frontend
`EXO_MAX_CONCURRENT_REQUESTS=8` (constants.py:117), runner `_work_queue`, BatchGenerator deque — the backend can run 8 concurrent generations. But the dashboard disables the input while loading (`ChatForm.svelte:206,:442`, `app.svelte.ts:2410`), so users can't queue. A FIFO queue is frontend-only work.

### N7: `pendingAutoMessage` is a single-slot auto-launch queue (proof the pattern works)
`+page.svelte:3112` — parks a message while a model launches, sends when ready. A real message queue generalizes this.

### N8: EXO_RUNTIME_DIR hot-swap (dev superpower)
`ExoProcessController.swift:398` checks `EXO_RUNTIME_DIR` env var FIRST — point the installed app at a freshly built `dist/exo` to skip xcodebuild. Dashboard override: `EXO_DASHBOARD_DIR`. See `../research/build-architecture-dev-loop.md`.

### N9: The error pipeline already produces rich data — it's lost at the edges
The backend has ErrorResponse, RunnerTerminationError (full traceback), stderr diagnostics. But: router `publish_bytes` crashes on bad events, Swift discards stderr, dashboard shows only "FAILED". See `../research/error-ux-research.md` + `error-ux-implementation-research.md`.

### N10: Message queueing for offline models = queue up prompts, walk away
Offline inference is slow (minutes per response on a 30B model) — a message queue lets users fire several prompts and collect responses later. Sequential FIFO is the right default (memory-safe). Steering via request params (temp/seed/stop) works fine offline; activation-level steering doesn't. See `../research/message-queueing-steering-research.md`.

### N6: Error surfacing work happened in ANOTHER chat session (72b9e67d)
The commits `4f850371` (keep failed instances visible), `85f1b024` (menu bar startup errors), `5acd6d6c` (error codes), `d6fcef17` (model picker order) came from a parallel agent session, driven by the same user. They're on this branch but not "ours" — tracked separately in GIT-HISTORY.

### N7: The user's workflow pattern
- User gives terse directives; expects agent to EXPAND them (do more, not minimum)
- User wants EVERYTHING tracked in promptsANDplans
- User says "just researching" when they want zero code changes
- User reverses decisions freely (eject auto-delete was added then removed in the same day)

### N8: Qwen3.5-2B-MLX-8bit failure question (08-05, pre-deployment)
Before today's deployment, the user asked why Qwen3.5-2B-MLX-8bit failed on the Linux box while loading on Macs. That was the same root cause chain documented in linux-deployment-log (mlx runtime missing / CUDA headers / driver mismatch). Today's verified inference (Llama 1B/3B) proves the path now works.

### N9: Verify with LIVE evidence, not just code
The "do we have these set up?" question proved the value: code said placement handles CUDA, but LIVE state showed MlxCpu only. Always check /state on running systems.

### N10: mini2 access
`ssh mini2` (BatchMode, key-based) works for remote ops on the other Mac.

---

## Notes from earlier sessions (mined from chat history)

### N11: force_override feature origin (08-03 session adfdd74a)
User wanted to manually override exo's "model too big" block. Research → implementation → commits 6b8d3a78/a6b99564 (pre-campaign WIP on this branch). This is why `force_override` exists in the API.

### N12: Model loading research (08-03 session 94821bd3)
"Recognize new changes to the app" — the session that surveyed app changes before the PR campaign.

### N13: Error UX origin (error-ux-research.md)
User: "I would like to know why [a model fails], and not just have the model disappear immediately after it fails" — drove the error-surfacing work.
