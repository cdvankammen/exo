# Track C: Tier-1 Wave-3 PRs

> **Created:** 2026-08-05 (during the PR triage)
> **Status:** ⚠️ **OBSOLETE — superseded by direct ports**

## What this folder was for
The original plan split Wave-3 tier-1 PR ports into a dedicated track folder (like Track A for #2063/#2064 and Track B for #1598).

## What actually happened
When the user pivoted to **single-branch merging**, all tier-1 Wave-3 ports went directly onto `fix-memory-error` instead of a dedicated branch/track. **This folder is intentionally empty** — the work it would have tracked is recorded in:

| Ported PR | Commit |
|---|---|
| #1999 (DSv4 fused MoE) | `3abc8623` |
| #2254 (link latency) | `8ccbe8c8` |
| #2155 (prefill step size) | `86927717` |
| #2222 (Claude system role) | `15647f8a` |
| #2237 (log viewer) | `b80ffbf1` |
| #2179 (use_prefix_cache) | `366c5302` |
| #2035 (liveness tuning) | `36acecdd` |
| #2060 (symlink-safe delete) | `91173df0` |

Full details: `../pr-merge-campaign-tracker.md`

## Decision
Keep this folder as a historical marker (why it exists, why it's empty) rather than deleting it — it documents the pivot from track-based to single-branch workflow.
