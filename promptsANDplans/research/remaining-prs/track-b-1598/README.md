# Track B: PR #1598 (Auto-Discover Models) — Core-Only Port

**Branch:** `fix-memory-error/track-b-model-discovery`
**Started:** 2026-08-05

## Status: CORE PORTED ✅ (commit 157a4c24)

## Why surgical port (not naive merge)
PR #1598 (Feb 2026) ballooned from a 3-file model-discovery feature into 70 commits / 33 files of UX overhaul:
- Complete onboarding rewrite (FirstLaunchPopout.swift, ContentView 419 deletions)
- DMG packaging iterations (9+ commits)
- Menu bar overhaul + Perplexity-style chat home (+2,096 lines in +page.svelte)
- Toast system, connection banner, skeleton loading
- MlxIbv→MlxJaccl renames

Naive merge would have DESTROYED 26+ features we ported since Feb (eject+delete #2238, generation stats #2239, v1/models filter #2247, peer-connect, KV disk persistence #2159, multi-rail TB #2160, CUDA VRAM #2216, native CUDA #2103, capacity-aware placement #2227, accelerator preference #2252, prefill headroom #2251, and 15 more).

## What was ported (the actual feature)
**HF Hub cache auto-discovery** — reuse models downloaded by other tools (HF CLI, transformers, LM Studio) from `~/.cache/huggingface/hub`:
- `constants.py`: HF cache auto-appended to `EXO_MODELS_READ_ONLY_DIRS` (only when dir exists)
- `download_utils.py`: `_resolve_hf_hub_model()` — resolves `models--org--name/snapshots/<commit>/` via `refs/main`, integrated into `resolve_existing_model()`
- Test adaptation: `test_empty_read_only_dirs_when_unset` allows well-known paths

## Adaptations
- PR's `EXO_MODELS_PATH` single-path system → our dual-tier `EXO_MODELS_DIRS`/`EXO_MODELS_READ_ONLY_DIRS` (HF cache appended to read-only)
- PR's `resolve_model_in_path()` → integrated into our `resolve_existing_model()`
- Swift "Model Search Paths" setting → SKIPPED (our branch already has full "Models Directories" settings section)

## Verified
- Functional test: fake HF cache layout resolves correctly (refs/main → snapshot)
- 124 passed, 1 skipped (download + shared suites)

## Not ported (deliberately)
Everything in the UX overhaul. If wanted later: layer-loading progress bars (#1598 commits ebf3cdf/86e42b8/441b12e) are the most valuable piece.
