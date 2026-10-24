# Learnings Log

> **Purpose:** Hard-won lessons from debugging, deployment, and research. Each entry = a mistake/insight + the lesson so we never repeat it.
> **Format:** Date | Lesson | Context/trigger | Prevention

## Historical records (dated)
- `2026-08-06.md` — Wave 4/5 lessons (L23-L29: Blackwell sm_120, apt clobbering, env-vs-code, lockfile drift, RDMA ghost edge → CORRECTED in L29, publish_bytes)

---

## Lessons Learned 2026-08-06

| # | Lesson | What triggered it | How to prevent |
|---|---|---|---|
| L1 | **mlx on Linux x86_64 is TWO packages** (`mlx` bindings + `mlx_cuda_12` runtime). `uv sync` installs bindings only. | `ImportError: libmlx.so` after `uv sync --extra mlx` | Always verify `import mlx.core as mx; mx.cuda.is_available()` after Linux setup; install both wheels explicitly |
| L2 | **uv cache can contain wrong-architecture wheels** (ARM aarch64 libmlx.so in x86_64 cache). `ldd` says "not a dynamic executable" — that's the tell. | Copied libmlx.so from cache → loader refused | Always `file` the .so before copying; download wheels explicitly from the release |
| L3 | **NVIDIA driver userspace MUST exactly match kernel module version.** apt 535/595.84 ≠ module 595.58.03. | `nvidia-smi: Driver/library version mismatch` | Check `/proc/driver/nvidia/version` first; install exact match from NVIDIA `.run` (extract only, no install) |
| L4 | **nvidia-smi binary name matters.** Installing as `nvidia-smi-595.58` broke `shutil.which("nvidia-smi")` → VRAM reporting silently fell back to system RAM. | Cluster state showed 26.7GB RAM instead of 32GB VRAM | Always symlink to plain `nvidia-smi`; verify `which nvidia-smi` resolves |
| L5 | **pynvml is a separate package** (`nvidia-ml-py`) only in mlx-cuda12/13 extras. Without it, node advertises MlxCpu only. | `_has_nvml_cuda()` returned False silently | `uv sync --extra mlx-cuda12` on Linux GPU boxes |
| L6 | **mlx CUDA JIT needs CUDA toolkit headers at RUNTIME.** "Can not find locations of CUDA headers" crashes at warmup, not import. | Runner loaded model then crashed at first GPU compute | Install `nvidia-cuda-toolkit` + set `CUDA_HOME` |
| L7 | **Stale app bundles break clusters silently.** Old EXO.app (missing `event_id`) crash-looped on BOTH Macs → phantom nodes → election churn on Linux. | Election churn: "Node X elected master - demoting self" every 20s | After ANY protocol change, rebuild ALL node apps from source |
| L8 | **`nohup &` dies when SSH session closes** — need `setsid` for detached servers. | exo stopped after SSH disconnect ("EXO Shutdown complete") | `setsid nohup env ... cmd > log 2>&1 < /dev/null &` |
| L9 | **Hot-swapping dashboard static files into a running app works for UI** but leaves build products stale; next rebuild wipes it. | New eject UI "worked" but not in build products | Do the full rebuild (pyinstaller → xcodebuild) for permanent changes |
| L10 | **macOS app bundle copied via `sudo cp -R` can end up 700/root** → launch "Permission denied" | App wouldn't open after install | `sudo chmod -R a+rX` after install + re-sign ad-hoc |
| L11 | **Svelte 5: barrel re-exports break tree-shaking for components** → `ReferenceError: X is not defined` at runtime (build passes!). | EjectButton not defined in served page | Direct import for components used in pages |
| L12 | **Svelte 5: `transition:` + `out:` on same element is a compile error** | Build failed | Use `transition:` only (handles both) |
| L13 | **VS Code Insiders auto-updates DAILY and reverts local patches** | Patch stopped working after update | Re-run `/Users/chris/scripts/vscode-patch-sensitive-input.sh` after updates |
| L14 | **SSH secrets should never touch shell history/process list** — temp file + umask 077 + delete after | Security best practice | `printf 'pw\n' > /tmp/.x; sshpass -f /tmp/.x ...; rm -f /tmp/.x` |
| L15 | **The election churn lesson is the same as L7** — protocol versioning matters | — | Same |
| L16 | **`basedpyright` strict mode rejects upstream PR code** written against looser configs — adaptation is normal, not failure | Every port needed type: ignore adaptations | Expect it; use project conventions |
| L17 | **pruning/GC ports (#2249, #2250) can look scary in logs** ("Manually removing node... due to inactivity") — usually WORKING AS INTENDED | Saw removal logs during election churn investigation | Check if the removed node is a phantom first |
| L18 | **rust build dirs: xcodebuild outputs to DerivedData, NOT app/EXO/build** | "I don't see it happening" | Look in `~/Library/Developer/Xcode/DerivedData/EXO-*/Build/Products/Release/` |
| L19 | **Svelte 5 compiles unimported `transition:` names to BARE refs** — build passes, runtime `ReferenceError: fly is not defined` the moment the element animates | EJECT click → NO/YES never appeared | Always `import { fade, fly } from "svelte/transition"`; grep compiled chunk for `()=>fly` |
| L20 | **A prop name mismatch compiles silently and sends `undefined`** — `<EjectButton {id}>` vs `instanceId` prop → `DELETE /instance/undefined` → 404 | YES eject gave 404 toast | Verify the ACTUAL network request in browser devtools; name props to match parents |
| L21 | **`uv sync` (no extra) PRUNES git deps like mlx_lm** → pyinstaller `_safe_collect` silently returns [] → bundle ships without mlx_lm → every model load fails with ModuleNotFoundError | Models stopped loading after rebuild | Always `uv sync --extra mlx` before building; verify `grep -c mlx_lm build/exo/Analysis-00.toc` > 0 |
| L22 | **pyinstaller refuses dirty output dir** (`dist/exo not empty`) — old build blocks new | Rebuild "didn't finish" | `rm -rf dist/exo` before `uv run pyinstaller` |
| L23 | **Stale test backends squat ports → app exit-code-1 on launch** | App opened, exited 1, mystery backend on 52415 | Always run isolated tests on non-default ports (52416) and kill after; `lsof -ti :52415` to find squatters |
| L24 | **Static JS inspection can't catch runtime prop bugs** — only a real click test with network capture does | EJECT YES 404 despite correct compiled chunk | Do end-to-end browser tests (Playwright) for UI changes, capture network requests |

---

## Lessons from earlier sessions (pre-08-06, mined from chat history)

| # | Lesson | Source session |
|---|---|---|
| L19 | force_override plumbing: "load model anyway" needed API → download coordinator → placement end-to-end + tests | adfdd74a (08-03) |
| L20 | Error UX: `RunnerFailed.error_message + diagnostics[]` existed but was discarded in dashboard; `publish_bytes` had no try/except → process crash exit 1 | error-ux-research.md |
| L21 | Settings: ~50 `EXO_*` env vars, NONE exposed in UI; no Settings page exists | settings-manager-design.md |
| L22 | Model load failures should show reason, not disappear (user request → other agent's commits 4f850371, 85f1b024, 5acd6d6c) | 72b9e67d (08-06) |
| L25 | **Driver version match must be explicit** — kernel module (595.58.03) vs userspace must match; the setup script now detects mismatch instead of silently failing | deployment saga 08-06 |
| L26 | **Bare ValueError messages lose users** — the placement "No cycle" error now lists each node's backends so the failing node is obvious | placement.py fix 9d906d32 |
| L27 | **Passwordless SSH is worth setting up** — one-time pubkey install + `Host tolinux` alias removes sshpass/secret-handling from every future command | user request 08-07 |
| L28 | **Re-verify before each task** — the other agent had already done improvements 1&2 (e6ec3e2c, 2a9e15a2); double-checking avoided redoing work | 08-07 workflow |
| L29 | **Multiple agents share one branch — coordinate like a repo admin** — always `git log`/`git status` before AND after each step; NEVER `git add .` or commit others' uncommitted files (uv.lock, AGENTS.md, .vscode/settings.json); re-check docs/commit counts before finishing because the branch moves under you (76 → 77 commits mid-session). Goal hooks after compaction can point at stale goal IDs — reload with goal_system_status using the CURRENT session id from VSCODE_TARGET_SESSION_LOG | 08-07: concurrent agent added Settings page (db3736e2) + queue-test on :52416 while we deployed | Assume drift; verify state at every boundary; treat uncommitted WIP as another agent's property |
