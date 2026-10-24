# Decisions Log

> **Purpose:** Every decision made in this project, with the reasoning. When someone asks "why did we do X?", the answer lives here.
> **Format per entry:** Date | Decision | Alternatives considered | Why chosen | Source (user/agent/PR)

## Historical records (dated)
- `2026-08-06.md` — Wave 4/5 decisions (D19-D27: #2129 port, CUDA 12.8 fix, publish_bytes guard, **torch fix landed (f48bdf86)**, RDMA corrected: no ghost edge exists — retest Mac↔Mac jaccl with fresh builds, see rdma-multi-node-investigation.md, README fix, setup script, no-cycle error, passwordless SSH)
- `2026-08-07.md` — Wave 5-7 decisions (D29-D36: T1/T2/T3 landed, **T29 jaccl = upstream darwin fork bug (MlxRing workaround)**, T24 message queue, final Wave-7 rebuild + cluster verification, don't-file-upstream-until-user-go)

---

## Decision Log 2026-08-05 → 2026-08-06

| # | Date | Decision | Alternatives | Why | Source |
|---|---|---|---|---|---|
| D1 | 08-05 | Port upstream PRs one-by-one onto `fix-memory-error` with full attribution | Separate branches per PR | User pivoted to single-branch merging; attribution kept in commit messages | User |
| D2 | 08-05 | Skip PR #2064 (asymmetric TP) | Port as infrastructure | Zero value on identical M4 Pros; Qwen3.5-only whitelist; high placement.py conflict | Agent analysis, user agreed |
| D3 | 08-05 | Port #1598 CORE-ONLY (HF cache discovery), skip UX overhaul | Naive merge | Would destroy 26+ of our ports; UX not wanted | Agent analysis |
| D4 | 08-05 | Skip #2229 (token relay) | Port unbundled | Highest-value skip; 1259 lines, needs 4-commit unbundle | Agent analysis |
| D5 | 08-05 | Skip #2255 (Ring Attention) | Port later | 3505 lines, 39 files conflicts | Agent analysis |
| D6 | 08-05 | Skip #1860 (RotatingKVCache) | Needs user decision | Aggressive defaults (MAX_KV_SIZE 3200→131072) without configurability | User decision needed |
| D7 | 08-06 | Modify VS Code Insiders source to disable sensitive-input guard in autopilot | Use Interactive mode / SSH keys | User explicitly wanted autopilot to pause for secrets, not cancel | User |
| D8 | 08-06 | Use `sshpass` with password in temp file (umask 077, deleted after) | Interactive password entry | Keeps secrets out of shell history/process list; enables automation | Agent security practice |
| D9 | 08-06 | Install exact driver 595.58.03 userspace from NVIDIA `.run` (not apt 595.84) | apt packages (535/595.84) | Kernel module is 595.58.03; version must match exactly | Agent diagnosis |
| D10 | 08-06 | Install mlx wheels explicitly from rltakashige fork (mlx + mlx_cuda_12) | rely on `uv sync` | Linux x86_64 mlx = TWO packages; sync never installs runtime | Agent diagnosis |
| D11 | 08-06 | Eject button: inline NO/YES split-button, no native confirm() popups | Keep native popups | User requested smooth inline UI | User |
| D12 | 08-06 | Eject = stop instance ONLY; NEVER auto-delete weights | Auto-delete weights | User explicitly wants manual deletion control | User (reversal of D13-era behavior) |
| D13 | 08-06 | (Reversed) Eject auto-deleted weights | — | Was from #2238 original; user reversed it | User |
| D14 | 08-06 | Rebuild app fully (pyinstaller → xcodebuild) instead of hot-swap only | Hot-swap static files | Hot-swap leaves build products stale; bundle signature invalid | Agent + user |
| D15 | 08-06 | SKIP #2129's IP-priority flip (ethernet>TB) | Port as-is | Contradicts our #2063/#2254 TB-preference for Mac↔Mac | Agent analysis |
| D16 | 08-06 | Defer #2200/#2129 port (research only) | Port now | User said "just researching" | User |
| D17 | 08-06 | Do NOT amend past commits to add detail | Amend | Other agent's commit 4f850371 touched same files after ours; history rewrite unsafe | Agent analysis |
| D18 | 08-06 | Track everything in promptsANDplans (gitignored, local) | Track in git | Folder is `.gitignore:52` by design; docs persist on disk | Repo design |
| D19 | 08-06 | EjectButton: `import { fade, fly }` from svelte/transition | Rely on Svelte auto | Svelte 5 compiles unimported transitions to bare refs → runtime ReferenceError | Agent (bug fix) |
| D20 | 08-06 | EjectButton prop named `id` (matches parent `{id}`) | `instanceId` | Parent passes `{id}`; mismatch sent `DELETE /instance/undefined` → 404 | Agent (bug fix) |
| D21 | 08-06 | `uv sync --extra mlx` REQUIRED before pyinstaller (installs mlx_lm) | Plain `uv sync` | mlx_lm is a git dep; plain sync prunes it → bundle lacks it → models fail to load | Agent (build fix) |
| D22 | 08-06 | Isolated test backends MUST use non-default ports (52416) + cleanup | Default 52415 | Stale test backend squatted on 52415 → app exit-code-1 on launch | Agent |
| D23 | 08-06 | Error UX research first, implementation deferred | Implement now | User said "just research for now" | User |
| D24 | 08-06 | Message queueing: research first (frontend-only FIFO, backend already batches 8) | Backend queue work | Backend has EXO_MAX_CONCURRENT_REQUESTS=8; gap is frontend blocking | User + agent research |

---

## Pending decisions (need user input)

| # | Question | Context |
|---|---|---|
| P1 | Apply the 2-command Linux fix (`uv sync --extra mlx-cuda12` + nvidia-smi symlink)? | ✅ **RESOLVED 08-07** — code fixes `e6ec3e2c` + `2a9e15a2` make manual steps optional; box live on MlxCuda+Vllm |
| P2 | Port #2200 (backend per shard)? | Foundation for heterogeneous; recommended next port (T4) |
| P3 | Port #2129 (TcpRelay etc.)? | ✅ **RESOLVED 08-06** — `72cf86bb` |
| P4 | #2064 asymmetric TP now or wait? | Agent rec: wait for >200B model need |
| P5 | Clean up empty branches (track-a/track-b)? | Empty bookmarks |
| P6 | #2246 PromoteMaster / #2135 --no-master? | Master control design |
| P7 | #2190 API host binding? | Security vs convenience |
| P8 | #1864 disable server-side tools? | Opt-out design |
| P9 | #2255 Ring Attention? | Long-context 65K+ |
| P10 | #2229 token relay? | ✅ **RESOLVED 08-07** — ported in 4 commits (`f0e59802` `e18a8f0e` `60f081e3` `e9011332`); live 152 tok/s Llama-3.2-3B |
| P11 | #1860 RotatingKVCache defaults? | Aggressive defaults need user decision — see `research/decision-support-2026-08-07.md` (rec: 16384 default + env override) |
| D25 | 08-07 | Fix README Linux section (CPU-only claim) | Leave stale | Misleads Linux users; CUDA works since #2103/#2216 | Agent (user asked for Linux improvements) |
| D26 | 08-07 | Add `scripts/setup_linux_gpu.sh` verify/repair script | None | Encodes all 5 deployment lessons; --fix mode | Agent |
| D27 | 08-07 | Improve placement "no cycle" error with node backend list | New event type | Message-only = no overlap risk, still actionable | Agent |
| D28 | 08-07 | Set up passwordless SSH (`ssh tolinux`) | Keep sshpass | User preference; eliminates secret handling | User |
