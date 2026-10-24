# Remaining PRs Triage Report — exo-explore/exo

**Generated:** 2026-08-05
**Scope:** All PRs from `/tmp/all_prs.tsv` minus the ~65 already-handled PRs
**Method:** State-based triage (MERGED = in main, no port needed; OPEN/CLOSED = candidates) + topic relevance to `fix-memory-error` branch goals (memory, perf, stability, correctness)

## Summary Statistics

| Category | Count |
|---|---|
| **MERGED upstream** (already in `main`, skip) | ~150 |
| **Already handled** (ported or analyzed) | ~65 |
| **Remaining OPEN** (triage candidates) | ~75 |
| **Remaining CLOSED unmerged** (rescue candidates) | ~40 |
| **Total triaged below** | ~115 |

## Tier 1 — High-Value, Likely Clean-Apply (18 PRs)

### Correctness bugs
| PR | Title | State | Author | Date | What | Difficulty |
|---|---|---|---|---|---|---|
| **2203** | Deterministic election recovery | OPEN | oliver0006 | 2026-07-03 | Split-brain fix. **✅ PORTED** | moderate |
| **2177** | Snapshot pending/in_progress before iterating | OPEN | rogerstom-lgtm | 2026-06-25 | Dict-mutation crash. **✅ PORTED** | trivial |
| **2001** | Avoid content leak when generation ends inside thinking | OPEN | mlpy0 | 2026-04-28 | Thinking-tag leak. **✅ PORTED** | moderate |
| **2123** | Correct stop_sequence in Claude API | OPEN | morisil | 2026-05-27 | Stop reporting bug. **✅ PORTED** | moderate |
| **2184** | Preserve real finish_reason for truncated tools | OPEN | mlpy0 | 2026-06-27 | Wrong finish reason. **✅ PORTED** | trivial |
| **2187** | Drop tool calls with leaked GLM arg markup | OPEN | mlpy0 | 2026-06-29 | Corrupted args. **✅ PORTED** | trivial |
| **2189** | Honor tool_choice | OPEN | michaelharrigan | 2026-06-29 | OpenAI-compat gap. **✅ PORTED** | moderate |
| **2134** | Block SSRF in fetch_image_url | OPEN | 5F0jd2vLq54RerYW | 2026-06-01 | Security. **✅ PORTED** | trivial |

### Stability / reliability
| PR | Title | State | Author | What | Difficulty |
|---|---|---|---|---|---|
| **2205** | Exit runner when parent worker dies | CLOSED | aidiffuser | Orphaned runners pin memory | heavy |
| **2107** | Wrap blocking pipe/join in to_thread | OPEN | raajkumars | Event loop stalls | moderate — **DEFERRED** (architectural conflict with #2244 restart loop) |
| **2231** | Harden event sends on shutdown + debounce re-elections | OPEN | caesarniseko-coder | Flapping connections | moderate |
| **2202** | Guard event log rotation by file ownership | OPEN | oliver0006 | Log corruption. **✅ PORTED** | moderate |
| **1804** | Fix master routing to dead runners | OPEN | mlpy0 | Hangs | moderate — conflicts |

### Performance
| PR | Title | State | Author | What | Difficulty |
|---|---|---|---|---|---|
| **2254** | Measure link latency + prefer low-latency ring hosts | OPEN | abendrothj | Latency-aware placement | heavy — conflicts |
| **1999** | DeepSeek V4 fuse switch_mlp gate+up proj | OPEN | adurham | MoE kernel fusion | moderate — conflicts |
| **2121** | Only rank 0 emits ChunkGenerated under TP | OPEN | adurham | Duplicate events. **✅ PORTED** | trivial |

### Documentation
| PR | Title | What | Difficulty |
|---|---|---|---|
| **2169** | Architecture overview | Useful for our ARCHITECTURE.md sync | trivial |
| **2148** | TurboQuant MLX PDD plan | Quantization research doc | trivial |

## Tier 2 — Valuable but Needs Work (22 PRs)
- **#2253** Manual pipeline layer allocation — API surface design
- **#2226** FLOPS + memory weighted partitioning — targets non-existent path, stale
- **#2143** Balance burst requests — needs master-side state
- **#1821** Asymmetric TP (third variant of #2064 topic)
- **#2028/#2027** Reconcile worker backoff/API streams from State — snapshot series
- **#2243** EXO_ZENOH_CONNECT unicast — overlaps our POST /peers WIP
- **#1813** Distinguish TB4 from TB5 — depends on #2063 context
- **#1992** Peer-to-peer model distribution — big feature
- **#2099** Prefer 192.168.100.x subnet — too opinionated
- **#2200** Select MLX backend per shard — needs backend abstraction
- **#2179** use_prefix_cache request flag — simple but interacts with #1886
- **#2155** EXO_PREFILL_STEP_SIZE — tuning knob proliferation
- **#2012** Unified model catalog — massive scope
- **#1985** Prometheus /metrics — useful, adds dependency
- **#2230** Cross-platform packaging — huge scope
- **#2163** Migrate CLI to Rust — big refactor
- **#1903** HF mirror fallback for China — locale-specific
- **#2053** CUDA Docker image — no Docker policy
- **#2041** ANE power profiling — niche
- **#2135/#2246** --no-master / PromoteMaster — design decision (TIER 4)
- **#2190** API host binding config — security tradeoff (TIER 4)
- **#1864** Disable server-side tools — opt-out design (TIER 4)

## Tier 3 — Skip/Redundant (~70 PRs)
### 3a. MERGED upstream (~150) — already in main, no port needed
### 3b. Superseded by our ports or duplicate author series
- abendrothj CLOSED series #2210-#2217 → reopened as #2249-#2255 (we ported the open ones)
- AlexCheema snapshot series #2016-#2035 → only #2011/#2012 remain relevant
- #2079/#2100 → covered by #2103 port
- #2217 → reopened as #2255 (Ring Attention, deferred)

### 3c. Low value / noise
- Docker PRs (#2256, #1954) — no Docker policy
- One-off scripts, vague commits, stale feature attempts

## Tier 4 — Needs User Input (5 PRs)
| PR | Title | Question |
|---|---|---|
| **2246** | PromoteMaster command | Do we want manual master control? |
| **2135** | --no-master flag | Same design surface as #2246 |
| **2190** | Configurable API host binding | Security vs convenience |
| **1864** | Disable server-side tools | Opt-out design |
| **2129** | 3-node heterogeneous cluster | Do we care about non-Apple hardware? |

## Ported in Wave 2 (2026-08-05/06)
#2063, #1598 (core), #2177, #2121, #2184, #2187, #2054, #2055, #2134, #2001, #2189, #2203, #2123, #2202, #2162, #2133, #2161 — all verified (basedpyright 0 errors, ruff pass, tests pass), committed with full attribution.

## Next Up
- **#2231** (harden event sends + debounce re-elections) — clean-apply candidate
- **#1804** (master routing dead runners) — needs conflict resolution
- **#2169/#2148** (docs) — trivial ports
- **#2107** (to_thread) — revisit after architectural decision on restart loop vs wait_stopped
- **#2161** ✅ done; **#2205** (orphan runner exit) — heavy, high value
