# Settings Manager — Research & Design Ideas

**Date:** 2026-08-06
**Status:** Research only — no implementation yet
**Branch:** fix-memory-error (60 commits ahead of main)

---

## 1. The Problem

exo has ~50 `EXO_*` environment variables controlling everything meaningful
(KV-cache quantization, disk persistence, prefill step size, memory thresholds,
max concurrent requests, image models, fused MoE, tracing, model dirs, etc.).
**None are exposed in any UI** — they only work as process-launch env vars
(edit launch script / plist / systemd unit, restart, pray).

The UI currently exposes only:
- `minNodes`, `sharding`, `instanceType`, `force_override` — launch dialog
  (persisted in `exo-launch-defaults-v2` localStorage, dashboard-only)
- `exo-debug-mode`, `exo-allow-memory-override`, `exo-topology-only-mode`,
  `exo-chat-sidebar-visible` — localStorage toggles (dashboard-only)
- `/v1/feature-flags` — currently just `disaggregation` (API → dashboard)

There is **no Settings page** anywhere (routes = advanced/downloads/integrations/
logs/traces), and **nothing in the macOS app**.

## 2. Complete Inventory of Configurable Knobs

### A. Memory / KV-cache (our ports!) — highest value
| Env var | Meaning | Suggested UI |
|---|---|---|
| `EXO_KV_CACHE_BITS` | KV cache quantization bits (4/8) | Dropdown (per-model or global) |
| `EXO_KV_CACHE_GROUP_SIZE` | Quantization group size | Number input (advanced) |
| `EXO_KV_DISK_PERSISTENCE` | SSD-as-RAM on/off | Toggle |
| `EXO_KV_DISK_PATH` | Cache directory | Path picker |
| `EXO_KV_DISK_MAX_SIZE_GB` | Disk budget | Number input |
| `EXO_KV_DISK_TTL_HOURS` | Cache expiry | Number input |
| `EXO_MEMORY_THRESHOLD` | RAM headroom before prefill | Slider (% or GB) |
| `EXO_PREFILL_MEMORY_THRESHOLD` | Prefill-specific headroom | Slider |
| `EXO_PREFILL_STEP_SIZE` | Prefill chunk tokens | Number input |
| `EXO_MAX_CHUNK_SIZE` | Token chunk size | Number input |

### B. Cluster / placement
| Env var | Meaning | Suggested UI |
|---|---|---|
| `EXO_MAX_CONCURRENT_REQUESTS` | API concurrency limit | Number input |
| `EXO_MAX_INSTANCE_RETRIES` | Runner retry budget | Number input |
| `EXO_BOOTSTRAP_PEERS` | Manual peer list | Textarea (advanced) |
| `EXO_NODE_ZID` | Fixed node identity | Text (advanced) |
| `EXO_ZENOH_NAMESPACE` | Cluster namespace isolation | Text (advanced) |
| interface priority table | TB vs ethernet preference (from #2129 discussion) | Reorderable list (advanced) |

### C. Models
| Env var | Meaning | Suggested UI |
|---|---|---|
| `EXO_MODELS_DIRS` | Writable model dirs | List editor |
| `EXO_MODELS_READ_ONLY_DIRS` | Read-only dirs (incl. HF cache auto-discovery) | List editor |
| `EXO_ENABLE_IMAGE_MODELS` | Show image model cards | Toggle |
| `EXO_OFFLINE` | No network (Hub disabled) | Toggle |
| Per-model sampling defaults | temperature/top_p (already on cards) | In composer (read-only today) |

### D. Debug / perf toggles
| Env var | Meaning | Suggested UI |
|---|---|---|
| `EXO_DSV4_FUSED_MOE` | DeepSeek V4 fused gate+up | Toggle |
| `EXO_DEBUG_PROMPT_TRIGGERS` | Debug prompts | Toggle |
| `EXO_NO_BATCH` | Disable batching | Toggle |
| `EXO_FAST_SYNCH` | Metal fast sync | Toggle (advanced) |
| `EXO_DISABLE_ORPHAN_WATCHDOG` | Disable orphan runner watchdog | Toggle (advanced) |
| `EXO_TRACING_ENABLED` | Per-request tracing | Toggle |

### E. Paths (lowest value — leave as env-only)
`EXO_HOME`, `EXO_CACHE_HOME`, `EXO_LOG_DIR`, `EXO_EVENT_LOG_DIR`,
`EXO_RUNNER_LOG_DIR`, `EXO_CUSTOM_MODEL_CARDS_DIR`, `EXO_IMAGE_CACHE_DIR`,
`EXO_DASHBOARD_DIR`, `EXO_CONFIG_HOME`, `EXO_DATA_HOME`, `EXO_RUNTIME_DIR`,
`EXO_PID_FILE`, `EXO_RESOURCES_DIR`, `EXO_TRACING_CACHE_DIR`

## 3. Proposed Architecture

```
┌─────────────────────────────────────────────────────┐
│  SettingsManager (exo/utils/settings.py)           │
│  • persisted JSON at EXO_CONFIG_HOME/settings.json  │
│  • typed Pydantic model (frozen, strict)            │
│  • precedence: UI setting > env var > default       │
│  • loaded once at startup; workers inherit at spawn │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼────────┐          ┌─────────▼─────────┐
│  API           │          │  Worker/runner    │
│  GET /v1/settings           (reads resolved    │
│  PUT /v1/settings│          │   settings, not   │
│  (mirrors       │          │   bare os.environ)│
│   feature-flags)│          └───────────────────┘
└───────┬────────┘
        │
┌───────▼───────────────────┐
│  Dashboard Settings page  │  ← NEW route (routes/settings/+page.svelte)
│  macOS app settings pane  │  ← NEW (SwiftUI form → same endpoints)
└───────────────────────────┘
```

### Key design decisions
1. **Precedence:** UI > env > default. Env vars remain as headless/CLI escape
   hatch. This also fixes "which value wins?" ambiguity for ops users.
2. **Change semantics:** some settings need process restart (paths, KV bits
   affect new runners), some can hot-apply (concurrency limits). Version the
   settings file; API returns `requires_restart` per changed key.
3. **Where knobs are read:** replace bare `os.environ.get("EXO_...")` calls in
   hot paths with `settings.get(...)` — a small helper keeps env fallback.
4. **Worker propagation:** runner processes spawn from the node process; pass
   resolved settings via env at spawn time (single source of truth in the
   master/node), OR read the same JSON file (needs shared config dir — already
   exists: `EXO_CONFIG_HOME`).

### UI placement ideas
- **Dashboard:** new `Settings` page (gear icon in HeaderNav) with sections:
  Memory/KV, Cluster, Models, Debug — plus a "Relaunch needed" badge per key.
  The existing `advanced` route is a good home for the advanced/experimental
  section.
- **macOS app:** Settings window (Cmd+,) with the same sections; SwiftUI form
  hitting the same `/v1/settings` endpoints on the local node.
- **Launch dialog (picker):** keep minNodes/sharding/instanceType there (they're
  per-launch), but persist defaults via the Settings API instead of localStorage
  so both UIs share them.

## 4. Implementation Phases (estimates)
1. **Core:** SettingsManager + Pydantic model + `/v1/settings` GET/PUT
   (~150-200 lines + tests) — 1 session
2. **Wire top knobs:** KV bits/disk/prefill/memory threshold reads
   (~100 lines across cache.py/generate.py + tests) — 1 session
3. **Dashboard page:** routes/settings + forms (~300 lines Svelte) — 1 session
4. **macOS app pane:** SwiftUI settings (~200 lines) — 1 session
5. **Advanced:** per-node settings override, requires_restart semantics — later

## 5. Related upstream work
- `#1985` (Prometheus) — not related to settings UI
- `#2145` (custom model cards + instance links) — touches API types, overlaps
  with the settings API surface if merged; keep an eye on it
- No upstream PR implements a settings UI; this is greenfield.
