# EXO — Git History, Branch Work & Upstream Sources

> **Companion to `FILE-INVENTORY.md`.** Documents the current branch (`fix-memory-error`), all ported upstream PRs with their GitHub URLs, the uncommitted work-in-progress, and every upstream source the code derives from. Verified 2026-08-05.

---

## 1. Repository

| Item | Value |
|---|---|
| **Upstream repo** | https://github.com/exo-explore/exo |
| **License** | Apache-2.0 |
| **Maintainer org** | exo labs (https://x.com/exolabs, Discord https://discord.gg/TJ4P57arEm) |
| **Current branch** | `fix-memory-error` (fork/working branch off `main`) |
| **Upstream default branch** | `main` (b5375f8c at fork point) |
| **CODEOWNERS** | @ToxicPine, @AlexCheema, @GeluVrabie |

---

## 2. Current Branch — `fix-memory-error`

The branch contains **79 commits** ahead of `main` (56+ PR ports + Wave 4-7 session commits incl. #2129 `72cf86bb`, #2229 4-commit port, error-UX hardening, Linux UX, Settings page `db3736e2`, message queue `9ea20961`, multi-GPU VRAM `a082b091`). Verified 2026-08-07. Older counts in this file (65/63) are historical snapshots.

### Commit inventory (`git log main..HEAD` — 79 commits, snapshot 08-07)

**Local WIP (pre-campaign):**
| Commit | Description |
|---|---|
| `6b8d3a78` | code supposedly dealing with the memory problem (force_override plumbing, dashboard force-launch UI, ModelCard memory estimates) |
| `a6b99564` | update to the memory override and download override (API → download coordinator → tests) |
| `014a23ee` | baseline: commit existing WIP before PR merge campaign (includes `connect_peer` Rust/Python groundwork) |

**Wave 1 — memory/network core (23 ports):**
| Commit | PR | Description |
|---|---|---|
| `7dd12e9f` | #2159 | KV cache disk persistence/offload (SSD-as-RAM) |
| `070acc7f` | #2242 | KV-cache quantization via `EXO_KV_CACHE_BITS` |
| `5260b384` | #2251 | prefill memory headroom (`evict_for_prefill`) |
| `fde5c298` | #1990 | skip KV quant single-node |
| `5f7682dc` | #2160 | multi-rail TB bonding (`MLX_JACCL_RING`) |
| `35f1e789` | #2227 | capacity-aware layer allocation |
| `38101807` | #2252 | accelerator preference |
| `ddc34a44` | #2216 | CUDA VRAM reporting |
| `02f17fb4` | #2103 | **native CUDA Linux support** |
| `26818670` | #2192 | EHOSTUNREACH discovery fix |
| `ed7ba52f` | #2238 | eject + optional weight deletion |
| `4bac50c9` | #2204 | debug prompt trigger gate |
| `2772b55d` | #2244 | runner reconnect cascade |
| `6f5458ba` | #2228 | ArraysCache mask tolerance |
| `53bb0f98` | #2158 | V4 trim_cache snapshot restore |
| `c1075b87` | #2152 | vision cache hash dims |
| `0756536e` | #2174 | GPT-OSS tiktoken vocab |
| `5fa86ef8` | #2218 | Gemma 4 12B card |
| `5913b0ac` | #2239+#2247 | gen stats UI + v1/models filter |
| `89921b38` | #2245 | docs mlx extra |
| `6f0261c2` | #2249 | orphaned download GC |
| `2d50d66d` | #2250 | stale node prune |
| `086787dd` | *(our own)* | force_override bypass layer caps |

**Wave 2 (21 ports):**
| Commit | PR | Description |
|---|---|---|
| `868088f7` | #2063 | TB5 nested-hub discovery + JACCL retry + IP preference |
| `157a4c24` | #1598 (core) | HF Hub cache auto-discovery (UX overhaul deliberately NOT ported) |
| `f4abf47e` | #2177 | snapshot dicts before iterating |
| `1102e28b` | #2121 | rank-0 only ChunkGenerated |
| `b2b3dd74` | #2184 | finish_reason for truncated tools |
| `5686da7c` | #2187 | drop GLM arg markup |
| `a503b2b6` | #2054 | Kimi processor_repo |
| `5c0480b2` | #2055 | vision text_config fix |
| `b4d02e38` | #2134 | SSRF block |
| `bf38d869` | #2001 | thinking-block content leak |
| `10876037` | #2189 | tool_choice |
| `b55c99b0` | #2203 | deterministic election |
| `1d83a244` | #2123 | stop_sequence |
| `ba7092d1` | #2202 | event log ownership |
| `ac7a170f` | #2162 | Xcode 26 fix |
| `e7e0d34b` | #2133 | events pagination |
| `c34c1589` | #2161 | macOS 26 psutil fallback |
| `6e71b7bd` | *(typing)* | strict-typing adaptations |
| `dbf76ecb` | #2205 | orphan watchdog |
| `143d70c2` | #2231 | event-send hardening |
| `68d32e8c` | #1804 | dead runners + error chunks |
| `165b614e` | #2169+#2148 | docs |

**Wave 3 (8 ports):**
| Commit | PR | Description |
|---|---|---|
| `3abc8623` | #1999 | DSv4 fused MoE (opt-in) |
| `8ccbe8c8` | #2254 | link-latency ring selection |
| `86927717` | #2155 | `EXO_PREFILL_STEP_SIZE` |
| `15647f8a` | #2222 | Claude system role |
| `b80ffbf1` | #2237 | dashboard log viewer |
| `366c5302` | #2179 | `use_prefix_cache` flag |
| `36acecdd` | #2035 | liveness tuning |
| `91173df0` | #2060 | symlink-safe model deletion |

**Session commits 2026-08-06 (this chat):**
| Commit | Type | Description |
|---|---|---|
| `ca8474c7` | user request | inline EJECT split-button (replaces confirm popups; fly animation NO/YES) |
| `72cf86bb` | other agent | 3-node heterogeneous Metal+CUDA pipeline support (port of PR #2129) |
| `d6fcef17` | other agent | show loaded/ready models at top of model picker |
| `4f850371` | other agent | keep failed instances visible with error details |
| `85f1b024` | other agent | surface real startup failure reasons in menu bar app |
| `5acd6d6c` | other agent | stable machine-readable error codes |
| `ef16c380` | other agent | guard publish_bytes + dashboard warning chip |
| `f48bdf86` | this session (agent) | fix torch/torchaudio/torchvision resolve on Linux in uv.lock |
| `3d1a675c` | user request | **eject no longer auto-deletes model weights** (reversal of ca8474c7's deletion behavior) |

---

## 3. Ported Upstream PRs (with URLs)

All five ports preserve the original author's work. **Original PR URLs are the source of truth for rationale.**

### 3.1 PR #2160 — RDMA multi-cable striping (commit `5f7682dc`)
- **URL:** https://github.com/exo-explore/exo/pull/2160
- **Author:** @jasonpaulso (Jason Schulz)
- **State:** OPEN DRAFT, MERGEABLE
- **What it does:** When 2+ Thunderbolt cables connect two Macs, uses ALL of them simultaneously for RDMA (`MLX_JACCL_RING=1`), instead of one. Fixes a 2-cable mesh deadlock (mismatched interface pairing across directions hung model load at 47/48 layers).
- **Key changes:**
  - `src/exo/master/placement_utils.py` — `get_mlx_jaccl_devices_matrix` now collects ALL RDMA links per node pair from both edge directions → `list[list[list[str]]]`; element k of `[i][j]` and `[j][i]` are the two endpoints of the SAME physical cable
  - `src/exo/shared/types/worker/instances.py` — `jaccl_devices` type: `list[list[str|None]]` → `list[list[list[str]]]`
  - `src/exo/worker/engines/mlx/utils_mlx.py` — assertion update + sets `MLX_JACCL_RING=1` when any pair has >1 link; matrix parser accepts string-or-array cells (mesh backend uses `[0]` so single-cable behavior unchanged)
  - `src/exo/master/tests/test_placement.py` — 2 new regression tests incl. crossed-interface case
- **Measured impact (from PR):** 2-node cluster, 3× TB5 cables, Qwen3-Coder-Next-4bit, Tensor+RDMA: 19.4 → 42.0 tok/s generation, TTFT 610 → ~430 ms.
- **⚠️ CAVEAT:** Original PR marked DRAFT pending upstream mlx/jaccl fix for multi-wire recv prefill deadlock: https://github.com/rltakashige/mlx-jaccl-fix-small-recv/issues/3 and https://github.com/ml-explore/mlx/issues/3654. Pipeline-parallel activation transfers may hang in warmup without that fix. Tensor collectives unaffected (all_reduce falls back to one wire ≤64 KiB).
- **Dependency:** forked mlx at `rltakashige/mlx-jaccl-fix-small-recv` (branch pinned in `pyproject.toml`)

### 3.2 PR #1990 — skip KV quantization in single-node batch (commit `fde5c298`)
- **URL:** https://github.com/exo-explore/exo/pull/1990
- **What it does:** Avoids quantizing the KV cache in single-node `BatchGenerator` mode (where it can hurt quality without memory benefit).
- **Files:** `src/exo/worker/engines/mlx/cache.py` (+39/−2)

### 3.3 PR #2251 — reserve memory headroom before prefill (commit `5260b384`)
- **URL:** https://github.com/exo-explore/exo/pull/2251
- **What it does:** Reserves memory headroom before prefill to prevent OOM during generation (KV cache growth).
- **Files:** `cache.py` (+32), `batch_generate.py` (+3), `test_kv_prefix_cache.py` (+25)

### 3.4 PR #2242 — `EXO_KV_CACHE_BITS` env var (commit `070acc7f`)
- **URL:** https://github.com/exo-explore/exo/pull/2242
- **What it does:** Exposes KV-cache quantization bits via `EXO_KV_CACHE_BITS` env var (default `4`), making the previously hard-coded `ATTENTION_KV_BITS` configurable.
- **Files:** `cache.py`, `constants.py` (+20), `generate.py`

### 3.5 PR #2159 — disk persistence/offload for KV prefix cache (commit `7dd12e9f`)
- **URL:** https://github.com/exo-explore/exo/pull/2159
- **What it does:** Offloads conversation KV state to disk, freeing RAM for weights (the closest thing to "SSD as RAM" in the codebase). Adds optional disk-backed KV prefix cache reload.
- **Files (990 insertions):**
  - `src/exo/worker/engines/mlx/cache.py` (+531 — disk persistence, save/load, LRU eviction)
  - `src/exo/worker/engines/mlx/builder.py`, `disaggregated/serve.py`, `generator/batch_generate.py`, `generator/generate.py`, `runner/runner.py`
  - New tests: `tests/unittests/test_mlx/test_kv_disk_persistence.py` (+409), `tests/unittests/test_runner/test_kv_flush_wiring.py` (+25)

---

## 4. Local WIP Commits

### 4.1 `6b8d3a78` — "code supposedly dealing with the memory problem"
- **Files (12, +367):** `.gitignore` (+7), `dashboard/src/lib/components/ModelCard.svelte` (+55), `ModelPickerModal.svelte` (+25), `stores/app.svelte.ts` (+50), `routes/+page.svelte` (+75), `src/exo/api/main.py`, `src/exo/api/types/api.py` (+6), `src/exo/master/placement.py` (+11), `placement_utils.py` (+23), `test_placement.py` (+32), `test_placement_utils.py` (+99), `src/exo/shared/types/commands.py` (+7)
- **Theme:** force_override plumbing for placement ("load the model anyway" when it exceeds reported memory), dashboard memory-estimate + force-launch UI, `.gitignore` additions

### 4.2 `a6b99564` — "update to the memory override and download override"
- **Files (21, +303):** dashboard selector/launch UI refinements, `docs/api.md` (+27 — documents `force_override` on `/instance`, `/instance/previews`, `/instance/placement`, `/place_instance`), `src/exo/api/main.py`, `types/api.py`, new test `test_create_instance_force_override.py` (+106), download subsystem (`coordinator.py`, `download_utils.py`, `impl_shard_downloader.py`, `shard_downloader.py`, +tests), `placement_utils.py`
- **Theme:** download override + memory override end-to-end (API → download coordinator → tests)

### 4.3 `014a23ee` — baseline commit (pre-PR-campaign WIP)
- **Files (6, +119):** `rust/exo_rs/src/networking.rs` (+28 — `connect_peer` PyO3 binding), `rust/networking/src/lib.rs` (+65 — `Session::connect_peer` manual TCP connect bypassing discovery), `rust/networking/src/swarm.rs` (+21 — `ToSwarm::ConnectPeer` command), `src/exo/api/main.py` (+3), `src/exo/api/types/api.py` (+14 — `AddPeerParams`/`AddPeerResponse`), `src/exo/routing/router.py` (+11 — `Router.connect_peer`)
- **Theme:** manual peer connection groundwork (the `connect_peer` feature continues uncommitted)

---

## 5. Uncommitted Changes — Manual Peer Connection Feature (WIP)

`git status` shows 5 modified + 1 untracked file. This is the **`POST /peers`** feature: manually connect a node by hostname/IP (Tailscale, cross-subnet) when multicast discovery isn't working.

### Full stack of the feature
| Layer | File | Change |
|---|---|---|
| Rust core | `rust/networking/src/lib.rs` | `Session::connect_peer(host, port) -> bool` — builds `Locator("tcp", host:port)`, calls `runtime.connect_peer`; returns false if already connected (mirrors zenoh) |
| Rust swarm | `rust/networking/src/swarm.rs` | `ToSwarm::ConnectPeer { host, port, result_sender }` command handled in `into_stream` loop |
| PyO3 binding | `rust/exo_rs/src/networking.rs` | `NetworkingHandle.connect_peer(host, port) -> PyResult<bool>` |
| Stub | `rust/exo_rs/exo_rs.pyi` | `async def connect_peer(self, host: str, port: int) -> bool` (uncommitted diff) |
| Router | `src/exo/routing/router.py` | `Router.connect_peer(host, port=52414) -> bool` — delegates to `self._net.connect_peer` |
| API | `src/exo/api/main.py` | `POST /peers` handler: `add_peer(payload: AddPeerParams) -> AddPeerResponse`; 503 if no router wired; 400 on connection failure |
| Types | `src/exo/api/types/api.py` | `AddPeerParams(host, zenoh_port=52414)`, `AddPeerResponse(host, port, connected)`; re-exported in `types/__init__.py` |
| Wiring | `src/exo/main.py` | `Node.create` now passes `router=router` to `API` (was `"Router | None"` string annotation → real import) |
| Dashboard | `dashboard/src/routes/+page.svelte` | "Add Node Manually" panel in the sidebar: host + port inputs, POST `/peers`, success/error toasts (+81) |
| Tests | `src/exo/api/tests/test_add_peer.py` (new) | 4 tests: connects (200 + connected=True), default port 52414, 503 without router, 400 on bad host |

### Type definitions (from `src/exo/api/types/api.py`)
```python
class AddPeerParams(BaseModel):
    """Parameters for manually adding a peer node by hostname/IP."""
    host: str
    # Port of the target's zenoh TCP listener (default 52414).
    zenoh_port: int = 52414

class AddPeerResponse(BaseModel):
    host: str
    port: int
    connected: bool
```

### API contract
```
POST /peers
{"host": "mini2", "zenoh_port": 52414}
→ 200 {"host": "mini2", "port": 52414, "connected": true}
→ 503 {"detail": "Peer connection is unavailable on this node"}   (no router)
→ 400 {"detail": "Failed to connect to peer: ..."}                 (failure)
```

---

## 6. Upstream Dependencies & Their GitHub URLs

| Dependency | URL | Why |
|---|---|---|
| **zenoh (forked)** | https://github.com/evanev7/zenoh (branch `exo`) | All ~25 `zenoh*` crates patched via `[patch.crates-io]`; networking substrate |
| **mlx** | https://github.com/ml-explore/mlx | Apple array framework (inference engine) |
| **mlx (forked)** | https://github.com/rltakashige/mlx-jaccl-fix-small-recv | JACCL/ring fixes for distributed mlx (pinned in `pyproject.toml` extras) |
| **mlx-lm (forked)** | https://github.com/leo/deepseek-v4 (branch) | DeepSeek V4 model support |
| **mflux (forked)** | https://github.com/ai-dock/mflux (exo fork) | FLUX.1 image generation |
| **mlx-vlm** | https://github.com/ml-explore/mlx-vlm | Vision-language models |
| **macmon** | https://github.com/vladkens/macmon (pinned rev `a1cd06b6cc0d5e61db24fd8832e74cd992097a7d`) | GPU/CPU/power monitoring |
| **pidfile-rs** | https://github.com/fluent-rs/pidfile-rs (git dep) | PID file locking (libbsd flopen) |
| **Sparkle** | https://github.com/sparkle-project/Sparkle (2.9.0-beta.1) | macOS app auto-update |
| **openai-harmony** | https://github.com/openai-harmony (pip pkg) | GPT-OSS Harmony encoding/tool parsing |
| **pyo3** | https://github.com/PyO3/pyo3 (0.28.3) | Rust↔Python bindings |
| **zenoh-plugin-storage-manager** | https://github.com/evanev7/zenoh (workspace) | Cluster KV storage plugin |
| **netwatcher** | https://github.com/nickel-org/netwatcher (0.6.0) | Network interface change monitoring |
| **basedpyright** | https://github.com/detachhead/basedpyright | Strict type checking |
| **rustworkx** | https://github.com/Qiskit/rustworkx | Topology graph (`PyDiGraph`) |

### Related upstream discussions
- Multi-wire jaccl recv prefill deadlock: https://github.com/rltakashige/mlx-jaccl-fix-small-recv/issues/3, https://github.com/ml-explore/mlx/issues/3654
- RDMA over Thunderbolt announcement: https://x.com/exolabs/status/2001817749744476256
- Benchmark source (Jeff Geerling 15TB VRAM article): https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5

---

## 7. CI/CD & GitHub Configuration

### Workflows (`.github/workflows/`)
| File | Trigger | Purpose |
|---|---|---|
| `pipeline.yml` | push / PR to staging+main | Matrix build (macos-26 aarch64-darwin, ubuntu-latest x86_64-linux, ubuntu-24.04-arm aarch64-linux): Nix build all outputs, `nix flake check`, pytest (macOS only, `-m "not slow"`, `EXO_TESTS=1`); builds `metal-toolchain` NAR on CI when cachix misses |
| `build-app.yml` | workflow_dispatch / tag `v*` / branch `test-app` | Full release pipeline: version derivation from semver tag, draft-release validation, Xcode 26.2 + Metal, keychain setup, notarization credential pre-flight (all-or-nothing), provisioning-profile expiry check, PyInstaller bundle, Swift app build, codesign, notarize, DMG, Sparkle appcast (with markdown release notes), S3 upload, GitHub Release publish |

### Composite actions (`.github/actions/`)
`conditional-commit`, `format`, `lint`, `lint-check`, `regenerate-protobufs`, `setup-python-uv`, `unit-test`, `verify-clean`

### Other
- `.github/CODEOWNERS` — @ToxicPine, @AlexCheema, @GeluVrabie
- `.github/pull_request_template.md` — Motivation / Changes / Why It Works / Test Plan (manual + automated)

---

## 8. Editor & Agent Tooling Config

| Folder | Contents |
|---|---|
| `.vscode/settings.json` | nix LSP (nixd), ruff formatter for Python, basedpyright config pointing at `pyproject.toml`, `.venv/bin/python` interpreter |
| `.zed/settings.json` | `nix_python` LSP running `nix run .#python-lsp` |
| `.idea/` | PyCharm/IntelliJ config (pyright overrides, modules, vcs) |
| `.typings/` | Generated `.pyi` stubs for `mlx_lm` (for basedpyright) |
| `.feeder/tasks.db` | SQLite DB (agent tool "feeder" task storage — local, gitignored) |
| `.mnemo/` | `graph.lbug` (19MB) + `parse-cache.json` — agent memory graph tool (local, gitignored) |
| `.githooks/` | Git LFS hooks (post-checkout, post-commit, post-merge, pre-push) wired via `core.hookspath` |

---

## 9. `tmp/` Folder (developer scratch)

| File | Purpose |
|---|---|
| `run_exo_on.sh` | Deploy exo to remote hosts over SSH (requires clean tree) |
| `set_rdma_network_config.sh` | Removes `bridge0`, disables Thunderbolt Bridge, creates `exo` network location with DHCP (same logic as the macOS app's LaunchDaemon) |
| `run_llm.sh` | SSH to a host and run a query |
| `run_llm.py` | Python variant |
| `test_trust_remote_code_attack.sh` | Security test: verifies API-added models get `trust_remote_code=false` (model `KevTheHermit/security-testing`) |
| `quantize_and_upload.py` | mflux model quantization + HF upload (run from mflux dir) |
| `gen_card.py` | Generate inference model cards from HF model IDs |
| `prompt.txt` | Scratch prompt |
| `config_examples/claude_code.sh` | Run Claude Code against exo (GPT-OSS 120B via Anthropic-compatible endpoint) |
| `config_examples/opencode.json` | opencode config pointing at exo provider |
| `old_tests/` | `auto_bench.sh`, `eval_tool_calls.sh`, `get_all_models_on_cluster.py`, `test_vision_cache.py` |