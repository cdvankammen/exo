# EXO — Known Issues, Limitations & Roadmap

> Companion to `PROJECT-OVERVIEW.md`. Compiled from `TODO.md`, `PLATFORMS.md`, code comments, and analysis.

---

## 1. Known Issues

### From `TODO.md`
1. **`EXO_BOOTSTRAP_PEERS` is currently broken** — manual peer list doesn't work; discovery is the only reliable path.
2. **Continuous batching prefill blocks decode** — when a new prompt arrives, it blocks decode of the current batch until prefill completes (TODO #7).
3. **Offline model copying is incomplete** — exo requires an internet connection once to cache files to check download completion. Desired: check for a non-empty model folder with no `.partial` files instead (TODO #8).
4. **Memory pressure not used** — placement uses memory *used* rather than memory *pressure* (TODO #13).
5. **Connection type not shown in UI** — the dashboard doesn't display the type of each connection (TB5, Ethernet, etc.) (TODO #14).
6. **No dynamic connection priority switching** — can't dynamically switch to a higher-priority connection when it becomes available (TODO #16).
7. **No streaming model loads** — models can't be streamed from other devices in the cluster during load (TODO #17).
8. **No network connection type selection in tests** (TODO #18).
9. **Log cleanup needed** — per-module log filters, default DEBUG log levels (TODO #27).
10. **RDMA validation missing** — no `ibv_devinfo` validation of RDMA connections in the info gatherer (TODO #28).

### 1.2 From code comments
- **Node ID persistence is commented out** — `get_node_zid()` returns a fresh random ID on every start (`os.urandom(16).hex()`); the keypair persistence code is disabled. This means node identity is not stable across restarts.
- **`system_custodian` is referenced in docs but does not exist** — `AGENTS.md`/`CLAUDE.md` mention a Rust `system_custodian` crate for system-level operations (NFS, low power mode); it is absent from the repository (stale doc entry).
- **`docs/api.md` references `src/exo/master/api.py`** — the API actually lives in `src/exo/api/main.py`; the doc is outdated.
- **`AGENTS.md` describes an older architecture** — it references `src/exo/components/`, `src/exo/inference/`, `src/exo/model/`, `src/exo/tools/`, `src/exo/inspect/` which no longer exist. Current layout: `api/`, `download/`, `master/`, `routing/`, `shared/`, `utils/`, `worker/`.

### 1.3 Security / operational notes
- **No authentication** — the API is unauthenticated by design (local network tool). Anyone on the network can place instances, download models, or run inference. Do not expose the API port (52415) to untrusted networks.
- **Large payload warning** — the router warns on >1 MB pub/sub payloads (image chunks are chunked to stay under this).
- **`--bootstrap-peers` broken** means clusters can't be formed across subnets without multicast.

### 1.4 Fixed bugs / regressions (from tests)
- **#1918 (fixed)** — download status must not revert from completed to pending: the periodic rescan compared local file sizes against HF API sizes, and text files (README, YAML, jinja) can have different local vs remote sizes due to encoding changes. Test: `test_download_status_not_lost.py`.
- **Safetensors index (fixed)** — `total_size` was a required `PositiveInt`, so image-model index files (FLUX/mflux, which use `quantization_level`/`mflux_version` instead) failed silently and could report incomplete models as complete. Now `ModelSafetensorsIndexMetadata.total_size` is optional. Test: `test_safetensors_index.py`.
- **HF 429 rate-limit handling** — `_parse_retry_after` parses HF's `RateLimit: "api";r=0;t=52` header; retries capped at 300s. Test: `test_rate_limit_handling.py` (with real captured 429 headers).
- **Anthropic volatile headers** — Claude Code prepends `x-anthropic-billing-header` telemetry that changes per request and breaks KV prefix caching; the Claude adapter strips them (`_strip_volatile_headers`).
- **Safari PDF attachments** — pdfjs-dist needs `ReadableStream[Symbol.asyncIterator]`; the dashboard polyfills it.
- **macOS app disk-cache bug (fixed)** — `ClusterStateService` polls `/state` at 2 Hz; `URLSession.shared`'s on-disk cache wrote ~500–620 KB/s to disk and tripped the per-process disk-writes resource limit; fixed with an ephemeral non-caching URLSession.

---

## 2. Platform Limitations (`PLATFORMS.md`)

### Tier 1 — tested and maintained
- Apple Silicon macOS: Mac Studio (M3 Ultra), Mac Mini (M4 Pro), MacBook Pro (M5, M4 Max)

### Tier 2 — checked occasionally
- (empty)

### Tier 3 — minimal support
- (empty)

### Planned
- **Tier 1:** Linux CUDA (Nvidia DGX Spark), Linux CPU
- **Tier 2:** Linux Vulkan (Framework Desktop), Linux CUDA (Framework Desktop)
- **Longer term:** Windows CUDA, Windows CPU

---

## 3. Known Limitations

1. **Python 3.13 only** — hard pin; no support for other Python versions.
2. **MLX-only inference** — no PyTorch/vLLM backend; models must be MLX-compatible (GGUF/MLX quantized).
3. **Local network assumption** — discovery is multicast-based (site-local IPv6); clusters across subnets/VPNs need manual configuration (currently broken).
4. **Node identity instability** — random node IDs per restart complicate long-running cluster identity.
5. **Single master bottleneck** — all events flow through the elected master for indexing; master failure triggers re-election (handled, but adds latency).
6. **Memory-based placement only** — placement doesn't consider memory pressure, only used memory.
7. **No GPU memory accounting** — placement uses system memory; MLX unified memory is accounted via system memory.
8. **Image models require the forked mflux** — image generation depends on a fork that must be maintained.
9. **zenoh fork dependency** — the entire networking layer depends on a personal fork of zenoh (`evanev7/zenoh`), a maintenance risk.
10. **`EXO_BOOTSTRAP_PEERS` broken** — no manual peer configuration path.

### 3.1 Session-discovered issues (2026-08-06)

**Linux CUDA box (10.2.0.16, 2× RTX 5060 Ti) — KNOWN GAPS:**
11. **`pynvml` not installed on Linux box** — `NodeBackends.gather()` needs `pynvml` (`nvidia-ml-py`) to advertise `MlxCuda`; it's only in the `mlx-cuda12`/`mlx-cuda13` extras, and we installed with plain `--extra mlx`. **Result: Linux node advertises `["MlxCpu"]` only — cluster treats it as a CPU node.** Fix: `uv sync --extra mlx-cuda12`.
12. **`nvidia-smi` named `nvidia-smi-595.58`** (version-suffixed to match kernel driver 595.58.03) — `_query_cuda_vram_bytes()` uses `shutil.which("nvidia-smi")` which fails → falls back to system RAM (26.7GB) instead of VRAM (32GB). Fix: `ln -sf /usr/bin/nvidia-smi-595.58 /usr/bin/nvidia-smi`.
12b. **torch-on-Linux gap — FIXED 2026-08-06 (commit `f48bdf86`)** — pyproject declared `torch==2.10.0; sys_platform == 'linux'` but uv.lock markers never resolved it on Linux → vision models silently disabled (`ModuleNotFoundError: No module named 'torch'`, graceful fallback vision.py:113). Fixed by MANUALLY simplifying the 3 torch/torchaudio/torchvision markers to `sys_platform == 'linux' and extra == 'mlx'` (the `extra != 'mlx-cpu'...` clauses from #2103's extra layout are mis-evaluated by uv 0.12.x). Verified on box: `torch 2.10.0+cu130`. ⚠️ `uv lock` REGENERATION RE-BREAKS IT — re-apply the 3-line patch after any regen (verified twice).
13. **CUDA driver/library version matching is fragile** — kernel module 595.58.03 (Proxmox host) required exact 595.58.03 userspace from NVIDIA's `.run`; apt's 595.84 mismatches. Any Proxmox host kernel/driver upgrade breaks it again. **Also: apt installs clobber the symlinks — re-verify after ANY NVIDIA package install.**
13b. **Blackwell (sm_120) needs CUDA ≥ 12.8 for JIT** — Ubuntu's nvidia-cuda-toolkit (12.0) can't compile kernels for RTX 5060 Ti → `nvrtc: error: invalid value for --gpu-architecture` at warmup. Fixed with cuda-nvrtc-12-8/cuda-nvvm-12-8/cuda-cudart-dev-12-8 + symlinks (box-level).
14. **mlx on Linux x86_64 = TWO packages** — `mlx` (bindings, 0.7MB) + `mlx_cuda_12` (runtime 337MB, contains `libmlx.so`). `uv sync` alone doesn't install the runtime; the fork's wheels (`rltakashige/mlx-jaccl-fix-small-recv`) must be installed explicitly. Watch for wrong-arch ARM wheels in uv cache.

**VS Code Insiders patch (this Mac):**
15. **`workbench.desktop.main.js` patched** (`if(u&&!1){...}`) so autopilot doesn't cancel password prompts (sensitive-input guard). **Insiders auto-updates daily and reverts the patch** — must re-run `/Users/chris/scripts/vscode-patch-sensitive-input.sh` after each update. App is ad-hoc signed.

**macOS app bundle (this Mac):**
16. **Stale-bundle protocol crash (FIXED by rebuild)** — old EXO.app lacked `event_id` on `BaseEvent`; crash-looped on both Macs when receiving events from newer nodes, causing cluster election churn + phantom nodes. **Lesson: after protocol changes, app must be rebuilt from source** (`uv sync --extra mlx` → pyinstaller → xcodebuild → install). The hot-swap of dashboard static files is NOT enough — build products go stale.
17. **App bundle permissions** — after `cp -R` to /Applications, bundle was `drwx------` (700/root) → launch "Permission denied". Fix: `sudo chmod -R a+rX` + re-sign ad-hoc.

**Eject behavior (by design, user-directed):**
18. **Eject = stop instance ONLY** — weights are NEVER auto-deleted (user request 2026-08-06, commit `3d1a675c`). Manual deletion via Downloads page only.

---

## 4. Future Improvements & Enhancements

### From TODO.md (verbatim intent)
1. Profiled network latency / bandwidth display (TODO #4)
2. Per-link bandwidth usage visualization (TODO #5)
3. Fix continuous batching prefill blocking decode (TODO #7)
4. Offline model copying without internet (TODO #8)
5. Memory pressure instead of memory used (TODO #13)
6. Show connection type (TB5, Ethernet) in UI (TODO #14)
7. Dynamic switch to higher-priority connection (TODO #16)
8. Stream models from other devices (TODO #17)
9. Network connection type in tests (TODO #18)
10. Log cleanup — per-module filters, DEBUG default (TODO #27)
11. Validate RDMA with `ibv_devinfo` (TODO #28)

### Suggested enhancements (from analysis)
- **Stable node identity** — re-enable keypair persistence for `get_node_zid()`.
- **Authentication layer** — optional API keys/tokens for multi-user or untrusted networks.
- **More backends** — PyTorch/vLLM, Vulkan, Windows support per roadmap.
- **Fix `EXO_BOOTSTRAP_PEERS`** — enable cross-subnet clusters.
- **Better observability** — Prometheus metrics endpoint, structured tracing.
- **Model streaming** — load models from peer devices to reduce cold-start latency.
- **GPU memory-aware placement** — use actual GPU memory pressure.
- **Reduce master bottleneck** — consider sharded event indexing or leaderless ordering.
- **Update stale docs** — `AGENTS.md`, `docs/api.md`, `PLATFORMS.md` need alignment with the current architecture.

### Session roadmap 2026-08-06 (from deep research)

**Immediate (2 commands to make Linux box a real CUDA node):**
1. `cd /root/exo && uv sync --extra mlx-cuda12` — installs `nvidia-ml-py` → node advertises `MlxCuda`
2. `ln -sf /usr/bin/nvidia-smi-595.58 /usr/bin/nvidia-smi` — enables VRAM reporting (32GB instead of system RAM)
3. Restart exo on the Linux box

**Then (recommended port order for Mac+Linux splitting):**
4. **#2200** backend-per-shard (small, clean foundation — `ShardMetadata.backend` + `assign_shard_backends()` + `apply_shard_backend()`)
5. **#2129 pieces:** `_linux_interface_types()` → post-prefill barrier → `_rotate_metal_to_middle` → `MLX_HOSTS_JSON` + CUDA rank probing → **TcpRelay** (the big one, ~220 lines)
6. **SKIP** #2129's IP-priority flip (ethernet>TB) — contradicts our #2063/#2254 TB preference

**Deferred (awaiting user decision):**
- #2064 asymmetric TP — only relevant when even split OOMs smallest node; Qwen3.5-only whitelist
- Non-MLX models (GGUF/PyTorch) — no upstream PR; needs a whole new inference engine subsystem
- #2255 Ring Attention (long-context), #2229 token relay (biggest perf win, +10-40% TPS)
- Tier 4: #2246 PromoteMaster, #2135 --no-master, #2190 API host binding, #1864 disable server-side tools

**Continuation-session additions (2026-08-06 night):**
- **Resolved:** EJECT button runtime bugs (missing `svelte/transition` import; prop named `id` instead of `instanceId`) — fixed + verified live in browser
- **Resolved:** mlx_lm missing from bundle (plain `uv sync` pruned the git dep → pyinstaller silently shipped without it → models failed to load). Fix: `uv sync --extra mlx` before building; verify `grep -c mlx_lm build/exo/Analysis-00.toc` > 0
- **Resolved:** exit-code-1 on launch = stale test backend squatting port 52415 (not a code bug)
- **Known:** frontend blocks message sending while generating (ChatForm.svelte:206,:442, app.svelte.ts:2410) — no message queue despite backend supporting 8 concurrent (`EXO_MAX_CONCURRENT_REQUESTS`). Researched: `research/message-queueing-steering-research.md`
- **Roadmap additions:** frontend FIFO message queue (T24-T26), sampling controls panel (T27), guided decoding (T28), dashboard runner-error rendering (T29), Swift exit-reason mapping (T30) — see `tasks/README.md`

---

## 5. Testing Status

### Integration tests (`tests/`)
| File | Coverage |
|---|---|
| `test_1node.py` | Single-node: place instance + chat, multi-turn, delete instance |
| `test_2node.py` | 2-node: tensor+JACCL, pipeline+RING, multi-turn |
| `test_4node.py` | 4-node: pipeline+RING, tensor+JACCL |
| `test_resilience.py` | Node disconnect/reconnect recovery cycle |
| `test_dashboard.py` | Dashboard smoke tests |

### Test framework
- Marker-driven: `@pytest.mark.cluster(count=2, thunderbolt='a2a')`, `@pytest.mark.instance(model, sharding=..., comm=..., min_nodes=...)`
- `session` fixture deploys the cluster, places the instance, provides a `Session` object
- Orchestration lives in `exo_tools.harness`

### Commands
```bash
uv run pytest                    # excludes slow tests by default
uv run pytest -m ""              # all tests including slow
uv run pytest src/exo/shared/tests/test_election.py
uv run basedpyright              # strict type checking
uv run ruff check                # linting
nix fmt                          # formatting
```

### Unit test inventory (from exhaustive read)
- **shared**: `test_apply/*` (custom model cards, instance links, node download dedup, RDMA gating via thunderbolt connections + rdma_ctl, runner deleted), `test_election.py` (campaign/clock/seniority), `test_node_id_persistence.py` (**skipped** — keypair persistence disabled), `test_resolve_reasoning_params.py`, `test_state_serialization.py` (Topology JSON round-trip), `test_xdg_paths.py` (XDG compliance)
- **routing**: `test_event_buffer.py` (OrderedBuffer: sequential, out-of-order, gap handling)
- **master**: `test_master.py` (inject NodeGatheredInfo → verify TaskCreated), `test_placement.py`, `test_placement_utils.py` (layer allocation, CFG, tensor, jaccl), `test_topology.py`
- **api**: `test_api_error_handling.py` (OpenAI-style errors), `test_cancel_command.py`, `test_chat_completions_stream.py` (SSE delta shape), `test_claude_api.py`, `test_claude_tool_use.py`, `test_instance_deleted_stream_cleanup.py`, `test_openai_responses_api.py`
- **download**: `test_cancel_download.py`, `test_download_status_not_lost.py`, `test_download_verification.py` (size-mismatch re-download), `test_model_dirs.py` (multi-dir resolution, read-only), `test_offline_mode.py`, `test_rate_limit_handling.py`, `test_re_download.py`, `test_safetensors_index.py`
- **worker plan**: `test_download_and_loading.py`, `test_runner_lifecycle.py`, `test_task_forwarding.py`, `test_warmup.py`
- **worker runner**: `test_dsml_e2e.py`, `test_event_ordering.py`, `test_finish_reason_sse.py`, `test_glm_tool_parsing.py`, `test_parse_gpt_oss.py`, `test_parse_tool_calls.py`, `test_runner_supervisor.py`, `test_serve_prefill.py`
- **worker mlx**: `test_auto_parallel.py` (distributed, slow), `test_distributed_fix.py` (GPT-OSS, slow), `test_kv_prefix_cache.py`, `test_pipeline_prefill_callbacks.py`, `test_prefix_cache_architectures.py`, `test_tokenizers.py` (all model families), `test_tp_bit_exact.py` (slow)
- **worker image**: `test_cancellation.py` (NaN sentinel protocol)
- **mlx disaggregated**: `test_end_to_end.py`, `test_mlx_adapter.py` (dtype round-trips incl. bf16 bitcast), `test_protocol_roundtrip.py`, `test_server_drain.py`
- **utils**: `test_async_process.py`, `test_channels.py`, `test_event_log.py`, `test_pidfile.py`, `test_power_sampler.py`, `test_tagged.py` (tagged-union serialization), `info_gatherer/test_tb_parsing.py` (macOS only)
- **rust**: `exo_rs/tests/dummy.rs` (tokio channel), `exo_rs/tests/test_python.py` (NetworkingHandle + Pidfile via pytest)

---

## 11. Pre-Commit Checklist (from AGENTS.md)

```bash
uv run basedpyright && uv run ruff check && nix fmt && uv run pytest
```

CI runs `nix flake check` (formatting, linting, Rust tests).