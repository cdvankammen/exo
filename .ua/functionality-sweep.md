# exo Functionality Sweep

Complete functionality inventory for the exo distributed inference system.

<!-- source: config-cli-env.md -->

# exo: CLI + Environment + Config

Generated: 2026-09-10 (sweep)

---

## 1. CLI Arguments

All flags are parsed in `src/exo/main.py` lines 497–623 via `argparse`.

| Flag | Short | Type | Default | Description |
|---|---|---|---|---|
| `--quiet` | `-q` | const → -1 | 0 (info) | Suppress all log output (verbosity = -1) |
| `--verbose` | `-v` | count | 0 | Increase log verbosity; each `-v` adds 1 |
| `--force-master` | `-m` | bool | False | Force this node to become cluster master |
| `--no-master` | — | bool | False | Worker-only mode; never become master |
| `--no-api` | — | bool | False | Do not spawn the OpenAI-compatible HTTP API |
| `--api-port` | — | int | 52415 | TCP port for the API server |
| `--max-in-flight-text-generations` | — | int | 2 | Max concurrent text-gen streams before HTTP 429 |
| `--no-worker` | — | bool | False | Disable the worker (no model execution) |
| `--no-downloads` | — | bool | False | Disable the download coordinator |
| `--offline` | — | bool | `EXO_OFFLINE` env | Air-gapped mode: skip internet checks, local models only |
| `--no-batch` | — | bool | False | Disable continuous batching (sequential gen) |
| `--legacy-daemon` | — | bool | False | Run as SysV-style double-fork daemon |
| `--bootstrap-peers` | — | CSV list | `EXO_BOOTSTRAP_PEERS` env | Comma-separated `host[:zenoh_port]` for cross-subnet clusters |
| `--metrics-port` | — | int | None | Port for Prometheus `/metrics` endpoint |
| `--namespace` | — | str | `__version__` | Discovery namespace; mismatched names isolate clusters |
| `--zenoh-port` | — | int | 52414 | TCP port for Zenoh listener |
| `--discovery-port` | — | int | 52413 | UDP port for multicast discovery |
| `--fast-synch` | — | bool (mutually exclusive) | None | Force MLX FAST_SYNCH on (JACCL backend) |
| `--no-fast-synch` | — | bool (mutually exclusive) | None | Force MLX FAST_SYNCH off |

---

## 2. Environment Variables

### 2.1 Core / Behavior

| Variable | Default | Source | Purpose |
|---|---|---|---|
| `EXO_HOME` | None | `constants.py` | Override home directory (all XDG paths become children of this) |
| `EXO_OFFLINE` | `"false"` | `constants.py` | Air-gapped mode (also a CLI flag) |
| `EXO_API_TOKEN` | None | `constants.py` | Bearer token required for API routes |
| `EXO_API_HOST` | `"0.0.0.0"` | `constants.py` | API bind address; `127.0.0.1` for localhost-only |
| `EXO_MAX_INPUT_TOKENS` | `128000` | `constants.py` | Max input tokens before API rejects (char heuristic) |
| `EXO_CHARS_PER_TOKEN` | `4` | `constants.py` | Chars-per-token heuristic for input length check |
| `EXO_MAX_CONCURRENT_REQUESTS` | `8` | `constants.py` | Global concurrent request limit |
| `EXO_ENABLE_IMAGE_MODELS` | `"false"` | `constants.py` | Enable image model support |
| `EXO_TRACING_ENABLED` | `"false"` | `constants.py` | Enable distributed tracing |
| `EXO_ENABLE_SERVERSIDE_TOOLCALLS` | `"1"` | `constants.py` | Enable server-side tool execution (T13) |
| `EXO_BOOTSTRAP_PEERS` | `""` | `main.py` | Comma-separated seed peers for cluster join |
| `EXO_API_PORT` | `52415` | `info_gatherer.py` | API port (read by info gatherer) |
| `EXO_METRICS_PORT` | None | `metrics.py` | Prometheus metrics port |

### 2.2 Logging

| Variable | Default | Source | Purpose |
|---|---|---|---|
| `EXO_LOG_LEVEL` | None (→ INFO/DEBUG) | `logging.py` | Base log level override (TRACE/DEBUG/INFO/WARNING/ERROR/CRITICAL) |
| `EXO_LOG_LEVELS` | None | `logging.py` | Per-module overrides: `"exo.master=DEBUG,exo.worker=WARNING"` |

### 2.3 Path / XDG

| Variable | Default | Source | Purpose |
|---|---|---|---|
| `EXO_HOME` | None | `constants.py` | Overrides all XDG base dirs (non-Linux default: `~/.exo`) |
| `XDG_CONFIG_HOME` | `~/.config` | `constants.py` | Config home (Linux only) → `XDG_CONFIG_HOME/exo/` |
| `XDG_DATA_HOME` | `~/.local/share` | `constants.py` | Data home (Linux only) → `XDG_DATA_HOME/exo/` |
| `XDG_CACHE_HOME` | `~/.cache` | `constants.py` | Cache home (Linux only) → `XDG_CACHE_HOME/exo/` |
| `EXO_DEFAULT_MODELS_DIR` | `EXO_DATA_HOME/models` | `constants.py` | Default writable models directory |
| `EXO_MODELS_DIRS` | None | `constants.py` | Colon-separated additional writable model dirs |
| `EXO_MODELS_READ_ONLY_DIRS` | None | `constants.py` | Colon-separated read-only model dirs (never deleted from) |
| `EXO_RESOURCES_DIR` | auto-detect | `constants.py` | Path to bundled resources |
| `EXO_DASHBOARD_DIR` | auto-detect | `constants.py` | Path to dashboard static files |

### 2.4 Ring / Placement

| Variable | Default | Source | Purpose |
|---|---|---|---|
| `EXO_RING_KV_WORKING_SET_MULTIPLIER_TIERED` | `"2"` | `placement_utils.py` | KV cache working-set multiplier when tiered |
| `EXO_RING_ADMISSION_CONTEXT` | `16384` | `placement_utils.py` | Context tokens for ring admission |
| `EXO_TIERED_KV` | `"0"` | `placement_utils.py` | Enable tiered KV cache offload (`"1"` = on) |
| `EXO_SINGLE_NODE_PREFILL_DISABLED` | — | `placement.py` | Disable single-node prefill optimization |

### 2.5 MLX / Multi-node Training

| Variable | Default | Source | Purpose |
|---|---|---|---|
| `MLX_HOSTS_JSON` | None | `auto_parallel.py` | JSON-encoded host list for MLX parallel |
| `MLX_HOSTFILE` | None | `auto_parallel.py` | Path to hostfile (JSON format) |
| `MLX_RANK` | `"0"` | `auto_parallel.py` | Current rank in multi-node setup |
| `MLX_TCPRELAY_PORT` | `40000` | `auto_parallel.py` | Base TCP port for inter-rank relay (port = base + rank) |
| `MLX_CUDA_RANKS` | `""` | `auto_parallel.py` | Comma-separated ranks using CUDA |
| `MLX_METAL_FAST_SYNCH` | auto | `bootstrap.py` | Force FAST_SYNCH on/off for MLX Metal |
| `EXO_FAST_SYNCH` | None | `bootstrap.py` | Override fast-synch (maps to MLX_METAL_FAST_SYNCH) |
| `EXO_DISABLE_ORPHAN_WATCHDOG` | None | `bootstrap.py` | Set `"1"` to disable orphan-process watchdog |

### 2.6 Download / Network

| Variable | Default | Source | Purpose |
|---|---|---|---|
| `HF_ENDPOINT` | `https://huggingface.co` | `huggingface_utils.py` | HuggingFace mirror endpoint |
| `HF_HOME` | `~/.cache/huggingface` | `huggingface_utils.py` | HuggingFace cache root |
| `HF_TOKEN` | None | `huggingface_utils.py` | HuggingFace API token |
| `SSL_CERT_FILE` | certifi CA | `download_utils.py` | Custom CA bundle for TLS |
| `HTTPS_PROXY` / `HTTP_PROXY` | None | `download_utils.py` | HTTP(S) proxy for downloads |

### 2.7 Debug / Dev

| Variable | Default | Source | Purpose |
|---|---|---|---|
| `EXO_DEBUG_PROMPT_TRIGGERS` | None | `batch_generator.py` | Set `"1"` to debug prompt trigger detection |
| `EXO_DSV4_FUSED_MOE` | `"0"` | `auto_parallel.py` | Enable DSv4 fused MoE kernel |
| `EXO_MACMON_PATH` | auto-detect | `info_gatherer.py` | Path to `macmon` binary |
| `OVERRIDE_MEMORY_MB` | None | `info_gatherer.py` | Override reported memory size |
| `EXO_RUNTIME_DIR` | None | `banner.py` | Runtime directory (non-empty = not first boot) |
| `CUDA_HOME` / `CUDA_PATH` | None | `bootstrap.py` | CUDA toolkit location |

---

## 3. Configuration Files

### 3.1 Build & Packaging

| File | Purpose |
|---|---|
| `pyproject.toml` | Python package metadata, dependencies, entry points |
| `Cargo.toml` (root) | Rust workspace root; builds `exo_rs` Python extension |
| `rust/exo_rs/Cargo.toml` | Rust crate config for the native extension |
| `bench/pyproject.toml` | Benchmarking sub-project |
| `tools/pyproject.toml` | Developer tooling sub-project |
| `package.json` / `package-lock.json` (root) | Node.js deps (dashboard build tooling) |

### 3.2 Dashboard

| File | Purpose |
|---|---|
| `dashboard/package.json` | Dashboard Node.js deps |
| `dashboard/tsconfig.json` | TypeScript config |
| `dashboard/svelte.config.js` | SvelteKit config |
| `dashboard/vite.config.js` | Vite bundler config (if present) |

### 3.3 Nix

| File | Purpose |
|---|---|
| `flake.nix` | Nix flake definition |
| `nix/apple-sdk-overlay.nix` | macOS SDK overlay for Nix |
| `nix/metal-toolchain.nix` | Metal toolchain for Nix builds |
| `nix/darwin-build-fixes.patch` | macOS build patches |
| `python/parts.nix` | Python component for Nix |
| `rust/parts.nix` | Rust component for Nix |
| `dashboard/dashboard.nix` | Dashboard component for Nix |
| `dashboard/parts.nix` | Dashboard parts for Nix |

### 3.4 Git Hooks

| File | Purpose |
|---|---|
| `.githooks/post-checkout` | Runs after `git checkout` / worktree switch |
| `.githooks/post-commit` | Runs after each commit |
| `.githooks/post-merge` | Runs after `git merge` / pull |
| `.githooks/pre-push` | Gate before `git push` |

### 3.5 CI/CD

| File | Purpose |
|---|---|
| `.github/workflows/pipeline.yml` | Main CI/CD pipeline |
| `.github/workflows/build-app.yml` | Application build workflow |

### 3.6 Editor / Tooling

| File | Purpose |
|---|---|
| `.zed/settings.json` | Zed editor settings |
| `.vscode/settings.json` | VS Code workspace settings |
| `.vscode/extensions.json` | Recommended VS Code extensions |

### 3.7 Benchmark Configs

| File | Purpose |
|---|---|
| `bench/scenarios.toml` | Benchmark scenario definitions |
| `bench/bench.toml` | Benchmark runner config |
| `bench/single-m3-ultra.toml` | M3 Ultra single-node config |
| `bench/prefill-decode.toml` | Prefill/decode profiling config |

### 3.8 Runtime Config

| Path | Purpose |
|---|---|
| `EXO_CONFIG_HOME/config.toml` | User config file (e.g. `~/.config/exo/config.toml`) |
| `EXO_CACHE_HOME/exo.pid` | PID file for daemon mode |
| `EXO_CACHE_HOME/node_zid` | Zenoh node identity |

---

## 4. Network Ports

| Port | Protocol | Service | Source |
|---|---|---|---|
| **52413** | UDP | Multicast discovery | `main.py` `--discovery-port` |
| **52414** | TCP | Zenoh peer-to-peer | `main.py` `--zenoh-port` |
| **52415** | TCP | OpenAI-compatible API (Hypercorn/ASGI) | `main.py` `--api-port` |
| **40000 + rank** | TCP | MLX TCP relay (multi-node training) | `auto_parallel.py` `MLX_TCPRELAY_PORT` |
| **49153–65535** | TCP | Ephemeral ports for runner processes | `utils/ports.py` (avoids 52415) |
| **dynamic** | HTTP | Prometheus `/metrics` (when `--metrics-port` or `EXO_METRICS_PORT` set) | `main.py`, `metrics.py` |

**Port collision note**: `random_ephemeral_port()` in `utils/ports.py` deliberately avoids port 52415 to prevent clashing with the API server.

---

## Status: COMPLETE — all four subsections written.


---

<!-- source: api-dashboard-settings.md -->

# exo Functionality Sweep — API Routes + Dashboard + Settings

## HTTP API Routes (src/exo/api/main.py)

| Method | Path | Handler |
|---|---|---|
| GET | /node_id | lambda → self.node_id |
| POST | /peers | add_peer |
| POST | /instance | create_instance |
| POST | /place_instance | place_instance |
| POST | /master/promote/{node_id} | promote_master |
| GET | /instance/placement | get_placement |
| GET | /instance/previews | get_placement_previews |
| GET | /instance/node-compatibility | get_node_compatibility |
| GET | /instance/await | await_instance |
| GET | /instance/{instance_id} | get_instance |
| DELETE | /instance/{instance_id} | delete_instance |
| GET | /v1/instance-links | list_instance_links |
| POST | /v1/instance-links | create_instance_link |
| PUT | /v1/instance-links/{link_id} | update_instance_link |
| DELETE | /v1/instance-links/{link_id} | delete_instance_link |
| GET | /v1/feature-flags | get_feature_flags |
| GET | /v1/settings | get_settings |
| PUT | /v1/settings | update_settings |
| GET | /v1/warnings | get_warnings |
| GET | /models | get_models |
| GET | /v1/models | get_v1_models |
| POST | /models/add | add_custom_model |
| DELETE | /models/custom/{model_id:path} | delete_custom_model |
| GET | /models/search | search_models |
| POST | /v1/chat/completions | chat_completions |
| POST | /bench/chat/completions | bench_chat_completions |
| POST | /v1/images/generations | image_generations |
| POST | /bench/images/generations | bench_image_generations |
| POST | /v1/images/edits | image_edits |
| POST | /bench/images/edits | bench_image_edits |
| GET | /images | list_images |
| GET | /images/{image_id} | get_image |
| POST | /v1/messages | claude_messages |
| POST | /v1/responses | openai_responses |
| POST | /v1/cancel/{command_id} | cancel_command |
| POST | /ollama/v1/chat/completions | ollama_chat_completions |
| POST | /ollama/api/chat | ollama_chat |
| POST | /ollama/api/api/chat | ollama_chat |
| POST | /ollama/api/v1/chat | ollama_chat |
| POST | /ollama/api/generate | ollama_generate |
| GET | /ollama/api/tags | ollama_tags |
| GET | /ollama/api/api/tags | ollama_tags |
| GET | /ollama/api/v1/tags | ollama_tags |
| POST | /ollama/api/show | ollama_show |
| GET | /ollama/api/ps | ollama_ps |
| GET | /ollama/api/version | ollama_version |
| GET | /state | get_state |
| GET | /state/{path:path} | get_state |
| GET | /v1/bandwidth-probe | get_bandwidth_probe |
| GET | /events | stream_events |
| POST | /download/start | start_download |
| DELETE | /download/{node_id}/{model_id:path} | delete_download |
| POST | /download/cancel | cancel_download |
| GET | /v1/traces | list_traces |
| POST | /v1/traces/delete | delete_traces |
| GET | /v1/traces/{task_id} | get_trace |
| GET | /v1/traces/{task_id}/stats | get_trace_stats |
| GET | /v1/traces/{task_id}/raw | get_trace_raw |
| GET | /v1/logs | list_logs |
| GET | /v1/logs/errors | get_log_errors |
| GET | /v1/logs/{name}/raw | get_log_raw |
| GET | /v1/logs/{node_id}/{name} | get_node_log_tail |
| GET | /v1/logs/{node_id}/{name}/raw | get_node_log_raw |
| GET | /v1/logs/{name} | get_log_tail |
| GET | /onboarding | get_onboarding |
| POST | /onboarding | complete_onboarding |

## Dashboard Pages (dashboard/src/routes/)

| Route | File | Purpose |
|---|---|---|
| / | +page.svelte | Main dashboard — cluster overview, chat |
| /advanced | advanced/+page.svelte | Advanced settings / internals |
| /downloads | downloads/+page.svelte | Model download management |
| /integrations | integrations/+page.svelte | API integrations (OpenAI/Claude/Ollama compat) |
| /logs | logs/+page.svelte | Live log viewer |
| /settings | settings/+page.svelte | Settings editor |
| /traces | traces/+page.svelte | Request traces listing |
| /traces/[taskId] | traces/[taskId]/+page.svelte | Single trace detail |

## Settings Model (src/exo/utils/settings.py — SettingsManager / SettingSpec)

Precedence: persisted UI override > process environment > default. Persisted at `EXO_CONFIG_HOME/settings.json`.

| Var | Type | Description | Default |
|---|---|---|---|
| EXO_KV_CACHE_BITS | str | KV cache quantization bits (4/8) | 4 |
| EXO_KV_CACHE_GROUP_SIZE | int | KV cache quantization group size | 64 |
| EXO_KV_DISK_PERSISTENCE | bool | SSD-as-RAM: persist KV cache to disk | 0 |
| EXO_KV_DISK_PATH | str | KV cache disk directory | None |
| EXO_KV_DISK_MAX_SIZE_GB | int | KV cache disk budget (GB) | None |
| EXO_KV_DISK_TTL_HOURS | int | KV cache expiry (hours) | None |
| EXO_MEMORY_THRESHOLD | str | RAM headroom reserved before prefill | None |
| EXO_PREFILL_MEMORY_THRESHOLD | str | Prefill-specific RAM headroom | None |
| EXO_PREFILL_STEP_SIZE | int | Prefill chunk size in tokens | 512 |
| EXO_MAX_CHUNK_SIZE | int | Token chunk size | None |
| EXO_MAX_CONCURRENT_REQUESTS | int | API concurrency limit | None |
| EXO_MAX_INSTANCE_RETRIES | int | Runner retry budget | 5 |
| EXO_BOOTSTRAP_PEERS | str | Manual peer list (comma-separated) | None |
| EXO_NODE_ZID | str | Fixed node identity (keypair seed) | None |
| EXO_ZENOH_NAMESPACE | str | Cluster namespace isolation | None |
| EXO_MODELS_DIRS | str | Writable model directories (colon-separated) | None |
| EXO_MODELS_READ_ONLY_DIRS | str | Read-only model dirs (colon-separated) | None |
| EXO_ENABLE_IMAGE_MODELS | bool | Show image model cards | 0 |
| EXO_OFFLINE | bool | No network (Hub disabled) | 0 |
| EXO_DSV4_FUSED_MOE | bool | DeepSeek V4 fused gate+up MoE | 1 |
| EXO_NO_BATCH | bool | Disable request batching | 0 |
| EXO_FAST_SYNCH | bool | Metal fast synchronisation | 1 |
| EXO_TRACING_ENABLED | bool | Per-request tracing | 0 |
| EXO_ENABLE_SERVERSIDE_TOOLCALLS | bool | Server-side tool execution (strip tools when 0) | 1 |

## Feature Flags (src/exo/api/main.py get_feature_flags)

| Flag | Source | Description |
|---|---|---|
| disaggregation | ENABLE_DISAGGREGATION env | Prefill/decode disaggregation experimental mode |
| server_side_tools | tools_enabled() | Server-side tool execution (EXO_ENABLE_SERVERSIDE_TOOLCALLS) |

STATUS: complete

---

<!-- source: models-modules-tests.md -->

# exo Functionality Sweep — Models + Modules + Tests

## Python Modules (src/exo/)

| Module | Purpose |
|---|---|
| api | FastAPI HTTP server: adapters (chat/claude/ollama/responses), main route registry, auth, keepalive, latency tracking |
| download | HuggingFace model downloading, shard downloaders, download coordinator |
| master | Master node: image store, placement, placement_utils, main orchestration |
| routing | Event router, Zenoh router, connection messages, topics |
| shared | Constants, XDG dirs, election, logging, metrics, topology, tracing, shared type definitions |
| utils | Async helpers, channels, dashboard path, port utils, power sampler, keyed backoff, SettingsManager |
| worker | MLX/CUDA inference engines, runner, plan, disaggregated prefill |

## Rust Crates (Cargo workspace)

| Crate | Path | Purpose |
|---|---|---|
| exo_rs | rust/exo_rs | PyO3 bindings: exposes Rust to Python (Pidfile, etc.) |
| networking | rust/networking | Low-level networking transport, Zenoh integration |

## Model Card Data (resources/inference_model_cards/ — 124 TOML files)

Families: DeepSeek V3.1/V3.2/V4, GLM-4.5/4.7/5, Kimi-K2, Qwen3, Llama-3, openai-harmony, etc.

Sample cards:
- mlx-community--DeepSeek-V3.2-4bit.toml
- mlx-community--DeepSeek-V3.1-8bit.toml
- mlx-community--GLM-4.7-Flash-4bit.toml
- mlx-community--Qwen3-0.6B-4bit.toml

For a complete list: `ls resources/inference_model_cards/*.toml | xargs -n1 basename`

## Tests

### Top-level tests (tests/)

| File | Purpose |
|---|---|
| tests/test_1node.py | Single-node integration tests |
| tests/test_2node.py | Two-node (distribution/placement) tests |
| tests/test_4node.py | Four-node cluster tests |
| tests/test_dashboard.py | Dashboard integration tests |
| tests/test_resilience.py | Failure/resilience scenarios |
| tests/conftest.py | Shared fixtures (framework.py) |
| tests/framework.py | Custom multi-node test harness |

### API tests (src/exo/api/tests/)

| File | Focus |
|---|---|
| test_add_peer.py | Peer addition |
| test_api_auth.py | Authentication |
| test_api_error_handling.py | Error responses |
| test_cancel_command.py | Cancel in-flight commands |
| test_chat_completions_stream.py | Streaming chat completir |
| test_claude_api.py | Claude Messages compatibility |
| test_claude_stop_sequence.py | Stop sequences |
| test_claude_tool_use.py | Claude tool-call flow |
| test_create_instance_force_override.py | Instance creation override |
| test_get_v1_models.py | Model listing endpoint |
| test_input_length_guard.py | Token-length guard |
| test_instance_deleted_stream_cleanup.py | Stream cleanup on instance delete |
| test_latency_tracker.py | Latency tracking |
| test_logs_api.py | Logs API |
| test_openai_responses_api.py | OpenAI Responses compat |
| test_promote_master.py | Master promotion |
| test_settings_api.py | Settings management |
| test_stream_events.py | Event streaming |
| test_text_generation_backpressure.py | Backpressure |
| conftest.py | Shared API test fixtures |

### Benchmarks (bench/)

| File | Purpose |
|---|---|
| exo_bench.py | Load benchmark harness (bench.toml, single-m3-ultra.toml, prefill-decode.toml, scenarios.toml) |
| exo_eval.py | Evaluation runner (eval_configs/) |
| eval_tool_calls.py | Tool-call benchmark |
| parallel_requests.py | Parallel request benchmark |
| prefill_decode_bench.py | Prefill/decode disaggregation benchmark |
| bench.toml / scenarios.toml | Benchmark configs |

STATUS: complete