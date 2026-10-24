# EXO — API Reference & Usage Examples

> Companion to `PROJECT-OVERVIEW.md`. Covers the REST API surface, the dashboard, CLI usage, and code examples with expected output.

**Base URL:** `http://localhost:52415` (default `--api-port`)

---

## 1. OpenAI-Compatible Endpoints

### `POST /v1/chat/completions`

Standard OpenAI chat completions, streaming or non-streaming.

**Request:**
```bash
curl http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "stream": true
  }'
```

**Response (streaming, SSE):**
```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":"4"},"index":0}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":""},"finish_reason":"stop","index":0}]}

data: [DONE]
```

**Python (OpenAI SDK):**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:52415/v1", api_key="exo")
resp = client.chat.completions.create(
    model="mlx-community/Llama-3.2-1B-Instruct-4bit",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(resp.choices[0].message.content)
# e.g. "Hello! I'm here to help you with anything you need."
```

### `POST /v1/responses` (OpenAI Responses API)
```bash
curl http://localhost:52415/v1/responses \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Llama-3.2-1B-Instruct-4bit","input":"Tell me a joke"}'
```

### `POST /v1/messages` (Anthropic Messages API)
```bash
curl http://localhost:52415/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: exo" \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Ollama-compatible endpoints
- `POST /ollama/api/chat`
- `POST /ollama/api/generate`
- `GET /ollama/api/tags`
- `POST /ollama/api/show`
- `GET /ollama/api/ps`
- `GET /ollama/api/version`

```bash
curl http://localhost:52415/ollama/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Llama-3.2-1B-Instruct-4bit","messages":[{"role":"user","content":"hi"}]}'
```

---

## 2. Image Generation & Editing

### `POST /v1/images/generations`
```bash
curl http://localhost:52415/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "exolabs/FLUX.1-dev-4bit",
    "prompt": "a cat astronaut on the moon",
    "n": 1,
    "size": "1024x1024"
  }'
```

### `POST /v1/images/edits`
Multipart form with `image` file + `prompt` + `model` (e.g. `exolabs/Qwen-Image-Edit-2509-4bit`).

---

## 3. Cluster & Instance Management

### `GET /node_id`
```json
{"node_id": "3f2a9c1e..."}
```

### `GET /state`
Full cluster state dump (topology, nodes, instances, runners, downloads, tasks). Used by the dashboard. Also supports path navigation: `GET /state/{path}` returns a clipped subtree (e.g. `/state/instances`).

### `GET /events`
Internal event log (debugging/observability) — JSON array from the API's `DiskEventLog`.

### `POST /instance` — create an instance (manual)
```bash
curl http://localhost:52415/instance \
  -H "Content-Type: application/json" \
  -d '{"instance": {"instanceId": "...", "shardAssignments": {...}}, "force_override": false}'
```
Performs a total-available-memory check across all nodes (400 error if insufficient) unless `force_override: true`.

### `POST /place_instance` — server-computed placement
```bash
curl http://localhost:52415/place_instance \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "sharding": "Pipeline",
    "instance_meta": "MlxRing",
    "min_nodes": 1,
    "force_override": false
  }'
```
Response includes the placed `Instance` with `shardAssignments` (runner→shard, node→runner maps).

### `GET /instance/placement?model_id=...&sharding=...&instance_meta=...&min_nodes=...`
Computes a placement without creating it (400 on `ValueError` from placement logic).

### `GET /instance/previews?model_id=...`
Placement previews — iterates all combinations of (Pipeline/Tensor sharding) × (MlxRing/MlxJaccl meta) × (1..N min_nodes) and reports each with instance or error + `memory_delta_by_node`.

### `GET /instance/{instance_id}` / `DELETE /instance/{instance_id}`
Get or delete an instance.

### `GET /instance/await?model_id=...&timeout_seconds=0` (SSE long-poll)
Waits until the instance is ready (all runners loaded), streaming progress:
```
data: {"type": "ready", "instance": {...}}
data: {"type": "timeout", "message": "No instance found for model ..."}
```
`timeout_seconds=0` waits indefinitely (max 300 for positive values).

### `GET/POST/PUT/DELETE /v1/instance-links` — disaggregated prefill/decode links
```json
{
  "prefill_instances": ["..."],
  "decode_instances": ["..."]
}
```
Returns 404 unless `ENABLE_DISAGGREGATION=true`.

### `GET /v1/feature-flags`
`{"disaggregation": true|false}` — gates the dashboard's Advanced page.

### `POST /v1/cancel/{command_id}`
Cancel an in-flight command by `command_id` (text or image). 404 if not found or already completed.

---

## 4. Models

### `GET /models`
List built-in + custom model cards.

### `POST /models/custom` / `DELETE /models/custom/{model_id}`
Add/remove custom model cards (TOML).

### `GET /models/search?query=...`
Search HuggingFace for models.

---

## 5. Downloads

### `POST /download/start`
```json
{
  "node_id": "<target node>",
  "model_id": "mlx-community/Llama-3.2-1B-Instruct-4bit"
}
```

### `POST /download/cancel` / `DELETE /download/{node_id}/{model_id}`

### `GET /state` → `downloads` field
Poll download progress (`DownloadPending/Ongoing/Completed/Failed`).

---

## 6. Traces

- `GET /v1/traces` — list traces
- `GET /v1/traces/{task_id}` — trace detail (list of events)
- `GET /v1/traces/{task_id}/stats` — timing stats by category and rank
- `GET /v1/traces/{task_id}/raw` — raw Chrome-trace-format JSON file
- `POST /v1/traces/delete` — delete traces (body: `{"task_ids": [...]}`)

Enabled with `EXO_TRACING_ENABLED=1`. Traces are produced by `shared/tracing.py` context managers inside the MLX engine (categories like `sync/compute`, `async/comms`), collected per-rank as `TracesCollected` events, merged by the Master into `TracesMerged`, and saved by the API to `EXO_TRACING_CACHE_DIR`.

## 7. Onboarding

- `GET /onboarding` — onboarding completion state
- `POST /onboarding` — mark onboarding complete (`EXO_CACHE_HOME/onboarding_complete` marker file)

---

## 8. Streaming Events (SSE)

### `GET /stream/events`
Server-Sent Events stream of `IndexedEvent`s — powers the live dashboard updates.

---

## 9. Bench Endpoints

### `POST /bench/chat/completions`
Benchmark variant of chat completions:
- KV prefix cache disabled by default (cold prefill) unless `use_prefix_cache: true`
- EOS tokens banned (forces exactly `max_tokens` generation)
- No model output parsing

Returns `BenchChatCompletionResponse` with `generation_stats` (prefill TPS, decode TPS, token counts, peak memory) and `power_usage` (per-node avg/prefill/gen power + energy from `PowerSampler`).

### `POST /bench/images/generations` / `POST /bench/images/edits`
Same pattern for images: `seconds_per_step`, `total_generation_time`, `num_inference_steps`, `num_images`, dimensions, `peak_memory_usage`, plus `power_usage`.

---

## 10. CLI Usage

```bash
# Run exo (master + worker + API + dashboard)
uv run exo

# Verbose logging
uv run exo -v        # or -vv

# Force this node to be master
uv run exo -m

# Headless (no API)
uv run exo --no-api

# Offline mode (no model downloads)
uv run exo --offline

# Custom ports
uv run exo --api-port 8080 --zenoh-port 52414 --discovery-port 52413
```

---

## 11. Dashboard

Built with Svelte 5 + TypeScript + Tailwind 4 + D3. Served at `/` by the API.

**Routes:**
| Route | Purpose |
|---|---|
| `/` | Main page: topology graph + chat |
| `/advanced` | Advanced controls (prefill/decode disaggregation, etc.) |
| `/downloads` | Model download manager |
| `/integrations` | API integration cards (OpenAI, Claude, Ollama) |
| `/traces` | Inference traces viewer |
| `/traces/[taskId]` | Individual trace detail |

**Key components** (`dashboard/src/lib/components/`):
- `TopologyGraph.svelte` — D3 live cluster topology
- `ChatForm.svelte`, `ChatMessages.svelte`, `ChatSidebar.svelte`, `ChatModelSelector.svelte`, `ChatAttachments.svelte` — chat UI
- `ModelPickerModal.svelte`, `ModelCard.svelte`, `ModelFilterPopover.svelte`, `HuggingFaceResultItem.svelte` — model selection
- `PrefillDecodeDisaggregation.svelte`, `PrefillProgressBar.svelte` — disaggregation controls
- `TokenHeatmap.svelte` — token throughput visualization
- `ImageLightbox.svelte`, `ImageParamsPanel.svelte` — image generation
- `MarkdownContent.svelte` — markdown/KaTeX/code rendering
- `ConnectionBanner.svelte`, `DeviceIcon.svelte`, `FamilyLogos.svelte`, `FamilySidebar.svelte`, `HeaderNav.svelte`, `ToastContainer.svelte`, `IntegrationCard.svelte`

**Stores** (`dashboard/src/lib/stores/`):
- `app.svelte.ts` — central state: topology, chat, instances, runners, downloads, placement previews, feature flags
- `favorites.svelte.ts` — favorite models (localStorage)
- `recents.svelte.ts` — recently launched models (localStorage)
- `toast.svelte.ts` — toast notifications

**Key API calls from the dashboard** (`app.svelte.ts`):
- `GET /v1/feature-flags`, `GET /state`
- `GET /instance/previews?model_id=...`
- `POST /v1/chat/completions` (streaming)
- `POST /v1/images/generations`, `POST /v1/images/edits`
- `POST /download/start`, `POST /download/cancel`, `DELETE /download/{node}/{model}`
- `GET/POST/DELETE /v1/instance-links`
- `GET /v1/traces`, `GET /v1/traces/{id}`, `GET /v1/traces/{id}/stats`, `GET /v1/traces/{id}/raw`, `POST /v1/traces/delete`

---

## 12. Test Harness Usage (`tools/src/exo_tools/`)

The `exo_tools` package provides programmatic cluster control used by tests and bench:

```python
from exo_tools.client import ExoClient
from exo_tools.harness import place_instance, cleanup_all_instances

client = ExoClient("localhost", 52415)
instance = place_instance(client, "mlx-community/Llama-3.2-1B-Instruct-4bit", min_nodes=1)
resp = client.request_json("POST", "/v1/chat/completions", body={...})
```

**Key modules:**
- `client.py` — `ExoClient` (HTTP client), `ExoHttpError`
- `cluster.py` — `Chip`, `ClusterInfo`, `EcoSession`, `Thunderbolt`, `make_client_from_url`
- `harness.py` — `Sharding` (PIPELINE/TENSOR), `Comm` (RING/JACCL), `place_instance`, `cleanup_all_instances`, `wait_for_instance_gone`, `is_model_downloaded`, `capture_cluster_snapshot`, `add_common_instance_args`

---

## 13. Bench Usage

```bash
# Tool-calling eval
uv run python bench/eval_tool_calls.py --model <model-id>

# General eval (Artificial Analysis methodology)
uv run python bench/exo_eval.py --model <model-id> --tasks gpqa_diamond
uv run python bench/exo_eval.py --model <model-id> --tasks humaneval,livecodebench --limit 50

# Throughput benchmark (uses /bench/chat/completions)
uv run python bench/exo_bench.py --model <model-id>

# Parallel requests
uv run python bench/parallel_requests.py --model <model-id>

# Prefill/decode disaggregation benchmark
uv run python bench/prefill_decode_bench.py --model <id> --pp 2048,8192 --tg 128
```

Supported eval tasks: `gpqa_diamond`, `mmlu_pro`, `aime_2024`, `aime_2025`, `humaneval`, `livecodebench`. Per-model configs in `bench/eval_configs/models.toml` (auto-detect reasoning vs non-reasoning). Bench manifests in `bench/*.toml` (constraints like `All(Chip(m3_ultra))`, defaults `pp = [512, 2048, 8192, 16384]`, `tg = 128`). Bench methodology is documented in `bench/METHODOLOGY.md` (prompt construction via binary search over repeated atoms, cold KV cache, banned EOS, no output parsing).

---

## 14. Expected Output Examples

### Placing an instance (Python)
```python
from exo_tools.harness import place_instance

instance = place_instance(client, "mlx-community/Llama-3.2-1B-Instruct-4bit", min_nodes=1)
print(instance)
# {
#   "MlxRingInstance": {
#     "instanceId": "...",
#     "modelId": "mlx-community/Llama-3.2-1B-Instruct-4bit",
#     "shardAssignments": {
#       "runnerToShard": {"runner-1": {"PipelineShardMetadata": {...}}},
#       "nodeToRunner": {"node-1": "runner-1"}
#     },
#     "hostsByNode": {"node-1": "192.168.1.10"},
#     "ephemeralPort": 54321
#   }
# }
```

### Streaming chat with raw HTTP (Python)
```python
import httpx

with httpx.stream(
    "POST",
    "http://localhost:52415/v1/chat/completions",
    json={
        "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "messages": [{"role": "user", "content": "Count to 3"}],
        "stream": True,
    },
    timeout=None,
) as r:
    for line in r.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            import json
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0]["delta"].get("content", "")
            print(delta, end="", flush=True)
# 1, 2, 3
```

---

## 15. Environment Variables

| Variable | Purpose |
|---|---|
| `EXO_HOME` | Base data directory (overrides XDG on all platforms) |
| `EXO_MODELS_DIRS` | Colon-separated writable model download dirs |
| `EXO_MODELS_READ_ONLY_DIRS` | Read-only model dirs (e.g. NFS); never written/deleted; wins over writable if duplicated |
| `EXO_DEFAULT_MODELS_DIR` | Default model dir (always first writable entry) |
| `EXO_CUSTOM_MODEL_CARDS_DIR` | Custom model card TOMLs |
| `EXO_EVENT_LOG_DIR` | Event log persistence |
| `EXO_IMAGE_CACHE_DIR` | Generated image cache |
| `EXO_TRACING_CACHE_DIR` | Trace storage |
| `EXO_ENABLE_IMAGE_MODELS` | Enable image models (`true`) |
| `EXO_OFFLINE` | Offline mode (local models only) |
| `EXO_TRACING_ENABLED` | Enable tracing (`true`) |
| `ENABLE_DISAGGREGATION` | Enable prefill/decode split (`true`) |
| `EXO_MAX_CONCURRENT_REQUESTS` | Continuous batching limit (default 8) |
| `EXO_MAX_INSTANCE_RETRIES` | Runner retry limit (default 5) |
| `EXO_MAX_CHUNK_SIZE` | Image chunk size (default 512KB) |
| `EXO_NO_BATCH` | Disable continuous batching (SequentialGenerator) |
| `EXO_FAST_SYNCH` | MLX fast-sync mode |
| `EXO_MEMORY_THRESHOLD` | KV prefix cache LRU eviction threshold (default 0.70–0.85 by RAM) |
| `EXO_BOOTSTRAP_PEERS` | Manual peer list (⚠️ broken) |
| `OVERRIDE_MEMORY_MB` | Override reported available memory (info gatherer, testing) |
| `HF_TOKEN` / `HF_ENDPOINT` / `HF_HOME` | HuggingFace auth/endpoint/cache |
| `EXO_RUNTIME_DIR` | Set by macOS app (skips browser auto-open) |
| `EXO_RESOURCES_DIR` / `EXO_DASHBOARD_DIR` | Override bundled resources/dashboard paths |
| `EXO_CONFIG_FILE` | Node config TOML (gathered at startup) |
| `EXO_PID_FILE` | PID file (Rust `Pidfile`) |

---

## 16. Complete Endpoint Quick Reference

```
# General
GET     /node_id
GET     /state, /state/{path}
GET     /events
GET     /v1/feature-flags
GET/POST /onboarding

# Instance Management
POST    /instance
POST    /place_instance
GET     /instance/placement
GET     /instance/previews
GET     /instance/await
GET     /instance/{instance_id}
DELETE  /instance/{instance_id}
GET/POST/PUT/DELETE /v1/instance-links

# Models
GET     /models, /v1/models
POST    /models/add
DELETE  /models/custom/{model_id}
GET     /models/search

# Text Generation
POST    /v1/chat/completions     (OpenAI)
POST    /v1/messages             (Claude)
POST    /v1/responses            (OpenAI Responses)
POST    /bench/chat/completions  (+ stats/power)
POST    /v1/cancel/{command_id}

# Image Generation
POST    /v1/images/generations
POST    /v1/images/edits         (multipart)
POST    /bench/images/generations
POST    /bench/images/edits
GET     /images
GET     /images/{image_id}

# Ollama Compatibility
POST    /ollama/api/chat (+ aliases)
POST    /ollama/api/generate
GET     /ollama/api/tags (+ aliases)
POST    /ollama/api/show
GET     /ollama/api/ps
GET     /ollama/api/version, HEAD /ollama/, HEAD /ollama/api/version

# Downloads
POST    /download/start
DELETE  /download/{node_id}/{model_id}
POST    /download/cancel

# Traces (EXO_TRACING_ENABLED)
GET     /v1/traces
GET     /v1/traces/{task_id}
GET     /v1/traces/{task_id}/stats
GET     /v1/traces/{task_id}/raw
POST    /v1/traces/delete

# Dashboard (static)
GET     /  → Svelte dashboard
```