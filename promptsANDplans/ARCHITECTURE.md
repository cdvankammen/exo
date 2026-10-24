# EXO — Architecture Deep Dive

> Companion to `PROJECT-OVERVIEW.md`. This document explains every major module, class, and function in the codebase, and how the pieces fit together.

---

## 1. The Node — Composition Root

**File:** `src/exo/main.py`

Every device runs a single `Node` process that composes all components. There is no separate "server" vs "client" binary — every node runs the full stack and the cluster self-organizes.

### `Args(FrozenModel)`
Pydantic-validated CLI arguments with `.parse()` (argparse). Key flags:

| Flag | Default | Purpose |
|---|---|---|
| `-v` / `-vv` / `-q` | — | Verbosity / quiet |
| `-m, --force-master` | off | Force this node to win elections (seniority 1,000,000) |
| `--no-api` | off | Don't start the FastAPI server |
| `--api-port` | `52415` | API port |
| `--no-worker` | off | Don't start the worker |
| `--no-downloads` | off | Don't start the download coordinator |
| `--offline` | off | No network model downloads (local models only) |
| `--no-batch` | off | Disable continuous batching |
| `--fast-synch/--no-fast-synch` | on | MLX fast-sync mode |
| `--legacy-daemon` | off | Legacy daemonization |
| `--bootstrap-peers` | — | Manual peer list (⚠️ currently broken, see KNOWN-ISSUES) |
| `--namespace` | `__version__` | Cluster namespace (isolates clusters) |
| `--zenoh-port` | `52414` | zenoh TCP port |
| `--discovery-port` | `52413` | UDP discovery port |

### `Node(dataclass)`
The composition root. Fields: `router`, `event_router`, `download_coordinator`, `worker`, `election`, `election_result_receiver`, `master`, `api`, `node_id`, `offline`, `_api_port`.

**`Node.create(args)`** builds, in order:
1. `Router.create(...)` — Rust `NetworkingHandle.new(identity, namespace, listen_port, discovery_service_port)`
2. Registers 6 topics: `GLOBAL_EVENTS`, `LOCAL_EVENTS`, `COMMANDS`, `ELECTION_MESSAGES`, `CONNECTION_MESSAGES`, `DOWNLOAD_COMMANDS`
3. `EventRouter` (command_sender → COMMANDS, external_outbound → LOCAL_EVENTS, external_inbound ← GLOBAL_EVENTS)
4. `DownloadCoordinator` (unless `--no-downloads`)
5. `API` (unless `--no-api`)
6. `Worker` (unless `--no-worker`)
7. `Master` — **every node starts as master**; election demotes/promotes
8. `Election` with `seniority=1_000_000 if args.force_master else 0`

**`Node.run()`** starts all components as anyio tasks in a `TaskGroup`: `router.run`, `event_router.run`, `election.run`, `download_coordinator.run`, `worker.run`, `master.run`, `api.run`, `_elect_loop`.

**`Node._elect_loop()`** — consumes `ElectionResult` messages:
- On `is_new_master`: rebuilds the `EventRouter` (new session), recreates `DownloadCoordinator`, `Worker`, resets API, restarts event router
- On self-promotion: spawns a `Master`
- On other-node win: shuts down its own `Master`
- API gets `reset(won_clock, receiver)` or `unpause(won_clock)`

### Entry points
- `src/exo/__main__.py` — console bootstrap: `freeze_support()`, multiprocessing `-c` flag handling, then `main()`
- `pyproject.toml` script: `exo = "exo.main:main"`

### `main()` / `main_inner()`
- `main()` — PID-file locking via Rust `exo_rs.Pidfile`, optional double-fork daemonization (`python-daemon`'s `DaemonContext`)
- `main_inner()` — raises NOFILE rlimit to 65535, sets `mp.set_start_method("spawn", force=True)`, configures loguru, handles env overrides (`EXO_NO_BATCH`, `EXO_FAST_SYNCH`), then `anyio.run(Node.create, args)` and `anyio.run(node.run)`

---

## 2. Routing Layer

### `src/exo/routing/topics.py`
Defines the typed pub/sub topics. Each topic binds a string + `PublishPolicy` + Pydantic model type, serialized as JSON bytes.

| Topic | Policy | Payload |
|---|---|---|
| `GLOBAL_EVENTS` | Always | `GlobalForwarderEvent` (master → all) |
| `LOCAL_EVENTS` | Always | `LocalForwarderEvent` (workers → master) |
| `COMMANDS` | Always | `ForwarderCommand` |
| `ELECTION_MESSAGES` | Always | `ElectionMessage` |
| `CONNECTION_MESSAGES` | Never (local only) | `ConnectionMessage` |
| `DOWNLOAD_COMMANDS` | Always | `ForwarderDownloadCommand` |

`PublishPolicy`: `Never` (local only), `Minimal` (publish only when no local receiver), `Always` (always to network).

### `src/exo/routing/router.py`
The networking bridge between Rust zenoh and Python channels.

- **`TopicRouter[T]`** — per-topic fan-out. `run()` reads from its internal receiver; per `PublishPolicy` forwards to `networking_sender` and/or `publish()` to all subscribed local `Sender`s. `new_sender()` returns a clone of the internal sender (what components call `router.sender(topic)` on).
- **`Router`** — owns a Rust `NetworkingHandle` and one `TopicRouter` per topic.
  - `create()` — `NetworkingHandle.new(identity, namespace, listen_port, discovery_service_port)`
  - `run()` — starts each topic router + `_networking_recv` + `_networking_publish`; subscribes via `gossipsub_subscribe(topic)`
  - `_networking_recv()` — receives `FromSwarm` variants: `Message(topic, data)` → `publish_bytes`; `Connection()` → `ConnectionMessage.from_update()` → published on `CONNECTION_MESSAGES`
  - `_networking_publish()` — drains `(topic, bytes)` channel → `gossipsub_publish(topic, data)`; warns on >1 MB payloads
  - `get_node_zid()` — random `NodeId` via `os.urandom(16).hex()` (keypair persistence is commented out)

### `src/exo/routing/event_router.py`
Per-node event ingestion, sequencing, and NACK logic — the heart of event sourcing.

- **`EventRouter(dataclass)`** — fields: `session_id`, `command_sender`, `external_inbound` (← GLOBAL_EVENTS), `external_outbound` (→ LOCAL_EVENTS), `_system_id`, `internal_outbound` (local consumers: Master/Worker/API), `event_buffer` (`OrderedBuffer[Event]`), `out_for_delivery`, NACK retry state.
  - `sender()` — returns a `Sender[Event]`; a background `_ingest` task wraps each event in `LocalForwarderEvent(origin_idx, origin=_system_id, session, event)` → `external_outbound`, tracking in `out_for_delivery`
  - `_simple_retry()` — every ~1–2s re-sends events older than 5s not yet acked
  - `_run_ext_in()` — consumes `GlobalForwarderEvent`s; filters to this session + master-originated; ingests into `OrderedBuffer` keyed by `origin_idx`; drains contiguous events to `internal_outbound` as `IndexedEvent(idx, event)`
  - `_nack_request(since_idx)` — on gaps, exponential backoff (0.5→10s cap) then sends `ForwarderCommand(origin, RequestEventLog(since_idx))` to the master
- Uses tagged `EventRouterClosedResourceError`/`EventRouterBrokenResourceError` so consumers distinguish shutdown from failure.

### `src/exo/routing/connection_message.py`
`ConnectionMessage(FrozenModel)` with `connected: bool`, built from `FromSwarm.Connection`. Consumed by Election to trigger re-elections.

---

## 3. Shared Layer (`src/exo/shared/`)

### `apply.py` — pure event application
The **only** place state is mutated. `apply(state, IndexedEvent)` asserts `state.last_event_applied_idx == event.idx - 1` (strict ordering), calls `event_apply(event.event, state)`, returns `state.model_copy(update={"last_event_applied_idx": event.idx})`.

**`event_apply()` dispatch:**
- **Pass-through:** `TestEvent`, `ChunkGenerated`, `TaskAcknowledged`, `InputChunkReceived`, `TracesCollected`, `TracesMerged`
- **Mutating handlers** (pure functions returning copied `State`):
  - `apply_custom_model_card_added/deleted`
  - `apply_instance_created/deleted` (also prunes `instance_links`)
  - `apply_node_timed_out`, `apply_node_download_progress` (dedupes by model_id), `apply_node_gathered_info` (updates node_* maps, topology, backends, thunderbolt info)
  - `apply_runner_status_updated` (updates `state.runners`; `RunnerShutdown` removes runner)
  - `apply_task_created/deleted/status_updated/failed`
  - `apply_topology_edge_created/deleted` (via `state.topology.add_connection/remove_connection`)
  - `apply_instance_link_created/deleted`

### `src/exo/shared/types/state.py` — `State`
Immutable global cluster state snapshot, replicated identically on every node.
Key fields: `instances`, `runners`, `downloads`, `tasks`, `last_seen`, `topology` (custom serialized to `TopologySnapshot`), `last_event_applied_idx`, plus granular node maps: `node_identities`, `node_memory`, `node_disk`, `node_system`, `node_network`, `node_thunderbolt`, `node_thunderbolt_bridge`, `node_rdma_ctl`, `node_backends`, `thunderbolt_bridge_cycles`, `instance_links`, `prefill_server_ports`, `custom_model_cards`.

### `src/exo/shared/types/` — the message vocabulary
- **`common.py`** — `Id` (uuid4-backed str newtype), `NodeId`, `SystemId`, `ModelId` (`normalize()` replaces `/`→`--`), `CommandId`, `TruncatingString`, `SessionId(master_node_id, election_clock)`, `Host(ip, port)`
- **`events.py`** — all events extend `BaseEvent(TaggedModel)` with `event_id` + `_master_time_stamp`. Union includes: `TaskCreated/Acknowledged/Deleted/StatusUpdated/Failed`, `InstanceCreated/Deleted`, `RunnerStatusUpdated`, `NodeTimedOut`, `NodeGatheredInfo`, `NodeDownloadProgress`, `ChunkGenerated`, `InputChunkReceived`, `TopologyEdgeCreated/Deleted`, `CustomModelCardAdded/Deleted`, `TracesCollected/Merged`, `InstanceLinkCreated/Deleted`, and wrappers `GlobalForwarderEvent` / `LocalForwarderEvent` / `IndexedEvent`
- **`commands.py`** — `BaseCommand(TaggedModel)` with `command_id`. Commands: `TextGeneration`, `ImageGeneration`, `ImageEdits`, `PlaceInstance`, `CreateInstance`, `DeleteInstance`, `TaskCancelled`, `TaskFinished`, `SendInputChunk`, `RequestEventLog`, download commands (`StartDownload`, `DeleteDownload`, `CancelDownload`), `AddCustomModelCard`/`DeleteCustomModelCard`, `SetInstanceLink`/`DeleteInstanceLink`. Wrapped as `ForwarderCommand(origin, command)` / `ForwarderDownloadCommand(origin, command)`
- **`tasks.py`** — `TaskId`, `TaskStatus` enum (`Pending/Running/Complete/TimedOut/Failed/Cancelled`), `CANCEL_ALL_TASKS` sentinel. Worker-emitted: `CreateRunner`, `DownloadModel`, `LoadModel`, `ConnectToGroup`, `StartWarmup`, `Shutdown(runner_id)`, `CancelTask`. Master-emitted: `TextGeneration`, `ImageGeneration`, `ImageEdits`. Union `Task`, `GenerationTask`
- **`chunks.py`** — streaming chunks: `TokenChunk`, `ErrorChunk` (with runner diagnostics), `ToolCallChunk`, `ImageChunk`, `InputImageChunk` (chunked base64 for image editing)
- **`text_generation.py`** — `TextGenerationTaskParams` (model, prompt, max tokens, temperature, tools, images, `image_hashes`, reasoning dialect, etc.) + `resolve_reasoning_params()`
- **`worker/instances.py`** — `InstanceMeta` (`MlxRing`, `MlxJaccl`), `MlxRingInstance` (hosts_by_node, ephemeral_port), `MlxJacclInstance` (jaccl_devices, jaccl_coordinators), union `Instance`, `BoundInstance(instance, bound_runner_id, bound_node_id)` with `bound_shard`
- **`worker/runners.py`** — runner status FSM: `RunnerIdle → Connecting → Connected → Loading → Loaded → WarmingUp → Ready → Running`, plus `RunnerShuttingDown`, `RunnerShutdown`, `RunnerFailed` (with `termination_cause`, `exit_code`, `diagnostics`). `ShardAssignments(model_id, runner_to_shard, node_to_runner)`
- **`worker/shards.py`** — `Sharding` enum (`Pipeline`, `CFG`, `Tensor`), `BaseShardMetadata` (model_card, device_rank, world_size), `PipelineShardMetadata` (start/end/n_layers), `CfgShardMetadata`, `TensorShardMetadata` (head_shards, expert_shards)
- **`worker/downloads.py`** — `DownloadProgress` variants `DownloadPending/Ongoing/Completed/Failed`
- **`worker/runner_response.py`** — `ModelLoadingResponse`, `GenerationResponse`, `CancelledResponse`, `FinishedResponse`
- **`instance_link.py`** — `InstanceLinkId`, `InstanceLink(link_id, prefill_instances, decode_instances)` — disaggregated prefill/decode
- Also `backends.py` (`Backend` enum: MlxMetal/MlxCuda/MlxCpu), `profiling.py` (node hardware/network/RDMA/thunderbolt models), `topology.py` (`Connection`, `SocketConnection`, `RDMAConnection`, `Cycle`), `memory.py`, `multiaddr.py`, `thunderbolt.py`

### `src/exo/shared/election.py` — bully-style master election
- **`ElectionMessage(FrozenModel)`** — `clock`, `seniority`, `proposed_session`, `commands_seen`. `__lt__` comparison: clock → seniority → commands_seen → master_node_id
- **`ElectionResult(FrozenModel)`** — `session_id`, `won_clock`, `is_new_master`
- **`Election`** — each node starts as its own master (`current_session = SessionId(node_id, 0)`).
  - `run()` — spawns `_election_receiver`, `_connection_receiver`, `_command_counter`, then runs an initial campaign with timeout 0 (instantly resolves to itself)
  - `_election_receiver()` — on a message with a **newer clock**, starts a new campaign (`DEFAULT_ELECTION_TIMEOUT=3.0s`); appends same-clock messages to candidates; drops stale/self messages
  - `_connection_receiver()` — on any `ConnectionMessage`, bumps `clock += 1` and starts a new campaign
  - `_command_counter()` — counts `ForwarderCommand`s seen (tie-breaker: busiest node wins)
  - `_campaign(candidates, timeout)` — posts `_election_status(clock)`, sleeps, rebroadcasts, elects `max(candidates)`. Winners raise seniority to `max(seniority, len(candidates))`
  - `elect(em)` — emits `ElectionResult` to the receiver consumed by `Node._elect_loop`

### `src/exo/shared/topology.py`
`Topology` — mutable `rustworkx.PyDiGraph[NodeId, SocketConnection|RDMAConnection]` wrapper. Methods: `add_node`, `neighbours`, `out_edges`, `add_connection`, `remove_connection`, `list_nodes`, `get_cycles()` (returns `Cycle` objects), `node_is_leaf`, `to_snapshot()`/`from_snapshot()`.

### `src/exo/shared/constants.py`
All paths/env vars: `EXO_HOME`, XDG dirs, `EXO_DEFAULT_MODELS_DIR`, `EXO_MODELS_DIRS`/`EXO_MODELS_READ_ONLY_DIRS` (colon-separated; a dir in both lists is treated as read-only), `RESOURCES_DIR`, `DASHBOARD_DIR`, `EXO_LOG_DIR`, `EXO_RUNNER_STDOUT/STDERR_LOG`, `EXO_PID_FILE`, `EXO_NODE_ZID`, `EXO_CONFIG_FILE`, `EXO_MAX_CHUNK_SIZE=512*1024`, `EXO_CUSTOM_MODEL_CARDS_DIR`, `EXO_EVENT_LOG_DIR`, `EXO_IMAGE_CACHE_DIR`, `EXO_TRACING_CACHE_DIR`, feature flags (`EXO_ENABLE_IMAGE_MODELS`, `EXO_OFFLINE`, `EXO_TRACING_ENABLED`, `ENABLE_DISAGGREGATION`), `EXO_MAX_CONCURRENT_REQUESTS=8`, `EXO_MAX_INSTANCE_RETRIES=5`.

### `src/exo/shared/tracing.py`
Chrome-trace-format instrumentation for distributed inference:
- `TraceEvent(name, start_us, duration_us, rank, category)` — frozen dataclass
- `trace(name, rank, category)` — context manager; nested traces inherit parent category (e.g. `sync/compute`)
- `get_trace_buffer()` / `clear_trace_buffer()` — global per-rank buffer
- `export_trace(traces, path)` — writes Chrome trace JSON (`ph: "X"` complete events + `thread_name` metadata)
- `load_trace_file(path)` / `compute_stats(traces)` — read back + aggregate (`TraceStats` with `by_category`/`by_rank`)
- Gated by `EXO_TRACING_ENABLED`

### `src/exo/shared/logging.py`
- `logger_setup(log_file, verbosity)` — loguru config; stdlib loggers (exo_rs, networking, httpx, httpcore) intercepted via `_InterceptHandler`; file rotation with zstd compression (`_zstd_compress`, `_MAX_LOG_ARCHIVES=5`)
- `logger_cleanup()` — flushes loguru queues
- `InterceptLogger` — hypercorn logger subclass routing to loguru

### `src/exo/shared/models/model_cards.py`
`ModelCard` — TOML-based model definitions loaded from `resources/inference_model_cards` + `resources/image_model_cards` + user custom dir. `_CardCache` (global `card_cache`) with `refresh/save/pop/list_all`. `ModelTask` enum (TextGeneration, TextToImage, ImageToImage), `Backend` support, `detect_vision_from_config`, `ConfigData` (HF config parsing with `supports_tensor` architecture allowlist), `fetch_from_hf` (pure fetch — persistence via event sourcing), `SamplingValues`/`SamplingDefaults` (thinking/non-thinking sampling presets).

---

## 4. Master (`src/exo/master/`)

### `main.py` — `Master`
Command processor + event indexer. Receives commands, produces events, orders them globally, persists, broadcasts.

**Wiring:** `command_receiver` (← COMMANDS), `event_sender` (→ EventRouter → LOCAL_EVENTS), `local_event_receiver` (← LOCAL_EVENTS), `global_event_sender` (→ GLOBAL_EVENTS), `download_command_sender` (→ DOWNLOAD_COMMANDS).

**Key methods:**
- `run()` — starts `_event_processor`, `_command_processor`, `_plan`
- `_command_processor()` — the command dispatch match:
  - `TextGeneration` → picks least-loaded **decode instance** (non-prefill-only) matching the model, computes `prefill_endpoint` via `_prefill_endpoint_for()`, emits `TaskCreated(TextGenerationTask)`; records `command_task_mapping[command_id] = task_id`
  - `ImageGeneration` / `ImageEdits` — same pattern → `TaskCreated` (with tracing rank tracking if `EXO_TRACING_ENABLED`)
  - `DeleteInstance` / `PlaceInstance` / `CreateInstance` — call placement functions, get `transition_events`, send download cancellations for unneeded shards
  - `SendInputChunk` → `InputChunkReceived` event
  - `TaskCancelled` → `TaskStatusUpdated(Cancelled)`; `TaskFinished` → `TaskDeleted` + pop mapping
  - `AddCustomModelCard`/`DeleteCustomModelCard` → `CustomModelCardAdded/Deleted`
  - `SetInstanceLink`/`DeleteInstanceLink` → `InstanceLinkCreated/Deleted`
  - `RequestEventLog(since_idx)` → reads `DiskEventLog` and sends up to 1000 `IndexedEvent`s directly
- `_event_processor()` — consumes `LocalForwarderEvent`s (own session only), feeds `MultiSourceBuffer`, stamps `_master_time_stamp`, computes `idx = len(self._event_log)`, **applies to state**, appends to `DiskEventLog`, broadcasts `GlobalForwarderEvent(origin, origin_idx=idx, session, event)` on GLOBAL_EVENTS. `TracesCollected` events buffered per rank and merged when all expected ranks arrive
- `_plan()` — every 10s: deletes instances whose nodes dropped out of the topology; emits `NodeTimedOut` for nodes unseen >30s
- `_prefill_endpoint_for(state, decode_instance_id)` — for disaggregated decode: finds linked prefill instances, picks the least-busy one (fewest in-flight tasks), finds its runner's `prefill_server_ports[runner_id]`, and returns `"{ip}:{port}"` using `find_ip_prioritised` (ring priority)

### `placement.py` — instance placement decisions
`place_instance(command, topology, instances, node_memory, node_network, node_backends, ...)`:
1. Finds topology **cycles** ≥ `min_nodes` with sufficient memory (`filter_cycles_by_memory`)
2. Filters by `required_nodes`
3. Validates tensor/pipeline constraints (hidden_size & kv_heads divisibility, DeepSeek V3.1 8-bit no-pipeline, Gemma 4 single-node)
4. Picks smallest cycles (`get_smallest_cycles`)
5. Filters by backend support (`MlxRing`/`MlxJaccl`)
6. Prefers RDMA-enabled nodes when applicable
7. Produces the new `Instance` with shard assignments (favoring nodes that already have downloads via `_cycle_download_score`)

Also: `add_instance_to_placements` (naive add for `CreateInstance`), `delete_instance`, `get_transition_events` (produces `InstanceCreated/InstanceDeleted/TaskStatusUpdated` events to reconcile tasks), `cancel_unnecessary_downloads`.

### `placement_utils.py` — sharding math
- `get_shard_assignments` → dispatch by `Sharding`:
  - `get_shard_assignments_for_pipeline_parallel` — layers split proportionally to RAM (`allocate_layers_proportionally`, largest-remainder method)
  - `_get_shard_assignments_for_cfg_parallel` — CFG parallel
  - `_get_shard_assignments_for_pure_pipeline`
  - `get_shard_assignments_for_tensor_parallel` — heads + MoE experts split
- `find_ip_prioritised` (network path selection for rings), `get_mlx_ring_hosts_by_node`, `get_mlx_jaccl_devices_matrix`, `get_mlx_jaccl_coordinators`

### `image_store.py` — `ImageStore`
Stores generated images (`store`, `get`, `list_images`, `cleanup_expired`) on disk under `EXO_IMAGE_CACHE_DIR` for `/images/{id}` serving. Images expire after a TTL; `_cleanup_expired_images` runs every 5 minutes in the API.

---

## 5. Worker (`src/exo/worker/`)

### `main.py` — `Worker`
Local executor: mirrors global state, plans runner lifecycle, drives model downloads, streams chunks.

**Wiring:** `event_receiver` (← EventRouter internal, `IndexedEvent`s), `event_sender` (→ EventRouter → LOCAL_EVENTS), `command_sender` (→ COMMANDS), `download_command_sender` (→ DOWNLOAD_COMMANDS), `api_port`.

**Key methods:**
- `run()` — starts `InfoGatherer` (hardware profiling → `NodeGatheredInfo` events), `_forward_info`, `plan_step`, `_event_applier`, `_poll_connection_updates`, `_reconcile_custom_cards`
- `_event_applier()` — `self.state = apply(self.state, event)` for each `IndexedEvent`; buffers `InputChunkReceived` chunks (assembles `Base64Image`s into `image_cache` by SHA256 hash); resets instance backoff on `InstanceDeleted`
- `plan_step()` — every 0.1s calls `plan(...)`; on non-None task sends `TaskCreated` then executes:
  - `CreateRunner` → `_create_supervisor(task)` (spawns `RunnerSupervisor`), marks complete; retries bounded by `_instance_backoff`/`EXO_MAX_INSTANCE_RETRIES` (then requests `DeleteInstance`)
  - `DownloadModel` → `resolve_existing_model()`; if found emits `DownloadCompleted` + complete; else sends `StartDownload` to the download coordinator and marks running
  - `Shutdown(runner_id)` → stops runner process (3s fail_after → `TimedOut`)
  - `CancelTask` → `runner.cancel_task`
  - `ImageEdits` with chunks → assembles image, `_start_runner_task`
  - `TextGeneration` with `image_hashes` → resolves cached images into task params, `_start_runner_task`
  - `LoadModel`/other → `_start_runner_task(task)` → `runners[instance.shard_assignments.node_to_runner[node_id]].start_task(task)`
- `_poll_connection_updates()` — every 10s pings neighbors via `check_reachable()` (HTTP probe of the API port) and emits `TopologyEdgeCreated/Deleted`
- `_reconcile_custom_cards()` — syncs custom model cards to disk cache

### `plan.py` — worker planning FSM
`plan(...)` returns the next `Task` or None using short-circuit OR of priorities:
1. `_cancel_tasks` — cancelled task → `CancelTask`
2. `_kill_runner` — instance deleted or runner failed (self or peer) → `Shutdown`
3. `_create_runner` — instance assigned to this node without a runner → `CreateRunner`
4. `_model_needs_download` → `DownloadModel` (respecting download backoff)
5. `_init_distributed_backend` — `ConnectToGroup` (MLX distributed init)
6. `_load_model` — all shards downloaded → `LoadModel`
7. `_ready_to_warmup` → `StartWarmup`
8. `_pending_tasks` — runnable `TextGeneration`/`ImageGeneration`/`ImageEdits` tasks → forward to runner

### `runner/supervisor.py`
- **`RunnerStdioHandler`** — tees runner stdout/stderr to log files and feeds lines to `RunnerDiagnosticCollector` (log mining for `RunnerMetalGpuTimeout` etc.)
- **`RunnerSupervisor`** — owns the runner **subprocess** (`AsyncProcess`), created via `create()` which sets up `mp_channel`s: `ev_send/ev_recv` (events ← runner), `task_sender` (→ runner), `cancel_sender` (→ runner).
  - `start_task(task)` — sends task, waits for `TaskAcknowledged` (anyio Event) or 60s prefill/5s decode timeouts
  - `cancel_task(task_id)` — sends to cancel channel with 0.5s timeout
  - `_forward_events()` — relays `Event | RunnerTerminationError` from the runner to the Worker; tracks `pending`/`in_progress`/`completed` task sets
  - `_watch_runner()` — every 5s checks `runner_process.is_alive()`
  - `_check_runner()` — on death, records exit code/diagnostics, sends `ErrorChunk` for all in-progress generation tasks

### `runner/bootstrap.py`
`entrypoint(bound_instance, event_sender, task_receiver, cancel_receiver, logger)` — subprocess main: sets MLX fast-synch env, imports `Runner`, picks the builder (`MfluxBuilder` for image models, else `MlxBuilder` with `apply_mlx_patches()`), runs `runner.main()`. On crash sends `RunnerTerminationError` and exits 1.

### `runner/runner.py`
**`Runner`** — synchronous engine driver in the subprocess.
- Loads `builder: Builder`; initial status `RunnerIdle`
- `main()` — background `_task_reader_thread` drains `task_receiver` into a `queue.Queue` (`WorkItem = Task | PrefillTask | _TaskStreamClosed`)
- `handle_first_task()` — state machine: `ConnectToGroup` (RunnerIdle→Connecting, `builder.connect()`), `LoadModel` (→Loading, `yield from builder.load()`), `StartWarmup` (→WarmingUp, `engine.warmup()`), generation tasks (→Running, `engine.submit()`), `Shutdown` (→ShuttingDown, `engine.close()`)
- `_start_prefill_server()` — if `ENABLE_DISAGGREGATION` and rank 0, spins a `PrefillServer` (threading TCP) on a random ephemeral port, enqueuing `PrefillTask`s; `serve_prefill` calls `engine.serve_prefill()`
- Every iteration pumps the engine `step()` and forwards `ChunkGenerated`/`TaskStatusUpdated`/`FinishedResponse` events

### `runner/llm_inference/`
- `batch_generator.py` — `BatchGenerator(Engine)` (continuous batching, `EXO_MAX_CONCURRENT_REQUESTS`) and `SequentialGenerator`; owns the `ExoBatchGenerator`, `GeneratorQueue`, KV prefix cache, tool-call parsing, disaggregated prefill wiring
- `model_output_parsers.py` — `apply_all_parsers`, `map_responses_to_chunks`, reasoning token counting, tool-call extraction (DeepSeek/`openai-harmony`)
- `tool_parsers.py` — `ToolParser` abstraction + `make_mlx_parser`

### `engines/` — inference backends
- **`base.py`** — `Engine` ABC: `warmup()`, `submit(task)`, `step()` (returns `(task_id, Chunk | CancelledResponse | FinishedResponse)` tuples), `close()`, `serve_prefill()`, `should_cancel()`. `Builder` ABC: `connect()`, `load()` (generator of `ModelLoadingResponse`), `build()`, `close()`
- **`mlx/`** — the main backend:
  - `builder.py` — `MlxBuilder`: `connect()` → `initialize_mlx()`; `load()` → `load_mlx_items()`; `build()` → `SequentialGenerator` or `BatchGenerator`
  - `utils_mlx.py` — `mlx_distributed_init` (creates `mx.distributed.Group` with hosts from the instance), `initialize_mlx`, `load_mlx_items`, `apply_chat_template`, `get_weights_size`; transformer 5.x monkey-patches
  - `auto_parallel.py` — `pipeline_auto_parallel`, `tensor_auto_parallel` (re-shards the MLX model across devices), `get_layers`
  - `generator/generate.py` — `mlx_generate`, `prefill`/`pipeline_parallel_prefill`, `warmup_inference`, logprobs extraction
  - `generator/batch_generate.py` — `ExoBatchGenerator` (batched prefill/decode loop with `_EngineTask` bookkeeping)
  - `cache.py` — `KVPrefixCache` and `copy_rotating_kv_cache`
  - `disaggregated/` — prefill/decode split: `server.py` (threading TCP `PrefillServer`, msgpack protocol), `client.py` (`PrefillResult`), `adapter.py` (KV-cache ↔ wire dtype), `serve.py`
  - `vision.py` — `VisionProcessor` (VLMs via `mlx-vlm`)
- **`image/`** — `builder.py` (`MfluxBuilder` for FLUX), `distributed_model.py` (`DistributedImageModel`), `models/` (`flux/` adapters & wrappers incl. Kontext, `qwen/` incl. image-edit adapters) — powered by a forked `mflux`

### `disaggregated/`
`protocol.py` (msgspec `Header`/`TensorBlob` framing, zstandard compression) + `server.py` (threading `PrefillServer` with `ResolveHandler`). Used when `ENABLE_DISAGGREGATION=1`.

---

## 6. API (`src/exo/api/`)

### `main.py` — `API`
FastAPI server (hypercorn ASGI) — OpenAI-compatible, Ollama, Anthropic, image generation, traces, downloads, instance management; serves the Svelte dashboard.

**Wiring:** `event_receiver` (← EventRouter internal `IndexedEvent`s), `command_sender` (→ COMMANDS), `download_command_sender` (→ DOWNLOAD_COMMANDS), `election_receiver` (← ELECTION_MESSAGES, for pausing during elections). Maintains local `state`, `_event_log` (`DiskEventLog`), per-command output queues.

**Complete route list (from `_setup_routes()`):**
```
# General
GET     /node_id                      → node id
GET     /state, /state/{path}         → full/clipped state JSON
GET     /events                       → all events from DiskEventLog (JSON array)
GET     /v1/feature-flags             → {"disaggregation": bool}
GET     /onboarding, POST /onboarding → onboarding state

# Instances
POST    /instance                     → create instance (memory check, force_override)
POST    /place_instance               → place instance (server placement logic)
GET     /instance/placement           → compute placement without creating
GET     /instance/previews            → all placement previews (sharding × meta × min_nodes)
GET     /instance/await               → SSE long-poll until instance exists
GET     /instance/{instance_id}       → get instance
DELETE  /instance/{instance_id}       → delete instance
GET/POST/PUT/DELETE /v1/instance-links → disaggregated prefill/decode links (gated by ENABLE_DISAGGREGATION)

# Models
GET     /models, /v1/models           → model cards (status=downloaded filter)
POST    /models/add                   → add custom model (fetches from HF)
DELETE  /models/custom/{model_id}     → delete custom model card
GET     /models/search                → HF search (mlx-community first, then all)

# Inference
POST    /v1/chat/completions          → OpenAI chat (stream/non-stream)
POST    /bench/chat/completions       → + generation stats + power usage
POST    /v1/messages                  → Anthropic Claude
POST    /v1/responses                 → OpenAI Responses
POST    /v1/cancel/{command_id}       → cancel command
POST    /v1/images/generations        → image gen (stream partials or collect)
POST    /bench/images/generations     → + stats + power
POST    /v1/images/edits              → img2img (multipart form)
POST    /bench/images/edits           → + stats + power
GET     /images                       → list stored images
GET     /images/{image_id}            → serve stored image

# Ollama compatibility
POST    /ollama/api/chat (+ /ollama/api/api/chat, /ollama/api/v1/chat)
POST    /ollama/api/generate
GET     /ollama/api/tags (+ aliases)
POST    /ollama/api/show
GET     /ollama/api/ps
GET     /ollama/api/version, HEAD /ollama/, HEAD /ollama/api/version

# Downloads
POST    /download/start               → start shard download
DELETE  /download/{node_id}/{model_id} → delete download
POST    /download/cancel              → cancel download

# Traces (EXO_TRACING_ENABLED)
GET     /v1/traces                    → list traces
GET     /v1/traces/{task_id}          → trace events
GET     /v1/traces/{task_id}/stats    → stats by category/rank
GET     /v1/traces/{task_id}/raw      → raw Chrome-trace JSON file
POST    /v1/traces/delete             → delete traces
```

**Key methods:**
- `_apply_state()` — applies `IndexedEvent`s; routes `ChunkGenerated` to the right command's queue (text vs image); closes streams on `InstanceDeleted`; saves `TracesMerged`
- `_token_chunk_stream(command_id)` — the internal low-level stream all adapters consume; on cancellation sends `TaskCancelled`; finally sends `TaskFinished` and cleans the queue
- `_send_text_generation_with_images()` — dedupes images by SHA-256 hash, sends images as chunked `SendInputChunk` commands (`EXO_MAX_CHUNK_SIZE` = 512KB)
- `_send_image_edits_command()` — reads UploadFile, base64-encodes, chunks, computes `image_strength` (0.7 high / 0.3 low fidelity)
- `_validate_model_has_instance()` — 404 if no instance for the model (with download-notification TODO)
- `_collect_text_generation_with_stats()` / `_collect_image_generation_with_stats()` — bench collection with `PowerSampler`
- `_pause_on_new_election()` — pauses request processing during elections (anyio Event gate in `_send`)
- `run_api()` — hypercorn `serve` with graceful shutdown (2s graceful / 3s shutdown timeouts), `InterceptLogger`
- Traces: `_get_trace_path` (path traversal guard), `list_traces`, `get_trace`, `get_trace_stats`, `get_trace_raw`, `delete_traces`

### `adapters/`
- `chat_completions.py` — OpenAI request → `TextGenerationTaskParams` (`chat_request_to_text_generation`), SSE stream (`generate_chat_stream`), response collection
- `claude.py`, `ollama.py`, `responses.py` — equivalent adapters for Anthropic, Ollama, OpenAI Responses

### `types/`
`api.py` (all request/response models incl. `PlaceInstanceParams`, `CreateInstanceParams`, `ImageGenerationTaskParams`, `ImageEditsTaskParams`, `AdvancedImageParams`, `normalize_image_size`), `claude_api.py`, `ollama_api.py`, `openai_responses.py`.

### `keepalive.py`
`with_sse_keepalive(generator, keepalive_message, interval)` — wraps SSE streams with periodic `: keep-alive` comments (10s) so proxies don't drop long generations.

---

## 7. Download Subsystem (`src/exo/download/`)

### `coordinator.py` — `DownloadCoordinator`
Per-node download manager driven by DOWNLOAD_COMMANDS.
- `run()` consumes `ForwarderDownloadCommand`s:
  - `StartDownload(target_node_id, shard_metadata)` — if targeting this node and not already active, calls `shard_downloader.ensure_shard(shard)` in a cancellable scope; only if it's the worker that requested it or the download is for an instance the node hosts
  - `DeleteDownload` / `CancelDownload` — cancel scopes + `delete_model`
- `_download_progress_callback()` — throttled (1s) `NodeDownloadProgress` events, with `resolve_existing_model()` for completed state

### `impl_shard_downloader.py`
- `exo_shard_downloader(offline)` → `SingletonShardDownloader(ResumableShardDownloader(...))`
- `SingletonShardDownloader` — dedupes concurrent `ensure_shard` calls per shard (asyncio task cache)
- `ResumableShardDownloader` — HuggingFace `snapshot_download` with `allow_patterns`, resumable, max 8 parallel downloads, `offline` mode
- Helpers `build_base_shard` / `build_full_shard`

### `download_utils.py`
`resolve_existing_model`, `select_download_dir` (picks a writable dir from `EXO_MODELS_DIRS` by free space), `delete_model`, `download_shard`, `fetch_file_list_with_cache`, `get_weight_map`, `download_file_with_retry` (retry-after handling), `calc_hash`, `is_model_directory_complete`, `is_read_only_model_dir`, `seed_models`, `get_download_headers`.

### `huggingface_utils.py` / `shard_downloader.py`
HF auth/headers helpers; `ShardDownloader` ABC (`ensure_shard`, `on_progress`, `get_shard_download_status`).

---

## 8. Utils (`src/exo/utils/`)

- **`channels.py`** — core async primitives: `Sender[T]`/`Receiver[T]` (anyio wrappers with `collect()`, `clone_sender`, `clone`, error overrides), `MpSender`/`MpReceiver` (multiprocessing-pipe-based channels with `send_async`/`send`/`__iter__`, used for runner IPC), `channel()`/`mp_channel()` factories. `ErrorOverride` patches exception types (ClosedResourceError, BrokenResourceError, EndOfStream, WouldBlock) so consumers can distinguish router shutdown from failure
- **`task_group.py`** — `TaskGroup` with `start_soon`/`queue` (deferring task starts until the group runs) and `cancel_tasks`
- **`event_buffer.py`** — `OrderedBuffer[T]` (in-order resequencing by index, `ingest`/`drain`/`drain_indexed`) and `MultiSourceBuffer[SourceId, T]` (per-source ordering for the master)
- **`disk_event_log.py`** — `DiskEventLog`: append-only event persistence. On-disk format: `[4-byte big-endian uint32 length][msgpack payload]` records. Uses a bounded LRU cache (128 entries) of event index → byte offset for efficient random access. `append`, `read_range`, `__len__`, `close`; rotates stale active files from previous sessions/crashes (max 5 archives). Serialization: msgpack → JSON → Pydantic (`validate_json`) for strict-mode round-trip correctness
- **`async_process.py`** — `AsyncProcess`: anyio-managed subprocess with stdout/stderr capture (64KB chunks) and graceful `stop()` (3s join → 5s terminate → 2s kill, 10 retries)
- **`keyed_backoff.py`** — `KeyedBackoff`: per-key exponential backoff with `attempts/record_attempt/reset`
- **`ports.py`** — `random_ephemeral_port` (49153–65535, avoids 52415)
- **`power_sampler.py`** — `PowerSampler`: samples `SystemPerformanceProfile` per node at 1 Hz, integrates energy via trapezoidal rule, splits prefill vs generation at `mark_prefill_done()`, returns `PowerUsage` (per-node avg/prefill/gen power + energy). Used by bench endpoints
- **`info_gatherer/`** — `InfoGatherer` + `macmon` (psutil+macOS metrics), `system_info`, `net_profile` (`check_reachable` for topology pings). Produces `GatheredInfo` (memory, disk, CPU/GPU, network interfaces, backends, thunderbolt bridge discovery, rdma_ctl status) — emitted as `NodeGatheredInfo` events for placement decisions
  - Poll intervals: macmon 1s, system_profiler thunderbolt 5s, thunderbolt bridge 10s, rdma_ctl 10s, network interfaces 10s, disk 30s, static info 60s, misc 60s
  - `NodeBackends.gather()` — always includes MlxCpu; MlxMetal on Darwin; MlxCuda + Vllm if NVML finds GPUs
  - `ThunderboltBridgeInfo.gather()` — detects Thunderbolt Bridge service via networksetup + ifconfig member analysis
  - `RdmaCtlStatus.gather()` — runs `rdma_ctl status`
- Others: `reactive.py` (`Reactive[T]`), `phantom.py` (`PhantomData`), `banner.py` (`print_startup_banner` — ASCII art + auto-open browser on first run, skips in macOS app), `dashboard_path.py` (`find_dashboard`, `find_resources` — repo or PyInstaller bundle), `fs.py` (delete/ensure/mkdir helpers), `pydantic_ext.py` (`FrozenModel` — camelCase aliases, strict, frozen, extra=forbid; `TaggedModel` — tagged-union wrap serialization)

---

## 9. Rust Layer

### `rust/networking/` — the mesh
- **`src/lib.rs`** — `is_valid_zid()`, `cfg()` (zenoh config: router mode, TCP listen, multicast scouting disabled, gossip multihop enabled, storage_manager plugin with in-memory `mem1` storage), `open()` (blake3 namespace hash → 8 bytes, zenoh session + background discovery loop; only lower-zid node dials), `Session` struct, `AbortOnDrop`
- **`src/discovery.rs`** — custom UDPv6 multicast peer discovery:
  - Multicast group `ff12::e0a1:de89`, magic bytes `"EXO"`
  - `Discovery::new(zid, namespace, listen_port, discovery_port)` — non-blocking socket2 IPv6 UDP socket, netwatcher callback joins multicast group per interface
  - `next()` — every 1s tick → `announce()`; on UDP receive → `respond()`; returns first valid `Discovered`
  - `announce()` — sends `Hello { nonce, namespace }` to every multicast interface addr
  - `respond()` — validates magic; on `Hello` replies `WhatsUp { nonce, zid, port_le }` (up to 5 retries, 300ms apart); on `WhatsUp` validates nonce == our last nonce, zid ≠ self, returns `Discovered { zid, addr }`
  - Wire types: `Header { magic, kind }`, `Hello { nonce, namespace }`, `WhatsUp { nonce, zid, port_le }` — all `#[repr(C)]`, `Pod`/`Zeroable` (bytemuck)
- **`src/swarm.rs`** — "Compat shim for the old libp2p code": `ToSwarm` (Subscribe/Unsubscribe/Publish commands), `FromSwarm` (Message/Discovered/Expired events), `Swarm::into_stream()` (liveliness token `live/{zid}` + subscriber `live/*`), `on_message()` (pub/sub on `topics/{topic}` with `CongestionControl::Block`, remote-only delivery)

### `rust/exo_rs/` — PyO3 bindings
- **`src/lib.rs`** — `#[pymodule(name = "exo_rs", gil_used = true)]`, `pyo3_log::init()`, multi-threaded tokio runtime via `pyo3_async_runtimes::tokio::init`, registers `pidfile_submodule` + `networking_submodule`. `ext` module: `ByteArrayExt`, `ResultExt`, `FutureExt` (`.allow_threads_py()`), `PyErrExt`, `PyResultExt`, `TokioRuntimeExt`, `TokioMpscSenderExt`/`TokioMpscReceiverExt`
- **`src/networking.rs`** — `NetworkingHandle` (Python class): `new(identity, namespace, listen_port, discovery_service_port)`, `recv() -> Awaitable[FromSwarm]`, `gossipsub_subscribe(topic)`, `gossipsub_unsubscribe(topic)`, `gossipsub_publish(topic, data)`. `FromSwarm`: `Connection { connected: bool }` and `Message { topic, data }`
- **`src/pidfile.rs`** — `Pidfile(path, mode)` (wraps `pidfile_rs::Pidfile`, libbsd/libutil `flopen`): `write()`, `as_raw_fd()`, `close()`. `PidfileError` — `#[pyclass(frozen, extends=PyException)]`
- **`src/allow_threading.rs`** — `AllowThreads<F>` — releases the GIL while the inner future is polled
- **`src/bin/stub_gen.rs`** — regenerates `exo_rs.pyi` (invoked by `just rust-rebuild`)
- **`exo_rs.pyi`** — generated Python surface: `__all__ = ["FromSwarm", "NetworkingHandle", "Pidfile", "PidfileError"]`

### Python call sites for Rust
| File | Usage |
|---|---|
| `src/exo/routing/router.py` | `NetworkingHandle.new(...)`, `handle.recv()`, `gossipsub_*` |
| `src/exo/routing/connection_message.py` | `FromSwarm.Connection` → `ConnectionMessage` |
| `src/exo/main.py` | `Pidfile(EXO_PID_FILE, 0o0600)`, `as_raw_fd()` for daemonization |
| `src/exo/utils/tests/test_pidfile.py` | Tests `Pidfile`/`PidfileError` |

---

## 10. Message Flow (End-to-End)

**Normal request flow:**
1. **Client → API** (`/v1/chat/completions`, Ollama, Claude, images) — API converts request to a `ForwarderCommand` (e.g. `TextGeneration`) on **COMMANDS**
2. **Master._command_processor** receives it, picks the least-loaded instance, creates `TaskCreated(TextGenerationTask)` event → `event_sender` → **LOCAL_EVENTS**
3. **Master._event_processor** indexes the event (assigns `idx`, applies to state, persists to `DiskEventLog`), broadcasts `GlobalForwarderEvent` on **GLOBAL_EVENTS**
4. **Every node's EventRouter** filters to its session, resequences via `OrderedBuffer` (NACKing gaps with `RequestEventLog`), delivers `IndexedEvent`s to Worker, API (and its own Master)
5. **Worker._event_applier** applies events to local state; **plan_step** drives the runner FSM (`CreateRunner → DownloadModel → ConnectToGroup → LoadModel → StartWarmup → tasks`); downloads go through **DOWNLOAD_COMMANDS** → `DownloadCoordinator`
6. **Runner** (subprocess, MLX engine) generates → emits `ChunkGenerated` events → Worker → EventRouter → **LOCAL_EVENTS** → Master indexes → **GLOBAL_EVENTS** → API's `_apply_state` routes chunks to the right streaming queue → SSE to client
7. **Election** — on peer connect/disconnect (`CONNECTION_MESSAGES`) or new clock, nodes campaign with `ElectionMessage`s; winner's `SessionId` wins; `Node._elect_loop` promotes/demotes masters, rebuilds EventRouter/Worker/DownloadCoordinator, resets the API

**Worker→Master direct channels:** `RequestEventLog` (gap recovery), `DeleteInstance` (retry exhaustion), `NodeGatheredInfo` (via LOCAL_EVENTS), `TopologyEdgeCreated/Deleted` (topology maintenance from ping probing).

---

## 11. Key Design Principles (from RULES.md)

- **Type-level discipline** — eliminate try-catch/if-statements at runtime using type-level discipline
- **Descriptive names** — no three-letter acronyms, no non-standard contractions; the average layman should understand a function from its signature
- **Pure functions** — same inputs → same outputs; no hidden state; side-effects only via injectable effect handlers
- **Pydantic everywhere** — `frozen=True` and `strict=True` ConfigDicts; inherit from `UUID4` for UUID classes; idempotency tags from salted hashes of persisted state
- **Event sourcing** — state is derived from an ordered event log; `apply()` is the sole mutator