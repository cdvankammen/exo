# EXO — Complete File-by-File Inventory

> **Companion to `ARCHITECTURE.md`.** This is the exhaustive, file-by-file map of the entire repository, generated from a full line-by-line read of every source file (2026-08-04). Use this as the definitive "where is X" reference.

---

## 1. Python Backend — `src/exo/`

### Entry points
| File | Purpose |
|---|---|
| `src/exo/__init__.py` | Package init |
| `src/exo/__main__.py` | Console bootstrap: `freeze_support()`, multiprocessing `-c` flag handling, calls `main()` |
| `src/exo/main.py` | `Args` (CLI), `Node` (composition root), `main()`/`main_inner()` (pidfile, daemonize, rlimit, spawn start method) |

### `src/exo/api/`
| File | Purpose |
|---|---|
| `main.py` | `API` class — FastAPI server, all routes, streaming, image store, traces, downloads, election pause/reset |
| `keepalive.py` | `with_sse_keepalive()` — periodic `: keep-alive` comments on SSE streams |
| `adapters/__init__.py` | Adapter package init |
| `adapters/chat_completions.py` | OpenAI Chat Completions ↔ `TextGenerationTaskParams`; `chat_request_to_text_generation`, `generate_chat_stream`, `collect_chat_response`, `fetch_image_url`, `extract_base64_from_data_url` |
| `adapters/claude.py` | Claude Messages ↔ internal; `claude_request_to_text_generation`, `generate_claude_stream`, `collect_claude_response`, `finish_reason_to_claude_stop_reason`, `_strip_volatile_headers` (removes Anthropic billing headers that break KV prefix caching) |
| `adapters/ollama.py` | Ollama chat/generate ↔ internal; `ollama_request_to_text_generation`, `ollama_generate_request_to_text_generation`, `generate_ollama_chat_stream`, `generate_ollama_generate_stream`, `collect_ollama_chat_response`, `collect_ollama_generate_response` |
| `adapters/responses.py` | OpenAI Responses ↔ internal; `responses_request_to_text_generation`, `generate_responses_stream`, `collect_responses_response` |
| `types/__init__.py` | Re-exports all API types |
| `types/api.py` | All request/response models: `ChatCompletionRequest/Response`, `PlaceInstanceParams`, `CreateInstanceParams`, `ImageGenerationTaskParams`, `ImageEditsTaskParams`, `AdvancedImageParams`, `BenchChatCompletionRequest/Response`, `BenchImageGenerationTaskParams/Response`, `ImageData`, `ImageListResponse`, `ModelList`, `HuggingFaceSearchResult`, `TraceListResponse`, `TraceResponse`, `TraceStatsResponse`, `PowerUsage`, `NodePowerStats`, `normalize_image_size` |
| `types/claude_api.py` | Claude wire types: `ClaudeMessagesRequest/Response`, `ClaudeMessage`, `ClaudeTextBlock`, `ClaudeImageBlock`, `ClaudeThinkingBlock`, `ClaudeToolUseBlock`, `ClaudeToolResultBlock`, `ClaudeToolDefinition`, streaming event types |
| `types/ollama_api.py` | Ollama wire types: `OllamaChatRequest/Response`, `OllamaGenerateRequest/Response`, `OllamaMessage`, `OllamaToolCall`, `OllamaOptions`, `OllamaTagsResponse`, `OllamaShowResponse`, `OllamaPsResponse`, `OllamaModelDetails`, `OllamaModelTag` |
| `types/openai_responses.py` | Responses wire types: `ResponsesRequest/Response`, `ResponseInputMessage`, `ResponseContentPart`, `FunctionCallInputItem`, `FunctionCallOutputInputItem`, `ReasoningInputItem`, `ComputerCallInputItem`, `WebSearchCallInputItem`, `FileSearchCallInputItem`, `CodeInterpreterCallInputItem`, `McpCallInputItem`, `CustomToolCallInputItem`, `ShellCallInputItem`, `ToolSearchCallInputItem`, `LocalShellCallInputItem`, `ApplyPatchCallInputItem`, `ImageGenerationCallInputItem`, `CompactionInputItem`, `ItemReferenceInputItem`, `McpApprovalRequestInputItem`, `McpApprovalResponseInputItem`, `McpListToolsInputItem`, streaming event types (`ResponseCreatedEvent`, `ResponseInProgressEvent`, `ResponseOutputItemAddedEvent`, `ResponseOutputItemDoneEvent`, `ResponseTextDeltaEvent`, `ResponseTextDoneEvent`, `ResponseFunctionCallArgumentsDeltaEvent`, `ResponseFunctionCallArgumentsDoneEvent`, `ResponseReasoningSummaryTextDeltaEvent`, `ResponseReasoningSummaryTextDoneEvent`, `ResponseReasoningSummaryPartAddedEvent`, `ResponseReasoningSummaryPartDoneEvent`, `ResponseContentPartAddedEvent`, `ResponseContentPartDoneEvent`, `ResponseCompletedEvent`), `ResponseUsage`, `InputTokensDetails`, `OutputTokensDetails` |
| `tests/test_api_error_handling.py` | OpenAI-style error format for HTTPException |
| `tests/test_cancel_command.py` | Cancel endpoint: 404 for unknown, 200 + sender.close() for active |
| `tests/test_chat_completions_stream.py` | OpenAI SSE delta wire-shape compliance (no disallowed nulls) |
| `tests/test_claude_api.py` | Claude adapter conversion + stop-reason mapping |
| `tests/test_claude_tool_use.py` | Claude tool_use blocks in stream/collect |
| `tests/test_instance_deleted_stream_cleanup.py` | `InstanceDeleted` closes active generation streams |
| `tests/test_openai_responses_api.py` | Responses request validation + stream/collect |

### `src/exo/master/`
| File | Purpose |
|---|---|
| `main.py` | `Master` — command processor (dispatch match), event indexer (`_event_processor`), `_plan` (10s: kill broken instances, timeout dead nodes), `_prefill_endpoint_for`, trace merge (`_handle_traces_collected`, `_merge_and_save_traces`), `RequestEventLog` replay (1000 at a time) |
| `placement.py` | `place_instance` (cycle finding, memory filtering, tensor/pipeline constraints, RDMA preference, download-score), `add_instance_to_placements`, `delete_instance`, `get_transition_events`, `cancel_unnecessary_downloads`, `_get_node_download_fraction`, `_cycle_download_score`, `INSTANCE_META_BACKENDS` |
| `placement_utils.py` | `filter_cycles_by_memory`, `get_smallest_cycles`, `allocate_layers_proportionally` (largest-remainder), `get_shard_assignments` (dispatch), `get_shard_assignments_for_pipeline_parallel`, `_get_shard_assignments_for_cfg_parallel` (2 pipelines, reversed ring), `_get_shard_assignments_for_pure_pipeline`, `get_shard_assignments_for_tensor_parallel`, `get_mlx_jaccl_devices_matrix`, `get_mlx_jaccl_coordinators`, `get_mlx_ring_hosts_by_node` (0.0.0.0 self, RFC 5737 placeholder for non-neighbors), `find_ip_prioritised` (ring: TB > ethernet > wifi; RDMA: ethernet > wifi > TB) |
| `image_store.py` | `ImageStore` — store/get/list/cleanup_expired images under `EXO_IMAGE_CACHE_DIR` |
| `tests/conftest.py` | Fixtures: `create_node_memory`, `create_node_network`, `create_socket_connection`, `create_rdma_connection` |
| `tests/test_master.py` | Master end-to-end: inject `NodeGatheredInfo`, verify `TaskCreated` for `TextGeneration` |
| `tests/test_placement.py` | Placement: memory filtering, transition events, instance creation/deletion |
| `tests/test_placement_utils.py` | Sharding math: layer allocation, CFG parallel, tensor parallel, jaccl coordinators |
| `tests/test_topology.py` | Topology: add/remove nodes & connections, cycles, leaves |

### `src/exo/worker/`
| File | Purpose |
|---|---|
| `main.py` | `Worker` — `run()` (InfoGatherer, `_forward_info`, `plan_step`, `_event_applier`, `_poll_connection_updates`, `_reconcile_custom_cards`), image chunk buffering, `_start_runner_task`, `_create_supervisor`, `_poll_connection_updates` (10s ping → topology edges) |
| `plan.py` | `plan()` FSM — short-circuit OR: `_cancel_tasks` → `_kill_runner` → `_create_runner` → `_model_needs_download` → `_init_distributed_backend` → `_load_model` → `_ready_to_warmup` → `_pending_tasks` |
| `runner/__init__.py` | Runner package init |
| `runner/bootstrap.py` | `entrypoint()` — subprocess main: MLX fast-synch env, `Runner` import, `MfluxBuilder`/`MlxBuilder` selection, `RunnerTerminationError` |
| `runner/diagnostics.py` | `RunnerDiagnosticCollector` — stderr log mining: `RunnerMetalGpuTimeout`, `RunnerRingTransportError`, `RunnerRingSocketReceivingError`, `RunnerUnknown` |
| `runner/runner.py` | `Runner` — subprocess engine driver: state machine (`ConnectToGroup`→`LoadModel`→`StartWarmup`→generation→`Shutdown`), `_start_prefill_server`, `handle_generation_tasks`, `send_chunk`, `ExitCode` enum |
| `runner/supervisor.py` | `RunnerSupervisor` (owns subprocess via `AsyncProcess`, mp channels), `RunnerStdioHandler` (tees stdout/stderr to logs + diagnostics), `PREFILL_TIMEOUT_SECONDS=60`, `DECODE_TIMEOUT_SECONDS=5` |
| `runner/image_models/__init__.py` | Empty (placeholder) |
| `runner/llm_inference/__init__.py` | LLM inference package init |
| `runner/llm_inference/batch_generator.py` | `SequentialGenerator` (Engine), `BatchGenerator` (Engine), `GeneratorQueue`, `_check_for_debug_prompts` (EXO_RUNNER_MUST_FAIL/OOM/TIMEOUT), `agree_on_tasks`/`agree_on_cancellations` (all-gather task ordering for TP) |
| `runner/llm_inference/model_output_parsers.py` | `apply_all_parsers` (dispatch by model type: GPT-OSS, DeepSeek V3.2/V4, thinking models, tool calls), `parse_thinking_models`, `parse_deepseek_v32`, `parse_deepseek_v4`, `parse_gpt_oss`, `parse_tool_calls`, `count_reasoning_tokens`, `map_responses_to_chunks` |
| `runner/llm_inference/tool_parsers.py` | `ToolParser` (start/end markers + inner parser), `make_mlx_parser`, `_coerce_tool_calls_to_schema` (JSON schema coercion) |
| `disaggregated/__init__.py` | Disaggregated package init |
| `disaggregated/protocol.py` | Wire protocol: `Header`, `TensorBlob`, `KVChunk`, `ArraysState`, `Done`, `ErrorMessage`, `Message` union; msgspec msgpack framing (4-byte length prefix); `write_frame`/`read_frame`/`write_header`/`read_header`/`write_message`/`read_message`/`write_kv_chunk`/`write_arrays_state`/`write_done`/`write_error` |
| `disaggregated/server.py` | `PrefillServer` (threading TCP), `_PrefillHandler`, `PrefillRequest` (msgspec), `ResolveHandler` callback, `write_request`/`read_request` |
| `engines/__init__.py` | Engines package init |
| `engines/base.py` | `Engine` ABC (`warmup`, `submit`, `step`, `close`, `serve_prefill`, `should_cancel`), `Builder` ABC (`connect`, `load`, `build`, `close`) |
| `engines/image/__init__.py` | Image engine package init |
| `engines/image/builder.py` | `MfluxBuilder` (Builder), `ImageEngine` (Engine), `_is_primary_output_node`, `_send_traces_if_enabled` |
| `engines/image/config.py` | `BlockType` (JOINT/SINGLE), `TransformerBlockConfig`, `ImageModelConfig` (block configs, default steps, sync steps, guidance scale) |
| `engines/image/distributed_model.py` | `DistributedImageModel` — adapter creation, layer slicing, `DiffusionRunner` init, `from_shard_metadata`, `get_steps_for_quality`, `generate` |
| `engines/image/generate.py` | `parse_size`, `warmup_image_generator`, `generate_image` (partial images, seeds, bench stats) |
| `engines/image/models/__init__.py` | `create_adapter_for_model`, `get_config_for_model` |
| `engines/image/models/base.py` | `ModelAdapter` ABC, `PromptData` ABC (prompt embeds, CFG data, conditioning latents, Kontext image IDs) |
| `engines/image/models/flux/__init__.py` | FLUX adapter package |
| `engines/image/models/flux/adapter.py` | FLUX model adapter (mflux) |
| `engines/image/models/flux/config.py` | FLUX config (block types, steps) |
| `engines/image/models/flux/kontext_adapter.py` | FLUX.1-Kontext adapter (image conditioning) |
| `engines/image/models/flux/wrappers.py` | FLUX transformer wrappers |
| `engines/image/models/qwen/__init__.py` | Qwen image adapter package |
| `engines/image/models/qwen/adapter.py` | Qwen-Image adapter |
| `engines/image/models/qwen/config.py` | Qwen-Image config |
| `engines/image/models/qwen/edit_adapter.py` | Qwen-Image-Edit adapter (img2img) |
| `engines/image/models/qwen/wrappers.py` | Qwen transformer wrappers |
| `engines/image/pipeline/__init__.py` | Diffusion pipeline package |
| `engines/image/pipeline/block_wrapper.py` | `JointBlockWrapper`, `SingleBlockWrapper`, `BlockWrapperMode` |
| `engines/image/pipeline/kv_cache.py` | Diffusion KV cache handling |
| `engines/image/pipeline/runner.py` | `DiffusionRunner` — PipeFusion: sync pipeline (initial timesteps) + async pipeline (patches), `CfgBranch`, `calculate_patch_heights`, `calculate_token_indices`, NaN sentinel cancellation |
| `engines/mlx/__init__.py` | MLX engine package init |
| `engines/mlx/auto_parallel.py` | `CustomMlxLayer`, `PipelineFirstLayer` (recv from prev rank), `PipelineLastLayer` (send to next), `patch_pipeline_model`, `pipeline_auto_parallel`, `tensor_auto_parallel`, `get_inner_model`, `get_layers`, `flush_prefill_sends`, `clear_prefill_sends`, `_pending_prefill_sends`; supports Llama, Qwen3/3.5/3.6/Next/MoE/VL, DeepSeek V3/V3.2/V4, GLM4-MoE/Lite, GPT-OSS, Kimi K2.5, MiniMax, Ministral3, Nemotron-H, Step3.5, Gemma4 |
| `engines/mlx/builder.py` | `MlxBuilder` — `connect` (initialize_mlx), `load` (load_mlx_items), `build` (SequentialGenerator or BatchGenerator based on `EXO_NO_BATCH`), tool parser setup |
| `engines/mlx/cache.py` | `KVPrefixCache` (LRU eviction by memory threshold), `CacheSnapshot`, `copy_rotating_kv_cache` (detached copies to avoid leaks), `_copy_arrays_cache`, `_copy_cache_list`, `_copy_compressor_branch`, `_copy_v4_cache`, `encode_prompt`, `make_kv_cache`, `cache_length`, `get_prefix_length`, `snapshot_ssm_states`, `has_non_kv_caches`, `is_non_trimmable_cache_entry`, `_default_memory_threshold` (0.85/0.80/0.75/0.70 by RAM) |
| `engines/mlx/constants.py` | `KV_GROUP_SIZE=32`, `ATTENTION_KV_BITS=4`, `MAX_TOKENS=32168`, `MAX_KV_SIZE=3200`, `KEEP_KV_SIZE=1600`, `QUANTIZE_MODEL_MODE="affine"`, `CACHE_GROUP_SIZE=64`, `DEFAULT_TOP_LOGPROBS=5`, `TRUST_REMOTE_CODE=True` |
| `engines/mlx/types.py` | `KVCacheType` (sequence of cache types incl. DeepseekV4Cache), `Model` (nn.Module wrapper with `__call__`) |
| `engines/mlx/utils_mlx.py` | `mlx_distributed_init` (ring: MLX_HOSTFILE; jaccl: MLX_IBV_DEVICES + MLX_JACCL_COORDINATOR), `initialize_mlx` (seed 42), `load_mlx_items`, `shard_and_load` (tensor/pipeline auto-parallel), `get_tokenizer`, `get_eos_token_ids_for_model` (Kimi: 163586; GLM-5: 154820/154827/154829; GLM: 151336/151329/151338; GPT-OSS: 200002/200012; Qwen3.5/3.6: 248046/248044; Gemma-4/3: 1/106/50), `load_tokenizer_for_model_id`, `apply_chat_template`, `fix_unmatched_think_end_tokens`, `mx_barrier`, `mx_all_gather_tasks`, `mx_any`, `system_prompt_token_count`, `render_chat_template`, `set_wired_limit_for_model`, `mlx_force_oom`, transformers 5.x monkey-patch (bytes_to_unicode) |
| `engines/mlx/vision.py` | `VisionProcessor` (mlx-vlm), `MediaRegion`, `VisionResult`, `prepare_vision`, `_run_processor` (variable-resolution vs fixed-token-budget), `_instantiate_projector`, `_patch_video_processor`, `_MLX_VLM_MODEL_TYPE_ALIASES` (kimi_k25/k26 → kimi_vl) |
| `engines/mlx/generator/__init__.py` | Generator package init |
| `engines/mlx/generator/generate.py` | `mlx_generate`, `prefill`, `pipeline_parallel_prefill`, `warmup_inference`, `extract_top_logprobs`, `ban_token_ids`, `eos_ids_from_tokenizer`, `patch_embed_tokens` (vision embedding injection incl. Gemma4 per-layer table), `REMOTE_PREFILL_MIN_TOKENS=1000`, `_MIN_PREFIX_HIT_RATIO_TO_UPDATE=0.5` |
| `engines/mlx/generator/batch_generate.py` | `ExoBatchGenerator` (wraps mlx `BatchGenerator`, `_EngineTask` bookkeeping, `has_work`), `_stop_sequences` |
| `engines/mlx/generator/remote_prefill.py` | `remote_prefill` — fetch KV cache from prefill server, `ingest_into_mlx_cache`, progress callbacks |
| `engines/mlx/patches/__init__.py` | Patches package init |
| `engines/mlx/patches/opt_batch_gen.py` | `BatchTopKLogprobs`, `set_needs_topk`, `take_ready_topk`, `_patched_step` (top-k logprobs in batch generation) |
| `engines/mlx/patches/standard_yarn_rope.py` | YarnRoPE patch to match vLLM inverse-frequency blending |
| `engines/mlx/disaggregated/__init__.py` | MLX disaggregated package init |
| `engines/mlx/disaggregated/adapter.py` | `send_mlx_kv_cache`, `write_cache_to_wire`, `chunk_to_mlx_nhd`, `inject_kv_chunk`, `inject_rotating_kv_chunk`, `inject_arrays_cache`, `array_to_bytes`/`bytes_to_array` (bf16 via uint16 bitcast), `bhsd_to_nhd`/`nhd_to_bhsd`, `wire_dtype_from_cache`, `mx_dtype_to_str`/`str_to_mx_dtype` |
| `engines/mlx/disaggregated/client.py` | `PrefillResult`, `remote_prefill_fetch` (TCP client, 60s timeout, 4MB recv buffer), `ingest_into_mlx_cache` |
| `engines/mlx/disaggregated/serve.py` | `run_prefill_for_request` — prefill server side: prefix cache lookup, `mlx_prefill`, cache update |
| `engines/mlx/disaggregated/tests/` | `test_end_to_end.py`, `test_mlx_adapter.py`, `test_protocol_roundtrip.py`, `test_server_drain.py` |
| `engines/mlx/tests/` | `test_batch_generate.py` (B=1 vs B=2 equivalence), `test_extract_top_logprobs.py` |
| `engines/mlx/vendor/__init__.py` | Vendor package init |
| `engines/mlx/vendor/deepseek_v4_encoding.py` | DeepSeek-V4 token templates (DSML, thinking, tool calls) |
| `engines/mlx/vendor/dsml_encoding.py` | DeepSeek V3.2/V4 DSML encoding: `encode_messages`, `parse_dsml_output`, `_INVOKE_PATTERN`, `_PARAM_PATTERN`, thinking-block fixes |
| `tests/__init__.py` | Worker tests package init |
| `tests/constants.py` | Test constants: NODE_A/B/C, RUNNER_1/2/3, INSTANCE_1/2, MODEL_A/B, TASK_1/2, COMMAND_1/2, SHUTDOWN_TASK_ID, etc. |
| `tests/unittests/conftest.py` | `FakeRunnerSupervisor`, `OtherTask`, `get_pipeline_shard_metadata`, `get_mlx_ring_instance`, `get_bound_mlx_ring_instance`, `get_shard_assignments` |
| `tests/unittests/test_download/` | (empty init) |
| `tests/unittests/test_image/test_cancellation.py` | Image cancellation: NaN sentinel protocol, `DiffusionRunner` cancel_checker |
| `tests/unittests/test_mlx/conftest.py` | `MockLayer`, `PipelineTestConfig`, `create_hostfile`, `DEFAULT_GPT_OSS_CONFIG` (gpt-oss-20b-MXFP4-Q8) |
| `tests/unittests/test_mlx/test_auto_parallel.py` | Pipeline first/last layer distributed test |
| `tests/unittests/test_mlx/test_distributed_fix.py` | GPT-OSS pipeline + tensor parallel distributed tests (slow) |
| `tests/unittests/test_mlx/test_kv_prefix_cache.py` | `get_prefix_length` + KV prefix cache tests |
| `tests/unittests/test_mlx/test_pipeline_prefill_callbacks.py` | Pipeline prefill callbacks match stream_generate |
| `tests/unittests/test_mlx/test_prefix_cache_architectures.py` | Prefix cache across model architectures (config reduction) |
| `tests/unittests/test_mlx/test_tokenizers.py` | Tokenizer loading for all supported models (downloads tokenizer files only) |
| `tests/unittests/test_mlx/test_tp_bit_exact.py` | Tensor parallel bit-exactness (llama, qwen3_5_moe) |
| `tests/unittests/test_plan/test_download_and_loading.py` | Plan: DownloadModel when shard not downloaded |
| `tests/unittests/test_plan/test_runner_lifecycle.py` | Plan: Shutdown when instance missing / sibling failed |
| `tests/unittests/test_plan/test_task_forwarding.py` | Plan: forward pending TextGeneration when runners ready |
| `tests/unittests/test_plan/test_warmup.py` | Plan: StartWarmup when all loaded/warming |
| `tests/unittests/test_runner/test_dsml_e2e.py` | DeepSeek DSML tool-call parsing end-to-end |
| `tests/unittests/test_runner/test_event_ordering.py` | Runner event ordering (tasks complete before ready) |
| `tests/unittests/test_runner/test_finish_reason_sse.py` | Finish reason → SSE mapping |
| `tests/unittests/test_runner/test_glm_tool_parsing.py` | GLM `<arg_key>`/`<arg_value>` tool parsing |
| `tests/unittests/test_runner/test_parse_gpt_oss.py` | GPT-OSS `<|channel|>`/`<|message|>`/`<|call|>` parsing |
| `tests/unittests/test_runner/test_parse_tool_calls.py` | Tool call parsing incl. unclosed calls |
| `tests/unittests/test_runner/test_runner_supervisor.py` | Supervisor emits ErrorChunk on runner death |
| `tests/unittests/test_runner/test_serve_prefill.py` | Prefill server producer path with KVPrefixCache |

### `src/exo/routing/`
| File | Purpose |
|---|---|
| `__init__.py` | Routing package init |
| `topics.py` | `PublishPolicy` (Never/Minimal/Always), `TypedTopic[T]`, topics: `GLOBAL_EVENTS`, `LOCAL_EVENTS`, `COMMANDS`, `ELECTION_MESSAGES`, `CONNECTION_MESSAGES` (Never), `DOWNLOAD_COMMANDS` |
| `router.py` | `TopicRouter[T]` (per-topic fan-out), `Router` (owns `NetworkingHandle`, `_networking_recv`, `_networking_publish`, `get_node_zid` = `os.urandom(16).hex()`) |
| `event_router.py` | `EventRouter` — `sender()`, `_ingest`, `_simple_retry` (5s), `_run_ext_in` (session filter, OrderedBuffer), `_nack_request` (exponential backoff 0.5→10s), `EventRouterClosedResourceError`/`EventRouterBrokenResourceError` |
| `connection_message.py` | `ConnectionMessage(connected: bool)` from `FromSwarm.Connection` |
| `tests/test_event_buffer.py` | OrderedBuffer: sequential, out-of-order, gaps |

### `src/exo/download/`
| File | Purpose |
|---|---|
| `coordinator.py` | `DownloadCoordinator` — `run()` (command processor + `_emit_existing_download_progress`), `_download_progress_callback` (1s throttle), `_completed_from_path`, `_default_model_dir` |
| `download_utils.py` | `HuggingFaceAuthenticationError`, `HuggingFaceRateLimitError` (429 with retry_after), `_parse_retry_after`, `_RATE_LIMIT_MAX_SLEEP_SECS=300`, `_FILE_LIST_CACHE_TTL_SECS=86400`, `trim_etag`, `map_repo_file_download_progress_to_download_progress_data`, `map_repo_download_progress_to_download_progress_data`, `InsufficientDiskSpaceError`, `resolve_existing_model`, `select_download_dir`, `delete_model`, `download_shard`, `fetch_file_list_with_cache`, `get_weight_map`, `download_file_with_retry`, `calc_hash`, `is_model_directory_complete`, `is_read_only_model_dir`, `seed_models`, `get_download_headers`, `create_http_session`, `build_model_path`, `resolve_model_dir`, `_download_file`, `_fetch_file_list`, `file_meta` |
| `huggingface_utils.py` | `filter_repo_objects`, `_add_wildcard_to_directories`, `get_hf_endpoint`, `get_hf_home`, `get_hf_token`, `get_auth_headers`, `get_allow_patterns` |
| `impl_shard_downloader.py` | `exo_shard_downloader(max_parallel=8, offline)`, `SingletonShardDownloader` (dedupe), `ResumableShardDownloader` (HF snapshot_download, allow_patterns, offline), `build_base_shard`, `build_full_shard` |
| `shard_downloader.py` | `ShardDownloader` ABC (`ensure_shard`, `on_progress`, `get_shard_download_status`, `get_shard_download_status_for_shard`), `NoopShardDownloader` |
| `tests/` | `test_cancel_download.py`, `test_download_status_not_lost.py` (regression #1918), `test_download_verification.py`, `test_model_dirs.py`, `test_offline_mode.py`, `test_rate_limit_handling.py`, `test_re_download.py`, `test_safetensors_index.py` |

### `src/exo/shared/`
| File | Purpose |
|---|---|
| `__init__.py` | Shared package init |
| `apply.py` | `apply(state, IndexedEvent)` — sole state mutator; `event_apply` dispatch; `apply_custom_model_card_added/deleted`, `apply_instance_created/deleted`, `apply_node_timed_out`, `apply_node_download_progress`, `apply_node_gathered_info` (incl. RDMA gating via `MacThunderboltConnections` + `RdmaCtlStatus`), `apply_runner_status_updated`, `apply_task_created/deleted/status_updated/failed`, `apply_topology_edge_created/deleted`, `apply_instance_link_created/deleted` |
| `constants.py` | All paths/env vars: `EXO_HOME`, XDG dirs, `EXO_DEFAULT_MODELS_DIR`, `EXO_MODELS_DIRS`/`EXO_MODELS_READ_ONLY_DIRS` (colon-separated, read-only wins), `RESOURCES_DIR`, `DASHBOARD_DIR`, `EXO_LOG_DIR`, `EXO_RUNNER_STDOUT/STDERR_LOG`, `EXO_PID_FILE`, `EXO_NODE_ZID`, `EXO_CONFIG_FILE`, `LIBP2P_*` (legacy), `EXO_MAX_CHUNK_SIZE=512*1024`, `EXO_CUSTOM_MODEL_CARDS_DIR`, `EXO_EVENT_LOG_DIR`, `EXO_IMAGE_CACHE_DIR`, `EXO_TRACING_CACHE_DIR`, `EXO_ENABLE_IMAGE_MODELS`, `EXO_OFFLINE`, `EXO_TRACING_ENABLED`, `ENABLE_DISAGGREGATION`, `EXO_MAX_CONCURRENT_REQUESTS=8`, `EXO_MAX_INSTANCE_RETRIES=5` |
| `election.py` | `ElectionMessage` (clock, seniority, proposed_session, commands_seen; `__lt__`), `ElectionResult`, `Election` (bully: `_election_receiver`, `_connection_receiver`, `_command_counter`, `_campaign`, `DEFAULT_ELECTION_TIMEOUT=3.0`) |
| `logging.py` | `logger_setup` (loguru + `_InterceptHandler` for stdlib/hypercorn), `logger_cleanup`, `_zstd_compress` (rotation), `_MAX_LOG_ARCHIVES=5` |
| `topology.py` | `Topology` (rustworkx `PyDiGraph` wrapper: `add_node`, `neighbours`, `out_edges`, `add_connection`, `remove_connection`, `list_nodes`, `get_cycles`, `node_is_leaf`, `to_snapshot`/`from_snapshot`), `TopologySnapshot` |
| `tracing.py` | `TraceEvent`, `CategoryStats`, `TraceStats`, `trace()` context manager (hierarchical categories), `get_trace_buffer`, `clear_trace_buffer`, `export_trace` (Chrome trace format), `load_trace_file`, `compute_stats` |
| `models/model_cards.py` | `ModelCard` (TOML), `_CardCache` (global `card_cache`), `ModelTask` (TextGeneration/TextToImage/ImageToImage), `ComponentInfo`, `VisionCardConfig`, `SamplingValues`/`SamplingDefaults`, `ConfigData` (HF config parsing, `supports_tensor` architecture list), `detect_vision_from_config`, `_is_image_card`, `fetch_from_hf`, `load`, `save`, `save_to_custom_dir` |
| `types/__init__.py` | Types package init |
| `types/backends.py` | `Backend` enum: MlxMetal, MlxCpu, MlxCuda, Vllm |
| `types/chunks.py` | `BaseChunk`, `TokenChunk`, `ErrorChunk` (with diagnostics), `ToolCallChunk`, `ImageChunk` (chunked base64), `InputImageChunk`, `PrefillProgressChunk`, `StatusChunk`, `GenerationChunk`, `Chunk` |
| `types/commands.py` | `BaseCommand`, `TestCommand`, `TextGeneration`, `ImageGeneration`, `ImageEdits`, `PlaceInstance` (force_override), `CreateInstance`, `DeleteInstance`, `TaskCancelled`, `TaskFinished`, `SendInputChunk`, `RequestEventLog`, `StartDownload`, `DeleteDownload`, `CancelDownload`, `AddCustomModelCard`, `DeleteCustomModelCard`, `SetInstanceLink`, `DeleteInstanceLink`, `DownloadCommand`, `Command`, `ForwarderCommand`, `ForwarderDownloadCommand` |
| `types/common.py` | `Id` (uuid4 str), `NodeId`, `SystemId`, `ModelId` (normalize/short), `CommandId`, `TruncatingString`, `SessionId(master_node_id, election_clock)`, `Host(ip, port)` |
| `types/events.py` | `EventId`, `BaseEvent` (event_id, `_master_time_stamp`), `TestEvent`, `TaskCreated/Acknowledged/Deleted/StatusUpdated/Failed`, `InstanceCreated/Deleted`, `RunnerStatusUpdated`, `NodeTimedOut`, `NodeGatheredInfo`, `NodeDownloadProgress`, `ChunkGenerated`, `InputChunkReceived`, `TopologyEdgeCreated/Deleted`, `CustomModelCardAdded/Deleted`, `TraceEventData`, `TracesCollected/Merged`, `InstanceLinkCreated/Deleted`, `Event` union, `GlobalForwarderEvent`, `LocalForwarderEvent`, `IndexedEvent` |
| `types/instance_link.py` | `InstanceLinkId`, `InstanceLink(link_id, prefill_instances, decode_instances)` |
| `types/memory.py` | `Memory` — byte arithmetic with kb/mb/gb properties, comparison operators, `__str__` (GiB/MiB/KiB/B) |
| `types/multiaddr.py` | `Multiaddr` — `/ip4/.../tcp/...`, `/ip6/...`, `/dns.../...` validation, `ip_address`, `port` |
| `types/profiling.py` | `MemoryUsage`, `DiskUsage`, `SystemPerformanceProfile` (gpu_usage, temp, sys_power, pcpu/ecpu_usage), `InterfaceType`, `NetworkInterfaceInfo`, `NodeIdentity`, `NodeNetworkInfo`, `NodeThunderboltInfo`, `NodeRdmaCtlStatus`, `ThunderboltBridgeStatus` |
| `types/state.py` | `State` — instances, runners, downloads, tasks, last_seen, topology (custom serialization), last_event_applied_idx, node_identities/memory/disk/system/network/thunderbolt/thunderbolt_bridge/rdma_ctl/backends, thunderbolt_bridge_cycles, instance_links, prefill_server_ports, custom_model_cards |
| `types/tasks.py` | `TaskId`, `CANCEL_ALL_TASKS`, `TaskStatus` (Pending/Running/Complete/TimedOut/Failed/Cancelled), `BaseTask`, `CreateRunner`, `DownloadModel`, `LoadModel`, `ConnectToGroup`, `StartWarmup`, `TextGeneration`, `CancelTask`, `ImageGeneration`, `ImageEdits`, `Shutdown`, `Task` union, `TextTask`, `ImageTask`, `GenerationTask` |
| `types/text_generation.py` | `MessageRole`, `ReasoningEffort`, `ReasoningDialect` (none/post_last_user/suffix/channel/tool_conditional), `resolve_reasoning_params`, `InputMessageContent`, `InputMessage`, `Base64Image`, `Base64ImageHash`, `ChatTemplateValue`, `TextGenerationTaskParams` (canonical internal type), `with_card_sampling_defaults` |
| `types/thunderbolt.py` | `ThunderboltConnection`, `ThunderboltIdentifier`, `_ReceptacleTag`, `_ConnectivityItem`, `ThunderboltConnectivityData` (system_profiler parsing), `ThunderboltConnectivity.gather()` |
| `types/topology.py` | `Cycle`, `RDMAConnection`, `SocketConnection`, `Connection` |
| `types/worker/__init__.py` | Worker types package init |
| `types/worker/downloads.py` | `DownloadProgressData`, `BaseDownloadProgress`, `DownloadPending/Completed/Failed/Ongoing`, `DownloadProgress` union, `ModelSafetensorsIndexMetadata`, `ModelSafetensorsIndex`, `FileListEntry`, `RepoFileDownloadProgress`, `RepoDownloadProgress` |
| `types/worker/instances.py` | `InstanceId`, `InstanceMeta` (MlxRing/MlxJaccl), `BaseInstance`, `MlxRingInstance` (hosts_by_node, ephemeral_port), `MlxJacclInstance` (jaccl_devices, jaccl_coordinators), `Instance` union, `BoundInstance` (bound_shard, is_image_model) |
| `types/worker/runner_response.py` | `ModelLoadingResponse`, `GenerationResponse`, `ToolCallResponse`, `CancelledResponse`, `FinishedResponse` |
| `types/worker/runners.py` | `RunnerId`, `RunnerError`, `BaseRunnerStatus`, `RunnerIdle/Connecting/Connected/Loading/Loaded/WarmingUp/Ready/Running/ShuttingDown/Shutdown/Failed`, `RunnerStatus` union, `ShardAssignments` (model_id, runner_to_shard, node_to_runner) |
| `types/worker/shards.py` | `Sharding` (Tensor/Pipeline), `BaseShardMetadata` (model_card, device_rank, world_size, start/end/n_layers, immediate_exception, should_timeout), `PipelineShardMetadata`, `CfgShardMetadata` (cfg_rank, cfg_world_size, pipeline_rank, pipeline_world_size), `TensorShardMetadata`, `ShardMetadata` alias |
| `tests/` | `conftest.py`, `test_apply/` (custom_model_cards, instance_link, node_download, rdma_gating, runner_deleted), `test_election.py`, `test_node_id_persistence.py` (skipped — keypair disabled), `test_resolve_reasoning_params.py`, `test_state_serialization.py`, `test_xdg_paths.py` |

### `src/exo/utils/`
| File | Purpose |
|---|---|
| `__init__.py` | Utils package init (STDIO_FDS, STDOUT_FD, STDERR_FD) |
| `async_process.py` | `AsyncProcess` — anyio-managed subprocess with stdout/stderr capture, graceful stop (join 3s → terminate 5s → kill 2s), `_READ_CHUNK_SIZE=64KB` |
| `banner.py` | `print_startup_banner` — ASCII art + dashboard URL, auto-opens browser on first run (skips in macOS app via `EXO_RUNTIME_DIR`) |
| `channels.py` | `ErrorOverride` (exception patching), `Sender[T]`/`Receiver[T]` (anyio memory streams with error overrides), `MpSender`/`MpReceiver` (multiprocessing pipes), `channel()`/`mp_channel()` factories |
| `dashboard_path.py` | `find_resources`, `find_dashboard` (repo or PyInstaller bundle) |
| `disk_event_log.py` | `DiskEventLog` — append-only length-prefixed msgpack records, LRU offset cache (128), rotation, `_MAX_ARCHIVES=5`, `_serialize_event`/`_deserialize_event` (msgpack→JSON→Pydantic for strict-mode round-trip) |
| `event_buffer.py` | `OrderedBuffer[T]` (in-order resequencing), `MultiSourceBuffer[SourceId, T]` |
| `fs.py` | `delete_if_exists`, `ensure_parent_directory_exists`, `ensure_directory_exists`, `make_temp_path`, `StrPath`/`BytesPath` type aliases |
| `info_gatherer/__init__.py` | Info gatherer package init |
| `info_gatherer/info_gatherer.py` | `InfoGatherer` (macmon monitor, system_profiler thunderbolt, bridge status, rdma_ctl, memory, network, static info, misc, disk, backends), `StaticNodeInformation`, `NodeNetworkInterfaces`, `MacThunderboltIdentifiers`, `MacThunderboltConnections`, `RdmaCtlStatus`, `ThunderboltBridgeInfo`, `NodeConfig`, `MiscData`, `NodeDiskUsage`, `NodeBackends`, `GatheredInfo` union, `_get_thunderbolt_devices`, `_get_bridge_services`, `_get_bridge_members`, `_find_thunderbolt_bridge`, `_is_service_enabled`, `_gather_iface_map`, `_has_nvml_cuda` |
| `info_gatherer/macmon.py` | `RawMacmonMetrics`, `MacmonMetrics` (system_profile + memory from macmon JSON) |
| `info_gatherer/net_profile.py` | `check_reachability` (HTTP probe of `/node_id`, 3 attempts), `check_reachable` (concurrent probes, yields (ip, node_id)) |
| `info_gatherer/system_info.py` | `get_os_version`, `get_os_build_version`, `get_friendly_name`, `_get_interface_types_from_networksetup`, `get_network_interfaces`, `get_model_and_chip` |
| `info_gatherer/tests/test_tb_parsing.py` | Thunderbolt parsing test (macOS only) |
| `keyed_backoff.py` | `KeyedBackoff[K]` — per-key exponential backoff (`should_proceed`, `record_attempt`, `attempts`, `reset`) |
| `phantom.py` | `PhantomData[*T]` type (generic phantom data) |
| `ports.py` | `random_ephemeral_port` (49153–65535, avoids 52415) |
| `power_sampler.py` | `PowerSampler` — samples `SystemPerformanceProfile` per node, trapezoidal energy integration, prefill/generation split (`mark_prefill_done`), `result()` → `PowerUsage` |
| `pydantic_ext.py` | `FrozenModel` (camelCase aliases, strict, frozen, extra=forbid), `TaggedModel` (tagged-union serialization) |
| `reactive.py` | `Reactive[T]` (value + on_change callback), `OnChange` protocol |
| `task_group.py` | `TaskGroup` — deferred task queue (`queue`/`start_soon`), `cancel_tasks`, `cancel_called` |
| `tests/` | `conftest.py`, `test_async_process.py`, `test_channels.py`, `test_event_log.py`, `test_pidfile.py`, `test_power_sampler.py`, `test_tagged.py` |

---

## 2. Rust — `rust/`

### `rust/networking/`
| File | Purpose |
|---|---|
| `Cargo.toml` | zenoh (internal/plugins/unstable), storage-manager, netwatcher, bytemuck, socket2, blake3, parking_lot, tokio, async-stream, futures-lite, rand, log |
| `src/lib.rs` | `is_valid_zid`, `cfg` (router mode, TCP listen, multicast scouting off, gossip multihop on, storage_manager mem1), `open` (blake3 namespace → 8 bytes, discovery loop, lower-zid dials), `Session`, `AbortOnDrop` |
| `src/discovery.rs` | `Discovery` — UDPv6 multicast (`ff12::e0a1:de89`), `Hello`/`WhatsUp` protocol, netwatcher interface tracking, `next()`, `announce()`, `respond()`, wire types (`Header`, `Hello`, `WhatsUp`), `Kind` enum (`Hello = 0`, `WhatsUp = 1` with `TryFrom<u8>`; invalid discriminants → `UnknownKind`) |
| `src/swarm.rs` | `ToSwarm` (Subscribe/Unsubscribe/Publish), `FromSwarm` (Message/Discovered/Expired), `Swarm::into_stream` (liveliness `live/{zid}` + `live/*`), `on_message` (pub/sub `topics/{topic}`, CongestionControl::Block, remote-only), `create_swarm` |
| `examples/do_nothing.rs` | Minimal session + liveliness |
| `examples/heartbeat.rs` | Liveliness token + put loop |
| `examples/put_string.rs` | Pub throughput benchmark (5000 puts) |
| `examples/serve_storage.rs` | Storage plugin demo (get/subscribe) |
| `examples/z_get.rs` | Query network keys |

### `rust/exo_rs/`
| File | Purpose |
|---|---|
| `Cargo.toml` | cdylib+rlib, pyo3 (experimental-async), pyo3-stub-gen, pyo3-async-runtimes, pyo3-log, pidfile-rs, tokio, futures-lite, pin-project, zenoh, rand, serde_json, parking_lot |
| `pyproject.toml` | maturin build, module-name `exo_rs`, requires-python >=3.13 |
| `README.md` | TODO placeholder |
| `src/lib.rs` | `#[pymodule(name="exo_rs", gil_used=true)]`, tokio runtime init, `pidfile_submodule`, `networking_submodule`, `ext` module (ByteArrayExt, ResultExt, FutureExt, PyErrExt, PyResultExt, TokioRuntimeExt, TokioMpscSenderExt/ReceiverExt) |
| `src/networking.rs` | Rust types: `PyNetworkingHandle` (the class exported as `NetworkingHandle`: new/recv/gossipsub_subscribe/gossipsub_unsubscribe/gossipsub_publish/connect_peer), `PyFromSwarm` (exported as `FromSwarm`: Connection/Message); internal `from_session` constructor used by tests |
| `src/pidfile.rs` | Rust types `PyPidfile` (exported as `Pidfile`: write/as_raw_fd/close) and `PyPidfileError` (exported as `PidfileError`, PyException subclass) |
| `src/allow_threading.rs` | `AllowThreads<F>` — GIL release across awaits |
| `src/bin/stub_gen.rs` | Regenerates `exo_rs.pyi` |
| `exo_rs.pyi` | Generated stubs: `FromSwarm`, `NetworkingHandle`, `Pidfile`, `PidfileError` |
| `tests/dummy.rs` | mpsc channel drop/heartbeat test |
| `tests/test_python.py` | Python: NetworkingHandle pub/sub, Pidfile lifecycle |

---

## 3. Frontend — `dashboard/`

### `src/lib/components/`
| File | Purpose |
|---|---|
| `ChatAttachments.svelte` | Uploaded file chips (image/text/pdf/audio) |
| `ChatForm.svelte` | Message input, file upload, image params, model selector |
| `ChatMessages.svelte` | Message list, scroll management, regenerate, token heatmap, prefill progress |
| `ChatModelSelector.svelte` | Model dropdown + AUTO_TIERS (5-tier frontier ranking) + `pickAutoModel` |
| `ChatSidebar.svelte` | Conversation list, search, rename, delete, debug/topology toggles |
| `ConnectionBanner.svelte` | "Connection lost — Reconnecting" banner |
| `DeviceIcon.svelte` | SVG device glyphs (Mac Studio/Mini/Book) with Apple logo |
| `FamilyLogos.svelte` | SVG logos per model family (Meta, Qwen, DeepSeek, OpenAI, GLM, MiniMax, Kimi, FLUX, NVIDIA, Google) |
| `FamilySidebar.svelte` | Family filter sidebar with display names |
| `HeaderNav.svelte` | Top nav with hash router, advanced link, download progress |
| `HuggingFaceResultItem.svelte` | HF search result row with add button |
| `ImageLightbox.svelte` | Fullscreen image viewer with download |
| `ImageParamsPanel.svelte` | Image gen params (size, quality, fidelity, format) |
| `IntegrationCard.svelte` | Copy-to-clipboard API config cards |
| `MarkdownContent.svelte` | Markdown + KaTeX + highlight.js rendering |
| `ModelCard.svelte` | Model launch card with memory estimate, download status, force-launch |
| `ModelFilterPopover.svelte` | Capability/size/downloaded filters |
| `ModelPickerGroup.svelte` | Grouped model variants with fit status |
| `ModelPickerModal.svelte` | Full model picker with family sidebar, filters, HF search |
| `PrefillDecodeDisaggregation.svelte` | Instance-link management UI |
| `PrefillProgressBar.svelte` | Prefill progress with ETA |
| `ToastContainer.svelte` | Toast notifications |
| `TokenHeatmap.svelte` | Token confidence heatmap (Apple-style) |
| `TopologyGraph.svelte` | D3 live topology graph with device glyphs, RDMA/TB indicators |
| `index.ts` | Component barrel export |

### `src/lib/stores/`
| File | Purpose |
|---|---|
| `app.svelte.ts` | Central store: topology, chat, instances, runners, downloads, placement previews, feature flags, traces, image gen, conversations, debug/topology modes |
| `favorites.svelte.ts` | Favorite models (localStorage `exo-favorite-models`) |
| `recents.svelte.ts` | Recent models (localStorage `exo-recent-models`, max 20) |
| `toast.svelte.ts` | Toast state (success/info 4s, warning/error 6s, persistent option) |

### `src/lib/types/` & `src/lib/utils/`
| File | Purpose |
|---|---|
| `types/files.ts` | File attachment types, PDF extraction (pdfjs-dist), ReadableStream async-iterator polyfill for Safari |
| `utils/clipboard.ts` | `copyText` (navigator.clipboard + execCommand fallback) |
| `utils/downloads.ts` | Download state parsing (tagged unions, model ID extraction) |
| `utils/model_family.ts` | `deriveBaseModel`, `deriveFamily`, `baseModelsCompatible` (mirrors Python) |

### `src/routes/`
| File | Purpose |
|---|---|
| `+layout.svelte` | Root layout: ConnectionBanner + ToastContainer |
| `+page.svelte` | Main page: topology graph + chat |
| `advanced/+page.svelte` | Advanced tab (Prefill/Decode) — bounces home if disaggregation disabled |
| `downloads/+page.svelte` | Download manager grid (per-node status cells) |
| `integrations/+page.svelte` | API integration cards (OpenAI/Claude/Ollama) with tier mapping |
| `traces/+page.svelte` | Trace list with selection/delete |
| `traces/[taskId]/+page.svelte` | Trace detail with phase breakdown |

### Other dashboard files
| File | Purpose |
|---|---|
| `src/app.d.ts` | SvelteKit ambient type declarations |
| `src/app.css` | Global styles (exo dark theme, Tailwind) |
| `src/app.html` | HTML shell |
| `svelte.config.js`, `vite.config.ts`, `tsconfig.json` | Build config |

---

## 4. Tests — `tests/`
| File | Purpose |
|---|---|
| `__init__.py` | Tests package init |
| `conftest.py` | Session fixture (marker-driven cluster deploy) |
| `framework.py` | `ClusterSpec`, `InstanceSpec`, `parse_cluster_marker`, `_extract_content`, `DEFAULT_MODEL` |
| `test_1node.py` | Single-node: place+chat, multi-turn, delete instance |
| `test_2node.py` | 2-node: tensor+JACCL, pipeline+RING, multi-turn |
| `test_4node.py` | 4-node: pipeline+RING, tensor+JACCL |
| `test_dashboard.py` | Dashboard smoke tests |
| `test_resilience.py` | Node disconnect/reconnect recovery |

---

## 5. Tools — `tools/src/exo_tools/`
| File | Purpose |
|---|---|
| `__init__.py` | Package init |
| `client.py` | `ExoClient` (HTTP client via http.client), `ExoHttpError` |
| `cluster.py` | `Thunderbolt` (A2A/RING), `Chip` (M1–M4 Ultra), `ClusterInfo`, `EcoSession` (eco subprocess wrapper with atexit/signal cleanup), `make_client_from_url` |
| `harness.py` | `Sharding` (PIPELINE/TENSOR), `Comm` (RING/JACCL), `unwrap_instance`, `instance_id_from_instance`, `nodes_used_in_instance`, `runner_ids_from_instance`, `place_instance`, `cleanup_all_instances`, `wait_for_instance_gone`, `wait_for_instance_ready`, `is_model_downloaded`, `capture_cluster_snapshot`, `find_existing_instance`, `resolve_model_short_id`, `run_planning_phase`, `settle_and_fetch_placements`, `add_common_instance_args`, `_SETTLE_INITIAL_BACKOFF_S=1.0` etc. |

---

## 6. Bench — `bench/`
| File | Purpose |
|---|---|
| `METHODOLOGY.md` | Benchmark methodology (prompt construction, cold KV cache, banned EOS, no parsing) |
| `bench.toml` | Canary manifest (includes single-m3-ultra.toml) |
| `single-m3-ultra.toml` | Single M3 Ultra benchmark suite (constraints, defaults pp/tg) |
| `prefill-decode.toml` | Disaggregated prefill/decode bench config |
| `eval_configs/models.toml` | Per-model eval configs (reasoning/non-reasoning, temperature, max_tokens) |
| `scenarios.toml` | Tool-call eval scenarios |
| `exo_bench.py` | Throughput benchmark (PromptSizer, /bench/chat/completions) |
| `exo_eval.py` | Quality eval (Artificial Analysis methodology: gpqa_diamond, mmlu_pro, aime_2024/2025, humaneval, livecodebench) |
| `eval_tool_calls.py` | Tool-calling eval |
| `parallel_requests.py` | Parallel request load test |
| `prefill_decode_bench.py` | Disaggregated prefill/decode benchmark |
| `vendor/__init__.py`, `vendor/lcb_testing_util.py` | LiveCodeBench testing utilities |
| `pyproject.toml` | Bench package deps |

---

## 7. Packaging — `packaging/`
| File | Purpose |
|---|---|
| `pyinstaller/exo.spec` | PyInstaller spec: entrypoint `__main__.py`, dashboard build, resources, mlx libs, hidden imports |
| `dmg/create-dmg.sh` | DMG builder (background, icons, Applications alias) |
| `dmg/generate-background.py` | DMG background PNG generator (1600×740 retina, drag arrow) |
| `dmg/background.png` | Generated background asset (output of generate-background.py) |

---

## 8. Scripts — `scripts/`
| File | Purpose |
|---|---|
| `download_model_to_cluster.py` | Download a model to every node (bypasses placement) |
| `fetch_kv_heads.py` | Fetch `num_key_value_heads` from HF and update TOML cards |

---

## 9. Native App — `app/EXO/`
| File | Purpose |
|---|---|
| `EXO/EXOApp.swift` | App entry: services wiring, Sparkle updater, termination observer |
| `EXO/main.swift` | CLI parsing (--install/--uninstall/--help) |
| `EXO/ContentView.swift` | Main menu bar view: status, topology, nodes, instances, controls |
| `EXO/ExoProcessController.swift` | Manages exo subprocess, settings persistence (namespace, HF token, image models, offline, fast-synch, model dirs, custom env vars) |
| `EXO/Models/ClusterState.swift` | ClusterState decoding (instances, runners, tasks, topology, downloads, node_* maps, nodeProfiles computed) |
| `EXO/Services/ClusterStateService.swift` | Polls `/state` at 2 Hz (ephemeral non-caching URLSession), models, node_id |
| `EXO/Services/LocalNetworkChecker.swift` | mDNS multicast connectivity check (NWConnection) |
| `EXO/Services/NetworkSetupHelper.swift` | LaunchDaemon install for Thunderbolt bridge disable + DHCP setup |
| `EXO/Services/NetworkStatusService.swift` | ibv_devices, bridge state, interface IPs |
| `EXO/Services/ThunderboltBridgeDetector.swift` | Bridge service detection (mirrors Python info_gatherer) |
| `EXO/Services/ThunderboltBridgeService.swift` | Detects >2-node TB bridge cycles, prompts to disable |
| `EXO/Services/BugReportService.swift` | Collects logs/state, uploads via presigned URLs |
| `EXO/ViewModels/NodeViewModel.swift` | Node metrics view model |
| `EXO/ViewModels/InstanceViewModel.swift` | Instance/download progress view model |
| `EXO/Views/BugReportWindowController.swift` | Bug report window |
| `EXO/Views/FirstLaunchPopout.swift` | Onboarding popover with 10s countdown |
| `EXO/Views/InstanceRowView.swift` | Instance row with download progress |
| `EXO/Views/NodeDetailView.swift` | Node detail metrics |
| `EXO/Views/NodeRowView.swift` | Node row |
| `EXO/Views/SettingsView.swift` | Settings tabs (General/Model/Advanced/Environment/About) |
| `EXO/Views/SettingsWindowController.swift` | Settings window |
| `EXO/Views/TopologyMiniView.swift` | Mini topology graph |
| `EXO/EXO.entitlements`, `EXO/Info.plist` | App entitlements/config |
| `EXOTests/EXOTests.swift` | Unit test stub |
| `EXOUITests/EXOUITests.swift`, `EXOUITestsLaunchTests.swift` | UI test stubs |
| `uninstall-exo.sh` | Uninstaller (LaunchDaemon, scripts, logs, network location, ~/.exo) |

---

## 10. Resources — `resources/`
| File | Purpose |
|---|---|
| `inference_model_cards/*.toml` | ~150 model cards (Llama, Qwen3/3.5/3.6/Next/Coder/VL, DeepSeek V3.1/V3.2/V4, GLM 4.5/4.7/5/5.1, Kimi K2/K2.5/K2.6/K2.7, MiniMax M2.1/2.5/2.7, GPT-OSS, Nemotron, Gemma 4, Step 3.5, Moonshot) |
| `image_model_cards/*.toml` | FLUX.1 (dev/Kontext/Krea/schnell, 4/8-bit), Qwen-Image, Qwen-Image-Edit |

---

## 11. Docs — `docs/`
| File | Purpose |
|---|---|
| `api.md` | REST API reference (⚠️ references `src/exo/master/api.py` — actual: `src/exo/api/main.py`) |
| `benchmarks/jeffgeerling/` | Benchmark images (Qwen3-235B, DeepSeek 3.1 671B, Kimi K2) |
| `imgs/` | Logo + dashboard screenshots |

---

## 12. Nix — `nix/`
| File | Purpose |
|---|---|
| `apple-sdk-overlay.nix` | Patches nixpkgs apple-sdk to SDK 26.2 |
| `apple-sdk/metadata/versions.json` | SDK version metadata |
| `darwin-build-fixes.patch` | MLX hermetic build patches (no xcrun) |
| `metal-toolchain.nix` | Metal toolchain NAR repackage (metal + metallib) |

---

## 13. Root config files
| File | Purpose |
|---|---|
| `pyproject.toml` | Python package (0.3.70, py3.13, uv_build, exo script, deps, mlx extras, uv workspace) |
| `Cargo.toml` | Rust workspace (exo_rs, networking; zenoh fork patch) |
| `flake.nix` | Nix flake (nixpkgs 25.11, crane, fenix, uv2nix, cachix) |
| `justfile` | Command runner (lint/fmt/test/check/sync/rust-rebuild/build-dashboard/package/build-app/clean) |
| `AGENTS.md` | Agent instructions (⚠️ outdated architecture references) |
| `RULES.md` | Code style rules (type discipline, naming, purity, pydantic, UUIDs) |
| `CONTRIBUTING.md` | Contribution guide (uv, rust nightly, macmon pinned fork) |
| `PLATFORMS.md` | Platform support tiers |
| `TODO.md` | Known issues/roadmap |
| `.gitignore` | Ignore rules (incl. `/understand`, `promptsANDplans` is tracked) |

---

## 14. Hidden / Tooling Folders (verified 2026-08-05)

### GitHub CI — `.github/`
| File | Purpose |
|---|---|
| `workflows/pipeline.yml` | CI: Nix build all outputs on 3-platform matrix (macos-26 aarch64-darwin / ubuntu-latest x86_64-linux / ubuntu-24.04-arm aarch64-linux), `nix flake check`, pytest `-m "not slow"` (macOS, `EXO_TESTS=1`); builds metal-toolchain NAR on CI when cachix misses |
| `workflows/build-app.yml` | Release: version derivation, draft-release validation, Xcode 26.2, keychain setup, notarization pre-flight (all-or-nothing creds), provisioning-profile expiry check, PyInstaller bundle, Swift app build, codesign/notarize, DMG, Sparkle appcast, S3 upload, GitHub Release publish |
| `actions/conditional-commit/action.yml` | Commit-if-dirty composite action |
| `actions/format/action.yml` | Format code |
| `actions/lint/action.yml` + `lint-check/` | Lint + lint check |
| `actions/regenerate-protobufs/action.yml` | Protobuf regen (legacy) |
| `actions/setup-python-uv/action.yml` | Python/uv env setup |
| `actions/unit-test/action.yml` | Unit tests |
| `actions/verify-clean/action.yml` | Fail if working tree dirty |
| `CODEOWNERS` | @ToxicPine, @AlexCheema, @GeluVrabie |
| `pull_request_template.md` | Motivation / Changes / Why It Works / Test Plan |

### Git LFS hooks — `.githooks/`
`post-checkout`, `post-commit`, `post-merge`, `pre-push` — all invoke `git lfs <hook>` (wired via `core.hookspath`)

### Editor config
| Folder | Contents |
|---|---|
| `.vscode/settings.json` | nix LSP (nixd) + `nixpkgs-fmt`, ruff Python formatter, basedpyright config, `.venv` interpreter |
| `.zed/settings.json` | `nix_python` LSP via `nix run .#python-lsp` |
| `.idea/` | PyCharm config (pyright, modules, vcs, misc) |

### Agent tooling data (gitignored)
| Folder | Contents |
|---|---|
| `.typings/` | Generated `mlx_lm` `.pyi` stubs for basedpyright |
| `.feeder/tasks.db` | SQLite DB (feeder agent tool task storage, 20KB) |
| `.mnemo/` | `graph.lbug` (19MB agent memory graph) + `parse-cache.json` |

### Developer scratch — `tmp/`
| File | Purpose |
|---|---|
| `run_exo_on.sh` | SSH-deploy exo to remote hosts |
| `set_rdma_network_config.sh` | RDMA network setup (remove bridge0, disable TB Bridge, exo network location) |
| `run_llm.sh` / `run_llm.py` | Remote LLM query helpers |
| `test_trust_remote_code_attack.sh` | Security test: API models get `trust_remote_code=false` |
| `quantize_and_upload.py` | mflux quantization + HF upload |
| `gen_card.py` | Model card generator |
| `prompt.txt` | Scratch |
| `config_examples/` | `claude_code.sh` (Claude Code → exo), `opencode.json` (opencode → exo) |
| `old_tests/` | Legacy: `auto_bench.sh`, `eval_tool_calls.sh`, `get_all_models_on_cluster.py`, `test_vision_cache.py` |

---

## 15. Branch-Specific Additions (`fix-memory-error`, verified 2026-08-05)

> See `GIT-HISTORY-AND-UPSTREAM.md` for full commit/PR documentation with URLs.

### Build artifact folders (gitignored, not source)
| Folder | Contents |
|---|---|
| `build/exo`, `dist/exo` | PyInstaller bundles (`just package`) |
| `target/` | Cargo build output (debug/release/maturin/wheels) |
| `dashboard/build/` | Built Svelte dashboard (required before `uv run exo`) |
| `python/` | Contains only `parts.nix` — the Python flake module (documented under Nix config) |

### New test files from ported PRs
| File | Origin |
|---|---|
| `src/exo/worker/tests/unittests/test_mlx/test_kv_disk_persistence.py` | PR #2159 port (KV cache disk persistence, +409 lines) |
| `src/exo/worker/tests/unittests/test_runner/test_kv_flush_wiring.py` | PR #2159 port (+25) |
| `src/exo/api/tests/test_create_instance_force_override.py` | WIP commit a6b99564 (+106) |
| `src/exo/api/tests/test_add_peer.py` | **Uncommitted** — manual peer endpoint tests (4 tests) |

### Modified-in-branch files
| File | Change |
|---|---|
| `src/exo/worker/engines/mlx/cache.py` | PR #2159 (+531 disk persistence), #2242 (KV cache bits), #2251 (headroom), #1990 (skip quant single-node) |
| `src/exo/worker/engines/mlx/constants.py` | #2242: `EXO_KV_CACHE_BITS` env var, `KV_CACHE_BITS` configurable |
| `src/exo/worker/engines/mlx/generator/generate.py` | #2242, #2159 |
| `src/exo/worker/engines/mlx/generator/batch_generate.py` | #2159, #2251 |
| `src/exo/worker/engines/mlx/builder.py`, `disaggregated/serve.py`, `runner/runner.py` | #2159 wiring |
| `src/exo/master/placement_utils.py` | #2160 (multi-link jaccl matrix) + WIP memory overrides |
| `src/exo/shared/types/worker/instances.py` | #2160: `jaccl_devices` → `list[list[list[str]]]` |
| `src/exo/worker/engines/mlx/utils_mlx.py` | #2160: `MLX_JACCL_RING=1` |
| `src/exo/master/placement.py`, `src/exo/shared/types/commands.py` | WIP: force_override placement |
| `src/exo/download/` (coordinator, download_utils, impl_shard_downloader, shard_downloader) | WIP: download override |
| `src/exo/api/main.py`, `src/exo/api/types/api.py`, `src/exo/main.py` | WIP: force_override + uncommitted `POST /peers` |
| `rust/networking/src/lib.rs`, `swarm.rs`, `rust/exo_rs/src/networking.rs`, `exo_rs.pyi`, `src/exo/routing/router.py` | `connect_peer` manual peer connect (baseline + uncommitted) |
| `dashboard/src/routes/+page.svelte`, `dashboard/src/lib/components/*` | WIP: force-launch UI + uncommitted "Add Node Manually" panel |
| `docs/api.md` | WIP a6b99564: force_override documentation |
| `.gitignore` | WIP 6b8d3a78: +7 lines |
# Appendix A — Complete Public Symbol Index (machine-generated, verified 2026-08-05)

> Every public class/function/async-function in every non-test Python file under `src/exo/`, extracted via AST. This is the authoritative completeness check: 1,245 definitions across 123 files.

### src/exo/main.py (8)
- `class Node`
- `def main`
- `def main_inner`
- `class Args`
- `async def create`
- `async def run`
- `def shutdown`
- `def parse`

### src/exo/utils/__init__.py (2)
- `def ensure_type`
- `def todo`

### src/exo/utils/async_process.py (11)
- `class AsyncProcess`
- `async def run`
- `async def stop`
- `async def aclose`
- `async def wait`
- `def pid`
- `def exitcode`
- `def is_alive`
- `def stdout`
- `def stderr`
- `def cleanup_stdio_fd`

### src/exo/utils/banner.py (1)
- `def print_startup_banner`

### src/exo/utils/channels.py (31)
- `class ErrorOverride`
- `class Sender`
- `class Receiver`
- `class MpState`
- `class MpSender`
- `class MpReceiver`
- `class channel`
- `class mp_channel`
- `def patch`
- `def patch`
- `def patch`
- `def clone`
- `def clone_receiver`
- `def clone`
- `def clone_sender`
- `def collect`
- `async def receive_at_least`
- `def send_nowait`
- `def send`
- `async def send_async`
- `def close`
- `def join`
- `def receive_nowait`
- `def receive`
- `async def receive_async`
- `def close`
- `def join`
- `def collect`
- `def receive_at_least`
- `async def async_wrapper`
- `def sync_wrapper`

### src/exo/utils/dashboard_path.py (2)
- `def find_resources`
- `def find_dashboard`

### src/exo/utils/disk_event_log.py (5)
- `class DiskEventLog`
- `def append`
- `def read_range`
- `def read_all`
- `def close`

### src/exo/utils/event_buffer.py (7)
- `class OrderedBuffer`
- `class MultiSourceBuffer`
- `def ingest`
- `def drain`
- `def drain_indexed`
- `def ingest`
- `def drain`

### src/exo/utils/fs.py (4)
- `def delete_if_exists`
- `def ensure_parent_directory_exists`
- `def ensure_directory_exists`
- `def make_temp_path`

### src/exo/utils/keyed_backoff.py (5)
- `class KeyedBackoff`
- `def should_proceed`
- `def record_attempt`
- `def attempts`
- `def reset`

### src/exo/utils/ports.py (1)
- `def random_ephemeral_port`

### src/exo/utils/power_sampler.py (7)
- `class PowerSampler`
- `def trapezoidal_energy`
- `def trapezoidal_energy_range`
- `async def run`
- `def mark_prefill_done`
- `def result`
- `def power_at`

### src/exo/utils/pydantic_ext.py (2)
- `class FrozenModel`
- `class TaggedModel`

### src/exo/utils/reactive.py (4)
- `class OnChange`
- `class Reactive`
- `def value`
- `def value`

### src/exo/utils/task_group.py (6)
- `class TaskGroup`
- `def is_running`
- `def cancel_tasks`
- `def cancel_called`
- `def start_soon`
- `def queue`

### src/exo/utils/info_gatherer/info_gatherer.py (20)
- `class StaticNodeInformation`
- `class NodeNetworkInterfaces`
- `class MacThunderboltIdentifiers`
- `class MacThunderboltConnections`
- `class RdmaCtlStatus`
- `class ThunderboltBridgeInfo`
- `class NodeConfig`
- `class MiscData`
- `class NodeDiskUsage`
- `class NodeBackends`
- `class InfoGatherer`
- `async def gather`
- `async def gather`
- `async def gather`
- `async def gather`
- `async def gather`
- `async def gather`
- `async def gather`
- `async def run`
- `def shutdown`

### src/exo/utils/info_gatherer/macmon.py (4)
- `class RawMacmonMetrics`
- `class MacmonMetrics`
- `def from_raw`
- `def from_raw_json`

### src/exo/utils/info_gatherer/net_profile.py (2)
- `async def check_reachability`
- `async def check_reachable`

### src/exo/utils/info_gatherer/system_info.py (5)
- `def get_os_version`
- `async def get_os_build_version`
- `async def get_friendly_name`
- `async def get_network_interfaces`
- `async def get_model_and_chip`

### src/exo/shared/apply.py (18)
- `def event_apply`
- `def apply`
- `def apply_node_download_progress`
- `def apply_task_created`
- `def apply_task_deleted`
- `def apply_task_status_updated`
- `def apply_task_failed`
- `def apply_instance_created`
- `def apply_instance_deleted`
- `def apply_instance_link_created`
- `def apply_instance_link_deleted`
- `def apply_runner_status_updated`
- `def apply_node_timed_out`
- `def apply_node_gathered_info`
- `def apply_topology_edge_created`
- `def apply_topology_edge_deleted`
- `def apply_custom_model_card_added`
- `def apply_custom_model_card_deleted`

### src/exo/shared/election.py (6)
- `class ElectionMessage`
- `class ElectionResult`
- `class Election`
- `async def run`
- `async def elect`
- `async def shutdown`

### src/exo/shared/logging.py (4)
- `class InterceptLogger`
- `def logger_setup`
- `def logger_cleanup`
- `def emit`

### src/exo/shared/topology.py (23)
- `class TopologySnapshot`
- `class Topology`
- `def to_snapshot`
- `def from_snapshot`
- `def add_node`
- `def node_is_leaf`
- `def neighbours`
- `def out_edges`
- `def contains_node`
- `def add_connection`
- `def get_all_connections_between`
- `def list_nodes`
- `def map_connections`
- `def list_connections`
- `def remove_node`
- `def replace_all_out_rdma_connections`
- `def remove_all_rdma_connections_touching`
- `def remove_connection`
- `def get_cycles`
- `def get_rdma_cycles`
- `def get_subgraph_from_nodes`
- `def is_rdma_cycle`
- `def get_thunderbolt_bridge_cycles`

### src/exo/shared/tracing.py (11)
- `class TraceEvent`
- `class CategoryStats`
- `class TraceStats`
- `def trace`
- `def get_trace_buffer`
- `def clear_trace_buffer`
- `def export_trace`
- `def load_trace_file`
- `def compute_stats`
- `def add`
- `def avg_us`

### src/exo/shared/types/backends.py (1)
- `class Backend`

### src/exo/shared/types/chunks.py (7)
- `class BaseChunk`
- `class TokenChunk`
- `class ErrorChunk`
- `class ToolCallChunk`
- `class ImageChunk`
- `class InputImageChunk`
- `class PrefillProgressChunk`

### src/exo/shared/types/commands.py (21)
- `class BaseCommand`
- `class TestCommand`
- `class TextGeneration`
- `class ImageGeneration`
- `class ImageEdits`
- `class PlaceInstance`
- `class CreateInstance`
- `class DeleteInstance`
- `class TaskCancelled`
- `class TaskFinished`
- `class SendInputChunk`
- `class RequestEventLog`
- `class StartDownload`
- `class DeleteDownload`
- `class CancelDownload`
- `class AddCustomModelCard`
- `class DeleteCustomModelCard`
- `class SetInstanceLink`
- `class DeleteInstanceLink`
- `class ForwarderCommand`
- `class ForwarderDownloadCommand`

### src/exo/shared/types/common.py (11)
- `class Id`
- `class NodeId`
- `class SystemId`
- `class ModelId`
- `class CommandId`
- `class TruncatingString`
- `class SessionId`
- `class Host`
- `def normalize`
- `def short`
- `def check_port`

### src/exo/shared/types/events.py (28)
- `class EventId`
- `class BaseEvent`
- `class TestEvent`
- `class TaskCreated`
- `class TaskAcknowledged`
- `class TaskDeleted`
- `class TaskStatusUpdated`
- `class TaskFailed`
- `class InstanceCreated`
- `class InstanceDeleted`
- `class RunnerStatusUpdated`
- `class NodeTimedOut`
- `class NodeGatheredInfo`
- `class NodeDownloadProgress`
- `class ChunkGenerated`
- `class InputChunkReceived`
- `class TopologyEdgeCreated`
- `class TopologyEdgeDeleted`
- `class CustomModelCardAdded`
- `class CustomModelCardDeleted`
- `class TraceEventData`
- `class TracesCollected`
- `class TracesMerged`
- `class InstanceLinkCreated`
- `class InstanceLinkDeleted`
- `class IndexedEvent`
- `class GlobalForwarderEvent`
- `class LocalForwarderEvent`

### src/exo/shared/types/instance_link.py (2)
- `class InstanceLinkId`
- `class InstanceLink`

### src/exo/shared/types/memory.py (13)
- `class Memory`
- `def from_bytes`
- `def in_kb`
- `def in_kb`
- `def from_kb`
- `def from_float_kb`
- `def in_mb`
- `def in_mb`
- `def in_float_mb`
- `def in_float_mb`
- `def from_mb`
- `def from_gb`
- `def in_gb`

### src/exo/shared/types/multiaddr.py (7)
- `class Multiaddr`
- `def validate_format`
- `def address_type`
- `def ipv6_address`
- `def ipv4_address`
- `def ip_address`
- `def port`

### src/exo/shared/types/profiling.py (13)
- `class MemoryUsage`
- `class DiskUsage`
- `class SystemPerformanceProfile`
- `class NetworkInterfaceInfo`
- `class NodeIdentity`
- `class NodeNetworkInfo`
- `class NodeThunderboltInfo`
- `class NodeRdmaCtlStatus`
- `class ThunderboltBridgeStatus`
- `def from_bytes`
- `def from_psutil`
- `def from_cuda`
- `def from_path`

### src/exo/shared/types/state.py (1)
- `class State`

### src/exo/shared/types/tasks.py (13)
- `class TaskId`
- `class TaskStatus`
- `class BaseTask`
- `class CreateRunner`
- `class DownloadModel`
- `class LoadModel`
- `class ConnectToGroup`
- `class StartWarmup`
- `class TextGeneration`
- `class CancelTask`
- `class ImageGeneration`
- `class ImageEdits`
- `class Shutdown`

### src/exo/shared/types/text_generation.py (8)
- `def resolve_reasoning_params`
- `class InputMessageContent`
- `class InputMessage`
- `class Base64Image`
- `class Base64ImageHash`
- `class TextGenerationTaskParams`
- `def with_card_sampling_defaults`
- `def resolve`

### src/exo/shared/types/thunderbolt.py (7)
- `class ThunderboltConnection`
- `class ThunderboltIdentifier`
- `class ThunderboltConnectivityData`
- `class ThunderboltConnectivity`
- `def ident`
- `def conn`
- `async def gather`

### src/exo/shared/types/topology.py (4)
- `class Cycle`
- `class RDMAConnection`
- `class SocketConnection`
- `class Connection`

### src/exo/shared/types/worker/downloads.py (11)
- `class DownloadProgressData`
- `class BaseDownloadProgress`
- `class DownloadPending`
- `class DownloadCompleted`
- `class DownloadFailed`
- `class DownloadOngoing`
- `class ModelSafetensorsIndexMetadata`
- `class ModelSafetensorsIndex`
- `class FileListEntry`
- `class RepoFileDownloadProgress`
- `class RepoDownloadProgress`

### src/exo/shared/types/worker/instances.py (10)
- `class InstanceId`
- `class InstanceMeta`
- `class BaseInstance`
- `class MlxRingInstance`
- `class MlxJacclInstance`
- `class BoundInstance`
- `def shard`
- `def bound_shard`
- `def is_image_model`
- `def validate_shard_exists`

### src/exo/shared/types/worker/runner_response.py (9)
- `class BaseRunnerResponse`
- `class GenerationResponse`
- `class ImageGenerationResponse`
- `class PartialImageResponse`
- `class ToolCallResponse`
- `class FinishedResponse`
- `class ModelLoadingResponse`
- `class CancelledResponse`
- `class PrefillProgressResponse`

### src/exo/shared/types/worker/runners.py (17)
- `class RunnerId`
- `class RunnerError`
- `class BaseRunnerStatus`
- `class RunnerIdle`
- `class RunnerConnecting`
- `class RunnerConnected`
- `class RunnerLoading`
- `class RunnerLoaded`
- `class RunnerWarmingUp`
- `class RunnerReady`
- `class RunnerRunning`
- `class RunnerShuttingDown`
- `class RunnerShutdown`
- `class RunnerFailed`
- `class ShardAssignments`
- `def is_running`
- `def validate_runners_exist`

### src/exo/shared/types/worker/shards.py (7)
- `class Sharding`
- `class BaseShardMetadata`
- `class PipelineShardMetadata`
- `class CfgShardMetadata`
- `class TensorShardMetadata`
- `def is_first_layer`
- `def is_last_layer`

### src/exo/shared/models/model_cards.py (22)
- `def detect_vision_from_config`
- `class ModelTask`
- `class ComponentInfo`
- `class VisionCardConfig`
- `class SamplingValues`
- `class SamplingDefaults`
- `class ModelCard`
- `class ConfigData`
- `async def fetch_config_data`
- `async def fetch_safetensors_size`
- `def get`
- `async def save`
- `async def pop`
- `async def list_all`
- `async def refresh`
- `async def save`
- `async def save_to_custom_dir`
- `async def load_from_path`
- `async def load`
- `async def fetch_from_hf`
- `def supports_tensor`
- `def defer_to_text_config`

### src/exo/master/image_store.py (6)
- `class StoredImage`
- `class ImageStore`
- `def store`
- `def get`
- `def list_images`
- `def cleanup_expired`

### src/exo/master/main.py (3)
- `class Master`
- `async def run`
- `async def shutdown`

### src/exo/master/placement.py (6)
- `def add_instance_to_placements`
- `def place_instance`
- `def delete_instance`
- `def get_transition_events`
- `def cancel_unnecessary_downloads`
- `def get_device_rank`

### src/exo/master/placement_utils.py (11)
- `def filter_cycles_by_memory`
- `def get_smallest_cycles`
- `def allocate_layers_proportionally`
- `def get_shard_assignments_for_pipeline_parallel`
- `def get_shard_assignments_for_tensor_parallel`
- `def get_shard_assignments`
- `def get_mlx_jaccl_devices_matrix`
- `def find_ip_prioritised`
- `def get_mlx_ring_hosts_by_node`
- `def get_mlx_jaccl_coordinators`
- `def get_ip_for_node`

### src/exo/download/coordinator.py (4)
- `class DownloadCoordinator`
- `async def run`
- `async def shutdown`
- `async def download_wrapper`

### src/exo/download/download_utils.py (32)
- `class HuggingFaceAuthenticationError`
- `class HuggingFaceRateLimitError`
- `def trim_etag`
- `def map_repo_file_download_progress_to_download_progress_data`
- `def map_repo_download_progress_to_download_progress_data`
- `class InsufficientDiskSpaceError`
- `def resolve_existing_model`
- `def is_read_only_model_dir`
- `def build_model_path`
- `def select_download_dir`
- `async def select_download_dir_for_shard`
- `async def resolve_model_dir`
- `async def ensure_cache_dir`
- `async def delete_model`
- `async def seed_models`
- `def is_model_directory_complete`
- `async def fetch_file_list_with_cache`
- `async def fetch_file_list_with_retry`
- `async def get_download_headers`
- `def create_http_session`
- `async def calc_hash`
- `async def file_meta`
- `async def download_file_with_retry`
- `def calculate_repo_progress`
- `async def get_weight_map`
- `async def resolve_allow_patterns`
- `def is_image_model`
- `async def get_downloaded_size`
- `async def download_shard`
- `async def on_progress_wrapper`
- `def schedule_progress`
- `async def download_with_semaphore`

### src/exo/download/huggingface_utils.py (7)
- `def filter_repo_objects`
- `def get_hf_endpoint`
- `def get_hf_home`
- `async def get_hf_token`
- `async def get_auth_headers`
- `def extract_layer_num`
- `def get_allow_patterns`

### src/exo/download/impl_shard_downloader.py (17)
- `def exo_shard_downloader`
- `async def build_base_shard`
- `async def build_full_shard`
- `class SingletonShardDownloader`
- `class ResumableShardDownloader`
- `def on_progress`
- `async def ensure_shard`
- `async def get_shard_download_status`
- `async def get_shard_download_status_for_shard`
- `async def on_progress_wrapper`
- `def on_progress`
- `async def ensure_shard`
- `async def get_shard_download_status`
- `async def get_shard_download_status_for_shard`
- `async def main_progress`
- `async def download_with_semaphore`
- `async def vision_progress`

### src/exo/download/shard_downloader.py (10)
- `class ShardDownloader`
- `class NoopShardDownloader`
- `async def ensure_shard`
- `def on_progress`
- `async def get_shard_download_status`
- `async def get_shard_download_status_for_shard`
- `async def ensure_shard`
- `def on_progress`
- `async def get_shard_download_status`
- `async def get_shard_download_status_for_shard`

### src/exo/api/keepalive.py (1)
- `async def with_sse_keepalive`

### src/exo/api/main.py (52)
- `class API`
- `def reset`
- `def unpause`
- `async def http_exception_handler`
- `def get_state`
- `async def place_instance`
- `async def create_instance`
- `async def get_placement`
- `async def get_placement_previews`
- `def get_instance`
- `async def await_instance`
- `async def delete_instance`
- `async def get_feature_flags`
- `async def list_instance_links`
- `async def create_instance_link`
- `async def update_instance_link`
- `async def delete_instance_link`
- `async def cancel_command`
- `async def chat_completions`
- `async def bench_chat_completions`
- `def stream_events`
- `async def get_image`
- `async def list_images`
- `async def image_generations`
- `async def bench_image_generations`
- `async def image_edits`
- `async def bench_image_edits`
- `async def claude_messages`
- `async def openai_responses`
- `async def ollama_chat`
- `async def ollama_generate`
- `async def ollama_tags`
- `async def ollama_show`
- `async def ollama_ps`
- `async def ollama_version`
- `async def get_models`
- `async def add_custom_model`
- `async def add_peer`
- `async def delete_custom_model`
- `async def search_models`
- `async def run`
- `async def run_api`
- `async def start_download`
- `async def delete_download`
- `async def cancel_download`
- `async def list_traces`
- `async def get_trace`
- `async def get_trace_stats`
- `async def get_trace_raw`
- `async def delete_traces`
- `async def get_onboarding`
- `async def complete_onboarding`

### src/exo/api/types/api.py (69)
- `class ErrorInfo`
- `class ErrorResponse`
- `class ModelListModel`
- `class ModelList`
- `class ChatCompletionMessageText`
- `class ChatCompletionMessageImageUrl`
- `class ToolCallItem`
- `class ToolCall`
- `class ChatCompletionMessage`
- `class BenchChatCompletionMessage`
- `class TopLogprobItem`
- `class LogprobsContentItem`
- `class Logprobs`
- `class PromptTokensDetails`
- `class CompletionTokensDetails`
- `class Usage`
- `class StreamingChoiceResponse`
- `class ChatCompletionChoice`
- `class ChatCompletionResponse`
- `class GenerationStats`
- `class ImageGenerationStats`
- `class NodePowerStats`
- `class PowerUsage`
- `class BenchChatCompletionResponse`
- `class StreamOptions`
- `class ChatCompletionRequest`
- `class BenchChatCompletionRequest`
- `class AddCustomModelParams`
- `class HuggingFaceSearchResult`
- `class PlaceInstanceParams`
- `class CreateInstanceParams`
- `class PlacementPreview`
- `class PlacementPreviewResponse`
- `class DeleteInstanceTaskParams`
- `class AddPeerParams`
- `class AddPeerResponse`
- `class CreateInstanceResponse`
- `class DeleteInstanceResponse`
- `class AwaitInstanceReadyMessage`
- `class AwaitInstanceTimeoutMessage`
- `class CancelCommandResponse`
- `class InstanceLinkBody`
- `class InstanceLinkResponse`
- `def normalize_image_size`
- `class AdvancedImageParams`
- `class ImageGenerationTaskParams`
- `class BenchImageGenerationTaskParams`
- `class ImageEditsTaskParams`
- `class ImageData`
- `class ImageGenerationResponse`
- `class BenchImageGenerationResponse`
- `class ImageListItem`
- `class ImageListResponse`
- `class StartDownloadParams`
- `class StartDownloadResponse`
- `class DeleteDownloadResponse`
- `class CancelDownloadParams`
- `class CancelDownloadResponse`
- `class TraceEventResponse`
- `class TraceResponse`
- `class TraceCategoryStats`
- `class TraceRankStats`
- `class TraceStatsResponse`
- `class TraceListItem`
- `class TraceListResponse`
- `class DeleteTracesRequest`
- `class DeleteTracesResponse`
- `def normalize_size`
- `def normalize_size`

### src/exo/api/types/claude_api.py (24)
- `class ClaudeToolDefinition`
- `class ClaudeTextBlock`
- `class ClaudeImageSource`
- `class ClaudeImageBlock`
- `class ClaudeThinkingBlock`
- `class ClaudeToolUseBlock`
- `class ClaudeToolResultBlock`
- `class ClaudeMessage`
- `class ClaudeThinkingConfig`
- `class ClaudeMessagesRequest`
- `class ClaudeUsage`
- `class ClaudeMessagesResponse`
- `class ClaudeMessageStart`
- `class ClaudeMessageStartEvent`
- `class ClaudeContentBlockStartEvent`
- `class ClaudeTextDelta`
- `class ClaudeThinkingDelta`
- `class ClaudeInputJsonDelta`
- `class ClaudeContentBlockDeltaEvent`
- `class ClaudeContentBlockStopEvent`
- `class ClaudeMessageDeltaUsage`
- `class ClaudeMessageDelta`
- `class ClaudeMessageDeltaEvent`
- `class ClaudeMessageStopEvent`

### src/exo/api/types/ollama_api.py (15)
- `class OllamaToolFunction`
- `class OllamaToolCall`
- `class OllamaMessage`
- `class OllamaOptions`
- `class OllamaChatRequest`
- `class OllamaGenerateRequest`
- `class OllamaGenerateResponse`
- `class OllamaShowRequest`
- `class OllamaChatResponse`
- `class OllamaModelDetails`
- `class OllamaModelTag`
- `class OllamaTagsResponse`
- `class OllamaShowResponse`
- `class OllamaPsModel`
- `class OllamaPsResponse`

### src/exo/api/types/openai_responses.py (55)
- `class ResponseInputTextPart`
- `class ResponseInputImagePart`
- `class ResponseOutputTextPart`
- `class ResponseInputMessage`
- `class FunctionCallInputItem`
- `class FunctionCallOutputInputItem`
- `class ReasoningInputItem`
- `class ComputerCallInputItem`
- `class ComputerCallOutputInputItem`
- `class WebSearchCallInputItem`
- `class FileSearchCallInputItem`
- `class CodeInterpreterCallInputItem`
- `class ImageGenerationCallInputItem`
- `class LocalShellCallInputItem`
- `class LocalShellCallOutputInputItem`
- `class ShellCallInputItem`
- `class ShellCallOutputInputItem`
- `class ApplyPatchCallInputItem`
- `class ApplyPatchCallOutputInputItem`
- `class ToolSearchCallInputItem`
- `class ToolSearchOutputInputItem`
- `class McpCallInputItem`
- `class McpListToolsInputItem`
- `class McpApprovalRequestInputItem`
- `class McpApprovalResponseInputItem`
- `class CustomToolCallInputItem`
- `class CustomToolCallOutputInputItem`
- `class CompactionInputItem`
- `class ItemReferenceInputItem`
- `class Reasoning`
- `class ResponsesRequest`
- `class ResponseOutputText`
- `class ResponseMessageItem`
- `class ResponseFunctionCallItem`
- `class ResponseReasoningSummaryText`
- `class ResponseReasoningItem`
- `class InputTokensDetails`
- `class OutputTokensDetails`
- `class ResponseUsage`
- `class ResponsesResponse`
- `class ResponseCreatedEvent`
- `class ResponseInProgressEvent`
- `class ResponseOutputItemAddedEvent`
- `class ResponseContentPartAddedEvent`
- `class ResponseTextDeltaEvent`
- `class ResponseTextDoneEvent`
- `class ResponseContentPartDoneEvent`
- `class ResponseOutputItemDoneEvent`
- `class ResponseFunctionCallArgumentsDeltaEvent`
- `class ResponseFunctionCallArgumentsDoneEvent`
- `class ResponseReasoningSummaryPartAddedEvent`
- `class ResponseReasoningSummaryTextDeltaEvent`
- `class ResponseReasoningSummaryTextDoneEvent`
- `class ResponseReasoningSummaryPartDoneEvent`
- `class ResponseCompletedEvent`

### src/exo/api/adapters/chat_completions.py (6)
- `def extract_base64_from_data_url`
- `async def fetch_image_url`
- `async def chat_request_to_text_generation`
- `def chunk_to_response`
- `async def generate_chat_stream`
- `async def collect_chat_response`

### src/exo/api/adapters/claude.py (5)
- `def finish_reason_to_claude_stop_reason`
- `async def handle_image_block`
- `async def claude_request_to_text_generation`
- `async def collect_claude_response`
- `async def generate_claude_stream`

### src/exo/api/adapters/ollama.py (6)
- `def ollama_request_to_text_generation`
- `async def generate_ollama_chat_stream`
- `async def collect_ollama_chat_response`
- `def ollama_generate_request_to_text_generation`
- `async def generate_ollama_generate_stream`
- `async def collect_ollama_generate_response`

### src/exo/api/adapters/responses.py (3)
- `async def responses_request_to_text_generation`
- `async def collect_responses_response`
- `async def generate_responses_stream`

### src/exo/routing/connection_message.py (2)
- `class ConnectionMessage`
- `def from_update`

### src/exo/routing/event_router.py (7)
- `class EventRouterClosedResourceError`
- `class EventRouterBrokenResourceError`
- `class EventRouter`
- `async def run`
- `def sender`
- `def receiver`
- `def shutdown`

### src/exo/routing/router.py (15)
- `class TopicRouter`
- `class Router`
- `def get_node_zid`
- `async def run`
- `async def shutdown`
- `async def publish`
- `async def publish_bytes`
- `def new_sender`
- `def create`
- `async def register_topic`
- `def sender`
- `def receiver`
- `async def run`
- `async def shutdown`
- `async def connect_peer`

### src/exo/routing/topics.py (4)
- `class PublishPolicy`
- `class TypedTopic`
- `def serialize`
- `def deserialize`

### src/exo/worker/main.py (4)
- `class Worker`
- `async def run`
- `async def plan_step`
- `async def shutdown`

### src/exo/worker/plan.py (2)
- `def instance_to_reset_backoff`
- `def plan`

### src/exo/worker/runner/bootstrap.py (3)
- `class RunnerTerminationError`
- `def entrypoint`
- `def from_exception`

### src/exo/worker/runner/diagnostics.py (8)
- `class BaseRunnerDiagnostic`
- `class RunnerMetalGpuTimeout`
- `class RunnerRingTransportError`
- `class RunnerRingSocketReceivingError`
- `class RunnerUnknown`
- `class RunnerDiagnosticCollector`
- `def record_line`
- `def diagnostics`

### src/exo/worker/runner/runner.py (14)
- `class PrefillTask`
- `class ExitCode`
- `class Runner`
- `def update_status`
- `def send_task_status`
- `def acknowledge_task`
- `def main`
- `def handle_first_task`
- `def shutdown`
- `def submit_generation`
- `def handle_generation_tasks`
- `def send_chunk`
- `def resolve`
- `def loop`

### src/exo/worker/runner/supervisor.py (12)
- `class RunnerStdioHandler`
- `class RunnerSupervisor`
- `async def create`
- `async def run`
- `async def create`
- `async def run`
- `def shutdown`
- `async def wait_stopped`
- `async def start_task`
- `async def cancel_task`
- `async def handle_line`
- `async def handle_text`

### src/exo/worker/runner/llm_inference/batch_generator.py (25)
- `class GeneratorQueue`
- `class SequentialGenerator`
- `class BatchGenerator`
- `def push`
- `def gen`
- `def warmup`
- `def submit`
- `def agree_on_tasks`
- `def agree_on_cancellations`
- `def step`
- `def close`
- `def serve_prefill`
- `def warmup`
- `def submit`
- `def agree_on_tasks`
- `def agree_on_cancellations`
- `def step`
- `def close`
- `def serve_prefill`
- `def on_prefill_progress`
- `def distributed_prompt_progress_callback`
- `def on_generation_token`
- `def on_prefill_progress`
- `def distributed_prompt_progress_callback`
- `def on_generation_token`

### src/exo/worker/runner/llm_inference/model_output_parsers.py (10)
- `def get_gpt_oss_encoding`
- `def count_reasoning_tokens`
- `def apply_all_parsers`
- `def map_responses_to_chunks`
- `def parse_gpt_oss`
- `def parse_deepseek_v32`
- `def parse_deepseek_v4`
- `def parse_thinking_models`
- `def parse_tool_calls`
- `def drain_pending`

### src/exo/worker/runner/llm_inference/tool_parsers.py (6)
- `class ToolParser`
- `def make_mlx_parser`
- `def make_json_parser`
- `def infer_tool_parser`
- `def parse`
- `def parse_tool_calls`

### src/exo/worker/disaggregated/protocol.py (18)
- `class ProtocolError`
- `class Header`
- `class TensorBlob`
- `class KVChunk`
- `class ArraysState`
- `class Done`
- `class ErrorMessage`
- `def write_frame`
- `def read_frame`
- `def write_header`
- `def read_header`
- `def write_message`
- `def read_message`
- `def write_kv_chunk`
- `def write_arrays_state`
- `def write_done`
- `def write_error`
- `def shape`

### src/exo/worker/disaggregated/server.py (7)
- `class PrefillRequest`
- `def write_request`
- `def read_request`
- `class PrefillServer`
- `def setup`
- `def handle`
- `def stop`

### src/exo/worker/engines/base.py (12)
- `class Engine`
- `class Builder`
- `def should_cancel`
- `def warmup`
- `def submit`
- `def step`
- `def close`
- `def serve_prefill`
- `def connect`
- `def load`
- `def build`
- `def close`

### src/exo/worker/engines/image/builder.py (12)
- `class MfluxBuilder`
- `class ImageEngine`
- `def connect`
- `def load`
- `def close`
- `def build`
- `def warmup`
- `def submit`
- `def step`
- `def close`
- `def serve_prefill`
- `def cancel_checker`

### src/exo/worker/engines/image/config.py (7)
- `class BlockType`
- `class TransformerBlockConfig`
- `class ImageModelConfig`
- `def total_blocks`
- `def joint_block_count`
- `def single_block_count`
- `def get_steps_for_quality`

### src/exo/worker/engines/image/distributed_model.py (4)
- `class DistributedImageModel`
- `def from_shard_metadata`
- `def get_steps_for_quality`
- `def generate`

### src/exo/worker/engines/image/generate.py (3)
- `def parse_size`
- `def warmup_image_generator`
- `def generate_image`

### src/exo/worker/engines/image/pipeline/block_wrapper.py (8)
- `class BlockWrapperMode`
- `class BlockWrapperMixin`
- `class JointBlockWrapper`
- `class SingleBlockWrapper`
- `def set_patch`
- `def set_text_seq_len`
- `def reset_cache`
- `def set_encoder_mask`

### src/exo/worker/engines/image/pipeline/kv_cache.py (4)
- `class ImagePatchKVCache`
- `def update_image_patch`
- `def get_full_kv`
- `def reset`

### src/exo/worker/engines/image/pipeline/runner.py (8)
- `class CfgBranch`
- `def calculate_patch_heights`
- `def calculate_token_indices`
- `class DiffusionRunner`
- `def is_first_stage`
- `def is_last_stage`
- `def is_distributed`
- `def generate_image`

### src/exo/worker/engines/image/models/__init__.py (2)
- `def get_config_for_model`
- `def create_adapter_for_model`

### src/exo/worker/engines/image/models/base.py (30)
- `class PromptData`
- `class ModelAdapter`
- `def prompt_embeds`
- `def pooled_prompt_embeds`
- `def negative_prompt_embeds`
- `def negative_pooled_prompt_embeds`
- `def get_encoder_hidden_states_mask`
- `def cond_image_grid`
- `def conditioning_latents`
- `def kontext_image_ids`
- `def get_batched_cfg_data`
- `def get_cfg_branch_data`
- `def config`
- `def model`
- `def transformer`
- `def hidden_dim`
- `def needs_cfg`
- `def get_joint_block_wrappers`
- `def get_single_block_wrappers`
- `def slice_transformer_blocks`
- `def set_image_dimensions`
- `def create_latents`
- `def decode_latents`
- `def encode_prompt`
- `def compute_embeddings`
- `def compute_text_embeddings`
- `def compute_rotary_embeddings`
- `def merge_streams`
- `def apply_guidance`
- `def final_projection`

### src/exo/worker/engines/image/models/flux/adapter.py (22)
- `class FluxPromptData`
- `class FluxModelAdapter`
- `def prompt_embeds`
- `def pooled_prompt_embeds`
- `def negative_prompt_embeds`
- `def negative_pooled_prompt_embeds`
- `def get_encoder_hidden_states_mask`
- `def cond_image_grid`
- `def conditioning_latents`
- `def kontext_image_ids`
- `def get_batched_cfg_data`
- `def get_cfg_branch_data`
- `def hidden_dim`
- `def needs_cfg`
- `def get_joint_block_wrappers`
- `def get_single_block_wrappers`
- `def slice_transformer_blocks`
- `def encode_prompt`
- `def compute_embeddings`
- `def compute_text_embeddings`
- `def compute_rotary_embeddings`
- `def apply_guidance`

### src/exo/worker/engines/image/models/flux/kontext_adapter.py (24)
- `class FluxKontextPromptData`
- `class FluxKontextModelAdapter`
- `def prompt_embeds`
- `def pooled_prompt_embeds`
- `def negative_prompt_embeds`
- `def negative_pooled_prompt_embeds`
- `def get_encoder_hidden_states_mask`
- `def cond_image_grid`
- `def conditioning_latents`
- `def kontext_image_ids`
- `def get_cfg_branch_data`
- `def get_batched_cfg_data`
- `def hidden_dim`
- `def needs_cfg`
- `def get_joint_block_wrappers`
- `def get_single_block_wrappers`
- `def slice_transformer_blocks`
- `def set_image_dimensions`
- `def create_latents`
- `def encode_prompt`
- `def compute_embeddings`
- `def compute_text_embeddings`
- `def compute_rotary_embeddings`
- `def apply_guidance`

### src/exo/worker/engines/image/models/flux/wrappers.py (4)
- `class FluxModulationParams`
- `class FluxNormGateState`
- `class FluxJointBlockWrapper`
- `class FluxSingleBlockWrapper`

### src/exo/worker/engines/image/models/qwen/adapter.py (22)
- `class QwenPromptData`
- `class QwenModelAdapter`
- `def prompt_embeds`
- `def pooled_prompt_embeds`
- `def negative_prompt_embeds`
- `def negative_pooled_prompt_embeds`
- `def get_encoder_hidden_states_mask`
- `def cond_image_grid`
- `def conditioning_latents`
- `def kontext_image_ids`
- `def get_batched_cfg_data`
- `def get_cfg_branch_data`
- `def hidden_dim`
- `def needs_cfg`
- `def get_joint_block_wrappers`
- `def get_single_block_wrappers`
- `def slice_transformer_blocks`
- `def encode_prompt`
- `def compute_embeddings`
- `def compute_text_embeddings`
- `def compute_rotary_embeddings`
- `def apply_guidance`

### src/exo/worker/engines/image/models/qwen/edit_adapter.py (30)
- `class EditImageDimensions`
- `class QwenEditPromptData`
- `class QwenEditModelAdapter`
- `def prompt_embeds`
- `def pooled_prompt_embeds`
- `def negative_prompt_embeds`
- `def negative_pooled_prompt_embeds`
- `def get_encoder_hidden_states_mask`
- `def cond_image_grid`
- `def conditioning_latents`
- `def qwen_image_ids`
- `def kontext_image_ids`
- `def is_edit_mode`
- `def get_batched_cfg_data`
- `def get_cfg_branch_data`
- `def config`
- `def model`
- `def transformer`
- `def hidden_dim`
- `def needs_cfg`
- `def get_joint_block_wrappers`
- `def get_single_block_wrappers`
- `def slice_transformer_blocks`
- `def set_image_dimensions`
- `def create_latents`
- `def encode_prompt`
- `def compute_embeddings`
- `def compute_text_embeddings`
- `def compute_rotary_embeddings`
- `def apply_guidance`

### src/exo/worker/engines/image/models/qwen/wrappers.py (3)
- `class QwenStreamModulation`
- `class QwenJointBlockWrapper`
- `def set_encoder_mask`

### src/exo/worker/engines/mlx/auto_parallel.py (47)
- `def flush_prefill_sends`
- `def clear_prefill_sends`
- `class CustomMlxLayer`
- `class PipelineFirstLayer`
- `class PipelineLastLayer`
- `def set_pipeline_prefill`
- `def set_pipeline_queue_sends`
- `def get_inner_model`
- `def get_layers`
- `def pipeline_auto_parallel`
- `def patch_pipeline_model`
- `def patch_tensor_model`
- `def tensor_auto_parallel`
- `class TensorParallelShardingStrategy`
- `class LlamaShardingStrategy`
- `class DeepSeekShardingStrategy`
- `class ShardedMoE`
- `class ShardedMoEV4`
- `class DeepseekV4ShardingStrategy`
- `class GLM4MoeLiteShardingStrategy`
- `class WrappedMiniMaxAttention`
- `class MiniMaxShardingStrategy`
- `class QwenShardingStrategy`
- `class Glm4MoeShardingStrategy`
- `class GptOssShardingStrategy`
- `class Step35ShardingStrategy`
- `class NemotronHShardingStrategy`
- `class WrappedGemma4Experts`
- `class Gemma4ShardingStrategy`
- `def original_layer`
- `def patched`
- `def patched_call`
- `def patched_call`
- `def shard_model`
- `def shard_model`
- `def shard_model`
- `def shard_model`
- `def shard_model`
- `def shard_model`
- `def shard_model`
- `def shard_model`
- `def shard_model`
- `def shard_model`
- `def shard_model`
- `def shard_model`
- `def shard_heads`
- `def shard_heads`

### src/exo/worker/engines/mlx/builder.py (5)
- `class MlxBuilder`
- `def connect`
- `def load`
- `def close`
- `def build`

### src/exo/worker/engines/mlx/cache.py (22)
- `class CacheSnapshot`
- `def copy_rotating_kv_cache`
- `def copy_snapshot_entry`
- `def snapshot_ssm_states`
- `def is_non_trimmable_cache_entry`
- `def has_non_kv_caches`
- `class KVPrefixCache`
- `def trim_cache`
- `def encode_prompt`
- `def cache_length`
- `def get_prefix_length`
- `def get_available_memory`
- `def get_memory_used_percentage`
- `def make_kv_cache`
- `def clear`
- `def add_kv_cache`
- `def update_kv_cache`
- `def should_update_entry`
- `def get_kv_cache`
- `def evict_for_prefill`
- `def flush_to_disk`
- `def get_memory_used_percentage`

### src/exo/worker/engines/mlx/types.py (1)
- `class Model`

### src/exo/worker/engines/mlx/utils_mlx.py (27)
- `def get_weights_size`
- `class HostList`
- `def mlx_distributed_init`
- `def initialize_mlx`
- `def load_mlx_items`
- `def shard_and_load`
- `def get_tokenizer`
- `def get_eos_token_ids_for_model`
- `def load_tokenizer_for_model_id`
- `def consolidate_system_messages`
- `def render_chat_template`
- `def apply_chat_template`
- `def system_prompt_token_count`
- `def detect_thinking_prompt_suffix`
- `def fix_unmatched_think_end_tokens`
- `class NullKVCache`
- `def mlx_force_oom`
- `def set_wired_limit_for_model`
- `def mlx_cleanup`
- `def mx_any`
- `def mx_barrier`
- `def mx_all_gather_tasks`
- `def from_hosts`
- `def state`
- `def state`
- `def encode_task_id`
- `def decode_task_id`

### src/exo/worker/engines/mlx/vision.py (13)
- `def decode_base64_image`
- `def build_vision_prompt`
- `class MediaRegion`
- `class VisionResult`
- `class VisionEncoder`
- `def get_inner_model`
- `def create_vision_embeddings`
- `class VisionProcessor`
- `def prepare_vision`
- `def ensure_loaded`
- `def encode_images`
- `def load`
- `def process`

### src/exo/worker/engines/mlx/patches/__init__.py (1)
- `def apply_mlx_patches`

### src/exo/worker/engines/mlx/patches/cuda_compat.py (1)
- `def apply_cuda_compat_patches`

### src/exo/worker/engines/mlx/patches/opt_batch_gen.py (5)
- `class BatchTopKLogprobs`
- `def set_needs_topk`
- `def take_ready_topk`
- `def apply_batch_gen_patch`
- `def for_uid`

### src/exo/worker/engines/mlx/patches/standard_yarn_rope.py (5)
- `def patch_yarn_rope`
- `def yarn_find_correction_dim`
- `def yarn_find_correction_range`
- `def yarn_get_mscale`
- `def yarn_linear_ramp_mask`

### src/exo/worker/engines/mlx/disaggregated/adapter.py (14)
- `def mx_dtype_to_str`
- `def wire_dtype_from_cache`
- `def str_to_mx_dtype`
- `def array_to_bytes`
- `def bytes_to_array`
- `def bhsd_to_nhd`
- `def nhd_to_bhsd`
- `def send_mlx_kv_cache`
- `def chunk_to_mlx_nhd`
- `def blob_to_mlx`
- `def inject_kv_chunk`
- `def inject_rotating_kv_chunk`
- `def inject_arrays_cache`
- `def write_cache_to_wire`

### src/exo/worker/engines/mlx/disaggregated/client.py (3)
- `class PrefillResult`
- `def remote_prefill_fetch`
- `def ingest_into_mlx_cache`

### src/exo/worker/engines/mlx/disaggregated/serve.py (1)
- `def run_prefill_for_request`

### src/exo/worker/engines/mlx/generator/batch_generate.py (6)
- `class ExoBatchGenerator`
- `def has_work`
- `def submit`
- `def step`
- `def cancel`
- `def close`

### src/exo/worker/engines/mlx/generator/generate.py (12)
- `def patch_embed_tokens`
- `class PrefillCancelled`
- `def pipeline_parallel_prefill`
- `def prefill`
- `def warmup_inference`
- `def ban_token_ids`
- `def eos_ids_from_tokenizer`
- `def extract_top_logprobs`
- `def mlx_generate`
- `def progress_callback`
- `def combined_progress_callback`
- `def proc`

### src/exo/worker/engines/mlx/generator/remote_prefill.py (1)
- `def remote_prefill`

### src/exo/worker/engines/mlx/vendor/deepseek_v4_encoding.py (14)
- `def to_json`
- `def tools_from_openai_format`
- `def tool_calls_from_openai_format`
- `def tool_calls_to_openai_format`
- `def encode_arguments_to_dsml`
- `def decode_dsml_to_arguments`
- `def render_tools`
- `def find_last_user_index`
- `def render_message`
- `def merge_tool_messages`
- `def sort_tool_results_by_call_order`
- `def encode_messages`
- `def parse_tool_calls`
- `def parse_message_from_completion_text`

### src/exo/worker/engines/mlx/vendor/dsml_encoding.py (2)
- `def encode_messages`
- `def parse_dsml_output`

