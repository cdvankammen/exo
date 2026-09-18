"""Pydantic schemas for exo's HTTP API wire format.

Models the chat-completion, image-generation, instance placement/creation, download, trace, and log endpoints. These types are shared by the FastAPI routes, the OpenAI-compatible adapters, and the internal command/event pipelines."""

import time
from collections.abc import Generator
from typing import Annotated, Any, Literal, get_args
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.text_generation import ReasoningDialect, ReasoningEffort
from exo.shared.types.worker.instances import Instance, InstanceId, InstanceMeta
from exo.shared.types.worker.shards import Sharding, ShardMetadata
from exo.utils.extra_fields_warner import WarnExtraModel
from exo.utils.pydantic_ext import FrozenModel

FinishReason = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call", "error"
]

# Stable, machine-readable error codes. Clients can branch on these without
# parsing the human message; `ErrorInfo.type` keeps the HTTP status phrase and
# `code` keeps the HTTP status code for backward compatibility.
ErrorCode = Literal[
    "INVALID_REQUEST",
    "UNAUTHORIZED",
    "INSUFFICIENT_MEMORY",
    "PLACEMENT_FAILED",
    "MODEL_NOT_FOUND",
    "INSTANCE_NOT_FOUND",
    "IMAGE_NOT_FOUND",
    "PEER_UNREACHABLE",
    "NODE_NOT_FOUND",
    "INSTANCES_RUNNING",
    "NOT_FOUND",
    "INPUT_TOO_LONG",
    "RATE_LIMITED",
    "QUOTA_EXCEEDED",
    "INTERNAL_ERROR",
]


class ErrorInfo(BaseModel):
    """Structured error payload returned on API failures.

    Carries a stable machine-readable ``error_code`` (see ``ErrorCode``)
    alongside the HTTP-style ``code``/``type`` so clients can branch
    programmatically without parsing the human ``message``.
    """

    message: str
    type: str
    param: str | None = None
    code: int
    error_code: ErrorCode = "INTERNAL_ERROR"


class ErrorResponse(BaseModel):
    """Uniform error envelope wrapping an :class:`ErrorInfo`."""

    error: ErrorInfo


class SettingsUpdateParams(BaseModel):
    """Body for PUT /v1/settings.

    ``value=None`` clears a persisted override, falling back to env/default.
    """

    var: str
    value: str | None = None


class TenantCreateParams(BaseModel):
    """Body for POST /v1/tenants (admin key creation).

    ``key`` is optional — when omitted a random key is generated and returned
    once in the response body.
    """

    key: str | None = None
    display_name: str = ""
    rate_limit: int = 60
    daily_tokens: int = 1_000_000
    models: str | list[str] = "*"
    note: str = ""


class TenantCreateResponse(BaseModel):
    """Response for POST /v1/tenants.

    ``key`` is only returned in full on creation for newly generated keys;
    listing keys omits secrets.
    """

    key: str
    display_name: str = ""
    created: bool = True


class TenantListResponse(BaseModel):
    """List of tenant keys without secrets (only metadata + quotas)."""

    data: list[dict[str, object]]


class ModelListModel(BaseModel):
    """A single model entry in ``GET /v1/models``.

    Mirrors the OpenAI ``model`` object for compatibility (``id``, ``object``,
    ``created``, ``owned_by``) and extends it with Open WebUI fields plus
    exo-specific metadata (storage size, task support, reasoning dialect).
    """

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "exo"
    # openwebui fields
    hugging_face_id: str = Field(default="")
    name: str = Field(default="")
    description: str = Field(default="")
    context_length: int = Field(default=0)
    tags: list[str] = Field(default=[])
    storage_size_megabytes: int = Field(default=0)
    supports_tensor: bool = Field(default=False)
    supports_ring: bool = Field(default=False)
    tasks: list[str] = Field(default=[])
    is_custom: bool = Field(default=False)
    family: str = Field(default="")
    quantization: str = Field(default="")
    base_model: str = Field(default="")
    capabilities: list[str] = Field(default_factory=list)
    reasoning_dialect: ReasoningDialect = "none"


class ModelList(BaseModel):
    """``GET /v1/models`` response body: the list payload wrapper."""

    object: Literal["list"] = "list"
    data: list[ModelListModel]


class ChatCompletionMessageText(BaseModel):
    """Text content part of a chat message."""

    type: Literal["text"] = "text"
    text: str


class ChatCompletionMessageImageUrl(BaseModel):
    """Image content part of a chat message (data URL or http(s) URL)."""

    type: Literal["image_url"] = "image_url"
    image_url: dict[str, str]  # {"url": "data:image/png;base64,..."}


ChatCompletionContentPart = ChatCompletionMessageText | ChatCompletionMessageImageUrl


class ToolCallItem(BaseModel):
    """A single tool invocation: name plus JSON-encoded arguments."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call attached to an assistant message (OpenAI wire shape)."""

    id: str
    index: int | None = None
    type: Literal["function"] = "function"
    function: ToolCallItem


class ChatCompletionMessage(BaseModel):
    """A chat message in OpenAI-compatible wire format.

    Supports text / image content parts, reasoning content (thinking), and
    tool calls; used for both requests and streamed deltas.
    """

    role: Literal["system", "user", "assistant", "developer", "tool", "function"]
    content: (
        str | ChatCompletionContentPart | list[ChatCompletionContentPart] | None
    ) = None
    reasoning_content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    function_call: dict[str, Any] | None = None


class BenchChatCompletionMessage(ChatCompletionMessage):
    """Chat message variant used by the benchmark mode (identical schema)."""



class TopLogprobItem(BaseModel):
    """Token-level logprob entry (``bytes`` is the UTF-8 encoding)."""

    token: str
    logprob: float
    bytes: list[int] | None = None


class LogprobsContentItem(BaseModel):
    """Per-token logprob with the top-k alternative tokens."""

    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprobItem]


class Logprobs(BaseModel):
    """Logprob payload attached to a message/choice (OpenAI wire shape)."""

    content: list[LogprobsContentItem] | None = None


class PromptTokensDetails(BaseModel):
    """Breakdown of prompt-token counts reported in ``usage``."""

    cached_tokens: int = 0
    audio_tokens: int = 0


class CompletionTokensDetails(BaseModel):
    """Breakdown of completion-token counts reported in ``usage``."""

    reasoning_tokens: int = 0
    audio_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


class Usage(BaseModel):
    """Token usage totals with per-category details."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails
    completion_tokens_details: CompletionTokensDetails


class StreamingChoiceResponse(BaseModel):
    """A single streaming chunk choice carrying a ``delta`` message."""

    index: int
    delta: ChatCompletionMessage
    logprobs: Logprobs | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None


class ChatCompletionChoice(BaseModel):
    """A single non-streamed completion choice."""

    index: int
    message: ChatCompletionMessage
    logprobs: Logprobs | None = None
    finish_reason: FinishReason | None = None


class GenerationStats(BaseModel):
    """Throughput and memory statistics for one generation run.

    ``*_tps`` are tokens per second; ``prefix_cache_hit`` describes how much
    of the prompt matched the shared KV prefix cache. ``time_to_first_token``
    and ``total_time`` are wall-clock seconds measured in the runner/API;
    ``decode_latency_ms`` is the average per-token decode time in milliseconds.
    """

    prompt_tps: float
    generation_tps: float
    prompt_tokens: int
    generation_tokens: int
    peak_memory_usage: Memory
    prefix_cache_hit: Literal["none", "partial", "exact"] = "none"
    # --- Per-request latency tracking (seconds unless noted) ---
    time_to_first_token: float | None = None
    total_time: float | None = None
    decode_latency_ms: float | None = None


class ChatCompletionResponse(BaseModel):
    """Non-streamed ``/v1/chat/completions`` response body."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice | StreamingChoiceResponse]
    usage: Usage | None = None
    service_tier: str | None = None
    generation_stats: GenerationStats | None = None


class ImageGenerationStats(BaseModel):
    """Timing, sizing, and memory statistics for an image generation run."""

    seconds_per_step: float
    total_generation_time: float

    num_inference_steps: int
    num_images: int

    image_width: int
    image_height: int

    peak_memory_usage: Memory


class NodePowerStats(BaseModel, frozen=True):
    """Per-node power sampling stats (system power, optional phase split)."""

    node_id: NodeId
    samples: int
    avg_sys_power: float
    # Per-phase breakdown. Populated only when the caller marks a phase
    # boundary (e.g. prefill -> generation); None otherwise.
    prefill_avg_sys_power: float | None = None
    generation_avg_sys_power: float | None = None
    prefill_energy_joules: float | None = None
    generation_energy_joules: float | None = None


class PowerUsage(BaseModel, frozen=True):
    """Aggregated power/energy report across all nodes in a run."""

    elapsed_seconds: float
    nodes: list[NodePowerStats]
    total_avg_sys_power_watts: float
    total_energy_joules: float
    # Split between the prefill (prompt-processing) phase and the
    # generation/decode phase. Populated only when the caller marks a phase
    # boundary; None otherwise. The two phase energies should sum to
    # approximately `total_energy_joules` (modulo interpolation rounding).
    prefill_seconds: float | None = None
    generation_seconds: float | None = None
    prefill_energy_joules: float | None = None
    generation_energy_joules: float | None = None
    prefill_avg_sys_power_watts: float | None = None
    generation_avg_sys_power_watts: float | None = None


class BenchChatCompletionResponse(ChatCompletionResponse):
    """Completion response extended with power usage (benchmark mode)."""

    power_usage: PowerUsage | None = None


class StreamOptions(BaseModel):
    """Options controlling streaming behavior (e.g. include usage chunks)."""

    include_usage: bool = False


class ChatCompletionRequest(WarnExtraModel):
    """``/v1/chat/completions`` request body (OpenAI-compatible superset).

    Extends the upstream schema with exo sampling extras (``top_k``, ``min_p``,
    ``repetition_penalty``), thinking toggles (``enable_thinking``), and the
    ``use_prefix_cache`` flag controlling shared KV-prefix-cache participation.
    Unknown extra fields produce a warning instead of an error (WarnExtraModel).
    """

    model: ModelId
    frequency_penalty: float | None = None
    messages: list[ChatCompletionMessage]
    logit_bias: dict[str, int] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    max_tokens: int | None = None
    n: int | None = None
    presence_penalty: float | None = None
    response_format: dict[str, Any] | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    tools: list[dict[str, Any]] | None = None
    reasoning_effort: ReasoningEffort | None = None
    enable_thinking: bool | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    repetition_context_size: int | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    user: str | None = None
    # Whether this request may use the shared KV prefix cache. Defaults to True.
    # Set to False to run a full prefill+decode on a fresh per-request cache that
    # is discarded afterwards, for one-off or throwaway requests that should not
    # evict other clients' cached prefixes or leave a cache entry behind.
    use_prefix_cache: bool = True


class BenchChatCompletionRequest(ChatCompletionRequest):
    """Benchmark variant: never participates in the shared prefix cache."""

    use_prefix_cache: bool = False


class AddCustomModelParams(WarnExtraModel):
    """Body for registering a custom model by id."""

    model_id: ModelId


class HuggingFaceSearchResult(BaseModel):
    """One hit from a Hugging Face model search."""

    id: str
    author: str = ""
    downloads: int = 0
    likes: int = 0
    last_modified: str = ""
    tags: list[str] = Field(default_factory=list)


class PlaceInstanceParams(WarnExtraModel):
    """Placement request: materialize a model onto a set of nodes.

    Controls sharding strategy, instance type, minimum participating nodes,
    optional explicit node/layer constraints, and memory-sufficiency tolerance
    (``force_override`` / ``memory_tolerance``).
    """

    model_id: ModelId
    sharding: Sharding = Sharding.Pipeline
    instance_meta: InstanceMeta = InstanceMeta.MlxRing
    min_nodes: int = 1
    # When True, bypass memory-sufficiency checks and attempt to place/load
    # the model anyway.
    force_override: bool = False
    # Memory tolerance multiplier (0.0–1.0). 1.0 = strict (model must fit
    # available memory). Lower values relax the check (e.g. 0.5 admits a
    # cycle holding at least half the model's size). Ignored when
    # force_override is True.
    memory_tolerance: float = 1.0
    # Optional explicit per-node layer allocation (Pipeline sharding only).
    # Keys must exactly match a connected pipeline cycle and sum to the
    # model's layer count.
    node_layers: dict[NodeId, int] | None = None
    # Optional explicit node subset for placement. When set, only cycles that
    # exactly match these nodes are considered.
    node_ids: set[NodeId] | None = None


class CreateInstanceParams(WarnExtraModel):
    """Direct instance-creation request with a fully-formed :class:`Instance`."""

    instance: Instance
    # When True, bypass the total-available-memory check when creating this
    # instance ("load the model anyway").
    force_override: bool = False


class PlacementPreview(BaseModel):
    """One candidate placement outcome: instance + per-node memory delta."""

    model_id: ModelId
    sharding: Sharding
    instance_meta: InstanceMeta
    instance: Instance | None = None
    # Keys are NodeId strings, values are additional bytes that would be used on that node
    memory_delta_by_node: dict[str, int] | None = None
    error: str | None = None


class PlacementPreviewResponse(BaseModel):
    """``POST /v1/placement/preview`` response: list of candidate placements."""

    previews: list[PlacementPreview]


class NodeCompatibilityEntry(BaseModel):
    """Per-node compatibility for a given model: green/red + reason."""

    node_id: NodeId
    friendly_name: str
    compatible: bool
    reason: str | None = None
    # Memory info for display
    ram_available_gb: float | None = None
    ram_total_gb: float | None = None
    backends: list[str] | None = None
    # Whether this node is in the topology (connected)
    in_topology: bool = True

class NodeCompatibilityResponse(BaseModel):
    """Compatibility report for placing ``model_id``: per-node verdicts.

    ``per_node_size_gb`` is the memory each node needs given the sharding
    strategy (Pipeline splits the model; Tensor/Ring replicate it).
    """

    model_id: ModelId
    storage_size_gb: float
    # Per-node memory requirement given the sharding strategy (Pipeline splits
    # the model across nodes; Tensor/Ring replicate it).
    per_node_size_gb: float | None = None
    sharding: str | None = None
    num_nodes: int | None = None
    required_backends: list[str]
    nodes: list[NodeCompatibilityEntry]

class DeleteInstanceTaskParams(WarnExtraModel):
    """Body for requesting instance deletion."""

    instance_id: str


class AddPeerParams(BaseModel):
    """Parameters for manually adding a peer node by hostname/IP."""

    host: str
    # Port of the target's zenoh TCP listener (default 52414).
    zenoh_port: int = 52414
    # Port of the target's HTTP API (default 52415). Used to verify the peer
    # is genuinely reachable and to fetch its node id.
    # NOTE (manual-peer-join): this exists because topology edges are built
    # from HTTP pings to the API port (see _poll_connection_updates), not from
    # the zenoh dial. Verifying it here is what makes "connected" truthful.
    api_port: int = 52415


class AddPeerResponse(BaseModel):
    """Result of manually adding a peer: connectivity + verified node id."""

    host: str
    port: int
    connected: bool
    # Node id of the peer, if it was verified reachable.
    node_id: str | None = None
    # Error detail if the peer could not be verified.
    error: str | None = None


class CreateInstanceResponse(BaseModel):
    """Response after a create-instance command is dispatched."""

    message: str
    command_id: CommandId
    model_card: ModelCard


class DeleteInstanceResponse(BaseModel):
    """Response after a delete-instance command is dispatched."""

    message: str
    command_id: CommandId
    instance_id: InstanceId


class PromoteMasterResponse(BaseModel):
    """Response after a promote-master command is dispatched."""

    message: str
    command_id: CommandId
    target_node_id: NodeId


class AwaitInstanceReadyMessage(BaseModel):
    """SSE payload: the awaited instance is ready."""

    type: Literal["ready"] = "ready"
    instance: Instance


class AwaitInstanceTimeoutMessage(BaseModel):
    """SSE payload: the awaited instance did not become ready in time."""

    type: Literal["timeout"] = "timeout"
    message: str


class CancelCommandResponse(BaseModel):
    """Response confirming a command cancellation was dispatched."""

    message: str
    command_id: CommandId


class InstanceLinkBody(WarnExtraModel):
    """Body for linking prefill and decode instances (disaggregation)."""

    prefill_instances: list[InstanceId]
    decode_instances: list[InstanceId]


class InstanceLinkResponse(BaseModel):
    """Response confirming an instance link was created."""

    message: str
    command_id: CommandId


ImageSize = Literal[
    "auto",
    "512x512",
    "768x768",
    "1024x768",
    "768x1024",
    "1024x1024",
    "1024x1536",
    "1536x1024",
]


def normalize_image_size(v: object) -> ImageSize:
    """Shared validator for ImageSize fields: maps None → "auto" and rejects invalid values."""
    if v is None:
        return "auto"
    if v not in get_args(ImageSize):
        raise ValueError(f"Invalid size: {v!r}. Must be one of {get_args(ImageSize)}")
    return v  # pyright: ignore[reportReturnType]


class AdvancedImageParams(BaseModel):
    """Advanced knobs for image generation (seed, steps, guidance, negatives)."""

    seed: Annotated[int, Field(ge=0)] | None = None
    num_inference_steps: Annotated[int, Field(ge=1, le=100)] | None = None
    guidance: Annotated[float, Field(ge=1.0, le=20.0)] | None = None
    negative_prompt: str | None = None
    num_sync_steps: Annotated[int, Field(ge=1, le=100)] | None = None


class ImageGenerationTaskParams(WarnExtraModel):
    """``/v1/image/generations`` request body (OpenAI images superset).

    Supports base64/URL responses, streaming partial images, output format and
    compression, and advanced sampling params; ``bench`` marks benchmark
    requests and is preserved through serialization.
    """

    prompt: str
    background: str | None = None
    model: str
    moderation: str | None = None
    n: int | None = 1
    output_compression: int | None = None
    output_format: Literal["png", "jpeg", "webp"] = "png"
    partial_images: int | None = 0
    quality: Literal["high", "medium", "low"] | None = "medium"
    response_format: Literal["url", "b64_json"] | None = "b64_json"
    size: ImageSize = "auto"
    stream: bool | None = False
    style: str | None = "vivid"
    user: str | None = None
    advanced_params: AdvancedImageParams | None = None
    # Internal flag for benchmark mode - set by API, preserved through serialization
    bench: bool = False

    @field_validator("size", mode="before")
    @classmethod
    def normalize_size(cls, v: object) -> ImageSize:
        """Coerce ``None``/invalid sizes to the canonical ``ImageSize`` value."""
        return normalize_image_size(v)


class BenchImageGenerationTaskParams(ImageGenerationTaskParams):
    """Image-generation benchmark variant (``bench=True`` by default)."""

    bench: bool = True


class ImageEditsTaskParams(WarnExtraModel):
    """Internal task params for image-editing requests."""

    image_data: str = ""  # Base64-encoded image (empty when using chunked transfer)
    total_input_chunks: int = 0
    prompt: str
    model: str
    n: int | None = 1
    quality: Literal["high", "medium", "low"] | None = "medium"
    output_format: Literal["png", "jpeg", "webp"] = "png"
    response_format: Literal["url", "b64_json"] | None = "b64_json"
    size: ImageSize = "auto"
    image_strength: float | None = 0.7
    stream: bool = False
    partial_images: int | None = 0
    advanced_params: AdvancedImageParams | None = None
    bench: bool = False

    @field_validator("size", mode="before")
    @classmethod
    def normalize_size(cls, v: object) -> ImageSize:
        """Coerce ``None``/invalid sizes to the canonical ``ImageSize`` value."""
        return normalize_image_size(v)

    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        """Redact the base64 ``image_data`` payload from repr output."""
        for name, value in super().__repr_args__():  # pyright: ignore[reportAny]
            if name == "image_data":
                yield name, f"<{len(self.image_data)} chars>"
            elif name is not None:
                yield name, value


class ImageData(BaseModel):
    """One generated image: base64 payload or URL, plus optional revised prompt."""

    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None

    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        """Redact the base64 ``b64_json`` payload from repr output."""
        for name, value in super().__repr_args__():  # pyright: ignore[reportAny]
            if name == "b64_json" and self.b64_json is not None:
                yield name, f"<{len(self.b64_json)} chars>"
            elif name is not None:
                yield name, value


class ImageGenerationResponse(BaseModel):
    """``/v1/image/generations`` response body."""

    created: int = Field(default_factory=lambda: int(time.time()))
    data: list[ImageData]


class BenchImageGenerationResponse(ImageGenerationResponse):
    """Image-generation benchmark response with stats + power usage."""

    generation_stats: ImageGenerationStats | None = None
    power_usage: PowerUsage | None = None


class ImageListItem(BaseModel, frozen=True):
    """One stored image entry in the image list endpoint."""

    image_id: str
    url: str
    content_type: str
    expires_at: float


class ImageListResponse(BaseModel, frozen=True):
    """``GET /v1/images`` response: stored image entries."""

    data: list[ImageListItem]


class StartDownloadParams(FrozenModel):
    """Request to start downloading a model shard onto a target node."""

    target_node_id: NodeId
    shard_metadata: ShardMetadata
    # When True, bypass the disk-space sufficiency check and attempt to
    # download the model anyway ("download the model anyway").
    force_override: bool = False


class StartDownloadResponse(FrozenModel):
    """Response confirming a download command was dispatched."""

    command_id: CommandId


class DeleteDownloadResponse(FrozenModel):
    """Response confirming a delete-download command was dispatched."""

    command_id: CommandId


class CancelDownloadParams(FrozenModel):
    """Request to cancel an in-flight download of a model shard."""

    target_node_id: NodeId
    model_id: ModelId


class CancelDownloadResponse(FrozenModel):
    """Response confirming a cancel-download command was dispatched."""

    command_id: CommandId


class TraceEventResponse(FrozenModel):
    """A single trace span: name, timing (microseconds), rank, category."""

    name: str
    start_us: int
    duration_us: int
    rank: int
    category: str


class TraceResponse(FrozenModel):
    """All trace spans collected for one task."""

    task_id: str
    traces: list[TraceEventResponse]


class TraceCategoryStats(FrozenModel):
    """Aggregate statistics for one trace category."""

    total_us: int
    count: int
    min_us: int
    max_us: int
    avg_us: float


class TraceRankStats(FrozenModel):
    """Aggregate trace statistics grouped by category, for one rank."""

    by_category: dict[str, TraceCategoryStats]


class TraceStatsResponse(FrozenModel):
    """Aggregate trace statistics for one task: total, by category, by rank."""

    task_id: str
    total_wall_time_us: int
    by_category: dict[str, TraceCategoryStats]
    by_rank: dict[int, TraceRankStats]


class TraceListItem(FrozenModel):
    """One stored trace file entry in the trace list endpoint."""

    task_id: str
    created_at: str
    file_size: int


class TraceListResponse(FrozenModel):
    """``GET /v1/traces`` response: stored trace entries."""

    traces: list[TraceListItem]


class DeleteTracesRequest(FrozenModel):
    """Body for bulk-deleting stored traces."""

    task_ids: list[str]


class DeleteTracesResponse(FrozenModel):
    """Result of a trace deletion: deleted ids and ids that were not found."""

    deleted: list[str]
    not_found: list[str]


class LogFileListItem(FrozenModel):
    """One stored log file entry in the log list endpoint."""

    name: str
    file_size: int
    modified_at: str


class LogFileListResponse(FrozenModel):
    """``GET /v1/logs`` response: stored log file entries."""

    logs: list[LogFileListItem]


class LogTailResponse(FrozenModel):
    """Tail of a log file: content plus a truncated flag."""

    name: str
    content: str
    truncated: bool


class LogErrorEntry(FrozenModel):
    """A single WARNING/ERROR/CRITICAL line parsed from a log file."""

    timestamp: str
    level: str
    source: str
    message: str
    source_log: str
    # Node that produced the entry (NodeId string). Local entries carry the
    # serving node's own id; remote entries carry the peer's id so the
    # dashboard can group/filter per node without parsing source_log.
    node_id: str | None = None
    # Surrounding log lines (±context_lines around the error) for debugging.
    context: list[str] | None = None


class LogErrorsResponse(FrozenModel):
    """``GET /v1/logs/errors`` response: parsed error entries."""

    errors: list[LogErrorEntry]
    truncated: bool
