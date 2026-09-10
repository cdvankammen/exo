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
    "INPUT_TOO_LONG",
    "INSUFFICIENT_MEMORY",
    "PLACEMENT_FAILED",
    "MODEL_NOT_FOUND",
    "INSTANCE_NOT_FOUND",
    "IMAGE_NOT_FOUND",
    "PEER_UNREACHABLE",
    "NODE_NOT_FOUND",
    "INSTANCES_RUNNING",
    "NOT_FOUND",
    "INTERNAL_ERROR",
]


class ErrorInfo(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: int
    error_code: ErrorCode = "INTERNAL_ERROR"


class ErrorResponse(BaseModel):
    error: ErrorInfo


class SettingsUpdateParams(BaseModel):
    """Body for PUT /v1/settings.

    ``value=None`` clears a persisted override, falling back to env/default.
    """

    var: str
    value: str | None = None


class ModelListModel(BaseModel):
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
    object: Literal["list"] = "list"
    data: list[ModelListModel]


class ChatCompletionMessageText(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ChatCompletionMessageImageUrl(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: dict[str, str]  # {"url": "data:image/png;base64,..."}


ChatCompletionContentPart = ChatCompletionMessageText | ChatCompletionMessageImageUrl


class ToolCallItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    index: int | None = None
    type: Literal["function"] = "function"
    function: ToolCallItem


class ChatCompletionMessage(BaseModel):
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
    pass


class TopLogprobItem(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None


class LogprobsContentItem(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprobItem]


class Logprobs(BaseModel):
    content: list[LogprobsContentItem] | None = None


class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0
    audio_tokens: int = 0


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int = 0
    audio_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails
    completion_tokens_details: CompletionTokensDetails


class StreamingChoiceResponse(BaseModel):
    index: int
    delta: ChatCompletionMessage
    logprobs: Logprobs | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    logprobs: Logprobs | None = None
    finish_reason: FinishReason | None = None


class GenerationStats(BaseModel):
    prompt_tps: float
    generation_tps: float
    prompt_tokens: int
    generation_tokens: int
    peak_memory_usage: Memory
    prefix_cache_hit: Literal["none", "partial", "exact"] = "none"
    # API-level latency metrics (populated by RequestLatencyTracker)
    time_to_first_token_ms: float | None = None
    decode_latency_ms: float | None = None
    total_time_ms: float | None = None
    tokens_per_second: float | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice | StreamingChoiceResponse]
    usage: Usage | None = None
    service_tier: str | None = None
    generation_stats: GenerationStats | None = None


class ImageGenerationStats(BaseModel):
    seconds_per_step: float
    total_generation_time: float

    num_inference_steps: int
    num_images: int

    image_width: int
    image_height: int

    peak_memory_usage: Memory


class NodePowerStats(BaseModel, frozen=True):
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
    power_usage: PowerUsage | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(WarnExtraModel):
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
    use_prefix_cache: bool = False


class AddCustomModelParams(WarnExtraModel):
    model_id: ModelId


class HuggingFaceSearchResult(BaseModel):
    id: str
    author: str = ""
    downloads: int = 0
    likes: int = 0
    last_modified: str = ""
    tags: list[str] = Field(default_factory=list)


class PlaceInstanceParams(WarnExtraModel):
    model_id: ModelId
    sharding: Sharding = Sharding.Pipeline
    instance_meta: InstanceMeta = InstanceMeta.MlxRing
    min_nodes: int = 1
    # When True, bypass memory-sufficiency checks and attempt to place/load
    # the model anyway.
    force_override: bool = False
    # Optional explicit per-node layer allocation (Pipeline sharding only).
    # Keys must exactly match a connected pipeline cycle and sum to the
    # model's layer count.
    node_layers: dict[NodeId, int] | None = None


class CreateInstanceParams(WarnExtraModel):
    instance: Instance
    # When True, bypass the total-available-memory check when creating this
    # instance ("load the model anyway").
    force_override: bool = False


class PlacementPreview(BaseModel):
    model_id: ModelId
    sharding: Sharding
    instance_meta: InstanceMeta
    instance: Instance | None = None
    # Keys are NodeId strings, values are additional bytes that would be used on that node
    memory_delta_by_node: dict[str, int] | None = None
    error: str | None = None


class PlacementPreviewResponse(BaseModel):
    previews: list[PlacementPreview]


class DeleteInstanceTaskParams(WarnExtraModel):
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
    host: str
    port: int
    connected: bool
    # Node id of the peer, if it was verified reachable.
    node_id: str | None = None
    # Error detail if the peer could not be verified.
    error: str | None = None


class CreateInstanceResponse(BaseModel):
    message: str
    command_id: CommandId
    model_card: ModelCard


class DeleteInstanceResponse(BaseModel):
    message: str
    command_id: CommandId
    instance_id: InstanceId


class PromoteMasterResponse(BaseModel):
    message: str
    command_id: CommandId
    target_node_id: NodeId


class AwaitInstanceReadyMessage(BaseModel):
    type: Literal["ready"] = "ready"
    instance: Instance


class AwaitInstanceTimeoutMessage(BaseModel):
    type: Literal["timeout"] = "timeout"
    message: str


class CancelCommandResponse(BaseModel):
    message: str
    command_id: CommandId


class InstanceLinkBody(WarnExtraModel):
    prefill_instances: list[InstanceId]
    decode_instances: list[InstanceId]


class InstanceLinkResponse(BaseModel):
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
    seed: Annotated[int, Field(ge=0)] | None = None
    num_inference_steps: Annotated[int, Field(ge=1, le=100)] | None = None
    guidance: Annotated[float, Field(ge=1.0, le=20.0)] | None = None
    negative_prompt: str | None = None
    num_sync_steps: Annotated[int, Field(ge=1, le=100)] | None = None


class ImageGenerationTaskParams(WarnExtraModel):
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
        return normalize_image_size(v)


class BenchImageGenerationTaskParams(ImageGenerationTaskParams):
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
        return normalize_image_size(v)

    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        for name, value in super().__repr_args__():  # pyright: ignore[reportAny]
            if name == "image_data":
                yield name, f"<{len(self.image_data)} chars>"
            elif name is not None:
                yield name, value


class ImageData(BaseModel):
    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None

    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        for name, value in super().__repr_args__():  # pyright: ignore[reportAny]
            if name == "b64_json" and self.b64_json is not None:
                yield name, f"<{len(self.b64_json)} chars>"
            elif name is not None:
                yield name, value


class ImageGenerationResponse(BaseModel):
    created: int = Field(default_factory=lambda: int(time.time()))
    data: list[ImageData]


class BenchImageGenerationResponse(ImageGenerationResponse):
    generation_stats: ImageGenerationStats | None = None
    power_usage: PowerUsage | None = None


class ImageListItem(BaseModel, frozen=True):
    image_id: str
    url: str
    content_type: str
    expires_at: float


class ImageListResponse(BaseModel, frozen=True):
    data: list[ImageListItem]


class StartDownloadParams(FrozenModel):
    target_node_id: NodeId
    shard_metadata: ShardMetadata
    # When True, bypass the disk-space sufficiency check and attempt to
    # download the model anyway ("download the model anyway").
    force_override: bool = False


class StartDownloadResponse(FrozenModel):
    command_id: CommandId


class DeleteDownloadResponse(FrozenModel):
    command_id: CommandId


class CancelDownloadParams(FrozenModel):
    target_node_id: NodeId
    model_id: ModelId


class CancelDownloadResponse(FrozenModel):
    command_id: CommandId


class TraceEventResponse(FrozenModel):
    name: str
    start_us: int
    duration_us: int
    rank: int
    category: str


class TraceResponse(FrozenModel):
    task_id: str
    traces: list[TraceEventResponse]


class TraceCategoryStats(FrozenModel):
    total_us: int
    count: int
    min_us: int
    max_us: int
    avg_us: float


class TraceRankStats(FrozenModel):
    by_category: dict[str, TraceCategoryStats]


class TraceStatsResponse(FrozenModel):
    task_id: str
    total_wall_time_us: int
    by_category: dict[str, TraceCategoryStats]
    by_rank: dict[int, TraceRankStats]


class TraceListItem(FrozenModel):
    task_id: str
    created_at: str
    file_size: int


class TraceListResponse(FrozenModel):
    traces: list[TraceListItem]


class DeleteTracesRequest(FrozenModel):
    task_ids: list[str]


class DeleteTracesResponse(FrozenModel):
    deleted: list[str]
    not_found: list[str]


class LogFileListItem(FrozenModel):
    name: str
    file_size: int
    modified_at: str


class LogFileListResponse(FrozenModel):
    logs: list[LogFileListItem]


class LogTailResponse(FrozenModel):
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
    # Surrounding log lines (±context_lines around the error) for debugging.
    context: list[str] | None = None


class LogErrorsResponse(FrozenModel):
    errors: list[LogErrorEntry]
    truncated: bool


class LogNodeSource(FrozenModel):
    """Metadata about a node whose main log was included in an AllNodesLog."""

    node_id: str
    label: str
    line_count: int


class AllNodesLogResponse(FrozenModel):
    """Aggregated main log from every reachable cluster node, sorted by timestamp."""

    content: str
    truncated: bool
    nodes: list[LogNodeSource]

class NodeCompatibilityEntry(FrozenModel):
    """Per-node compatibility status for the model-compatibility endpoint."""

    node_id: NodeId
    friendly_name: str
    compatible: bool
    reason: str | None = None
    ram_available_gb: float | None = None
    ram_total_gb: float | None = None
    backends: list[str] = Field(default_factory=list)
    in_topology: bool = False


class NodeCompatibilityResponse(FrozenModel):
    """Response for the model-compatibility endpoint."""

    model_id: str
    storage_size_gb: float
    per_node_size_gb: float
    sharding: str
    num_nodes: int
    required_backends: list[str] = Field(default_factory=list)
    nodes: list[NodeCompatibilityEntry] = Field(default_factory=list)
