"""Runner response types.

Generation, image, tool-call, finished, model-loading, cancelled, and prefill-progress responses from a runner."""

from collections.abc import Generator
from typing import Any, Literal

from exo.api.types import (
    FinishReason,
    GenerationStats,
    ImageGenerationStats,
    ToolCallItem,
    TopLogprobItem,
    Usage,
)
from exo.utils.pydantic_ext import TaggedModel


class BaseRunnerResponse(TaggedModel):
    """Base class for all runner responses (tagged by response type)."""


class GenerationResponse(BaseRunnerResponse):
    """A text-generation token response from a runner."""

    text: str
    token: int
    logprob: float | None = None
    top_logprobs: list[TopLogprobItem] | None = None
    finish_reason: FinishReason | None = None
    stats: GenerationStats | None = None
    usage: Usage | None
    is_thinking: bool = False
    # The stop sequence that terminated generation, when ``finish_reason`` is
    # "stop" because a user-supplied stop sequence matched (vs. a natural EOS).
    matched_stop_sequence: str | None = None


class ImageGenerationResponse(BaseRunnerResponse):
    """An image-generation response carrying raw image bytes."""

    image_data: bytes
    format: Literal["png", "jpeg", "webp"] = "png"
    stats: ImageGenerationStats | None = None
    image_index: int = 0

    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        """Redact the binary image payload from repr output."""
        for name, value in super().__repr_args__():  # pyright: ignore[reportAny]
            if name == "image_data":
                yield name, f"<{len(self.image_data)} bytes>"
            elif name is not None:
                yield name, value


class PartialImageResponse(BaseRunnerResponse):
    """A partial image response during streaming image generation."""

    image_data: bytes
    format: Literal["png", "jpeg", "webp"] = "png"
    partial_index: int
    total_partials: int
    image_index: int = 0

    def __repr_args__(self) -> Generator[tuple[str, Any], None, None]:
        """Redact the binary image payload from repr output."""
        for name, value in super().__repr_args__():  # pyright: ignore[reportAny]
            if name == "image_data":
                yield name, f"<{len(self.image_data)} bytes>"
            elif name is not None:
                yield name, value


class ToolCallResponse(BaseRunnerResponse):
    """A tool-call response from a runner."""

    tool_calls: list[ToolCallItem]
    usage: Usage | None
    stats: GenerationStats | None = None


class FinishedResponse(BaseRunnerResponse):
    """Signals the end of a runner's response stream."""


class ModelLoadingResponse(BaseRunnerResponse):
    """Progress notification while a model is loading layers."""

    layers_loaded: int
    total: int


class CancelledResponse(BaseRunnerResponse):
    """Signals that a runner's task was cancelled."""


class PrefillProgressResponse(BaseRunnerResponse):
    """Progress notification during prefill (token counts)."""

    processed_tokens: int
    total_tokens: int
