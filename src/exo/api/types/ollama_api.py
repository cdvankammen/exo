"""Pydantic models mirroring Ollama's REST wire format.

Covers the chat, generate, show, tags, and ps endpoints, used by :mod:`exo.api.adapters.ollama`."""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

from exo.shared.models.model_cards import ModelId

# https://github.com/ollama/ollama/blob/main/docs/api.md

OllamaRole = Literal["system", "user", "assistant", "tool"]
OllamaDoneReason = Literal["stop", "length", "tool_call", "error"]
OllamaCapability = Literal[
    "completion",
    "tools",
    "insert",
    "vision",
    "embedding",
    "thinking",
    "image",
    "audio",
]


class OllamaToolFunction(BaseModel, frozen=True):
    """Tool function name plus its JSON-serialized arguments."""

    name: str
    arguments: dict[str, Any] | str
    index: int | None = None


class OllamaToolCall(BaseModel, frozen=True):
    """A single tool invocation inside a message."""

    id: str | None = None
    type: Literal["function"] | None = None
    function: OllamaToolFunction


class OllamaMessage(BaseModel, frozen=True):
    """Chat message: role, content, optional thinking/tool calls/images."""

    role: OllamaRole
    content: str | None = None
    thinking: str | None = None
    tool_calls: list[OllamaToolCall] | None = None
    name: str | None = None
    tool_name: str | None = None
    images: list[str] | None = None


class OllamaOptions(BaseModel, frozen=True):
    """Sampling options mirroring Ollama's ``options`` block."""

    num_predict: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None


class OllamaChatRequest(BaseModel, frozen=True):
    """``POST /api/chat`` request body."""

    model: ModelId
    messages: list[OllamaMessage]
    stream: bool = True
    options: OllamaOptions | None = None
    tools: list[dict[str, Any]] | None = None
    format: Literal["json"] | dict[str, Any] | None = None
    keep_alive: str | int | None = None
    think: bool | None = None


class OllamaGenerateRequest(BaseModel, frozen=True):
    """``POST /api/generate`` request body."""

    model: ModelId
    prompt: str = ""
    system: str | None = None
    stream: bool = True
    options: OllamaOptions | None = None
    format: Literal["json"] | dict[str, Any] | None = None
    keep_alive: str | int | None = None
    think: bool | None = None
    raw: bool = False
    images: list[str] | None = None


class OllamaGenerateResponse(BaseModel, frozen=True, strict=True):
    """``POST /api/generate`` response body."""

    model: str
    created_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    response: str
    thinking: str | None = None
    done: bool
    done_reason: OllamaDoneReason | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


class OllamaShowRequest(BaseModel, frozen=True):
    """``POST /api/show`` request body."""

    name: str | None = None
    model: str | None = None
    verbose: bool | None = None


class OllamaChatResponse(BaseModel, frozen=True, strict=True):
    """``POST /api/chat`` response body."""

    model: str
    created_at: str = Field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    message: OllamaMessage
    done: bool
    done_reason: OllamaDoneReason | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None


class OllamaModelDetails(BaseModel, frozen=True, strict=True):
    """Model metadata: format, family, parameter size, quantization."""

    format: str | None = None
    family: str | None = None
    parameter_size: str | None = None
    quantization_level: str | None = None


class OllamaModelTag(BaseModel, frozen=True, strict=True):
    """One model entry in the ``/api/tags`` list."""

    name: str
    model: str | None = None
    modified_at: str | None = None
    size: int | None = None
    digest: str | None = None
    details: OllamaModelDetails | None = None


class OllamaTagsResponse(BaseModel, frozen=True, strict=True):
    """``GET /api/tags`` response body."""

    models: list[OllamaModelTag]


class OllamaShowResponse(BaseModel, frozen=True, strict=True):
    """``POST /api/show`` response: modelfile, template, capabilities."""

    modelfile: str | None = None
    parameters: str | None = None
    template: str | None = None
    details: OllamaModelDetails | None = None
    model_info: dict[str, Any] | None = None
    capabilities: list[OllamaCapability] = []


class OllamaPsModel(BaseModel, frozen=True, strict=True):
    """One loaded-model entry in the ``/api/ps`` list."""

    name: str
    model: str
    size: int
    digest: str | None = None
    details: OllamaModelDetails | None = None
    expires_at: str | None = None
    size_vram: int | None = None


class OllamaPsResponse(BaseModel, frozen=True, strict=True):
    """``GET /api/ps`` response: currently loaded models."""

    models: list[OllamaPsModel]
