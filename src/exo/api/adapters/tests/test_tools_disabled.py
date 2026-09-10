"""Tests for T13: server-side tool execution opt-out via EXO_ENABLE_SERVERSIDE_TOOLCALLS.

Verifies that:
1. `tools_enabled()` reads the env correctly.
2. Each adapter (chat_completions, claude, responses, ollama) strips tools
   when disabled and passes them through when enabled.
"""

import os
from typing import Any, cast
from unittest.mock import patch

import pytest

from exo.api.types import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    ToolCallItem,
)
from exo.shared.types.common import ModelId


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _chat_request(
    tools: list[dict[str, Any]] | None = None,
) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=ModelId("test-model"),
        messages=[ChatCompletionMessage(role="user", content="hi")],
        tools=tools,
    )


_SAMPLE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


# ---------------------------------------------------------------------------
# constants.tools_enabled() unit tests
# ---------------------------------------------------------------------------


class TestToolsEnabled:
    """Direct unit tests for the tools_enabled() predicate."""

    def test_default_enabled(self) -> None:
        """When the env var is unset, tools are enabled (backward compat)."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EXO_ENABLE_SERVERSIDE_TOOLCALLS", None)
            from exo.shared.constants import tools_enabled
            assert tools_enabled() is True

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", "False", "NO"])
    def test_disabled_variants(self, val: str) -> None:
        from exo.shared.constants import tools_enabled
        with patch.dict(os.environ, {"EXO_ENABLE_SERVERSIDE_TOOLCALLS": val}):
            assert tools_enabled() is False

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "True", "YES", "anything"])
    def test_enabled_variants(self, val: str) -> None:
        from exo.shared.constants import tools_enabled
        with patch.dict(os.environ, {"EXO_ENABLE_SERVERSIDE_TOOLCALLS": val}):
            assert tools_enabled() is True


# ---------------------------------------------------------------------------
# chat_completions adapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_completions_tools_strip() -> None:
    """When disabled, tools are set to None in TextGenerationTaskParams."""
    from exo.api.adapters.chat_completions import chat_request_to_text_generation

    req = _chat_request(tools=_SAMPLE_TOOLS)
    with patch.dict(os.environ, {"EXO_ENABLE_SERVERSIDE_TOOLCALLS": "0"}):
        result = await chat_request_to_text_generation(req)
    assert result.tools is None
    assert result.tool_choice is None


@pytest.mark.asyncio
async def test_chat_completions_tools_pass() -> None:
    """When enabled (default), tools are forwarded."""
    from exo.api.adapters.chat_completions import chat_request_to_text_generation

    req = _chat_request(tools=_SAMPLE_TOOLS)
    result = await chat_request_to_text_generation(req)
    assert result.tools is not None
    assert len(result.tools) == 1


# ---------------------------------------------------------------------------
# claude adapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claude_tools_strip() -> None:
    """When disabled, tools are not forwarded in the internal params."""
    from exo.api.adapters.claude import claude_request_to_text_generation
    from exo.api.types.claude_api import (
        ClaudeMessage,
        ClaudeMessagesRequest,
        ClaudeTextBlock,
        ClaudeToolDefinition,
    )

    req = ClaudeMessagesRequest(
        model=ModelId("test-model"),
        max_tokens=256,
        messages=[
            ClaudeMessage(role="user", content=[ClaudeTextBlock(type="text", text="hi")])
        ],
        tools=[
            ClaudeToolDefinition(
                name="search",
                description="Search",
                input_schema={"type": "object"},
            )
        ],
    )
    with patch.dict(os.environ, {"EXO_ENABLE_SERVERSIDE_TOOLCALLS": "0"}):
        result = await claude_request_to_text_generation(req)
    assert result.tools is None


@pytest.mark.asyncio
async def test_claude_tools_pass() -> None:
    """When enabled, Claude tools are converted and forwarded."""
    from exo.api.adapters.claude import claude_request_to_text_generation
    from exo.api.types.claude_api import (
        ClaudeMessage,
        ClaudeMessagesRequest,
        ClaudeTextBlock,
        ClaudeToolDefinition,
    )

    req = ClaudeMessagesRequest(
        model=ModelId("test-model"),
        max_tokens=256,
        messages=[
            ClaudeMessage(role="user", content=[ClaudeTextBlock(type="text", text="hi")])
        ],
        tools=[
            ClaudeToolDefinition(
                name="search",
                description="Search",
                input_schema={"type": "object"},
            )
        ],
    )
    result = await claude_request_to_text_generation(req)
    assert result.tools is not None
    assert result.tools[0]["function"]["name"] == "search"


# ---------------------------------------------------------------------------
# responses adapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_responses_tools_strip() -> None:
    """When disabled, normalised_tools is None."""
    from exo.api.adapters.responses import responses_request_to_text_generation
    from exo.api.types.openai_responses import ResponsesRequest

    req = ResponsesRequest(
        model=ModelId("test-model"),
        input="hello",
        tools=[{"name": "foo", "description": "bar", "parameters": {}}],
    )
    with patch.dict(os.environ, {"EXO_ENABLE_SERVERSIDE_TOOLCALLS": "0"}):
        result = await responses_request_to_text_generation(req)
    assert result.tools is None


@pytest.mark.asyncio
async def test_responses_tools_pass() -> None:
    """When enabled, tools are normalised and forwarded."""
    from exo.api.adapters.responses import responses_request_to_text_generation
    from exo.api.types.openai_responses import ResponsesRequest

    req = ResponsesRequest(
        model=ModelId("test-model"),
        input="hello",
        tools=[{"name": "foo", "description": "bar", "parameters": {}}],
    )
    result = await responses_request_to_text_generation(req)
    assert result.tools is not None
    assert len(result.tools) == 1


# ---------------------------------------------------------------------------
# ollama adapter
# ---------------------------------------------------------------------------


class TestOllamaToolsStrip:
    """Ollama chat: tools stripped when disabled."""

    def test_tools_stripped(self) -> None:
        from exo.api.adapters.ollama import ollama_request_to_text_generation
        from exo.api.types.ollama_api import OllamaChatRequest, OllamaMessage

        req = OllamaChatRequest(
            model="test-model",
            messages=[OllamaMessage(role="user", content="hi")],
            tools=_SAMPLE_TOOLS,
        )
        with patch.dict(os.environ, {"EXO_ENABLE_SERVERSIDE_TOOLCALLS": "0"}):
            result = ollama_request_to_text_generation(req)
        assert result.tools is None


class TestOllamaToolsPass:
    """Ollama chat: tools forwarded when enabled."""

    def test_tools_forwarded(self) -> None:
        from exo.api.adapters.ollama import ollama_request_to_text_generation
        from exo.api.types.ollama_api import OllamaChatRequest, OllamaMessage

        req = OllamaChatRequest(
            model="test-model",
            messages=[OllamaMessage(role="user", content="hi")],
            tools=_SAMPLE_TOOLS,
        )
        result = ollama_request_to_text_generation(req)
        assert result.tools is not None
        assert len(result.tools) == 1


# ---------------------------------------------------------------------------
# feature flags endpoint
# ---------------------------------------------------------------------------


class TestFeatureFlags:
    """Verify /v1/feature-flags includes server_side_tools."""

    def test_feature_flag_present(self) -> None:
        from exo.api.main import API
        import asyncio

        # We just verify the function returns the key; no need to start the server.
        api = object.__new__(API)
        result = asyncio.get_event_loop().run_until_complete(
            api.get_feature_flags()
        )
        assert "server_side_tools" in result
        assert isinstance(result["server_side_tools"], bool)

    def test_feature_flag_responds_to_env(self) -> None:
        from exo.api.main import API
        import asyncio

        api = object.__new__(API)
        with patch.dict(os.environ, {"EXO_ENABLE_SERVERSIDE_TOOLCALLS": "0"}):
            result = asyncio.get_event_loop().run_until_complete(
                api.get_feature_flags()
            )
        assert result["server_side_tools"] is False
