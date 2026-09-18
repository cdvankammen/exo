"""Tests for the T13 server-side tool-calling gate in the Chat Completions adapter.

Covers `chat_request_to_text_generation`: when
EXO_ENABLE_SERVERSIDE_TOOLCALLS disables server-side tool forwarding, both
`tools` and `tool_choice` must be None regardless of what the client sent.
"""

from typing import Any

import pytest

from exo.api.adapters.chat_completions import chat_request_to_text_generation
from exo.api.types.api import ChatCompletionMessage, ChatCompletionRequest
from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import TextGenerationTaskParams

_MODEL = ModelId("test-model")

_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {}},
}


def _request(tools: list[dict[str, Any]] | None = None) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=_MODEL,
        messages=[ChatCompletionMessage(role="user", content="hello")],
        tools=tools,
        tool_choice="auto" if tools else None,
    )


async def _convert(
    tools: list[dict[str, Any]] | None = None,
) -> TextGenerationTaskParams:
    return await chat_request_to_text_generation(_request(tools))


async def test_chat_tools_forwarded_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXO_ENABLE_SERVERSIDE_TOOLCALLS", raising=False)
    out = await _convert([_TOOL])
    assert out.tools == [_TOOL]


async def test_chat_tools_stripped_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_ENABLE_SERVERSIDE_TOOLCALLS", "0")
    out = await _convert([_TOOL])
    assert out.tools is None


async def test_chat_tool_choice_stripped_with_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_ENABLE_SERVERSIDE_TOOLCALLS", "0")
    out = await _convert([_TOOL])
    assert out.tool_choice is None


async def test_chat_tools_stripped_when_disabled_word_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_ENABLE_SERVERSIDE_TOOLCALLS", "off")
    out = await _convert([_TOOL])
    assert out.tools is None


async def test_chat_no_tools_stays_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXO_ENABLE_SERVERSIDE_TOOLCALLS", raising=False)
    out = await _convert(None)
    assert out.tools is None
    assert out.tool_choice is None
