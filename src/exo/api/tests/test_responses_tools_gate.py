"""Tests for the T13 server-side tool-calling gate in the Responses adapter.

Covers `responses_request_to_text_generation`: when
EXO_ENABLE_SERVERSIDE_TOOLCALLS disables server-side tool forwarding,
normalised tools must be None regardless of what the client sent.
"""

from typing import Any

import pytest

from exo.api.adapters.responses import responses_request_to_text_generation
from exo.api.types.openai_responses import ResponsesRequest
from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import TextGenerationTaskParams

_MODEL = ModelId("test-model")

_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {}},
}


async def _convert(
    tools: list[dict[str, Any]] | None = None,
) -> TextGenerationTaskParams:
    request = ResponsesRequest(
        model=_MODEL,
        input="hello",
        tools=tools,
    )
    return await responses_request_to_text_generation(request)


async def test_responses_tools_forwarded_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXO_ENABLE_SERVERSIDE_TOOLCALLS", raising=False)
    out = await _convert([_TOOL])
    assert out.tools == [_TOOL]


async def test_responses_tools_stripped_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_ENABLE_SERVERSIDE_TOOLCALLS", "0")
    out = await _convert([_TOOL])
    assert out.tools is None


async def test_responses_tools_stripped_when_disabled_word_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_ENABLE_SERVERSIDE_TOOLCALLS", "off")
    out = await _convert([_TOOL])
    assert out.tools is None


async def test_responses_no_tools_stays_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXO_ENABLE_SERVERSIDE_TOOLCALLS", raising=False)
    out = await _convert(None)
    assert out.tools is None
