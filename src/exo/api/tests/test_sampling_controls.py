# pyright: reportAny=false
"""Tests for sampling-control plumbing: ChatCompletionRequest -> TextGenerationTaskParams
(logit_bias and friends).

The card contract requires logit_bias to survive the adapter conversion and be
available to the mlx_lm sampler (make_logits_processors).
"""

import pytest

from exo.api.adapters.chat_completions import chat_request_to_text_generation
from exo.api.types import ChatCompletionMessage, ChatCompletionRequest
from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import TextGenerationTaskParams

_TEST_MODEL = ModelId("test-model")


def _make_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=_TEST_MODEL,
        messages=[ChatCompletionMessage(role="user", content="hello")],
    )


class TestLogitBiasPlumbing:
    async def test_logit_bias_is_mapped_to_task_params(self) -> None:
        req = _make_request()
        req = ChatCompletionRequest(
            **{**req.model_dump(), "logit_bias": {"1": 5, "100": -2}}
        )
        task = await chat_request_to_text_generation(req)
        assert task.logit_bias == {"1": 5, "100": -2}

    async def test_logit_bias_none_passthrough(self) -> None:
        req = _make_request()
        task = await chat_request_to_text_generation(req)
        assert task.logit_bias is None

    async def test_other_sampling_params_still_map(self) -> None:
        req = ChatCompletionRequest(
            model=_TEST_MODEL,
            messages=[ChatCompletionMessage(role="user", content="hello")],
            temperature=0.3,
            top_p=0.9,
            top_k=40,
            min_p=0.05,
            seed=42,
            repetition_penalty=1.1,
            presence_penalty=0.2,
            frequency_penalty=0.3,
        )
        task = await chat_request_to_text_generation(req)
        assert task.temperature == 0.3
        assert task.top_p == 0.9
        assert task.top_k == 40
        assert task.min_p == 0.05
        assert task.seed == 42
        assert task.repetition_penalty == 1.1
        assert task.presence_penalty == 0.2
        assert task.frequency_penalty == 0.3


class TestTaskParamsDefaults:
    async def test_task_params_carry_logit_bias_through_model_copy(self) -> None:
        """TextGenerationTaskParams is frozen; ensure construction with the field
        works and survives copy/equality (used by the pipeline)."""
        p = TextGenerationTaskParams(
            model="m",
            input=[],
            logit_bias={"10": -1, "99": 3},
        )
        assert p.logit_bias == {"10": -1, "99": 3}
        assert p == TextGenerationTaskParams(
            model="m", input=[], logit_bias={"10": -1, "99": 3}
        )