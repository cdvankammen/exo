"""Tests for the strict-extra mixin for public API request models.

The mixin started as ``WarnExtraModel`` (log-only, exo-explore/exo#2015) and
was flipped to reject unknown fields in the follow-up (exo-explore/exo#2313).
The old warn-only behavior is preserved in git history; ``WarnExtraModel``
remains as a backward-compatible alias of ``StrictExtraModel``.
"""

from collections.abc import Iterator

import pytest
from loguru import logger
from pydantic import AliasChoices, AliasPath, BaseModel, Field, ValidationError

from exo.utils.extra_fields_warner import StrictExtraModel, WarnExtraModel


@pytest.fixture
def captured_warnings() -> Iterator[list[str]]:
    """Capture loguru warnings for the duration of one test."""

    messages: list[str] = []
    sink_id = logger.add(
        lambda msg: messages.append(str(msg)),
        level="WARNING",
        format="{message}",
    )
    try:
        yield messages
    finally:
        logger.remove(sink_id)


class _ChatLikeRequest(StrictExtraModel):
    """Stand-in for a real public API request model."""

    model: str
    temperature: float | None = None
    max_tokens: int | None = None


def test_known_fields_parse_without_error() -> None:
    parsed = _ChatLikeRequest.model_validate(
        {"model": "qwen", "temperature": 0.5, "max_tokens": 100}
    )

    assert parsed.model == "qwen"
    assert parsed.temperature == 0.5
    assert parsed.max_tokens == 100


def test_unknown_field_is_rejected(captured_warnings: list[str]) -> None:
    # Mistyped field name — the historical bug case (e.g. `instance_meta`
    # arriving as `instanceMeta` on a snake_case model). Under the forbid
    # flip this is now a hard validation error, not a silent drop.
    with pytest.raises(ValidationError) as exc_info:
        _ChatLikeRequest.model_validate(
            {"model": "qwen", "tempreture": 0.5}  # typo, codespell-ignore
        )

    assert "tempreture" in str(exc_info.value)  # codespell-ignore
    assert "Extra inputs are not permitted" in str(exc_info.value)
    # No warning is emitted: the request never gets far enough to log.
    assert captured_warnings == []


def test_repeat_unknown_field_fails_every_time(captured_warnings: list[str]) -> None:
    for _ in range(3):
        with pytest.raises(ValidationError):
            _ChatLikeRequest.model_validate({"model": "qwen", "bogus": True})


def test_alias_choice_is_not_extra(captured_warnings: list[str]) -> None:
    class _MultiAlias(StrictExtraModel):
        thinking: bool = Field(
            default=False,
            validation_alias=AliasChoices("thinking", "enable_thinking"),
        )

    parsed_a = _MultiAlias.model_validate({"thinking": True})
    parsed_b = _MultiAlias.model_validate({"enable_thinking": True})

    assert parsed_a.thinking is True
    assert parsed_b.thinking is True


def test_alias_path_top_level_key_is_not_extra(
    captured_warnings: list[str],
) -> None:
    class _Pathed(StrictExtraModel):
        nested_value: int = Field(
            default=0, validation_alias=AliasPath("outer", "inner")
        )

    parsed = _Pathed.model_validate({"outer": {"inner": 7}})
    assert parsed.nested_value == 7


def test_subclass_inherits_strict_extra() -> None:
    # BenchChatCompletionRequest pattern: a subclass adds new fields and
    # still rejects undeclared keys, both inherited and added.
    class _BenchChatLike(_ChatLikeRequest):
        use_prefix_cache: bool = False

    parsed = _BenchChatLike.model_validate(
        {"model": "qwen", "use_prefix_cache": True}
    )
    assert parsed.use_prefix_cache is True

    with pytest.raises(ValidationError):
        _BenchChatLike.model_validate(
            {"model": "qwen", "use_prefix_cache": True, "wrong_key": "x"}
        )


def test_plain_basemodel_still_ignores_extras() -> None:
    # Sanity: a plain BaseModel keeps pydantic's historical default
    # (no warning, no error). Strictness is opt-in per model.
    class _Plain(BaseModel):
        x: int

    parsed = _Plain.model_validate({"x": 1, "extra": "ignored"})
    assert parsed.x == 1


def test_warnextramodel_alias_is_strict() -> None:
    # The old name still imports and still builds models — but those models
    # now forbid extra fields, matching StrictExtraModel.
    class _Legacy(WarnExtraModel):
        x: int

    assert _Legacy.model_config.get("extra") == "forbid"
    with pytest.raises(ValidationError):
        _Legacy.model_validate({"x": 1, "y": 2})