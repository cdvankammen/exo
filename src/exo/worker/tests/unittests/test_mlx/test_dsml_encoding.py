# type: ignore
"""Round-trip and edge-case tests for the strict-mode DSML encoder/decoder.

Targets ``exo.worker.engines.mlx.vendor.deepseek_v4_encoding``, which had
0.0% line/branch coverage (card t_8b65b1c1, gap report
test-coverage-gap-analysis-2026-09-17.md sections 2/4/7.2).

Covers: argument encode/decode, tool-call ordering, thinking-message
dropping, stop-token reads, message rendering, DSML tool-call parsing,
malformed-input handling and encode->parse round trips.

STRICT-MODE QUIRKS LOCKED IN BY THESE TESTS
-------------------------------------------
The vendored module declares ``bos_token = ""`` and ``eos_token = ""``.
Consequences (asserted below, not papered over):

* ``_read_until_stop`` with an empty stop string matches at the *current*
  index, so ``parse_message_from_completion_text`` reads zero-length
  summaries in chat mode and always trips either the trailing-length guard
  or the final special-token guard. No real completion text can round-trip
  through the outer parser.
* ``parse_tool_calls`` IS fully functional and round-trips the exact block
  that ``encode_messages`` emits (``\n\n<dstool_calls>...>``).

These tests intentionally document the current behavior; fixing the empty
tokens is an upstream change outside this card's scope (tests only).
"""

import json

import pytest

from exo.worker.engines.mlx.vendor.deepseek_v4_encoding import (
    ASSISTANT_SP_TOKEN,
    DS_TASK_SP_TOKENS,
    LATEST_REMINDER_SP_TOKEN,
    REASONING_EFFORT_MAX,
    USER_SP_TOKEN,
    _drop_thinking_messages,
    _read_until_stop,
    bos_token,
    decode_dsml_to_arguments,
    dsml_token,
    encode_arguments_to_dsml,
    encode_messages,
    eos_token,
    find_last_user_index,
    merge_tool_messages,
    parse_message_from_completion_text,
    parse_tool_calls,
    render_message,
    render_tools,
    sort_tool_results_by_call_order,
    thinking_end_token,
    thinking_start_token,
    tool_call_template,
    tool_calls_block_name,
    tool_calls_template,
    tool_calls_to_openai_format,
)

T = dsml_token
TCS = f"<{T}{tool_calls_block_name}"  # <dstool_calls (no open-tag '>')
TCE = f"</{T}{tool_calls_block_name}>"  # </dstool_calls>
INVOKE_OPEN = f"<{T}invoke"
INVOKE_CLOSE = f"</{T}invoke>"  # closing tag (parser stop string omits '>')
INVOKE_STOP = f"</{T}invoke"  # parser stop string, no '>'
PARAM_OPEN = f"<{T}parameter"
PARAM_CLOSE = f"</{T}parameter>"  # closing tag
PARAM_STOP = f"/{T}parameter"  # parser stop string (no leading '<')

_ALL_TOKENS = [
    ASSISTANT_SP_TOKEN,
    LATEST_REMINDER_SP_TOKEN,
    USER_SP_TOKEN,
    eos_token,
    dsml_token,
]


def _tool_call(name: str, arguments: str = "{}", **extra) -> dict:
    """Internal-format tool call plus optional id field."""
    tc: dict = {"name": name, "arguments": arguments}
    tc.update(extra)
    return tc


def _openai_tool_call(name: str, arguments: str = "{}", **extra) -> dict:
    """OpenAI-format tool call (as found in conversation history)."""
    tc: dict = {"type": "function", "function": {"name": name, "arguments": arguments}}
    if "id" in extra:
        tc["id"] = extra["id"]
    return tc


def _assistant_with_tool_calls(tool_calls: list[dict]) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": tool_calls}


def _parse_block(body: str) -> tuple[str, int]:
    """Build a full DSML tool-calls completion block.

    body must contain the ``<dstool_calls>`` inner content *without* the
    open-tag '>' -- i.e. the exact layout ``encode_messages`` produces:
    ``\n\n<dstool_calls>\\n<dstinvoke ...>\\n</dstinvoke>\\n</dstool_calls>``.
    Returns the full text and the index parse_tool_calls should start at.
    """
    prefix = f"\n\n{TCS}"
    return prefix + ">\n" + body + f"\n{TCE}", len(prefix)


def _invoke(name: str, params: str = "") -> str:
    """One encoded invoke element (matches encode_arguments_to_dsml output)."""
    middle = f"{params}\n" if params else ""
    return f"{INVOKE_OPEN} name=\"{name}\">\n{middle}{INVOKE_CLOSE}"


def _param(name: str, value: str, is_string: bool) -> str:
    flag = "true" if is_string else "false"
    return f"{PARAM_OPEN} name=\"{name}\" string=\"{flag}\">{value}{PARAM_CLOSE}"


# ──────────────────────────────────────────────────────────────────────────
# tool_calls_to_openai_format
# ──────────────────────────────────────────────────────────────────────────


class TestToolCallsToOpenAIFormat:
    def test_converts_internal_to_openai(self):
        calls = [_tool_call("get_weather", '{"city": "Tokyo"}')]
        assert tool_calls_to_openai_format(calls) == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Tokyo"}',
                },
            }
        ]

    def test_empty_list(self):
        assert tool_calls_to_openai_format([]) == []


# ──────────────────────────────────────────────────────────────────────────
# encode_arguments_to_dsml / decode_dsml_to_arguments
# ──────────────────────────────────────────────────────────────────────────


class TestEncodeArgumentsToDsml:
    def test_string_value_is_marked_string(self):
        out = encode_arguments_to_dsml(_tool_call("t", '{"city": "Tokyo"}'))
        assert out == _param("city", "Tokyo", True)

    def test_multiple_string_values(self):
        out = encode_arguments_to_dsml(
            _tool_call("t", '{"city": "Tokyo", "units": "celsius"}')
        )
        assert out.split("\n") == [
            _param("city", "Tokyo", True),
            _param("units", "celsius", True),
        ]

    def test_non_string_values_serialized_as_json(self):
        out = encode_arguments_to_dsml(
            _tool_call("t", '{"n": 5, "flag": true, "lst": [1, 2], "o": {"a": 1}}')
        )
        lines = out.split("\n")
        assert lines[0] == _param("n", "5", False)
        assert lines[1] == _param("flag", "true", False)
        assert lines[2] == _param("lst", "[1, 2]", False)
        assert lines[3] == _param("o", '{"a": 1}', False)

    def test_malformed_json_falls_back_to_wrapper(self):
        out = encode_arguments_to_dsml(_tool_call("t", "not json at all"))
        assert out == _param("arguments", "not json at all", True)

    def test_empty_arguments_produce_empty_string(self):
        assert encode_arguments_to_dsml(_tool_call("t", "{}")) == ""

    def test_unicode_value_survives(self):
        out = encode_arguments_to_dsml(_tool_call("t", '{"city": "北京"}'))
        assert "北京" in out
        assert 'string="true"' in out
        assert f"<{T}parameter" in out


class TestDecodeDsmlToArguments:
    def test_string_params_are_quoted(self):
        out = decode_dsml_to_arguments(
            "get_weather", {"city": ("Tokyo", "true")}
        )
        assert out == {"name": "get_weather", "arguments": '{"city": "Tokyo"}'}

    def test_non_string_params_kept_as_raw_json(self):
        out = decode_dsml_to_arguments(
            "t", {"n": ("5", "false"), "flag": ("true", "false")}
        )
        assert json.loads(out["arguments"]) == {"n": 5, "flag": True}

    def test_mixed_params_and_order_preserved(self):
        out = decode_dsml_to_arguments(
            "t",
            {
                "a": ("1", "true"),
                "b": ('[1, 2]', "false"),
                "c": ('{"x": 1}', "false"),
            },
        )
        assert out == {
            "name": "t",
            "arguments": '{"a": "1", "b": [1, 2], "c": {"x": 1}}',
        }

    def test_empty_tool_args(self):
        out = decode_dsml_to_arguments("noop", {})
        assert out == {"name": "noop", "arguments": "{}"}

    def test_quotes_inside_string_value_escape_correctly(self):
        out = decode_dsml_to_arguments("t", {"msg": ('he said "hi"', "true")})
        assert json.loads(out["arguments"]) == {"msg": 'he said "hi"'}


# ──────────────────────────────────────────────────────────────────────────
# render_tools / find_last_user_index
# ──────────────────────────────────────────────────────────────────────────


class TestRenderTools:
    def test_renders_schemas_into_template(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Current weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        out = render_tools(tools)
        assert "## Tools" in out
        assert "get_weather" in out
        assert '"description": "Current weather"' in out
        assert f"<{T}tool_calls>" in out
        assert f"{{thinking_start_token}}" not in out  # template placeholders filled

    def test_empty_tools(self):
        out = render_tools([])
        assert "## Tools" in out


class TestFindLastUserIndex:
    def test_finds_last_user(self):
        msgs = [
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "u2"},
        ]
        assert find_last_user_index(msgs) == 3

    def test_developer_counts_as_user(self):
        msgs = [{"role": "developer", "content": "d"}, {"role": "user", "content": "u"}]
        assert find_last_user_index(msgs) == 1

    def test_no_user_returns_minus_one(self):
        assert find_last_user_index([{"role": "assistant", "content": "a"}]) == -1


# ──────────────────────────────────────────────────────────────────────────
# merge_tool_messages
# ──────────────────────────────────────────────────────────────────────────


class TestMergeToolMessages:
    def test_single_tool_message_becomes_user_with_tool_result(self):
        msgs = [{"role": "tool", "content": "rain", "tool_call_id": "c1"}]
        merged = merge_tool_messages(msgs)
        assert merged == [
            {
                "role": "user",
                "content_blocks": [
                    {"type": "tool_result", "tool_use_id": "c1", "content": "rain"}
                ],
            }
        ]

    def test_tool_after_user_merges_into_user_blocks(self):
        msgs = [
            {"role": "user", "content": "Weather?"},
            {"role": "tool", "content": "72F", "tool_call_id": "c1"},
        ]
        merged = merge_tool_messages(msgs)
        assert len(merged) == 1
        assert merged[0]["role"] == "user"
        assert merged[0]["content_blocks"] == [
            {"type": "text", "text": "Weather?"},
            {"type": "tool_result", "tool_use_id": "c1", "content": "72F"},
        ]

    def test_consecutive_tool_messages_merge_into_one_user(self):
        msgs = [
            {"role": "tool", "content": "a", "tool_call_id": "c1"},
            {"role": "tool", "content": "b", "tool_call_id": "c2"},
        ]
        merged = merge_tool_messages(msgs)
        assert len(merged) == 1
        assert [b["tool_use_id"] for b in merged[0]["content_blocks"]] == ["c1", "c2"]

    def test_tool_after_assistant_starts_new_user(self):
        msgs = [
            {"role": "assistant", "content": "calling..."},
            {"role": "tool", "content": "done", "tool_call_id": "c1"},
        ]
        merged = merge_tool_messages(msgs)
        assert [m["role"] for m in merged] == ["assistant", "user"]
        assert merged[1]["content_blocks"][0]["type"] == "tool_result"

    def test_consecutive_users_merge_text_blocks(self):
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        merged = merge_tool_messages(msgs)
        assert len(merged) == 1
        assert [b["text"] for b in merged[0]["content_blocks"]] == ["first", "second"]
        assert merged[0]["content"] == "first"

    def test_user_with_task_does_not_merge(self):
        msgs = [
            {"role": "user", "content": "a", "task": "query"},
            {"role": "user", "content": "b"},
        ]
        merged = merge_tool_messages(msgs)
        assert len(merged) == 2

    def test_extra_fields_preserved_on_new_user(self):
        msgs = [{"role": "user", "content": "a", "wo_eos": True, "mask": "x"}]
        merged = merge_tool_messages(msgs)
        assert merged[0]["wo_eos"] is True
        assert merged[0]["mask"] == "x"

    def test_other_roles_pass_through(self):
        msgs = [{"role": "system", "content": "s"}]
        assert merge_tool_messages(msgs) == [{"role": "system", "content": "s"}]

    def test_input_not_mutated(self):
        msgs = [{"role": "user", "content": "x"}]
        snapshot = [dict(m) for m in msgs]
        merge_tool_messages(msgs)
        assert msgs == snapshot


# ──────────────────────────────────────────────────────────────────────────
# sort_tool_results_by_call_order
# ──────────────────────────────────────────────────────────────────────────


class TestSortToolResultsByCallOrder:
    def _user_with_results(self, ids: list[str]) -> dict:
        return {
            "role": "user",
            "content_blocks": [
                {"type": "tool_result", "tool_use_id": i, "content": f"r{i}"}
                for i in ids
            ],
        }

    def test_reorders_results_to_assistant_call_order(self):
        msgs = [
            _assistant_with_tool_calls(
                [_openai_tool_call("a", id="call_2"), _openai_tool_call("b", id="call_1")]
            ),
            self._user_with_results(["call_1", "call_2"]),
        ]
        out = sort_tool_results_by_call_order(msgs)
        assert [b["tool_use_id"] for b in out[1]["content_blocks"]] == [
            "call_2",
            "call_1",
        ]

    def test_function_id_fallback(self):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "x", "type": "function", "function": {"name": "a"}},
                ],
            },
            self._user_with_results(["x"]),
        ]
        out = sort_tool_results_by_call_order(msgs)
        assert [b["tool_use_id"] for b in out[1]["content_blocks"]] == ["x"]

    def test_unknown_tool_use_id_sorts_first(self):
        msgs = [
            _assistant_with_tool_calls([_openai_tool_call("a", id="call_2")]),
            self._user_with_results(["ghost", "call_2"]),
        ]
        out = sort_tool_results_by_call_order(msgs)
        assert [b["tool_use_id"] for b in out[1]["content_blocks"]] == [
            "ghost",
            "call_2",
        ]

    def test_text_blocks_keep_their_positions(self):
        msgs = [
            _assistant_with_tool_calls([_openai_tool_call("a", id="c1")]),
            {
                "role": "user",
                "content_blocks": [
                    {"type": "text", "text": "t"},
                    {"type": "tool_result", "tool_use_id": "c1", "content": "r"},
                    {"type": "text", "text": "t2"},
                ],
            },
        ]
        out = sort_tool_results_by_call_order(msgs)
        assert [b["type"] for b in out[1]["content_blocks"]] == [
            "text",
            "tool_result",
            "text",
        ]

    def test_single_tool_block_left_alone(self):
        msgs = [
            _assistant_with_tool_calls([_openai_tool_call("a", id="c1")]),
            self._user_with_results(["c1"]),
        ]
        out = sort_tool_results_by_call_order(msgs)
        assert [b["tool_use_id"] for b in out[1]["content_blocks"]] == ["c1"]

    def test_no_prior_calls_leaves_order_unchanged(self):
        msgs = [self._user_with_results(["c2", "c1"])]
        out = sort_tool_results_by_call_order(msgs)
        assert [b["tool_use_id"] for b in out[0]["content_blocks"]] == ["c2", "c1"]

    def test_mutates_and_returns_same_list(self):
        msgs = [
            _assistant_with_tool_calls([_openai_tool_call("a", id="c1")]),
            self._user_with_results(["c1"]),
        ]
        assert sort_tool_results_by_call_order(msgs) is msgs


# ──────────────────────────────────────────────────────────────────────────
# _drop_thinking_messages
# ──────────────────────────────────────────────────────────────────────────


class TestDropThinkingMessages:
    def test_assistant_reasoning_before_last_user_removed(self):
        msgs = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "a",
                "reasoning_content": "secret reasoning",
            },
            {"role": "user", "content": "q2"},
        ]
        out = _drop_thinking_messages(msgs)
        assert len(out) == 3
        assert "reasoning_content" not in out[1]

    def test_assistant_after_last_user_keeps_reasoning(self):
        msgs = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "a",
                "reasoning_content": "keep me",
            },
        ]
        out = _drop_thinking_messages(msgs)
        assert out[1].get("reasoning_content") == "keep me"

    def test_developer_before_last_user_dropped(self):
        msgs = [
            {"role": "developer", "content": "old dev"},
            {"role": "assistant", "content": "a", "reasoning_content": "r"},
            {"role": "user", "content": "q"},
            {"role": "developer", "content": "new dev"},
        ]
        out = _drop_thinking_messages(msgs)
        assert [m.get("content") for m in out] == ["a", "q", "new dev"]

    def test_keep_roles_always_survive(self):
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "latest_reminder", "content": "remember"},
            {"role": "tool", "content": "t"},
            {"role": "direct_search_results", "content": "hits"},
        ]
        out = _drop_thinking_messages(msgs)
        assert len(out) == 4

    def test_input_not_mutated(self):
        msg = {"role": "assistant", "content": "a", "reasoning_content": "r"}
        msgs = [{"role": "user", "content": "q"}, dict(msg)]
        _drop_thinking_messages(msgs)
        assert msgs[1] == msg


# ──────────────────────────────────────────────────────────────────────────
# _read_until_stop
# ──────────────────────────────────────────────────────────────────────────


class TestReadUntilStop:
    def test_finds_earliest_stop(self):
        idx, content, stop = _read_until_stop(0, "abcXdefXghi", ["X"])
        assert (idx, content, stop) == (4, "abc", "X")

    def test_earliest_of_multiple_stops_wins(self):
        idx, content, stop = _read_until_stop(0, "abcXdefY", ["Y", "X"])
        assert (idx, content, stop) == (4, "abc", "X")

    def test_no_stop_consumes_rest(self):
        idx, content, stop = _read_until_stop(0, "abcXdef", ["Z"])
        assert (idx, content, stop) == (7, "abcXdef", None)

    def test_match_at_current_index_returns_empty_content(self):
        idx, content, stop = _read_until_stop(3, "abcXdef", ["X"])
        assert (idx, content, stop) == (4, "", "X")

    def test_empty_stop_string_acts_as_no_match(self):
        # 'if matched_stop:' treats '' as falsy, so an empty stop string is
        # never "matched": the no-match branch returns the whole tail.
        idx, content, stop = _read_until_stop(0, "abc", [""])
        assert (idx, content, stop) == (3, "abc", None)


# ──────────────────────────────────────────────────────────────────────────
# render_message
# ──────────────────────────────────────────────────────────────────────────


class TestRenderMessage:
    def test_system_with_tools_and_response_format(self):
        msgs = [
            {
                "role": "system",
                "content": "be nice",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "d",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "response_format": {"type": "json_object"},
            }
        ]
        out = render_message(0, msgs, "chat")
        assert out.startswith("be nice")
        assert "## Tools" in out
        assert '"type": "json_object"' in out
        assert f"## Response Format:" in out

    def test_developer_prefixed_with_user_token(self):
        msgs = [{"role": "developer", "content": "dev instr"}]
        out = render_message(0, msgs, "chat")
        # developer messages render like assistant messages: USER + content,
        # then ASSISTANT + thinking_end when nothing follows.
        assert out == USER_SP_TOKEN + "dev instr" + ASSISTANT_SP_TOKEN + thinking_end_token

    def test_developer_with_tools_and_response_format(self):
        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "d",
                "parameters": {"type": "object"},
            },
        }
        msgs = [
            {
                "role": "developer",
                "content": "instr",
                "tools": [tool],
                "response_format": {"type": "text"},
            }
        ]
        out = render_message(0, msgs, "chat")
        assert out.startswith(USER_SP_TOKEN + "instr")
        assert "## Tools" in out
        assert '"type": "text"' in out
        assert "## Response Format:" in out

    def test_developer_empty_content_asserts(self):
        with pytest.raises(AssertionError):
            render_message(0, [{"role": "developer", "content": ""}], "chat")

    def test_user_plain_content(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = render_message(0, msgs, "chat")
        assert out == USER_SP_TOKEN + "hi" + ASSISTANT_SP_TOKEN + thinking_end_token

    def test_user_content_blocks_text_and_tool_result(self):
        msgs = [
            {
                "role": "user",
                "content_blocks": [
                    {"type": "text", "text": "look"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "c1",
                        "content": "42",
                    },
                ],
            }
        ]
        out = render_message(0, msgs, "chat")
        assert out.startswith(USER_SP_TOKEN + "look\n\n<tool_result>42</tool_result>")

    def test_user_tool_result_with_list_content(self):
        msgs = [
            {
                "role": "user",
                "content_blocks": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "c1",
                        "content": [
                            {"type": "text", "text": "one"},
                            {"type": "image", "url": "x"},
                        ],
                    }
                ],
            }
        ]
        out = render_message(0, msgs, "chat")
        assert "<tool_result>one\n\n[Unsupported image]</tool_result>" in out

    def test_user_unsupported_block_type(self):
        msgs = [
            {"role": "user", "content_blocks": [{"type": "video", "url": "v"}]}
        ]
        out = render_message(0, msgs, "chat")
        assert "[Unsupported video]" in out

    def test_latest_reminder(self):
        msgs = [{"role": "latest_reminder", "content": "remember to reply"}]
        out = render_message(0, msgs, "chat")
        assert out == LATEST_REMINDER_SP_TOKEN + "remember to reply"

    def test_tool_role_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="merge_tool_messages"):
            render_message(0, [{"role": "tool", "content": "x"}], "chat")

    def test_unknown_role_raises(self):
        with pytest.raises(NotImplementedError, match="Unknown role"):
            render_message(0, [{"role": "bot", "content": "x"}], "chat")

    def test_assistant_renders_tool_call_block(self):
        msgs = [
            _assistant_with_tool_calls(
                [_openai_tool_call("get_weather", '{"city": "Tokyo"}')]
            )
        ]
        out = render_message(0, msgs, "chat")
        expected = (
            f"\n\n{TCS}>\n"
            f"{_invoke('get_weather', _param('city', 'Tokyo', True))}\n"
            f"{TCE}"
            f"{eos_token}"
        )
        assert out == expected

    def test_assistant_thinking_shown_only_in_thinking_mode_after_last_user(self):
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "reasoning_content": "r"},
        ]
        chat = render_message(1, msgs, "chat")
        assert "r" not in chat
        think = render_message(1, msgs, "thinking")
        assert "r" + thinking_end_token in think

    def test_assistant_thinking_hidden_before_last_user_when_dropping(self):
        msgs = [
            {"role": "assistant", "content": "a", "reasoning_content": "secret"},
            {"role": "user", "content": "q"},
            {"role": "user", "content": "q2"},
        ]
        out = render_message(0, msgs, "thinking", drop_thinking=True)
        assert "secret" not in out

    def test_assistant_thinking_kept_when_not_dropping(self):
        msgs = [
            {"role": "assistant", "content": "a", "reasoning_content": "secret"},
            {"role": "user", "content": "q"},
        ]
        out = render_message(0, msgs, "thinking", drop_thinking=False)
        assert "secret" in out

    def test_assistant_after_task_message_has_no_thinking(self):
        msgs = [
            {"role": "user", "content": "q", "task": "query"},
            {"role": "assistant", "content": "a", "reasoning_content": "none"},
        ]
        assert "none" not in render_message(1, msgs, "thinking")

    def test_wo_eos_and_eos_branches_produce_same_output(self):
        # eos_token is non-empty here, so the default template appends it
        # while wo_eos=True uses the no-eos template.
        msgs = [{"role": "assistant", "content": "hi"}]
        a = render_message(0, msgs, "chat")
        b = render_message(0, [{"role": "assistant", "content": "hi", "wo_eos": True}], "chat")
        assert a == "hi" + eos_token
        assert b == "hi"
        assert a != b

    def test_user_followed_by_assistant_gets_transition_tokens(self):
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        out = render_message(0, msgs, "chat")
        assert out == USER_SP_TOKEN + "q" + ASSISTANT_SP_TOKEN + thinking_end_token

    def test_user_followed_by_user_returns_early_without_transition(self):
        msgs = [{"role": "user", "content": "q1"}, {"role": "user", "content": "q2"}]
        out = render_message(0, msgs, "chat")
        assert out == USER_SP_TOKEN + "q1"

    def test_thinking_start_appended_for_last_user_in_thinking_mode(self):
        msgs = [{"role": "user", "content": "q"}]
        out = render_message(0, msgs, "thinking")
        assert out == USER_SP_TOKEN + "q" + ASSISTANT_SP_TOKEN + thinking_start_token

    def test_early_user_gets_thinking_end_in_thinking_mode(self):
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "q2"},
        ]
        out = render_message(0, msgs, "thinking")
        assert out == USER_SP_TOKEN + "q1" + ASSISTANT_SP_TOKEN + thinking_end_token

    def test_non_action_task_appends_task_token(self):
        msgs = [{"role": "user", "content": "q", "task": "query"}]
        out = render_message(0, msgs, "thinking")
        assert out == USER_SP_TOKEN + "q" + DS_TASK_SP_TOKENS["query"]

    def test_action_task_appends_assistant_and_thinking(self):
        msgs = [{"role": "user", "content": "q", "task": "action"}]
        out = render_message(0, msgs, "thinking")
        assert out == (
            USER_SP_TOKEN
            + "q"
            + ASSISTANT_SP_TOKEN
            + thinking_start_token
            + DS_TASK_SP_TOKENS["action"]
        )

    def test_invalid_task_asserts(self):
        msgs = [{"role": "user", "content": "q", "task": "bogus"}]
        with pytest.raises(AssertionError):
            render_message(0, msgs, "chat")

    def test_bad_index_asserts(self):
        with pytest.raises(AssertionError):
            render_message(5, [{"role": "user", "content": "q"}], "chat")

    def test_bad_thinking_mode_asserts(self):
        with pytest.raises(AssertionError):
            render_message(0, [{"role": "user", "content": "q"}], "bogus")

    def test_bad_reasoning_effort_asserts(self):
        with pytest.raises(AssertionError):
            render_message(
                0, [{"role": "user", "content": "q"}], "chat", reasoning_effort="low"
            )

    def test_max_reasoning_effort_prefix(self):
        msgs = [{"role": "user", "content": "q"}]
        out = render_message(0, msgs, "thinking", reasoning_effort="max")
        assert out.startswith(REASONING_EFFORT_MAX)


# ──────────────────────────────────────────────────────────────────────────
# encode_messages
# ──────────────────────────────────────────────────────────────────────────


class TestEncodeMessages:
    def test_chat_mode_simple_user(self):
        out = encode_messages([{"role": "user", "content": "hi"}], "chat")
        assert out == (
            bos_token
            + USER_SP_TOKEN
            + "hi"
            + ASSISTANT_SP_TOKEN
            + thinking_end_token
        )

    def test_thinking_mode_ends_with_thinking_start(self):
        out = encode_messages([{"role": "user", "content": "hi"}], "thinking")
        assert out.endswith(thinking_start_token)
        assert out.startswith(bos_token + USER_SP_TOKEN + "hi")

    def test_max_effort_prefixed_in_thinking_mode(self):
        out = encode_messages(
            [{"role": "user", "content": "hi"}], "thinking", reasoning_effort="max"
        )
        assert out.startswith(bos_token + REASONING_EFFORT_MAX)

    def test_bad_reasoning_effort_raises(self):
        with pytest.raises(AssertionError):
            encode_messages(
                [{"role": "user", "content": "hi"}], "thinking", reasoning_effort="x"
            )

    def test_tools_force_keeping_thinking(self):
        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "d",
                "parameters": {"type": "object"},
            },
        }
        # merge_tool_messages strips `tools` from user messages, so put the
        # tools on the system message (which is preserved as-is).
        msgs = [
            {"role": "system", "content": "sys", "tools": [tool]},
            {"role": "assistant", "content": "a", "reasoning_content": "secret"},
            {"role": "user", "content": "q"},
        ]
        out = encode_messages(msgs, "thinking")
        assert "secret" in out

    def test_thinking_dropped_without_tools(self):
        msgs = [
            {"role": "assistant", "content": "a", "reasoning_content": "secret"},
            {"role": "user", "content": "q"},
        ]
        out = encode_messages(msgs, "thinking")
        assert "secret" not in out

    def test_context_offsets_rendered_correctly(self):
        context = [{"role": "user", "content": "old"}]
        out = encode_messages([{"role": "user", "content": "new"}], "chat", context=context)
        # Context is used only for the render offset; the new messages are
        # rendered with `context_len` indexing into full_messages.
        assert out == USER_SP_TOKEN + "new" + ASSISTANT_SP_TOKEN + thinking_end_token

    def test_context_development_message_dropped_in_thinking(self):
        context = [{"role": "developer", "content": "old dev"}]
        out = encode_messages(
            [{"role": "user", "content": "q"}], "thinking", context=context
        )
        # The dev context message survives _drop_thinking_messages (it IS the
        # last "user"), so num_to_render = len([user]) - len([dev]) = 0 and
        # the new user message is NOT rendered at all.
        assert "old dev" not in out
        assert out == ""

    def test_bos_flag_accepted_both_ways(self):
        # add_default_bos_token controls whether the BOS token is prepended.
        a = encode_messages([{"role": "user", "content": "hi"}], "chat")
        b = encode_messages(
            [{"role": "user", "content": "hi"}], "chat", add_default_bos_token=False
        )
        assert a == bos_token + b
        assert b.startswith(USER_SP_TOKEN)

    def test_full_system_user_tool_call_flow(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Weather?"},
            _assistant_with_tool_calls([_openai_tool_call("get_weather", '{"city": "Tokyo"}')]),
        ]
        out = encode_messages(msgs, "chat")
        # system content, then user (USER + content + ASSISTANT + thinking_end),
        # then the tool-call block, then EOS from the assistant template.
        assert out.startswith(bos_token + "sys")
        assert USER_SP_TOKEN + "Weather?" + ASSISTANT_SP_TOKEN + thinking_end_token in out
        assert f"\n\n{TCS}>\n" in out
        assert f"name=\"get_weather\"" in out
        assert out.endswith(eos_token)

    def test_encoded_block_round_trips_through_parse_tool_calls(self):
        msgs = [
            _assistant_with_tool_calls(
                [
                    _openai_tool_call(
                        "get_weather", '{"city": "Tokyo", "units": "celsius"}'
                    ),
                    _openai_tool_call("get_time", '{"timezone": "EST"}'),
                ]
            )
        ]
        prompt = encode_messages(msgs, "chat")
        anchor = f"\n\n{TCS}"
        start = prompt.index(anchor) + len(anchor)
        idx, stop, calls = parse_tool_calls(start, prompt)
        # The assistant template appends eos_token after the block, which
        # parse_tool_calls (correctly) does not consume.
        assert stop == TCE
        assert idx == len(prompt) - len(eos_token)
        assert [c["name"] for c in calls] == ["get_weather", "get_time"]
        assert json.loads(calls[0]["arguments"]) == {"city": "Tokyo", "units": "celsius"}
        assert json.loads(calls[1]["arguments"]) == {"timezone": "EST"}


# ──────────────────────────────────────────────────────────────────────────
# parse_tool_calls
# ──────────────────────────────────────────────────────────────────────────


class TestParseToolCalls:
    def test_single_call_single_param(self):
        body = _invoke("get_weather", _param("city", "Tokyo", True))
        text, index = _parse_block(body)
        idx, stop, calls = parse_tool_calls(index, text)
        assert stop == TCE
        assert idx == len(text)
        assert calls == [{"name": "get_weather", "arguments": '{"city": "Tokyo"}'}]

    def test_multi_param_with_non_string(self):
        body = _invoke(
            "t",
            _param("city", "Tokyo", True)
            + "\n"
            + _param("n", "[1, 2]", False),
        )
        text, index = _parse_block(body)
        _, _, calls = parse_tool_calls(index, text)
        assert calls[0]["arguments"] == '{"city": "Tokyo", "n": [1, 2]}'

    def test_multiple_invocations(self):
        body = (
            _invoke("a", _param("x", "1", True))
            + "\n"
            + _invoke("b", _param("y", "2", True))
        )
        text, index = _parse_block(body)
        _, _, calls = parse_tool_calls(index, text)
        assert [c["name"] for c in calls] == ["a", "b"]
        assert json.loads(calls[1]["arguments"]) == {"y": "2"}

    def test_no_param_invoke(self):
        body = _invoke("noop")
        text, index = _parse_block(body)
        _, _, calls = parse_tool_calls(index, text)
        assert calls == [{"name": "noop", "arguments": "{}"}]

    def test_duplicate_param_name_raises(self):
        body = _invoke(
            "t", _param("x", "1", True) + "\n" + _param("x", "2", True)
        )
        text, index = _parse_block(body)
        with pytest.raises(ValueError, match="Duplicate parameter name"):
            parse_tool_calls(index, text)

    def test_tool_name_format_error(self):
        # A name-less invoke (`<dstinvoke>` with no name attr) must fail.
        body = f"{INVOKE_OPEN}>\n{INVOKE_CLOSE}"
        text, index = _parse_block(body)
        with pytest.raises(ValueError, match="Tool name format error"):
            parse_tool_calls(index, text)

    def test_empty_arguments_invoke_round_trips(self):
        # tool_call_template with empty arguments emits name=\"x\">\n\n</dstinvoke>
        # which DOES re-parse (regex `$` accepts the trailing newline).
        body = f"{INVOKE_OPEN} name=\"x\">\n\n{INVOKE_CLOSE}"
        text, index = _parse_block(body)
        _, _, calls = parse_tool_calls(index, text)
        assert calls == [{"name": "x", "arguments": "{}"}]

    def test_parameter_format_error(self):
        body = (
            f"{INVOKE_OPEN} name=\"x\">\n"
            f"{PARAM_OPEN} name=\"city\">Tokyo\n"  # missing string="..." attribute
            f"{INVOKE_CLOSE}"
        )
        text, index = _parse_block(body)
        with pytest.raises(ValueError, match="Parameter format error"):
            parse_tool_calls(index, text)

    def test_missing_open_tag_newline_raises(self):
        # Opening '<dstool_calls>' must be followed by '>\n' (the '>' being
        # the open-tag bracket); here it is missing.
        text = f"\n\n{TCS}" + f"{INVOKE_OPEN} name=\"x\">\n{INVOKE_CLOSE}" + f"\n{TCE}"
        with pytest.raises(ValueError, match="Tool call format error"):
            parse_tool_calls(len(f"\n\n{TCS}"), text)

    def test_missing_special_token_raises(self):
        text = f"\n\n{TCS}>" + "\n"
        with pytest.raises(ValueError, match="Missing special token in tool calls"):
            parse_tool_calls(len(f"\n\n{TCS}"), text)

    def test_garbage_after_open_tag_raises(self):
        text = f"\n\n{TCS}garbage"
        with pytest.raises(ValueError, match="Tool call format error"):
            parse_tool_calls(len(f"\n\n{TCS}"), text)


# ──────────────────────────────────────────────────────────────────────────
# parse_message_from_completion_text (strict-mode quirks)
# ──────────────────────────────────────────────────────────────────────────
#
# bos_token == '' and eos_token == '' make this outer parser reject every
# real completion: '' always matches at the current index, so summaries are
# read as empty and either the trailing-length guard or the final
# special-token guard ("" in content is always True) fails. The block-level
# parser (parse_tool_calls) is the functional path and is covered above.
# These tests document the strict behavior; do not "fix" them without first
# fixing the module (upstream change, out of scope for this card).


class TestParseMessageFromCompletionText:
    def test_plain_text_chat_mode_raises(self):
        with pytest.raises(AssertionError):
            parse_message_from_completion_text("hello world", "chat")

    def test_empty_text_chat_mode_raises(self):
        with pytest.raises(AssertionError):
            parse_message_from_completion_text("", "chat")

    def test_tool_call_text_chat_mode_parses(self):
        # Probe-verified: a bare tool-calls block in chat mode parses OK.
        body = _invoke("get_weather", _param("city", "Tokyo", True))
        text, index = _parse_block(body)
        parsed = parse_message_from_completion_text(text, "chat")
        assert parsed["role"] == "assistant"
        assert parsed["content"] == ""
        assert parsed["tool_calls"] == [
            {
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
            }
        ]

    def test_tool_call_text_chat_mode_with_junk_after_block_raises(self):
        body = _invoke("get_weather", _param("city", "Tokyo", True))
        text, index = _parse_block(body)
        with pytest.raises(AssertionError, match="Unexpected content after tool calls"):
            parse_message_from_completion_text(text + "junk", "chat")

    def test_thinking_with_exact_end_still_raises(self):
        text = "My reasoning" + " response"
        with pytest.raises(AssertionError):
            parse_message_from_completion_text(text, "thinking")

    def test_thinking_missing_close_raises(self):
        with pytest.raises(AssertionError, match="Invalid thinking format"):
            parse_message_from_completion_text("no close token here", "thinking")

    def test_thinking_then_summary_raises(self):
        text = "My reasoning" + " response" + "my summary"
        with pytest.raises(AssertionError):
            parse_message_from_completion_text(text, "thinking")

    def test_thinking_then_tool_call_runs_block_parser_before_raising(self):
        body = _invoke("get_weather", _param("city", "Tokyo", True))
        text, index = _parse_block(body)
        with pytest.raises(AssertionError):
            parse_message_from_completion_text("reason" + " response" + text, "thinking")