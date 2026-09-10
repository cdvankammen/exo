"""Tests for constrained (JSON-schema guided) decoding — T28."""

from typing import cast

import mlx.core as mx
import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.generator.constrained_decoding import (
    ConstrainedDecodingProcessor,
    compile_json_schema,
)
from exo.worker.engines.mlx.generator.generate import make_constrained_processor


class _FakeTokenizer:
    """Minimal duck-typed tokenizer covering the processor's needs."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {
            "<unk>": 0,
            "<s>": 1,
            "</s>": 2,
            "{": 3,
            '"': 4,
            ":": 5,
            ",": 6,
            "}": 7,
            "true": 8,
            "false": 9,
            "null": 10,
            "a": 11,
            "b": 12,
            "hello": 13,
        }
        self.vocab_size: int = len(self._vocab)
        self.eos_token_id: int = 2
        self.name_or_path: str = "test-tokenizer"

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def decode(self, token_ids: list[int]) -> str:
        rev = {v: k for k, v in self._vocab.items()}
        return "".join(rev.get(i, "") for i in token_ids)


def _fake_tokenizer() -> TokenizerWrapper:
    """Minimal duck-typed tokenizer cast to the expected wrapper type."""
    return cast(TokenizerWrapper, cast(object, _FakeTokenizer()))


class TestCompileJsonSchema:
    def test_valid_schema_compiles(self) -> None:
        fsm = compile_json_schema({"type": "object"})
        assert fsm is not None

    def test_unsupported_keyword_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported JSON Schema keyword"):
            compile_json_schema({"$ref": "#/definitions/foo"})


class TestConstrainedDecodingProcessor:
    def test_masks_disallowed_tokens(self) -> None:
        tokenizer = _fake_tokenizer()
        proc = ConstrainedDecodingProcessor(tokenizer, {"type": "object"})

        # Fresh state: only structural tokens for "{" (start of object) + EOS
        # are allowed; "hello" must be masked.
        tokens = mx.array([1])  # <s>
        logits = mx.zeros((tokenizer.vocab_size,))
        out = proc(tokens, logits)

        hello_idx = tokenizer.get_vocab()["hello"]
        assert out[hello_idx].item() < -1e8

    def test_allows_eos_when_complete(self) -> None:
        tokenizer = _fake_tokenizer()
        proc = ConstrainedDecodingProcessor(tokenizer, {"type": "object"})

        # After generating "{", the FSM is in "waiting for key" — a completed
        # object state should allow EOS eventually; here just ensure the mask
        # never blows up the logits shape.
        tokens = mx.array([3])  # "{"
        logits = mx.zeros((tokenizer.vocab_size,))
        out = proc(tokens, logits)

        assert out.shape == logits.shape

    def test_vocab_mismatch_resizes_mask(self) -> None:
        """Logits with a larger vocab than the tokenizer's still get constrained.

        The mask is padded with -inf (blocking unknown ids) so the constraint
        applies to the overlapping ids instead of failing open.
        """
        tokenizer = _fake_tokenizer()
        proc = ConstrainedDecodingProcessor(tokenizer, {"type": "object"})
        vocab = tokenizer.get_vocab()

        tokens = mx.array([1])
        # logits of a LARGER vocab size than the processor's mask
        logits = mx.zeros((tokenizer.vocab_size + 7,))
        out = proc(tokens, logits)

        # output shape must match logits
        assert out.shape == logits.shape
        # the constraint still applies on the overlapping ids: "{" allowed,
        # "}" blocked at start state
        assert out[vocab["{"]].item() > -1e8
        assert out[vocab["}"]].item() < -1e8
        # unknown (padded) ids are blocked
        assert out[tokenizer.vocab_size + 3].item() < -1e8

    def test_vocab_mismatch_truncates_smaller(self) -> None:
        """Logits with a SMALLER vocab than the tokenizer's get truncated mask."""
        tokenizer = _fake_tokenizer()
        proc = ConstrainedDecodingProcessor(tokenizer, {"type": "object"})
        vocab = tokenizer.get_vocab()

        tokens = mx.array([1])
        logits = mx.zeros((tokenizer.vocab_size - 5,))
        out = proc(tokens, logits)

        assert out.shape == logits.shape
        # overlapping ids still constrained
        assert out[vocab["{"]].item() > -1e8
        assert out[vocab["}"]].item() < -1e8

    def test_reset_restores_start_state(self) -> None:
        tokenizer = _fake_tokenizer()
        vocab = tokenizer.get_vocab()
        proc = ConstrainedDecodingProcessor(tokenizer, {"type": "object"})

        # Fresh state allows "{"... (first call carries prompt context only;
        # nothing is walked, the mask constrains the first generated token)
        fresh = proc(mx.array([1]), mx.zeros((tokenizer.vocab_size,)))
        assert fresh[vocab["{"]].item() > -1e8
        # ...after consuming the generated "{", another "{" must be masked
        # (mid-object)...
        mid = proc(mx.array([1, 3]), mx.zeros((tokenizer.vocab_size,)))
        assert mid[vocab["{"]].item() < -1e8
        # ...and reset restores the fresh behavior.
        proc.reset()
        after_reset = proc(mx.array([1]), mx.zeros((tokenizer.vocab_size,)))
        assert after_reset[vocab["{"]].item() > -1e8

    def test_object_acceptance_path(self) -> None:
        """End-to-end FSM walk: { then the key's opening quote are allowed."""
        tokenizer = _fake_tokenizer()
        vocab = tokenizer.get_vocab()
        proc = ConstrainedDecodingProcessor(
            tokenizer,
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            },
        )

        # Start state: only "{" is structurally valid. First call carries
        # prompt context ([1]) only — nothing walked.
        out0 = proc(mx.array([1]), mx.zeros((tokenizer.vocab_size,)))
        assert out0[vocab["{"]].item() > -1e8
        assert out0[vocab["}"]].item() < -1e8

        # After the generated "{" is appended: the key must begin with a quote.
        out1 = proc(mx.array([1, 3]), mx.zeros((tokenizer.vocab_size,)))
        assert out1[vocab['"']].item() > -1e8
        assert out1[vocab["a"]].item() < -1e8

    def test_multi_property_object_requires_comma(self) -> None:
        """Two REQUIRED properties must be comma-separated (regression).

        After the first required value closes, ``}`` must NOT be reachable —
        closing early would skip the still-required second property.
        """
        tokenizer = _fake_tokenizer()
        vocab = tokenizer.get_vocab()
        proc = ConstrainedDecodingProcessor(
            tokenizer,
            {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "required": ["a", "b"],
            },
        )
        # Feed: {"a": "x"  — after the first value closes, only ',' is legal.
        # The FSM is driven byte-by-byte through tokens; simulate key/value.
        # First call is prompt context ([1]); each subsequent call appends one
        # generated token: { (3), " (4), a (11), " (4), : (5), " (4), a (11),
        # " (4).
        proc(mx.array([1]), mx.zeros((tokenizer.vocab_size,)))  # prompt
        proc(mx.array([1, 3]), mx.zeros((tokenizer.vocab_size,)))  # {
        proc(mx.array([1, 3, 4]), mx.zeros((tokenizer.vocab_size,)))  # "
        proc(mx.array([1, 3, 4, 11]), mx.zeros((tokenizer.vocab_size,)))  # a
        proc(mx.array([1, 3, 4, 11, 4]), mx.zeros((tokenizer.vocab_size,)))  # "
        proc(mx.array([1, 3, 4, 11, 4, 5]), mx.zeros((tokenizer.vocab_size,)))  # :
        proc(mx.array([1, 3, 4, 11, 4, 5, 4]), mx.zeros((tokenizer.vocab_size,)))  # "
        proc(mx.array([1, 3, 4, 11, 4, 5, 4, 11]), mx.zeros((tokenizer.vocab_size,)))  # a (string)
        out = proc(mx.array([1, 3, 4, 11, 4, 5, 4, 11, 4]), mx.zeros((tokenizer.vocab_size,)))  # closing "

        assert out[vocab[","]].item() > -1e8  # comma required between properties
        assert out[vocab["}"]].item() < -1e8  # ...but early close would skip "b"

    def test_optional_properties_can_close_early(self) -> None:
        """A non-required property may be omitted: after a value, both the
        comma (next property) and ``}`` (close) are legal."""
        tokenizer = _fake_tokenizer()
        vocab = tokenizer.get_vocab()
        proc = ConstrainedDecodingProcessor(
            tokenizer,
            {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "required": ["a"],
            },
        )
        proc(mx.array([1]), mx.zeros((tokenizer.vocab_size,)))  # prompt
        proc(mx.array([1, 3]), mx.zeros((tokenizer.vocab_size,)))  # {
        proc(mx.array([1, 3, 4]), mx.zeros((tokenizer.vocab_size,)))  # "
        proc(mx.array([1, 3, 4, 11]), mx.zeros((tokenizer.vocab_size,)))  # a
        proc(mx.array([1, 3, 4, 11, 4]), mx.zeros((tokenizer.vocab_size,)))  # "
        proc(mx.array([1, 3, 4, 11, 4, 5]), mx.zeros((tokenizer.vocab_size,)))  # :
        proc(mx.array([1, 3, 4, 11, 4, 5, 4]), mx.zeros((tokenizer.vocab_size,)))  # "
        proc(mx.array([1, 3, 4, 11, 4, 5, 4, 11]), mx.zeros((tokenizer.vocab_size,)))  # a (string)
        out = proc(mx.array([1, 3, 4, 11, 4, 5, 4, 11, 4]), mx.zeros((tokenizer.vocab_size,)))  # closing "

        assert out[vocab[","]].item() > -1e8  # comma -> next (optional) property
        assert out[vocab["}"]].item() > -1e8  # } -> close with only "a" present

    def test_required_object_rejects_empty_body(self) -> None:
        """With required properties, ``{}`` is not a valid completion."""
        tokenizer = _fake_tokenizer()
        vocab = tokenizer.get_vocab()
        proc = ConstrainedDecodingProcessor(
            tokenizer,
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            },
        )
        # After "{", the key's opening quote is forced — "}" (empty object)
        # must be masked since "a" is required.
        out = proc(mx.array([1, 3]), mx.zeros((tokenizer.vocab_size,)))
        assert out[vocab['"']].item() > -1e8
        assert out[vocab["}"]].item() < -1e8


class TestMakeConstrainedProcessor:
    def _task(self, **kw: object) -> TextGenerationTaskParams:
        return TextGenerationTaskParams(model=ModelId("test-model"), input=[], **kw)  # type: ignore[arg-type]

    def test_none_when_not_requested(self) -> None:
        proc = make_constrained_processor(self._task(), _fake_tokenizer())
        assert proc is None

    def test_builds_processor_for_dict_schema(self) -> None:
        proc = make_constrained_processor(
            self._task(response_format={"type": "object"}), _fake_tokenizer()
        )
        assert proc is not None
        assert isinstance(proc, ConstrainedDecodingProcessor)

    def test_parses_json_string_schema(self) -> None:
        proc = make_constrained_processor(
            self._task(response_format='{"type": "object"}'), _fake_tokenizer()
        )
        assert proc is not None

    def test_invalid_json_string_fails_open(self) -> None:
        proc = make_constrained_processor(
            self._task(response_format="{not json"), _fake_tokenizer()
        )
        assert proc is None

    def test_unsupported_schema_fails_open(self) -> None:
        proc = make_constrained_processor(
            self._task(response_format={"$ref": "#/definitions/x"}), _fake_tokenizer()
        )
        assert proc is None

    def test_non_dict_non_str_fails_open(self) -> None:
        task = self._task()
        # Bypass pydantic validation to exercise the helper's isinstance guard.
        object.__setattr__(task, "response_format", ["not", "a", "dict"])
        proc = make_constrained_processor(task, _fake_tokenizer())
        assert proc is None

    def test_json_object_wrapper_compiles_as_bare_object(self) -> None:
        """OpenAI json_object mode (no schema) must constrain to an OBJECT.

        Without the unwrap, {"type": "json_object"} compiles as an unknown
        type and falls back to the string FSM — output would be forced to
        start with a quote. It must start with '{' instead.
        """
        tokenizer = _fake_tokenizer()
        vocab = tokenizer.get_vocab()
        proc = make_constrained_processor(
            self._task(response_format={"type": "json_object"}), tokenizer
        )
        assert proc is not None

        out = proc(mx.array([1]), mx.zeros((tokenizer.vocab_size,)))
        assert out[vocab["{"]].item() > -1e8  # object start allowed
        assert out[vocab['"']].item() < -1e8  # string start blocked


class TestEosGating:
    """EOS must only be allowed at FINAL accept states, not intermediate ones."""

    def test_eos_not_allowed_mid_object(self) -> None:
        """Right after '{', the empty-object epsilon accept must NOT allow EOS
        (greedy sampling would stop immediately with a partial document)."""
        tokenizer = _fake_tokenizer()
        vocab = tokenizer.get_vocab()
        proc = ConstrainedDecodingProcessor(
            tokenizer,
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )
        # Consume "{": the FSM is at the body-accept state. The processor
        # mimics mlx-lm's BatchGenerator: growing full history per call, with
        # a leading prompt token on the first call (prompt-boundary rule).
        history = [vocab["<s>"]]  # prompt last token (not walked)
        proc(mx.array(history), mx.zeros((tokenizer.vocab_size,)))
        history.append(vocab["{"])
        out = proc(mx.array(history), mx.zeros((tokenizer.vocab_size,)))

        assert out[vocab["<unk>"]].item() < -1e8  # sanity: mask applied
        assert proc.eos_token_id is not None
        # EOS must be MASKED here (mid-object).
        assert out[proc.eos_token_id].item() < -1e8

    def test_eos_allowed_after_object_closed(self) -> None:
        """After the full object (…"}" consumed), EOS IS allowed."""
        tokenizer = _fake_tokenizer()
        vocab = tokenizer.get_vocab()
        proc = ConstrainedDecodingProcessor(
            tokenizer,
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            },
        )
        # Walk: { " a " : " a " }  — the final state after the closing brace.
        history: list[int] = [vocab["<s>"]]  # prompt last token (not walked)
        proc(mx.array(history), mx.zeros((tokenizer.vocab_size,)))
        for tok in ("{", '"', "a", '"', ":", '"', "a", '"', "}"):
            history.append(vocab[tok])
            proc(mx.array(history), mx.zeros((tokenizer.vocab_size,)))

        # No new tokens: mask reflects the current (final) state.
        final = proc(mx.array(history), mx.zeros((tokenizer.vocab_size,)))
        assert proc._state == proc.fsm.n_states - 1  # type: ignore[reportPrivateUsage]
        # EOS should be allowed at the terminal accept state.
        assert final[proc.eos_token_id].item() > -1e8
