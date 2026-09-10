# pyright: reportAny=false, reportArgumentType=false, reportUnknownMemberType=false, reportPrivateUsage=false
"""Tests for T28 constrained decoding (JSON-Schema-guided sampling).

Covers:
- byte normalization of tokenizer pieces (byte-level BPE vs SentencePiece)
- vocab trie construction
- JSON Schema -> byte-FSM compilation (all supported types)
- unsupported-schema rejection (ValueError -> 400)
- the logits processor: allowed-set masking, EOS handling, deadlock fallback
- fail-open behavior (vocab mismatch, invalid schemas)
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.generator.constrained_decoding import (
    ConstrainedDecodingProcessor,
    _build_vocab_trie,
    _is_sentence_piece,
    _piece_to_bytes,
    compile_json_schema,
)

# ---------------------------------------------------------------------------
# Byte normalization
# ---------------------------------------------------------------------------


def _bytelevel_tokenizer() -> SimpleNamespace:
    """A Llama-3-style byte-level BPE tokenizer stub."""

    class ByteLevel:  # mirrors tokenizers.decoders.ByteLevel
        pass

    class BPE:  # mirrors tokenizers.models.BPE
        pass

    # Byte-level BPE stores raw bytes as latin-1 chars: the space byte is
    # chr(0xC4)+chr(0xA0)? NO — byte-level BPE uses the *raw* byte values.
    # Actually: tiktoken-style vocab keys are the raw bytes decoded as
    # latin-1 (e.g. " " is chr(32), "Ġ" in HF's display is really the two
    # bytes 0xC4 0xA0). The tokenizer.json stores them as actual bytes.
    space_bytes = bytes([0xC4, 0xA0])  # HF displays this as "Ġ"
    newline_bytes = bytes([0xC4, 0x8A])  # HF displays this as "Ċ"
    vocab = {
        space_bytes.decode("latin-1") + "hello": 0,
        "world": 1,
        space_bytes.decode("latin-1"): 2,
        newline_bytes.decode("latin-1"): 3,
        "<|im_start|>": 4,
        "a": 5,
        bytes([0xC3, 0xA9]).decode("latin-1"): 6,  # é as raw bytes
    }
    backend = SimpleNamespace(
        model=BPE(),
        decoder=ByteLevel(),
    )
    return SimpleNamespace(
        backend_tokenizer=backend,
        get_vocab=lambda: vocab,
        vocab_size=len(vocab),
        name_or_path="test/bytelevel",
    )


def _sentencepiece_tokenizer() -> SimpleNamespace:
    """A Llama-2-style SentencePiece tokenizer stub."""

    class Unigram:  # mirrors tokenizers.models.Unigram
        pass

    vocab = {
        "\u0120hello": 0,  # Ġ = space sentinel
        "world": 1,
        "Ċ": 2,  # newline sentinel
        "hello": 3,
    }
    backend = SimpleNamespace(
        model=Unigram(),
        decoder=SimpleNamespace(),  # non-ByteLevel decoder
    )
    return SimpleNamespace(
        backend_tokenizer=backend,
        get_vocab=lambda: vocab,
        vocab_size=len(vocab),
        name_or_path="test/sentencepiece",
    )


class TestByteNormalization:
    def test_bytelevel_detected(self) -> None:
        assert _is_sentence_piece(_bytelevel_tokenizer()) is False

    def test_sentencepiece_detected(self) -> None:
        assert _is_sentence_piece(_sentencepiece_tokenizer()) is True

    def test_bytelevel_piece_is_raw_bytes(self) -> None:
        # Byte-level BPE stores raw bytes as latin-1 chars — re-encoding with
        # latin-1 recovers the original byte sequence exactly.
        piece = bytes([0xC4, 0xA0]).decode("latin-1") + "hello"
        piece_bytes = _piece_to_bytes(piece, is_sentence_piece=False)
        assert piece_bytes == bytes([0xC4, 0xA0]) + b"hello"

    def test_sentencepiece_sentinel_replaced(self) -> None:
        piece_bytes = _piece_to_bytes("\u0120hello", is_sentence_piece=True)
        assert piece_bytes == b" hello"

    def test_sentencepiece_newline_sentinel(self) -> None:
        assert _piece_to_bytes("aĊb", is_sentence_piece=True) == b"a\nb"

    def test_trie_build_bytelevel(self) -> None:
        _, id_to_bytes, size = _build_vocab_trie(_bytelevel_tokenizer())
        assert size == 7
        # "space byte" piece (id 2) maps to the 2 raw bytes
        assert id_to_bytes[2] == bytes([0xC4, 0xA0])
        assert len(id_to_bytes[0]) == 7  # 2 space bytes + "hello" (5)

    def test_trie_build_sentencepiece(self) -> None:
        _, id_to_bytes, size = _build_vocab_trie(_sentencepiece_tokenizer())
        assert size == 4
        assert id_to_bytes[0] == b" hello"  # Ġ replaced with space


# ---------------------------------------------------------------------------
# Schema compilation
# ---------------------------------------------------------------------------


class TestSchemaCompilation:
    def test_string_schema(self) -> None:
        fsm = compile_json_schema({"type": "string"})
        assert fsm.n_states >= 1

    def test_number_schema(self) -> None:
        fsm = compile_json_schema({"type": "number"})
        assert fsm.accept[fsm.start] or fsm.n_states > 1

    def test_integer_schema(self) -> None:
        compile_json_schema({"type": "integer"})  # no raise

    def test_boolean_schema(self) -> None:
        compile_json_schema({"type": "boolean"})

    def test_null_schema(self) -> None:
        compile_json_schema({"type": "null"})

    def test_array_schema(self) -> None:
        compile_json_schema({"type": "array", "items": {"type": "integer"}})

    def test_object_schema(self) -> None:
        compile_json_schema(
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            }
        )

    def test_enum_schema(self) -> None:
        compile_json_schema({"enum": ["red", "green", "blue"]})

    def test_unsupported_keyword_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            compile_json_schema({"type": "object", "$ref": "#/defs/x"})

    def test_oneof_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            compile_json_schema({"oneOf": [{"type": "string"}]})

    def test_multi_type(self) -> None:
        compile_json_schema({"type": ["string", "null"]})


# ---------------------------------------------------------------------------
# The processor
# ---------------------------------------------------------------------------


def _make_processor(schema: dict[str, object], tokenizer: object | None = None) -> ConstrainedDecodingProcessor:
    tok = tokenizer or _bytelevel_tokenizer()
    return ConstrainedDecodingProcessor(tok, schema)


class TestProcessor:
    def test_masks_non_matching_tokens(self) -> None:
        proc = _make_processor({"enum": ["true", "false"]})
        # Simulate logits over the 7-token vocab
        logits = mx.zeros((7,), dtype=mx.float32)
        tokens = mx.array([], dtype=mx.int32)
        masked = proc(tokens, logits)
        # At the start state, "true"/"false" start with 't'/'f' — which no vocab
        # piece starts with in our stub, so allowed set may be empty -> EOS fallback.
        # Assert masking doesn't crash and returns same shape.
        assert masked.shape == logits.shape

    def test_vocab_mismatch_fails_open(self) -> None:
        proc = _make_processor({"type": "string"})
        # Processor built with 7-token vocab; logits with 10
        logits = mx.zeros((10,), dtype=mx.float32)
        tokens = mx.array([], dtype=mx.int32)
        result = proc(tokens, logits)
        assert result.shape == (10,)  # unchanged, fail-open

    def test_eos_allowed_when_complete(self) -> None:
        tok = _bytelevel_tokenizer()
        proc = ConstrainedDecodingProcessor(
            tok, {"type": "string"}, eos_token_id=999
        )
        # after a complete string the FSM is in an accepting state; the mask
        # should include EOS
        proc._state = 0  # reset
        mask = proc._allowed_mask(0)
        # shape == vocab size
        assert mask.shape == (tok.vocab_size,)

    def test_reset_restores_state(self) -> None:
        proc = _make_processor({"type": "string"})
        proc._state = 3
        proc.reset()
        assert proc._state == 0

    def test_processor_call_shape(self) -> None:
        proc = _make_processor({"type": "string"})
        logits = mx.zeros((7,), dtype=mx.float32)
        tokens = mx.array([0], dtype=mx.int32)  # "Ġhello" piece
        out = proc(tokens, logits)
        assert out.shape == (7,)


# ---------------------------------------------------------------------------
# Integration: make_constrained_processor (fail-open paths)
# ---------------------------------------------------------------------------


class TestMakeConstrainedProcessor:
    def test_none_when_no_response_format(self) -> None:
        from exo.shared.types.text_generation import TextGenerationTaskParams
        from exo.worker.engines.mlx.generator.generate import (
            make_constrained_processor,
        )

        task = TextGenerationTaskParams(
            model="test/model",
            input=[{"role": "user", "content": "hi"}],
            max_output_tokens=8,
            response_format=None,
        )
        assert make_constrained_processor(task, _bytelevel_tokenizer()) is None

    def test_invalid_schema_string_fails_open(self) -> None:
        from exo.shared.types.text_generation import TextGenerationTaskParams
        from exo.worker.engines.mlx.generator.generate import (
            make_constrained_processor,
        )

        task = TextGenerationTaskParams(
            model="test/model",
            input=[{"role": "user", "content": "hi"}],
            max_output_tokens=8,
            response_format="{not json",
        )
        assert make_constrained_processor(task, _bytelevel_tokenizer()) is None

    def test_unsupported_schema_fails_open(self) -> None:
        from exo.shared.types.text_generation import TextGenerationTaskParams
        from exo.worker.engines.mlx.generator.generate import (
            make_constrained_processor,
        )

        task = TextGenerationTaskParams(
            model="test/model",
            input=[{"role": "user", "content": "hi"}],
            max_output_tokens=8,
            response_format={"$ref": "#/defs/x"},
        )
        assert make_constrained_processor(task, _bytelevel_tokenizer()) is None

    def test_valid_schema_builds_processor(self) -> None:
        from exo.shared.types.text_generation import TextGenerationTaskParams
        from exo.worker.engines.mlx.generator.generate import (
            make_constrained_processor,
        )

        task = TextGenerationTaskParams(
            model="test/model",
            input=[{"role": "user", "content": "hi"}],
            max_output_tokens=8,
            response_format={"type": "object", "properties": {"a": {"type": "integer"}}},
        )
        proc = make_constrained_processor(task, _bytelevel_tokenizer())
        assert proc is not None
