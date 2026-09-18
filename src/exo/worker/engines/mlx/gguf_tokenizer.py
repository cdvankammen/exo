"""GGUF tokenizer bridge.

Two strategies:
1. If the model directory contains HuggingFace tokenizer files
   (``tokenizer.json`` / ``tokenizer_config.json``), use the normal
   ``load_tokenizer_for_model_id`` path.  This is the common case for
   GGUF repos on Hugging Face (e.g. TheBloke/*-GGUF) which ship
   tokenizer files alongside the ``.gguf`` weights.
2. Otherwise, build a sentencepiece tokenizer from the GGUF metadata
   (``tokenizer.ggml.tokens`` / ``tokenizer.ggml.scores`` / ...) and wrap it
   in a minimal HF-like shim so it satisfies ``TokenizerWrapper``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from exo.shared.types.common import ModelId

logger = logging.getLogger(__name__)

_TOKENIZER_MARKERS = ("tokenizer.json", "tokenizer_config.json")

_BYTE_PIECES = frozenset(f"<0x{i:02X}>" for i in range(256))  # compare UPPERCASE


def _has_hf_tokenizer(model_path: Path) -> bool:
    return any((model_path / marker).exists() for marker in _TOKENIZER_MARKERS)


def _build_spm_from_metadata(metadata: dict[str, Any]):
    """Build a sentencepiece tokenizer from GGUF metadata."""
    import sentencepiece as spm
    import sentencepiece.sentencepiece_model_pb2 as model_pb2  # type: ignore

    tokens = metadata["tokenizer.ggml.tokens"]
    bos = metadata["tokenizer.ggml.bos_token_id"].item()
    eos = metadata["tokenizer.ggml.eos_token_id"].item()
    unk = metadata["tokenizer.ggml.unknown_token_id"].item()

    # Real GGUF tokenizers ship the 256 byte pieces (<0x00>..<0xFF>); keep
    # byte_fallback on unconditionally (harmless without byte pieces, matches
    # mlx-examples/gguf_llm).
    normalizer_spec = model_pb2.NormalizerSpec(
        name="identity",
        precompiled_charsmap=b"",
        add_dummy_prefix=True,
        remove_extra_whitespaces=False,
        normalization_rule_tsv=b"",
    )
    trainer_spec = model_pb2.TrainerSpec(
        model_type="BPE",
        vocab_size=len(tokens),
        input_format="text",
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_by_number=True,
        treat_whitespace_as_suffix=False,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        vocabulary_output_piece_score=True,
        byte_fallback=True,
        unk_id=unk,
        bos_id=bos,
        eos_id=eos,
        pad_id=-1,
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
        pretokenization_delimiter="",
    )
    proto = model_pb2.ModelProto(
        trainer_spec=trainer_spec, normalizer_spec=normalizer_spec
    )
    scores = metadata.get("tokenizer.ggml.scores")
    scores = scores.tolist() if scores is not None else None
    token_types = metadata.get("tokenizer.ggml.token_type")
    token_types = token_types.tolist() if token_types is not None else None

    for i, token in enumerate(tokens):
        score = scores[i] if scores else 0
        token_type = token_types[i] if token_types else 0
        # GGUF token_type uses llama.cpp's enum; sentencepiece's proto uses
        # its own.  The overlap: 1=NORMAL, 2=UNKNOWN, 3=CONTROL.
        # GGUF byte pieces (<0x00>..<0xFF>) are type 1 (NORMAL) in GGUF but
        # MUST be type 6 (BYTE) in the sentencepiece proto or load fails with
        # "there are not 256 byte pieces although byte_fallback is true".
        if token in _BYTE_PIECES:
            token_type = 6  # BYTE
        proto.pieces.append(
            model_pb2.ModelProto.SentencePiece(
                piece=token, score=score, type=token_type
            )
        )

    return spm.SentencePieceProcessor(model_proto=proto.SerializeToString())


class GgufSpmTokenizer:
    """Minimal HF-like shim over a sentencepiece tokenizer.

    Exposes only the surface exo's ``TokenizerWrapper`` and ``render_chat_template``
    need.  There is no chat template inside a GGUF file, so prompts are rendered
    with the plain fallback path (``chat_template is None``).
    """

    def __init__(self, sp: Any, eos_token_id: int, bos_token_id: int):
        self._sp = sp
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.chat_template: str | None = None
        self.vocab_size = sp.get_piece_size()

    def encode(self, text: str, add_special_tokens: bool = True, **_: object) -> list[int]:
        ids = self._sp.encode(text, out_type=int)
        if add_special_tokens:
            return [self.bos_token_id] + ids
        return ids

    def decode(self, token_ids: list[int], **_: object) -> str:
        return self._sp.decode(list(token_ids))

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._sp.piece_to_id(token)

    def get_vocab(self) -> dict[str, int]:
        """Return the full vocab as {piece: id} (required by mlx_lm internals)."""
        return {
            self._sp.id_to_piece(i): i for i in range(self._sp.get_piece_size())
        }

    def __len__(self) -> int:
        return self.vocab_size


def load_gguf_tokenizer(
    model_path: Path,
    metadata: dict[str, Any],
    model_id: ModelId,
    *,
    trust_remote_code: bool = True,
):
    """Load a tokenizer for a GGUF model.

    Prefers HF tokenizer files in ``model_path``; falls back to a
    sentencepiece tokenizer built from the GGUF metadata.
    """
    if _has_hf_tokenizer(model_path):
        logger.debug("GGUF model dir has HF tokenizer files; using HF tokenizer")
        from exo.worker.engines.mlx.utils_mlx import load_tokenizer_for_model_id

        return load_tokenizer_for_model_id(
            model_id, model_path, trust_remote_code=trust_remote_code
        )
    logger.info(
        "No HF tokenizer files found for GGUF model; building tokenizer from GGUF metadata"
    )
    sp = _build_spm_from_metadata(metadata)
    for key in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.bos_token_id"):
        if key not in metadata:
            raise ValueError(f"GGUF metadata missing required key: {key}")
    eos = metadata["tokenizer.ggml.eos_token_id"].item()
    bos = metadata["tokenizer.ggml.bos_token_id"].item()
    shim = GgufSpmTokenizer(sp, eos_token_id=eos, bos_token_id=bos)

    from mlx_lm.tokenizer_utils import TokenizerWrapper

    return TokenizerWrapper(shim)