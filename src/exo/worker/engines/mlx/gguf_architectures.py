"""Architecture registry for GGUF model loading.

Maps GGUF ``general.architecture`` metadata to mlx_lm model classes and
argument dataclasses.  Each entry is a ``(model_cls, args_cls)`` pair whose
``from_dict`` accepts GGUF-derived config dicts.

Currently supported: llama (covers Llama, Mistral, Mixtral, and most GGUF
releases).  Other architectures can be added incrementally following the same
pattern used by mlx_lm's own model registry.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazily imported so the module stays importable even when mlx_lm models have
# not been installed (e.g. during type-checking or CI on non-macOS hosts).
_MLX_LLAMA: type | None = None
_MODEL_ARGS_CLASSES: dict[str, Any] = {}


def _ensure_imports() -> None:
    global _MLX_LLAMA  # noqa: PLW0603
    if _MLX_LLAMA is not None:
        return
    try:
        from mlx_lm.models import llama as _llama

        _MLX_LLAMA = _llama
    except ImportError:
        raise ImportError(
            "mlx_lm is required for GGUF model loading. "
            "Install it with: pip install mlx-lm"
        ) from None


# ---------------------------------------------------------------------------
# Metadata → ModelArgs mapping helpers
# ---------------------------------------------------------------------------

# GGUF metadata keys that map directly to mlx_lm LlamaModelArgs fields.
# Missing keys will be filled with sensible defaults.
_GGUF_TO_LLAMA_ARGS: dict[str, str | None] = {
    "llama.context_length": "max_position_embeddings",
    "llama.embedding_length": "hidden_size",
    "llama.block_count": "num_hidden_layers",
    "llama.attention.head_count": "num_attention_heads",
    "llama.feed_forward_length": "intermediate_size",
    "llama.attention.head_count_kv": "num_key_value_heads",
    "llama.attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "llama.attention.bias": "attention_bias",
    "llama.attention.sliding_window": "sliding_window",
    "llama.rope.freq_base": "rope_theta",
    "llama.rope.scaling": "rope_scaling",
}


def _to_python(val: Any) -> Any:
    """Convert mlx array / numpy scalars to native Python."""
    import mlx.core as mx

    if isinstance(val, mx.array):
        return val.item()
    return val


def gguf_metadata_to_llama_args(metadata: dict[str, Any]) -> dict[str, Any]:
    """Convert GGUF metadata to a dict suitable for ``LlamaModelArgs(**d)``.

    Applies defaults for fields that GGUF metadata does not provide (``model_type``,
    ``tie_word_embeddings``, ``rope_traditional``).
    """
    args: dict[str, Any] = {"model_type": "llama", "rope_traditional": True}

    for gguf_key, llama_key in _GGUF_TO_LLAMA_ARGS.items():
        if llama_key is None:
            continue
        if gguf_key in metadata:
            args[llama_key] = _to_python(metadata[gguf_key])

    # Set rope_scaling handling — GGUF stores it as a dict if present.
    if "llama.rope.scaling.type" in metadata:
        rope_type = _to_python(metadata["llama.rope.scaling.type"])
        rope_factor = _to_python(metadata.get("llama.rope.scaling.factor", 1.0))
        args["rope_scaling"] = {"type": rope_type, "factor": rope_factor}

    # Tie embeddings by default (most Llama-family GGUF models use tied embeddings)
    args.setdefault("tie_word_embeddings", True)

    return args


# ---------------------------------------------------------------------------
# Architecture dispatch
# ---------------------------------------------------------------------------

# Mapping: GGUF general.architecture → factory function
_ARCH_REGISTRY: dict[str, Any] = {}


def _register_llama_arch() -> None:
    """Register the llama architecture (covers llama, mistral, mixtral)."""
    from mlx_lm.models.llama import Model, ModelArgs

    def _factory(metadata: dict[str, Any], weights: dict, *, tie_word_embeddings: bool) -> Any:
        args_dict = gguf_metadata_to_llama_args(metadata)
        args_dict["vocab_size"] = len(metadata.get("tokenizer.ggml.tokens", []))
        args_dict["tie_word_embeddings"] = tie_word_embeddings

        args = ModelArgs(**args_dict)
        model = Model(args)

        return model

    for arch in ("llama", "mistral", "mixtral", "llama2", "llama3", "phi3"):
        _ARCH_REGISTRY[arch] = _factory


def _ensure_arch_registry() -> None:
    if _ARCH_REGISTRY:
        return
    _register_llama_arch()


def build_model_from_gguf(
    metadata: dict[str, Any],
    weights: dict,
    *,
    tie_word_embeddings: bool = True,
) -> Any:
    """Build an mlx ``nn.Module`` from GGUF metadata and translated weights.

    Args:
        metadata: GGUF metadata dict.
        weights: Translated (HF/mlx_lm-named) weights.
        tie_word_embeddings: Whether the lm_head shares embed_tokens weights.
            GGUF files materialize ``output.weight``, so pass False for them.

    Returns:
        An ``nn.Module`` (typically ``mlx_lm.models.llama.Model`` or
        ``LlamaModel``) whose ``__call__`` signature is compatible with
        exo's ``Model`` type.

    Raises:
        ValueError: If the architecture is not yet supported.
    """
    _ensure_imports()
    _ensure_arch_registry()

    arch = metadata.get("general.architecture", "llama")
    if isinstance(arch, bytes):
        arch = arch.decode("utf-8")
    arch = arch.lower()

    factory = _ARCH_REGISTRY.get(arch)
    if factory is None:
        supported = ", ".join(sorted(_ARCH_REGISTRY.keys()))
        raise ValueError(
            f"GGUF architecture '{arch}' is not yet supported. "
            f"Supported: {supported}. "
            "Consider contributing an architecture entry to gguf_architectures.py"
        )

    logger.info("Building GGUF model for architecture: %s", arch)
    model = factory(metadata, weights, tie_word_embeddings=tie_word_embeddings)
    return model


# ---------------------------------------------------------------------------
# Quantization helper
# ---------------------------------------------------------------------------

# GGUF file_type values → MLX quantization config.
# See https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#quantization-types
_GGUF_QUANT_MAP: dict[int, dict[str, int] | None] = {
    0: None,  # ALL_F32
    1: None,  # MOSTLY_F16
    2: {"group_size": 32, "bits": 4},  # MOSTLY_Q4_0
    3: {"group_size": 32, "bits": 4},  # MOSTLY_Q4_1
    7: {"group_size": 32, "bits": 8},  # MOSTLY_Q8_0
}


def get_gguf_quantization(metadata: dict[str, Any]) -> dict[str, int] | None:
    """Return MLX quantization params from GGUF file_type, or None for F16/F32."""
    file_type = _to_python(metadata.get("general.file_type", 1))
    quant = _GGUF_QUANT_MAP.get(int(file_type))
    if quant is None and int(file_type) not in (0, 1):
        logger.warning(
            "Unsupported GGUF quantization (file_type=%d). "
            "Tensors will be cast to float16.",
            file_type,
        )
    return quant


# ---------------------------------------------------------------------------
# Weight name translation (GGUF → HuggingFace / mlx_lm format)
# ---------------------------------------------------------------------------

# These prefixes/suffixes are used in GGUF weight names.
# Source: gguf_llm/utils.py translate_weight_names
_GGUF_NAME_MAP: list[tuple[str, str]] = [
    ("token_embd", "model.embed_tokens"),
    ("blk.", "model.layers."),
    ("ffn_gate", "mlp.gate_proj"),
    ("ffn_down", "mlp.down_proj"),
    ("ffn_up", "mlp.up_proj"),
    ("attn_q", "self_attn.q_proj"),
    ("attn_k", "self_attn.k_proj"),
    ("attn_v", "self_attn.v_proj"),
    ("attn_output", "self_attn.o_proj"),
    ("attn_norm", "input_layernorm"),
    ("ffn_norm", "post_attention_layernorm"),
    ("output_norm", "model.norm"),
    ("output", "lm_head"),
]


def translate_gguf_weight_name(
    name: str, *, tie_word_embeddings: bool = False
) -> str | None:
    """Translate a GGUF weight name to the HuggingFace / mlx_lm equivalent.

    Returns ``None`` when the weight should be skipped (e.g. ``output.weight``
    when ``tie_word_embeddings=True`` — the mlx Model reuses
    ``embed_tokens.weight`` for the final projection).
    """
    if tie_word_embeddings and name == "output.weight":
        return None
    for gguf_pat, hf_repl in _GGUF_NAME_MAP:
        if gguf_pat in name:
            name = name.replace(gguf_pat, hf_repl)
    return name
