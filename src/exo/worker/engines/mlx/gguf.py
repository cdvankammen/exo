"""GGUF runtime model loader for exo.

Loads a ``.gguf`` file directly into an mlx ``nn.Module`` using MLX's native
GGUF support (``mx.load(..., return_metadata=True)``), following the pattern
from mlx-examples ``llms/gguf_llm``.

The loaded model reuses exo's existing MLX engine machinery (pipeline /
tensor / ring distributed sharding, KV caches, generators) because it is a
plain ``mlx.nn.Module`` with the same ``__call__`` contract as models loaded
through ``mlx_lm.utils.load_model``.

Supported quantizations (loaded natively by MLX): Q4_0, Q4_1, Q8_0.
All other GGUF quantizations are cast to float16 by MLX (warning logged).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlx.core as mx

from exo.worker.engines.mlx.gguf_architectures import (
    build_model_from_gguf,
    get_gguf_quantization,
    translate_gguf_weight_name,
)
from exo.worker.engines.mlx.gguf_tokenizer import load_gguf_tokenizer

logger = logging.getLogger(__name__)


def is_gguf_model(model_path: Path) -> bool:
    """Return True if ``model_path`` contains GGUF weights (and no safetensors)."""
    if not model_path.is_dir():
        return False
    has_safetensors = any(model_path.glob("model*.safetensors")) or any(
        model_path.glob("*.safetensors")
    )
    if has_safetensors:
        return False
    return any(model_path.glob("*.gguf"))


def find_gguf_file(model_path: Path) -> Path | None:
    """Return the first ``.gguf`` file in the model directory, if any."""
    files = sorted(model_path.glob("*.gguf"))
    if not files:
        return None
    if len(files) > 1:
        logger.debug("Multiple GGUF files found; using %s", files[0].name)
    return files[0]


def load_gguf(
    gguf_path: Path,
    model_path: Path,
    model_id: str,
    *,
    trust_remote_code: bool = True,
) -> tuple[Any, Any]:
    """Load a GGUF model and its tokenizer.

    Args:
        gguf_path: Path to the ``.gguf`` file.
        model_path: The model directory (used for tokenizer files + model_id).
        model_id: The exo ModelId string (e.g. ``TheBloke/Mistral-7B-v0.1-GGUF``).

    Returns:
        ``(model, tokenizer)`` where ``model`` is an ``mlx.nn.Module`` and
        ``tokenizer`` is a ``TokenizerWrapper``.
    """
    from exo.shared.types.common import ModelId

    logger.info("Loading GGUF model from %s", gguf_path)

    loaded = mx.load(str(gguf_path), return_metadata=True)
    weights, metadata = loaded  # type: ignore[misc]
    assert isinstance(metadata, dict)
    weights = {k: v for k, v in weights.items() if isinstance(k, str) and hasattr(v, "shape")}  # type: ignore[attr-defined]
    weight_dict: dict[str, mx.array] = weights  # type: ignore[assignment]

    quantization = get_gguf_quantization(metadata)
    if quantization is not None:
        logger.info("Applying GGUF quantization: %s", quantization)

    # Translate GGUF weight names to HF/mlx_lm format.
    weights = {
        translated: v
        for k, v in weight_dict.items()
        if (translated := translate_gguf_weight_name(k)) is not None
    }

    # llama.cpp GGUF conversions always materialize ``output.weight``.  Build
    # the model with an explicit lm_head (untied) when the GGUF provides it,
    # matching gguf_llm; fall back to tied embeddings otherwise.
    model = build_model_from_gguf(
        metadata,
        weights,
        tie_word_embeddings="output.weight" not in weights,
    )

    # Drop weights that have no parameter in the built model.  The common case
    # is ``output.weight`` on tied-embedding llama models (mlx reuses
    # ``embed_tokens.weight`` as the final projection); some GGUF conversions
    # materialize it anyway.  ``load_weights`` raises on extras, so filter.
    # NOTE: must run AFTER nn.quantize so the quantized layout's scales/bias
    # leaf names are present in the model's parameter tree.
    def _filter_weights_to_model(model: Any, weights: dict) -> dict:
        from mlx.utils import tree_flatten

        model_leaf_names = {k for k, _ in tree_flatten(model.parameters())}
        filtered = {k: v for k, v in weights.items() if k in model_leaf_names}
        missing = model_leaf_names - set(weights.keys())
        if missing:
            logger.warning(
                "GGUF weights missing for model params: %s", sorted(missing)[:8]
            )
        return filtered

    if quantization is not None:
        import mlx.nn as nn

        def class_predicate(p: str, m: Any) -> bool:
            return isinstance(m, (nn.Linear, nn.Embedding)) and f"{p}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            class_predicate=class_predicate,
        )

    weights = _filter_weights_to_model(model, weights)
    model.load_weights(list(weights.items()))

    tokenizer = load_gguf_tokenizer(
        model_path,
        metadata,
        ModelId(model_id),
        trust_remote_code=trust_remote_code,
    )

    logger.info(
        "GGUF model loaded: architecture=%s vocab=%d layers=%d",
        metadata.get("general.architecture", "unknown"),
        len(metadata.get("tokenizer.ggml.tokens", [])),
        metadata.get("llama.block_count", 0),
    )
    return model, tokenizer


__all__ = ["is_gguf_model", "find_gguf_file", "load_gguf"]