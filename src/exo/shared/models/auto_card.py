"""Auto-generate ModelCard from HuggingFace config.json.

Provides an ollama-like experience where `exo pull <model>` automatically
generates a complete ModelCard from the model's config.json without requiring
a pre-written TOML card. This enables zero-config support for any HF model.

Phase 1.5 of the chat-log-hive-mind epic.
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger

from exo.shared.constants import EXO_MODELS_DIRS
from exo.shared.models.model_cards import (
    ConfigData,
    ModelCard,
    ModelTask,
    VisionCardConfig,
)
from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory


# Architecture families mapped from HF config.json architectures field.
# Used to populate ModelCard.family for display and filtering.
_ARCHITECTURE_FAMILY_MAP: dict[str, str] = {
    "LlamaForCausalLM": "llama",
    "Qwen2ForCausalLM": "qwen2",
    "Qwen3ForCausalLM": "qwen3",
    "Qwen3MoeForCausalLM": "qwen3",
    "Qwen3NextForCausalLM": "qwen3",
    "Qwen3_5ForConditionalGeneration": "qwen3.5",
    "Qwen3_5MoeForConditionalGeneration": "qwen3.5",
    "Qwen3VLForConditionalGeneration": "qwen3-vl",
    "MistralForCausalLM": "mistral",
    "MixtralForCausalLM": "mixtral",
    "GemmaForCausalLM": "gemma",
    "Gemma2ForCausalLM": "gemma2",
    "Gemma4ForConditionalGeneration": "gemma4",
    "DeepseekV3ForCausalLM": "deepseek-v3",
    "DeepseekV32ForCausalLM": "deepseek-v3",
    "DeepseekV4ForCausalLM": "deepseek-v4",
    "Glm4MoeLiteForCausalLM": "glm4",
    "GlmMoeDsaForCausalLM": "glm",
    "MiniMaxM2ForCausalLM": "minimax",
    "GptOssForCausalLM": "gpt-oss",
    "Step3p5ForCausalLM": "step",
    "NemotronHForCausalLM": "nemotron",
    "Phi3ForCausalLM": "phi3",
    "CohereForCausalLM": "cohere",
    "DBRXForCausalLM": "dbrx",
}

# Architectures known to support reasoning/thinking mode.
_REASONING_ARCHITECTURES: set[str] = {
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen3NextForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
    "DeepseekV3ForCausalLM",
    "DeepseekV32ForCausalLM",
    "DeepseekV4ForCausalLM",
    "Glm4MoeLiteForCausalLM",
    "GlmMoeDsaForCausalLM",
}


def _detect_family(architectures: list[str] | None) -> str:
    """Map HF architectures list to a human-readable family name."""
    if not architectures:
        return ""
    for arch in architectures:
        if arch in _ARCHITECTURE_FAMILY_MAP:
            return _ARCHITECTURE_FAMILY_MAP[arch]
    # Fallback: use first architecture name stripped of suffixes
    first = architectures[0]
    for suffix in ("ForCausalLM", "ForConditionalGeneration", "Model"):
        if first.endswith(suffix):
            return first[: -len(suffix)].lower()
    return first.lower()


def _detect_reasoning_dialect(architectures: list[str] | None) -> str:
    """Detect reasoning dialect from architecture.

    Returns a ReasoningDialect string. Most reasoning models use 'suffix'
    style (thinking embedded in assistant content). Channel-based models
    can be added here as they appear.
    """
    if not architectures:
        return "none"
    for arch in architectures:
        if arch in _REASONING_ARCHITECTURES:
            return "suffix"
    return "none"


def _detect_capabilities(
    config_data: ConfigData,
    vision: VisionCardConfig | None,
) -> list[str]:
    """Build capabilities list from config features."""
    caps: list[str] = []
    if vision is not None:
        caps.append("vision")
    if config_data.supports_tensor:
        caps.append("tensor-parallel")
    if config_data.supports_ring:
        caps.append("ring-attention")
    reasoning = _detect_reasoning_dialect(config_data.architectures)
    if reasoning != "none":
        caps.append("reasoning")
    if config_data.max_position_embeddings > 32768:
        caps.append("long-context")
    return caps


def _find_local_config(model_id: ModelId) -> dict[str, Any] | None:
    """Find and parse config.json from local model directories.

    Searches EXO_MODELS_DIRS for the model's config.json. Returns the
    parsed JSON dict or None if not found.
    """
    normalized = model_id.normalize()
    for model_dir in EXO_MODELS_DIRS:
        config_path = model_dir / normalized / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f)  # type: ignore[no-any-return]
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to parse {config_path}: {e}")
                continue
    return None


async def generate_auto_card(
    model_id: ModelId,
    storage_size: Memory | None = None,
) -> ModelCard:
    """Generate a ModelCard automatically from HF config.json.

    This is the main entry point for auto-card generation. It:
    1. Finds config.json locally (already downloaded) or fetches from HF
    2. Parses it through ConfigData for standardized field extraction
    3. Detects vision, family, reasoning, capabilities
    4. Returns a complete ModelCard ready for caching/saving

    Args:
        model_id: The HuggingFace model identifier.
        storage_size: Pre-computed storage size. If None, will be fetched
                      separately via fetch_safetensors_size.

    Returns:
        A fully populated ModelCard with is_custom=True.
    """
    from exo.shared.models.model_cards import fetch_config_data, fetch_safetensors_size

    # Try local config first (fast path for already-downloaded models)
    raw_config = _find_local_config(model_id)
    if raw_config is not None:
        config_data = ConfigData.model_validate(
            raw_config, context={"model_id": str(model_id)}
        )
    else:
        # Fetch from HF (handles download + parse)
        config_data = await fetch_config_data(model_id)

    # Get storage size if not provided
    if storage_size is None:
        try:
            storage_size = await fetch_safetensors_size(model_id)
        except Exception as e:
            logger.warning(
                f"Could not determine storage size for {model_id}: {e}. "
                "Using 0 bytes placeholder."
            )
            storage_size = Memory.from_bytes(0)

    # Build the card
    family = _detect_family(config_data.architectures)
    reasoning = _detect_reasoning_dialect(config_data.architectures)
    capabilities = _detect_capabilities(config_data, config_data.vision)

    card = ModelCard(
        model_id=ModelId(model_id),
        storage_size=storage_size,
        n_layers=config_data.layer_count,
        hidden_size=config_data.hidden_size or 0,
        supports_tensor=config_data.supports_tensor,
        supports_ring=config_data.supports_ring,
        num_key_value_heads=config_data.num_key_value_heads,
        context_length=config_data.max_position_embeddings,
        tasks=[ModelTask.TextGeneration],
        trust_remote_code=False,
        is_custom=True,
        vision=config_data.vision,
        family=family,
        reasoning_dialect=reasoning,  # type: ignore[arg-type]
        capabilities=capabilities,
        backends=list(Backend),  # All backends; placement gate decides
    )

    logger.info(
        f"Auto-generated ModelCard for {model_id}: "
        f"family={family}, layers={config_data.layer_count}, "
        f"vision={'yes' if config_data.vision else 'no'}, "
        f"caps={capabilities}"
    )
    return card


async def ensure_auto_card(model_id: ModelId) -> ModelCard:
    """Ensure a ModelCard exists, generating one if needed.

    Integration point for the existing ModelCard.load() flow. Call this
    when no pre-written TOML card is found to auto-generate one.

    The generated card is saved to the custom cards directory so subsequent
    loads find it without re-generation.
    """
    from exo.shared.models.model_cards import card_cache

    # Check cache first
    cached = card_cache.get(model_id)
    if cached is not None:
        return cached

    # Generate and persist
    card = await generate_auto_card(model_id)
    await card_cache.save(card)
    return card
