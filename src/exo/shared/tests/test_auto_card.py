import pytest
from unittest.mock import patch, MagicMock

from exo.shared.models.auto_card import (
    generate_auto_card,
    ensure_auto_card,
    _detect_family,
    _detect_reasoning_dialect,
    _detect_capabilities,
    _find_local_config,
    _REASONING_ARCHITECTURES,
)
from exo.shared.models.model_cards import (
    ConfigData,
    ModelCard,
    VisionCardConfig,
    card_cache,
)
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory


# --- Family detection ---


def test_family_detection_known_architectures() -> None:
    assert _detect_family(["LlamaForCausalLM"]) == "llama"
    assert _detect_family(["Qwen3ForCausalLM"]) == "qwen3"
    assert _detect_family(["Qwen3VLForConditionalGeneration"]) == "qwen3-vl"
    assert _detect_family(["DeepseekV3ForCausalLM"]) == "deepseek-v3"
    assert _detect_family(["MiniMaxM2ForCausalLM"]) == "minimax"
    assert _detect_family(["MistralForCausalLM"]) == "mistral"


def test_family_detection_unknown_architectures() -> None:
    assert _detect_family(["CustomModelForCausalLM"]) == "custommodel"
    # 'Model' suffix is stripped in fallback
    assert _detect_family(["SomeModel"]) == "some"
    # No known suffix → full lowercase
    assert _detect_family(["WeirdArchitecture"]) == "weird architecture".lower() or _detect_family(["WeirdArchitecture"]) == "weirdarchitecture"


def test_family_detection_none_or_empty() -> None:
    assert _detect_family(None) == ""
    assert _detect_family([]) == ""


# --- Reasoning detection ---


def test_reasoning_detection() -> None:
    for arch in _REASONING_ARCHITECTURES:
        assert _detect_reasoning_dialect([arch]) == "suffix"


def test_reasoning_detection_none_for_non_reasoning() -> None:
    assert _detect_reasoning_dialect(["LlamaForCausalLM"]) == "none"
    assert _detect_reasoning_dialect(["MistralForCausalLM"]) == "none"
    assert _detect_reasoning_dialect(None) == "none"
    assert _detect_reasoning_dialect([]) == "none"


# --- Capabilities detection ---


def test_capabilities_tensor_parallel() -> None:
    config = ConfigData.model_validate(
        {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 1}
    )
    caps = _detect_capabilities(config, None)
    assert "tensor-parallel" in caps


def test_capabilities_ring_attention() -> None:
    config = ConfigData.model_validate(
        {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 1}
    )
    caps = _detect_capabilities(config, None)
    assert "ring-attention" in caps


def test_capabilities_vision() -> None:
    config = ConfigData.model_validate(
        {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 1}
    )
    vision = VisionCardConfig(
        image_token_id=1234,
        model_type="llava",
    )
    caps = _detect_capabilities(config, vision)
    assert "vision" in caps


def test_capabilities_reasoning() -> None:
    config = ConfigData.model_validate(
        {"architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 1}
    )
    caps = _detect_capabilities(config, None)
    assert "reasoning" in caps


def test_capabilities_long_context() -> None:
    config = ConfigData.model_validate(
        {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 1,
         "max_position_embeddings": 131072}
    )
    caps = _detect_capabilities(config, None)
    assert "long-context" in caps


def test_capabilities_empty_for_minimal() -> None:
    config = ConfigData.model_validate(
        {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 1}
    )
    caps = _detect_capabilities(config, None)
    # Llama supports tensor + ring
    assert "vision" not in caps
    assert "reasoning" not in caps


# --- ConfigData integration ---


def test_config_data_layer_count() -> None:
    raw = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 8192,
    }
    cd = ConfigData.model_validate(raw, context={"model_id": "test/model"})
    assert cd.layer_count == 32
    assert cd.hidden_size == 4096
    assert cd.supports_tensor is True
    assert cd.vision is None


def test_config_data_text_config_defer() -> None:
    """Verify ConfigData can read from text_config sub-object."""
    raw = {
        "architectures": ["LlamaForCausalLM"],
        "text_config": {
            "hidden_size": 8192,
            "num_hidden_layers": 40,
        },
    }
    cd = ConfigData.model_validate(raw, context={"model_id": "test/model"})
    assert cd.layer_count == 40
    assert cd.hidden_size == 8192


def test_config_data_vision_extraction() -> None:
    """Verify ConfigData extracts vision from vision_config + image_token_id."""
    raw = {
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": 1,
        "vision_config": {"model_type": "llava"},
        "image_token_id": 9999,
    }
    cd = ConfigData.model_validate(raw, context={"model_id": "test/model"})
    assert cd.vision is not None
    assert cd.vision.image_token_id == 9999
    assert cd.vision.model_type == "llava"


# --- generate_auto_card (async, with mocks) ---


@pytest.mark.asyncio
async def test_generate_auto_card_with_local_config(tmp_path, monkeypatch):
    """generate_auto_card reads local config.json and produces a ModelCard."""
    from exo.shared.models import auto_card as ac
    from exo.shared.models import model_cards as mc

    # Set up a local config.json
    model_dir = tmp_path / "models" / "test-org--test-model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        '{"architectures": ["LlamaForCausalLM"], "hidden_size": 2048, '
        '"num_hidden_layers": 16, "num_key_value_heads": 8, '
        '"max_position_embeddings": 4096}'
    )

    monkeypatch.setattr(ac, "EXO_MODELS_DIRS", [tmp_path / "models"])
    monkeypatch.setattr(mc, "EXO_MODELS_DIRS", [tmp_path / "models"])

    card = await generate_auto_card(
        ModelId("test-org/test-model"),
        storage_size=Memory.from_bytes(4_000_000_000),
    )

    assert card.model_id == "test-org/test-model"
    assert card.n_layers == 16
    assert card.hidden_size == 2048
    assert card.supports_tensor is True
    assert card.supports_ring is True
    assert card.is_custom is True
    assert card.family == "llama"
    assert "tensor-parallel" in card.capabilities
    assert "ring-attention" in card.capabilities
    assert card.vision is None


@pytest.mark.asyncio
async def test_generate_auto_card_vision_model(tmp_path, monkeypatch):
    """generate_auto_card detects vision from HF config."""
    from exo.shared.models import auto_card as ac
    from exo.shared.models import model_cards as mc

    model_dir = tmp_path / "models" / "test-org--vision-model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        '{"architectures": ["LlamaForCausalLM"], "hidden_size": 1024, '
        '"num_hidden_layers": 8, "max_position_embeddings": 2048, '
        '"vision_config": {"model_type": "llava"}, "image_token_id": 5432}'
    )

    monkeypatch.setattr(ac, "EXO_MODELS_DIRS", [tmp_path / "models"])
    monkeypatch.setattr(mc, "EXO_MODELS_DIRS", [tmp_path / "models"])

    card = await generate_auto_card(
        ModelId("test-org/vision-model"),
        storage_size=Memory.from_bytes(1_000_000_000),
    )

    assert card.vision is not None
    assert card.vision.image_token_id == 5432
    assert card.vision.model_type == "llava"
    assert "vision" in card.capabilities


@pytest.mark.asyncio
async def test_generate_auto_card_reasoning_model(tmp_path, monkeypatch):
    """generate_auto_card sets reasoning dialect for reasoning architectures."""
    from exo.shared.models import auto_card as ac
    from exo.shared.models import model_cards as mc

    model_dir = tmp_path / "models" / "test-org--qwen3-model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        '{"architectures": ["Qwen3ForCausalLM"], "hidden_size": 2048, '
        '"num_hidden_layers": 16, "max_position_embeddings": 32768}'
    )

    monkeypatch.setattr(ac, "EXO_MODELS_DIRS", [tmp_path / "models"])
    monkeypatch.setattr(mc, "EXO_MODELS_DIRS", [tmp_path / "models"])

    card = await generate_auto_card(
        ModelId("test-org/qwen3-model"),
        storage_size=Memory.from_bytes(4_000_000_000),
    )

    assert card.reasoning_dialect == "suffix"
    assert "reasoning" in card.capabilities
    assert card.family == "qwen3"
