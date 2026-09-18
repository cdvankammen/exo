from exo.shared.models.model_cards import ConfigData, ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.memory import Memory

# Must stay in lockstep with ring_attention._SUPPORTED_ATTENTION_TYPES:
# (mlx_lm module, attention class) -> HF config.json architecture name.
RING_ARCHITECTURES = [
    # (mlx_lm.models.llama, Attention)
    "LlamaForCausalLM",
    # (mlx_lm.models.qwen3, Attention)
    "Qwen3ForCausalLM",
    # (mlx_lm.models.qwen2, Attention)
    "Qwen2ForCausalLM",
    # (mlx_lm.models.glm4, Glm4Attention)
    "Glm4ForCausalLM",
    # (mlx_lm.models.glm4_moe, Attention)
    "Glm4MoeForCausalLM",
    # (mlx_lm.models.glm4_moe_lite, Glm4MoeLiteAttention)
    "Glm4MoeLiteForCausalLM",
    # (mlx_lm.models.minimax, MiniMaxAttention)
    "MiniMaxM2ForCausalLM",
    # (mlx_lm.models.nemotron_h, NemotronHAttention)
    "NemotronHForCausalLM",
]

NON_RING_ARCHITECTURES = [
    # Tensor-supported but Ring-unsupported (sliding window, MoE-gated,
    # softcapping, Mamba2-hybrid, etc. — see model_cards.py docstring).
    "GlmMoeDsaForCausalLM",
    "DeepseekV4ForCausalLM",
    "DeepseekV32ForCausalLM",
    "DeepseekV3ForCausalLM",
    "Qwen3NextForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3VLForConditionalGeneration",
    "MistralForCausalLM",
    # Llama4 (chunked attention + qk-norm + temperature scaling) is
    # tensor-supported but NOT ring-supported (ring engine is not verified
    # against the chunked/masked no-rope attention).
    "Llama4ForCausalLM",
    "Llama4TextForCausalLM",
    "GptOssForCausalLM",
    "Step3p5ForCausalLM",
    "Gemma4ForConditionalGeneration",
    # Ring-unsupported families entirely outside the whitelist.
    "Gemma2ForCausalLM",
    "FalconForCausalLM",
]


def test_config_detects_verified_ring_architectures() -> None:
    for architecture in RING_ARCHITECTURES:
        config = ConfigData.model_validate(
            {"architectures": [architecture], "num_hidden_layers": 1}
        )

        assert config.supports_ring is True, architecture


def test_config_rejects_unverified_ring_architecture() -> None:
    for architecture in NON_RING_ARCHITECTURES:
        config = ConfigData.model_validate(
            {"architectures": [architecture], "num_hidden_layers": 1}
        )

        assert config.supports_ring is False, architecture


# Must stay in lockstep with ConfigData.supports_tensor (model_cards.py):
# architectures validated for the tensor-parallel engine path
# (placement.py gates on supports_tensor; auto_parallel dispatches to
# GenericShardingStrategy, which structurally validates q/k/v/o_proj +
# mlp gate/down/up_proj before sharding).
TENSOR_ARCHITECTURES = [
    # Trained-on + verified through the tensor engine path (R&D t_2b5ba2eb:
    # mlx_lm gemma2 = full attention, no sliding window; softcapping is
    # rank-local per-tensor; GenericShardingStrategy structural validation
    # passes empirically).
    "Gemma2ForCausalLM",
]


def test_config_detects_tensor_supported_architectures() -> None:
    for architecture in TENSOR_ARCHITECTURES:
        config = ConfigData.model_validate(
            {"architectures": [architecture], "num_hidden_layers": 1}
        )

        assert config.supports_tensor is True, architecture


def test_ring_whitelist_matches_engine_supported_attention_modules() -> None:
    """Every ring engine whitelist entry must map to a ConfigData architecture.

    Guards against drift: when ring_attention.py adds a new supported
    (module, attention class) pair, this test fails until ConfigData
    supports_ring covers the corresponding HF architecture name.

    ring_attention.py imports mlx at module level, so skip gracefully in
    environments without the MLX stack (the static lists above still run).
    """
    try:
        from exo.worker.engines.mlx.ring_attention import _SUPPORTED_ATTENTION_TYPES
    except ImportError:
        return

    # Module -> HF architecture name mapping (verified against mlx_lm source).
    module_to_architecture = {
        "mlx_lm.models.llama": "LlamaForCausalLM",
        "mlx_lm.models.qwen3": "Qwen3ForCausalLM",
        "mlx_lm.models.qwen2": "Qwen2ForCausalLM",
        "mlx_lm.models.glm4": "Glm4ForCausalLM",
        "mlx_lm.models.glm4_moe": "Glm4MoeForCausalLM",
        "mlx_lm.models.glm4_moe_lite": "Glm4MoeLiteForCausalLM",
        "mlx_lm.models.minimax": "MiniMaxM2ForCausalLM",
        "mlx_lm.models.nemotron_h": "NemotronHForCausalLM",
    }

    for module, _attention_class in _SUPPORTED_ATTENTION_TYPES:
        architecture = module_to_architecture[module]
        config = ConfigData.model_validate(
            {"architectures": [architecture], "num_hidden_layers": 1}
        )

        assert config.supports_ring is True, (
            f"{module} is Ring-supported by the engine but "
            f"{architecture} is missing from ConfigData.supports_ring"
        )


def test_model_card_defaults_ring_support_to_false() -> None:
    card = ModelCard(
        model_id=ModelId("test/model"),
        storage_size=Memory.from_bytes(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )

    assert card.supports_ring is False
