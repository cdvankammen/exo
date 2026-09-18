"""Tests for GGUF runtime loading (src/exo/worker/engines/mlx/gguf*).

Creates synthetic GGUF files via ``mx.save_gguf`` (no network) at a tiny
scale that still exercises the full translate → build → quantize → load
path, plus focused unit tests for name translation, quantization mapping,
and the sentencepiece tokenizer shim.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.gguf import (
    find_gguf_file,
    is_gguf_model,
    load_gguf,
)
from exo.worker.engines.mlx.gguf_architectures import (
    get_gguf_quantization,
    gguf_metadata_to_llama_args,
    translate_gguf_weight_name,
)
from exo.worker.engines.mlx.gguf_tokenizer import (
    GgufSpmTokenizer,
    _build_spm_from_metadata,
    _has_hf_tokenizer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_gguf(path: Path, *, quant: str = "f16") -> dict:
    """Write a small llama-style GGUF file and return its metadata dict."""

    vocab_size = 320
    hidden_size = 32
    n_layers = 2
    n_heads = 4
    inter = 48

    # Real GGUF vocabularies always include the 256 byte pieces
    # (<0x00>..<0xFF>, type 6) — byte_fallback in the sentencepiece builder
    # requires them.
    byte_pieces = [f"<0x{i:02X}>" for i in range(256)]
    tokens = ["<unk>", "<s>", "</s>"] + [
        f"tok{i}" for i in range(vocab_size - 3 - 256)
    ] + byte_pieces
    token_types = [2, 3, 3] + [1] * (vocab_size - 3 - 256) + [6] * 256  # type: ignore[list-item]

    metadata: dict = {
        "general.architecture": "llama",
        "general.name": "tiny-test",
        "general.file_type": mx.array(1 if quant == "f16" else 7, mx.uint32),  # F16 or Q8_0
        "llama.context_length": mx.array(32, mx.uint32),
        "llama.embedding_length": mx.array(hidden_size, mx.uint32),
        "llama.block_count": mx.array(n_layers, mx.uint32),
        "llama.attention.head_count": mx.array(n_heads, mx.uint32),
        "llama.attention.head_count_kv": mx.array(n_heads, mx.uint32),
        "llama.feed_forward_length": mx.array(inter, mx.uint32),
        "llama.attention.layer_norm_rms_epsilon": mx.array(1e-5, mx.float32),
        "llama.rope.freq_base": mx.array(10000.0, mx.float32),
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.scores": mx.array([0.0] * vocab_size, mx.float32),
        "tokenizer.ggml.token_type": mx.array(token_types, mx.uint32),
        "tokenizer.ggml.bos_token_id": mx.array(1, mx.uint32),
        "tokenizer.ggml.eos_token_id": mx.array(2, mx.uint32),
        "tokenizer.ggml.unknown_token_id": mx.array(0, mx.uint32),
    }

    # Build GGUF weight names the way llama.cpp names them.
    def _w(name: str, shape: tuple[int, ...]) -> str:
        weights[name] = mx.random.normal(shape)
        return name

    weights: dict[str, mx.array] = {}
    _w("token_embd.weight", (vocab_size, hidden_size))
    for blk in range(n_layers):
        _w(f"blk.{blk}.attn_norm.weight", (hidden_size,))
        _w(f"blk.{blk}.attn_q.weight", (n_heads * (hidden_size // n_heads), hidden_size))
        _w(f"blk.{blk}.attn_k.weight", (n_heads * (hidden_size // n_heads), hidden_size))
        _w(f"blk.{blk}.attn_v.weight", (n_heads * (hidden_size // n_heads), hidden_size))
        _w(f"blk.{blk}.attn_output.weight", (hidden_size, n_heads * (hidden_size // n_heads)))
        _w(f"blk.{blk}.ffn_norm.weight", (hidden_size,))
        _w(f"blk.{blk}.ffn_gate.weight", (inter, hidden_size))
        _w(f"blk.{blk}.ffn_down.weight", (hidden_size, inter))
        _w(f"blk.{blk}.ffn_up.weight", (inter, hidden_size))
    _w("output_norm.weight", (hidden_size,))
    _w("output.weight", (vocab_size, hidden_size))

    mx.save_gguf(str(path), weights, metadata)
    return metadata


@pytest.fixture
def gguf_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "models--test--tiny-gguf"
    model_dir.mkdir(parents=True)
    _make_synthetic_gguf(model_dir / "model.gguf")
    return model_dir


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_is_gguf_model_detects_gguf_dir(gguf_dir: Path) -> None:
    assert is_gguf_model(gguf_dir) is True


def test_is_gguf_model_false_for_safetensors(tmp_path: Path) -> None:
    d = tmp_path / "models--x"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"x")
    assert is_gguf_model(d) is False


def test_find_gguf_file(gguf_dir: Path) -> None:
    f = find_gguf_file(gguf_dir)
    assert f is not None
    assert f.name == "model.gguf"


# ---------------------------------------------------------------------------
# Name translation
# ---------------------------------------------------------------------------


def test_translate_llama_weight_names() -> None:
    assert (
        translate_gguf_weight_name("blk.0.attn_norm.weight")
        == "model.layers.0.input_layernorm.weight"
    )
    assert (
        translate_gguf_weight_name("blk.3.attn_q.weight")
        == "model.layers.3.self_attn.q_proj.weight"
    )
    assert (
        translate_gguf_weight_name("blk.1.ffn_gate.weight")
        == "model.layers.1.mlp.gate_proj.weight"
    )
    assert (
        translate_gguf_weight_name("token_embd.weight") == "model.embed_tokens.weight"
    )
    assert translate_gguf_weight_name("output_norm.weight") == "model.norm.weight"
    assert translate_gguf_weight_name("output.weight") == "lm_head.weight"


# ---------------------------------------------------------------------------
# Quantization mapping
# ---------------------------------------------------------------------------


def test_get_gguf_quantization_none_for_f16() -> None:
    assert get_gguf_quantization({"general.file_type": 1}) is None


def test_get_gguf_quantization_q8() -> None:
    assert get_gguf_quantization({"general.file_type": 7}) == {
        "group_size": 32,
        "bits": 8,
    }


def test_get_gguf_quantization_q4() -> None:
    assert get_gguf_quantization({"general.file_type": 2}) == {
        "group_size": 32,
        "bits": 4,
    }


def test_get_gguf_quantization_unsupported() -> None:
    # file_type 8 (Q5_0) is not natively supported → falls back to None
    assert get_gguf_quantization({"general.file_type": 8}) is None


# ---------------------------------------------------------------------------
# Metadata → args
# ---------------------------------------------------------------------------


def test_gguf_metadata_to_llama_args() -> None:
    metadata = {
        "llama.context_length": 4096,
        "llama.embedding_length": 512,
        "llama.block_count": 4,
        "llama.attention.head_count": 8,
        "llama.attention.head_count_kv": 8,
        "llama.feed_forward_length": 1024,
        "llama.attention.layer_norm_rms_epsilon": 1e-5,
    }
    args = gguf_metadata_to_llama_args(metadata)
    assert args["hidden_size"] == 512
    assert args["num_hidden_layers"] == 4
    assert args["num_attention_heads"] == 8
    assert args["max_position_embeddings"] == 4096
    assert args["rope_traditional"] is True
    assert args["model_type"] == "llama"


# ---------------------------------------------------------------------------
# Tokenizer shim
# ---------------------------------------------------------------------------


def test_has_hf_tokenizer(tmp_path: Path) -> None:
    assert _has_hf_tokenizer(tmp_path) is False
    (tmp_path / "tokenizer.json").write_text("{}")
    assert _has_hf_tokenizer(tmp_path) is True


def test_gguf_spm_tokenizer_shim_roundtrip() -> None:
    """Build a shim from the production builder and validate its API shape.

    We do NOT assert a decode roundtrip here — a 6-token synthetic vocab is
    far too sparse for sentencepiece's byte_fallback and unicode_script
    normalizations to reconstruct exact text.  The real decode roundtrip is
    validated in ``test_gguf_integration`` on a real GGUF file.
    """
    import mlx.core as mx

    byte_pieces = [f"<0x{i:02X}>" for i in range(256)]
    tokens = ["<unk>", "<s>", "</s>", "▁hello", "▁world", "▁test"] + byte_pieces
    n = len(tokens)
    metadata = {
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.scores": mx.array([0.0] * n, mx.float32),
        "tokenizer.ggml.token_type": mx.array(
            [2, 3, 3, 1, 1, 1] + [6] * 256, mx.uint32
        ),
        "tokenizer.ggml.bos_token_id": mx.array(1, mx.uint32),
        "tokenizer.ggml.eos_token_id": mx.array(2, mx.uint32),
        "tokenizer.ggml.unknown_token_id": mx.array(0, mx.uint32),
    }
    sp = _build_spm_from_metadata(metadata)

    shim = GgufSpmTokenizer(sp, eos_token_id=2, bos_token_id=1)
    assert shim.eos_token_id == 2
    assert shim.chat_template is None
    assert shim.vocab_size == n

    ids = shim.encode("hello world")
    # BOS (1) is prepended; encode may map to unk on this sparse vocab
    assert len(ids) >= 2  # BOS + at least one token
    assert ids[0] == 1  # BOS
    assert isinstance(shim.decode(ids), str)


# ---------------------------------------------------------------------------
# Full load path (synthetic GGUF)
# ---------------------------------------------------------------------------


def test_load_gguf_roundtrip(gguf_dir: Path) -> None:
    gguf_file = find_gguf_file(gguf_dir)
    assert gguf_file is not None

    model, tokenizer = load_gguf(
        gguf_file,
        gguf_dir,
        "test/tiny-gguf",
    )

    # Model must be an mlx nn.Module with the expected structure.
    assert hasattr(model, "model")
    assert hasattr(model.model, "layers")
    assert len(model.model.layers) == 2

    # Tokenizer must be a TokenizerWrapper.
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    assert isinstance(tokenizer, TokenizerWrapper)

    # Forward pass smoke test: 1 token in → logits out.

    prompt_tokens = mx.array([[2, 3, 4]])
    logits = model(prompt_tokens)
    # mlx LlamaModel: __call__(inputs, cache=None) → logits for a fresh call.
    assert logits.shape == (1, 3, 320)


def test_load_gguf_unsupported_arch_raises(tmp_path: Path) -> None:
    gguf = tmp_path / "bad.gguf"
    _make_synthetic_gguf(gguf)
    # Overwrite architecture metadata by re-saving with a different arch → just
    # check the registry rejects unknown archs via a direct call.
    from exo.worker.engines.mlx.gguf_architectures import build_model_from_gguf

    with pytest.raises(ValueError):
        build_model_from_gguf({"general.architecture": "qwen3"}, {})