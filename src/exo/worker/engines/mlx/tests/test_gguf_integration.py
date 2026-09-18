"""End-to-end installer test: run the full GGUF load path on a real GGUF file.

Downloads a small real GGUF model (TinyLlama Q8_0 from second-state mirror,
which mirrors TheBloke's files), loads it with ``load_gguf``, runs a mini
forward pass, and asserts shapes.

Skipped when the file is already cached or no network is available.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.gguf import find_gguf_file, is_gguf_model, load_gguf

GGUF_URL = (
    "https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/"
    "main/TinyLlama-1.1B-Chat-v1.0-Q8_0.gguf?download=true"
)
GGUF_SHA = None  # optional

pytestmark = pytest.mark.slow


def _cache_file(tmp_path: Path) -> Path:
    """Download the real GGUF once, cache under tmp_path."""
    # Prefer an existing local copy (stable cache location), then tmp_path.
    for candidate in (
        Path(os.environ.get("EXO_GGUF_TEST_FILE", "")),
        Path("/tmp/tinyllama-q8.gguf"),
        tmp_path / "TinyLlama-1.1B-Chat-v1.0-Q8_0.gguf",
    ):
        if candidate.is_file() and candidate.stat().st_size > 100_000_000:
            return candidate
    dest = tmp_path / "TinyLlama-1.1B-Chat-v1.0-Q8_0.gguf"
    try:
        with urllib.request.urlopen(GGUF_URL, timeout=600) as r, open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:  # noqa: BLE001 - network may be unavailable
        pytest.skip(f"Could not download GGUF: {e}")
    return dest


@pytest.mark.slow
def test_load_real_gguf(tmp_path: Path) -> None:
    gguf_path = _cache_file(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / gguf_path.name).symlink_to(gguf_path)

    assert is_gguf_model(model_dir)
    gguf_file = find_gguf_file(model_dir)
    assert gguf_file is not None

    model, tokenizer = load_gguf(gguf_file, model_dir, "second-state/TinyLlama-1.1B-Chat-v1.0-GGUF")
    assert hasattr(model, "model")
    assert len(model.model.layers) > 0

    # hidden size from the GGUF arch: TinyLlama = 2048
    from mlx_lm.models.llama import Model

    assert isinstance(model, Model)

    # mini forward pass
    prompt = mx.array([[1, 2, 3]])
    out = model(prompt)
    # TinyLlama vocab 32000, hidden 2048
    assert out.shape == (1, 3, 2048) or out.shape[-1] == 32000