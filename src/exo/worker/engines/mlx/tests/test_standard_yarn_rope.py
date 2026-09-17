# pyright: reportAttributeAccessIssue=false, reportInvalidTypeForm=false
# pyright: reportReturnType=false, reportUnknownMemberType=false
# pyright: reportArgumentType=false
"""Unit tests for the standard_yarn_rope patch.

Covers the two responsibilities of
``src/exo/worker/engines/mlx/patches/standard_yarn_rope.py``:

1. Monkeypatch swap: ``patch_yarn_rope()`` must replace BOTH
   ``rope_utils.YarnRoPE.__init__`` and ``rope_utils.initialize_rope``
   on the live module, and the patched ``initialize_rope`` must route
   ``yarn`` / ``deepseek_yarn`` to the patched YarnRoPE path while
   delegating every other rope type to the original implementation.

2. Numeric equivalence: the patched YarnRoPE ``_freqs`` must match the
   stock mlx_lm folded formula within 1e-6 relative error over the four
   validated configurations (the parent analysis t_fdb502db verified the
   patch's inverse-frequency blending is numerically identical to the
   pinned mlx_lm fork up to 1.2e-10; this test pins that property
   against drift in either formula).

``rope_utils`` is faked via ``sys.modules`` injection — no real mlx_lm
model is needed. The stock-formula reference is implemented in pure
Python exactly as written in mlx_lm 0.31.3 ``rope_utils.py`` YarnRoPE.
If ``mlx.core`` is absent the whole file is skipped (mlx is a hard
dependency of the MLX engine and is present in CI).
"""

from __future__ import annotations

import importlib
import math
import sys
import types
from typing import Any

import pytest

pytest.importorskip("mlx.core")

# ---------------------------------------------------------------------------
# Stock mlx_lm YarnRoPE reference (pure Python).
#
# Mirrors .venv/lib/python3.13/site-packages/mlx_lm/models/rope_utils.py
# YarnRoPE.__init__ (mlx_lm 0.31.3, verified 2026-09-17). mlx_lm computes
# the folded form; the patch computes the inverse-frequency blend that is
# algebraically identical:
#   freq_extra = base ** (arange(0, dims, 2) / dims)
#   freq_inter = scaling_factor * freq_extra
#   freq_mask  = 1 - ramp(low, high, dims // 2)
#   freqs      = (freq_inter * freq_extra) / (
#                    freq_inter * freq_mask + freq_extra * (1 - freq_mask))
# ---------------------------------------------------------------------------
def _stock_yarn_freqs(
    dims: int,
    base: float,
    scaling_factor: float,
    original_max_position_embeddings: int,
    beta_fast: float,
    beta_slow: float,
) -> list[float]:
    def correction_dim(num_rotations: float) -> float:
        return (
            dims
            * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))
        ) / (2 * math.log(base))

    low = math.floor(correction_dim(beta_fast))
    high = math.ceil(correction_dim(beta_slow))
    low = max(low, 0)
    high = min(high, dims - 1)
    if low == high:
        high += 0.001

    n = dims // 2
    ramp = [max(0.0, min(1.0, (i - low) / (high - low))) for i in range(n)]
    freq_mask = [1.0 - v for v in ramp]

    freq_extra = [base ** (i / dims) for i in range(0, dims, 2)]
    freq_inter = [scaling_factor * v for v in freq_extra]
    return [
        (fi * fe) / (fi * fm + fe * (1.0 - fm))
        for fi, fe, fm in zip(freq_inter, freq_extra, freq_mask)
    ]


# Four validated configs from the parent analysis (t_fdb502db).
_YARN_CONFIGS: list[dict[str, Any]] = [
    dict(
        dims=128,
        base=10000,
        factor=4.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
    ),
    dict(
        dims=64,
        base=500000,
        factor=2.5,
        original_max_position_embeddings=8192,
        beta_fast=16,
        beta_slow=0.5,
    ),
    dict(
        dims=256,
        base=10000,
        factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
    ),
    dict(
        dims=96,
        base=10000,
        factor=8.0,
        original_max_position_embeddings=16384,
        beta_fast=64,
        beta_slow=2,
    ),
]

# Rope types that MUST be delegated to the original initialize_rope.
_OTHER_ROPE_TYPES = ["default", "linear", "llama3", "su", "telechat3-yarn", "unknown"]


def _make_fake_yarn_rope_class() -> type:
    """Build a FRESH YarnRoPE fake class.

    ``patch_yarn_rope()`` assigns ``YarnRoPE.__init__`` via a plain
    attribute write, so the class must be unique per fake module —
    otherwise the swap leaks across tests (patch residue from one test
    would break ``YarnRoPE()`` construction in the next).
    """

    class _FakeYarnRoPE:
        """Minimal fake mirror of mlx_lm.models.rope_utils.YarnRoPE.

        The patched __init__ is assigned by ``patch_yarn_rope()`` and
        sets ``_freqs`` / ``dims`` / ``traditional`` / ``mscale`` on the
        instance, exactly like the real class.
        """

        def __init__(self) -> None:  # pragma: no cover - super() target
            self._freqs: Any = None
            self.dims: int | None = None
            self.traditional: bool | None = None
            self.mscale: float | None = None

    return _FakeYarnRoPE


def _install_fake(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject a fake ``mlx_lm.models.rope_utils`` module into sys.modules.

    The fake carries a fresh ``YarnRoPE`` class and a recording
    ``initialize_rope``. Swapping ``sys.modules`` plus the parent package
    attribute makes ``from mlx_lm.models import rope_utils`` (used by the
    patch module) resolve to the fake on reload.
    """
    fake = types.ModuleType("mlx_lm.models.rope_utils")
    fake.YarnRoPE = _make_fake_yarn_rope_class()
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_original_initialize_rope(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return ("original", args, kwargs)

    fake.initialize_rope = fake_original_initialize_rope
    fake.__dict__["_calls"] = calls

    pkg = sys.modules.get("mlx_lm.models")
    if pkg is not None and hasattr(pkg, "rope_utils"):
        monkeypatch.setattr(pkg, "rope_utils", fake, raising=False)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.rope_utils", fake)
    return fake


def _load_patch_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Import the patch module with its ``_original_*`` globals bound to the fake.

    ``importlib.reload`` re-executes the module body, so the import-time
    captures ``_original_YarnRoPE_init`` / ``_original_initialize_rope``
    and the ``rope_utils`` module global all resolve to the injected fake.
    """
    mod = importlib.import_module("exo.worker.engines.mlx.patches.standard_yarn_rope")
    importlib.reload(mod)
    fake = sys.modules["mlx_lm.models.rope_utils"]
    # Defensive rebind in case reload served a cached module.
    mod._original_YarnRoPE_init = fake.YarnRoPE.__init__
    mod._original_initialize_rope = fake.initialize_rope
    return mod


def _freqs_list(freqs: Any) -> list[float]:
    """Normalize an mx.array (or list) of freqs to a Python list."""
    return freqs.tolist() if hasattr(freqs, "tolist") else list(freqs)


def test_patch_yarn_rope_swaps_both_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """patch_yarn_rope() replaces YarnRoPE.__init__ AND initialize_rope."""
    fake = _install_fake(monkeypatch)
    mod = _load_patch_module(monkeypatch)

    orig_init = fake.YarnRoPE.__init__
    orig_initialize = fake.initialize_rope

    mod.patch_yarn_rope()

    assert fake.YarnRoPE.__init__ is mod._patched_yarn_init
    assert fake.initialize_rope is mod._patched_initialize_rope
    # Module-level originals still point at the pre-patch implementations.
    assert mod._original_YarnRoPE_init is orig_init
    assert mod._original_initialize_rope is orig_initialize


def test_patched_initialize_rope_routes_yarn_and_deepseek_yarn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """yarn/deepseek_yarn build a YarnRoPE with the correct scaling config."""
    fake = _install_fake(monkeypatch)
    mod = _load_patch_module(monkeypatch)

    # Real-world order: the aggregator swaps YarnRoPE.__init__ before any
    # initialize_rope call, so the patched initialize_rope can construct
    # YarnRoPE instances with the patched (blend) formula.
    mod.patch_yarn_rope()

    for rope_type in ("yarn", "deepseek_yarn"):
        scaling_config = {
            "type": rope_type,
            "factor": 4.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1,
            "mscale_all_dim": 0,
        }
        rope = mod._patched_initialize_rope(
            dims=128,
            base=10000,
            traditional=False,
            scaling_config=scaling_config,
            max_position_embeddings=2048,
        )

        assert type(rope) is fake.YarnRoPE
        assert rope.dims == 128
        assert rope.traditional is False
        # mscale wired from mscale / mscale_all_dim:
        # yarn_get_mscale(4.0, 1) = 0.1*1*ln(4)+1 = 1.138629436..., and
        # yarn_get_mscale(4.0, 0) = 1.0 (scale > 1 keeps the log term).
        assert rope.mscale == pytest.approx(0.1 * math.log(4.0) + 1.0)
        # factor / original_max_position_embeddings / beta_fast / beta_slow
        # all feed the frequency formula: _freqs must exactly equal the
        # stock formula evaluated with THIS scaling config.
        got = _freqs_list(rope._freqs)
        want = _stock_yarn_freqs(
            dims=128,
            base=10000,
            scaling_factor=4.0,
            original_max_position_embeddings=4096,
            beta_fast=32,
            beta_slow=1,
        )
        assert len(got) == len(want) == 64
        assert all(
            abs(g - w) / max(abs(w), 1e-12) < 1e-6 for g, w in zip(got, want)
        )

        # No delegation to the original happened for either rope type.
        assert fake.__dict__["_calls"] == []


def test_patched_initialize_rope_delegates_other_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every non-yarn rope type goes to the original initialize_rope."""
    fake = _install_fake(monkeypatch)
    mod = _load_patch_module(monkeypatch)

    for rope_type in _OTHER_ROPE_TYPES:
        scaling_config = {"type": rope_type, "factor": 1.0}
        result = mod._patched_initialize_rope(
            dims=128,
            base=10000,
            traditional=False,
            scaling_config=scaling_config,
            max_position_embeddings=2048,
        )

        assert result[0] == "original"
        # Original was invoked with the exact positional signature.
        args, kwargs = fake.__dict__["_calls"][-1]
        assert args == (128, 10000, False, scaling_config, 2048)
        assert kwargs == {}
        fake.__dict__["_calls"].clear()


def test_patched_initialize_rope_delegates_none_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """scaling_config=None delegates (rope_type defaults to 'default')."""
    fake = _install_fake(monkeypatch)
    mod = _load_patch_module(monkeypatch)

    result = mod._patched_initialize_rope(
        dims=64,
        base=10000,
        traditional=True,
        scaling_config=None,
        max_position_embeddings=4096,
    )

    assert result[0] == "original"
    args, kwargs = fake.__dict__["_calls"][-1]
    assert args == (64, 10000, True, None, 4096)
    assert kwargs == {}


def test_patched_yarn_freqs_match_stock_mlx_lm_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_patched_yarn_init _freqs == stock mlx_lm folded formula <= 1e-6 rel."""
    fake = _install_fake(monkeypatch)
    mod = _load_patch_module(monkeypatch)

    for cfg in _YARN_CONFIGS:
        rope = fake.YarnRoPE()
        mod._patched_yarn_init(
            rope,
            dims=cfg["dims"],
            base=cfg["base"],
            scaling_factor=cfg["factor"],
            original_max_position_embeddings=cfg[
                "original_max_position_embeddings"
            ],
            beta_fast=cfg["beta_fast"],
            beta_slow=cfg["beta_slow"],
        )

        got = _freqs_list(rope._freqs)
        want = _stock_yarn_freqs(
            dims=cfg["dims"],
            base=cfg["base"],
            scaling_factor=cfg["factor"],
            original_max_position_embeddings=cfg[
                "original_max_position_embeddings"
            ],
            beta_fast=cfg["beta_fast"],
            beta_slow=cfg["beta_slow"],
        )

        assert len(got) == len(want) == cfg["dims"] // 2
        for g, w in zip(got, want):
            rel = abs(g - w) / max(abs(w), 1e-12)
            assert rel < 1e-6, (
                f"dims={cfg['dims']} base={cfg['base']} "
                f"factor={cfg['factor']}: rel err {rel:.2e} >= 1e-6"
            )

        fake.__dict__["_calls"].clear()