"""Preflight memory budget check for MLX prefill (exo issue #2374).

## Why this exists

Issue #2374: a ~19k-token prompt on a 2-node pipeline parallel placement
aborts the hub's runner partway through prefill with ``signal=6`` (SIGABRT)
on ``com.Metal.CompletionQueueDispatch``, and exo then resets the whole
instance. The user gets "Model failed" with no explanation.

The chain is:

1. ``set_wired_limit_for_model`` (utils_mlx.py) calls
   ``mx.set_wired_limit(max_recommended_working_set_size)``. That is the
   macOS default GPU wired limit -- roughly 75% of physical RAM.
2. MLX refuses to set a wired limit ABOVE ``max_recommended_working_set_size``
   (verified: raises ``ValueError: [metal::set_wired_limit] Setting a wired
   limit larger than the maximum working set size is not allowed``). So the
   only way to raise it is the system-wide ``sysctl iogpu.wired_limit_mb``,
   which requires root and is not something a user runner can do.
3. Once the wired limit is pinned at that value, GPU allocations that exceed
   it are not a catchable Python exception -- they abort the process from
   the Metal completion queue. There is no frame to catch.

So a prompt whose weights + KV cache + activations exceed the wired limit
cannot be served, and today exo finds that out by dying.

## What this module does

It estimates the peak prefill memory from things we already know locally
(weight shard size, KV geometry from the model config, prompt length) and
compares it against the wired limit BEFORE any prefill work starts. When
the estimate does not fit, the caller raises ``PrefillBudgetError`` with
the exact number of tokens that WOULD fit and the concrete remedy.

This is a guard, not a fix: a request that fits the budget still runs
unchanged, and the estimate is deliberately conservative (it over-counts)
so a borderline request is refused rather than killed.

The `iogpu.wired_limit_mb` sysctl is the real fix and is named in the error
message, because that is what the issue reporter verified works.
"""

from typing import NamedTuple

import mlx.core as mx

from exo.shared.types.memory import Memory
from exo.worker.runner.bootstrap import logger

# Bytes per element for the KV cache dtypes we might see. fp32 is the widest
# thing an unquantized cache can be; a quantized cache is always smaller, so
# reading fp32 for an unknown dtype over-estimates, which is the safe side.
_DTYPE_BYTES: tuple[tuple[str, int], ...] = (
    ("float32", 4),
    ("float16", 2),
    ("bfloat16", 2),
    ("float64", 8),
)

# Attention/activation workspace is allocated alongside the cache during a
# prefill chunk, on top of weights + cache. This factor prices that
# transient. Deliberately generous: a false refusal costs one smaller
# prompt, a false accept costs a runner abort plus an instance reset.
_ACTIVATION_RESERVE = 1.20

# The remaining fraction of the wired limit the runner may plan to use.
# Leaves room for the runtime's own allocations, ring/pipe buffers, and the
# image/tokenizer models that are resident on the same process.
_RUNTIME_RESERVE = 0.90

# Below this, the reported "limit" is not a meaningful ceiling to plan
# against, so callers skip the check instead of guessing.
_MIN_MEANINGFUL_LIMIT_MB = 1024


class PrefillBudgetError(RuntimeError):
    """Prefill would not fit in the GPU wired memory limit.

    A RuntimeError (not Exception) so the existing `except Exception` /
    `except RuntimeError` handlers around generation see it with no plumbing
    changes, while remaining catchable by type.
    """

    def __init__(
        self,
        *,
        prompt_tokens: int,
        max_prompt_tokens: int,
        estimated_peak: Memory,
        budget: Memory,
        weights: Memory,
        weights_source: str,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.estimated_peak = estimated_peak
        self.budget = budget
        self.weights = weights
        self.weights_source = weights_source
        super().__init__(str(self))

    def __str__(self) -> str:
        base = (
            f"Prefill does not fit in the GPU memory limit: this request needs "
            f"about {self.estimated_peak.in_float_mb:.0f} MB "
            f"({self.prompt_tokens:,} prompt tokens + "
            f"{self.weights.in_float_mb:.0f} MB of model weights) but only "
            f"{self.budget.in_float_mb:.0f} MB is usable on this node. "
        )
        if self.max_prompt_tokens <= 0:
            # Weights alone do not fit, so no prompt size would work. Saying
            # "at most 0 tokens" reads like a broken counter; the real cause
            # is that this shard is too large for the node it landed on.
            remedy = (
                "The model weights alone exceed that budget, so no prompt of "
                "any size can be prefilled here -- this shard is too large for "
                "this node. Use a smaller quantization, spread it across more "
                "nodes, or run it on a larger machine."
            )
        else:
            remedy = (
                f"At most about {self.max_prompt_tokens:,} tokens can be "
                f"prefilled in a single request here. Split the input into "
                f"smaller requests, lower EXO_MAX_KV_SIZE, or raise the system "
                f"wired limit with `sudo sysctl iogpu.wired_limit_mb=<MB>`."
            )
        return f"{base}{remedy} Weights measured from: {self.weights_source}."


class KvGeometry(NamedTuple):
    """What one token of KV cache costs, for the layers on this node.

    `kv_layers` is the layer count resident here (not the model's total) so
    the caller does not have to re-derive the shard split.
    """

    kv_layers: int
    kv_heads: int
    head_dim: int
    bytes_per_element: int

    @property
    def bytes_per_token(self) -> int:
        # K and V are both cached, so the factor of 2 is not optional.
        return (
            2 * self.kv_layers * self.kv_heads * self.head_dim * self.bytes_per_element
        )


def _int_attr(obj: object, names: tuple[str, ...]) -> int | None:
    """First present attribute from `names`, narrowed to a positive int.

    MLX model configs spell the same quantity several ways across
    architectures (`n_kv_heads` vs `num_key_value_heads`, and a config may be
    a pydantic model, a dataclass, or a plain object). Probing by name keeps
    this working across all of them without importing a per-architecture
    config schema.
    """
    for name in names:
        value = getattr(obj, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _dtype_bytes(cfg: object) -> int:
    dtype = getattr(cfg, "torch_dtype", None) or getattr(cfg, "dtype", None)
    name = (getattr(dtype, "__name__", None) or str(dtype) or "").lower()
    for key, size in _DTYPE_BYTES:
        if key in name:
            return size
    return 4


def _config_of(model: object) -> object | None:
    for name in ("args", "config", "model_config"):
        cfg = getattr(model, name, None)
        if cfg is not None:
            return cfg
    return None


def _resolve_geometry(
    model: object, kv_layers: int, kv_bits: int | None
) -> KvGeometry | None:
    """Best-effort KV geometry, or None when it cannot be read.

    None (rather than a guess) is the important part: a caller that cannot
    price the request must not refuse it, because an inflated guess would
    block requests that actually fit.
    """
    if kv_layers <= 0:
        return None
    cfg = _config_of(model)
    if cfg is None:
        return None

    if kv_bits is not None:
        # Quantized cache: packed weights plus fp32 scales per group.
        # 1 byte/element is close for 4-bit and still an over-estimate for
        # 8-bit, which is the safe direction.
        return KvGeometry(kv_layers, 1, 1, kv_bits // 8 or 1)

    layers = _int_attr(cfg, ("num_hidden_layers", "n_layer", "num_layers"))
    heads = _int_attr(cfg, ("num_attention_heads", "n_head"))
    kv_heads = _int_attr(cfg, ("num_key_value_heads", "n_kv_heads"))
    hidden = _int_attr(cfg, ("hidden_size", "n_embd", "d_model"))
    if layers is None or heads is None or kv_heads is None or hidden is None:
        return None

    # A GQA config with more kv heads than heads is malformed; the geometric
    # value is the authoritative one, so clamp rather than trust it.
    return KvGeometry(
        kv_layers=kv_layers,
        kv_heads=min(kv_heads, heads),
        head_dim=max(1, hidden // heads),
        bytes_per_element=_dtype_bytes(cfg),
    )


def _iter_arrays(obj: object, depth: int = 0):
    """Yield mx.array leaves reachable from `obj`, at bounded depth."""
    if depth > 4:
        return
    if isinstance(obj, mx.array):
        yield obj  # type: ignore[misc]
        return
    if isinstance(obj, dict):
        children: list[object] = list(obj.values())
    elif isinstance(obj, (list, tuple)):
        children = list(obj)
    elif hasattr(obj, "__dict__"):
        children = [
            v for v in vars(obj).values() if not callable(v) and not isinstance(v, type)
        ]
    else:
        return
    for child in children:
        yield from _iter_arrays(child, depth + 1)


def measure_weights(model: object) -> tuple[Memory, str]:
    """Resident weight bytes for a loaded model, plus how we found out.

    Prefers the allocator's own number -- it is the truth, and it excludes
    module attributes that never become GPU arrays. Falls back to a
    recursive sum of parameter nbytes.
    """
    try:
        active = mx.get_active_memory()
        if active > 0:
            return Memory.from_bytes(active), "mlx.get_active_memory()"
    except Exception:  # pragma: no cover - allocator not ready
        logger.debug("get_active_memory unavailable for weight sizing", exc_info=True)

    try:
        total = 0
        for a in _iter_arrays(model):
            n = getattr(a, "nbytes", 0)
            if n:
                total += n  # type: ignore[reportUnnecessaryCast]
    except Exception:  # pragma: no cover - exotic model objects
        logger.debug("recursive weight sizing failed", exc_info=True)
        return Memory.from_bytes(0), "unknown"

    if total <= 0:
        return Memory.from_bytes(0), "unknown"
    return Memory.from_bytes(int(total)), "sum of parameter nbytes"


def wired_limit() -> Memory | None:
    """The GPU wired limit exo pinned for this process, or None if unknown.

    ``max_recommended_working_set_size`` is what MLX reports and what
    ``set_wired_limit_for_model`` sets, so reading it back is the closest
    thing to asking the driver for the real ceiling.
    """
    try:
        if not mx.metal.is_available():
            return None
        raw = mx.device_info().get("max_recommended_working_set_size")
    except Exception:  # pragma: no cover - no Metal device
        return None
    if not isinstance(raw, int) or raw <= 0:
        return None
    limit = Memory.from_bytes(raw)
    if limit.in_mb < _MIN_MEANINGFUL_LIMIT_MB:
        return None
    return limit


def plan(
    *,
    limit: Memory,
    weights: Memory,
    geometry: KvGeometry,
    prompt_tokens: int,
    activation_reserve: float = _ACTIVATION_RESERVE,
    runtime_reserve: float = _RUNTIME_RESERVE,
) -> tuple[Memory, Memory]:
    """Return (effective_budget, estimated_peak) for a request.

    Kept pure so the arithmetic can be tested without a GPU.
    """
    budget = Memory.from_bytes(int(limit.in_bytes * runtime_reserve))
    peak = Memory.from_bytes(
        int(
            (weights.in_bytes + geometry.bytes_per_token * prompt_tokens)
            * activation_reserve
        )
    )
    return budget, peak


def max_prompt_tokens_for(
    *,
    budget: Memory,
    weights: Memory,
    geometry: KvGeometry,
    activation_reserve: float = _ACTIVATION_RESERVE,
) -> int:
    """Largest prompt that fits `budget` given the resident weights.

    Mirrors `plan`'s arithmetic exactly (same reserve applied to the
    weights term), so the advertised ceiling is actually the point where
    `plan` would start refusing. An off-by-reserve here would tell the
    user a number that immediately fails.
    """
    per_token = geometry.bytes_per_token * activation_reserve
    if per_token <= 0:
        return 0
    available = budget.in_bytes - int(weights.in_bytes * activation_reserve)
    if available <= 0:
        return 0
    return max(0, int(available // per_token))


def check_prefill_budget(
    *,
    model: object,
    prompt_tokens: int,
    kv_layers: int,
    kv_bits: int | None = None,
    weights: Memory | None = None,
    weights_source: str = "not measured",
    limit: Memory | None = None,
    activation_reserve: float = _ACTIVATION_RESERVE,
    runtime_reserve: float = _RUNTIME_RESERVE,
) -> PrefillBudgetError | None:
    """Return the error a prefill would hit, or None if it fits.

    Returning rather than raising keeps the caller's control flow explicit:
    it can log, raise, or still attempt the prefill. Every input is
    overridable so the arithmetic is testable without Metal.
    """
    if limit is None:
        limit = wired_limit()
    if limit is None:
        return None

    geometry = _resolve_geometry(model, kv_layers, kv_bits)
    if geometry is None:
        logger.debug(
            "Skipping prefill budget check: KV geometry unknown for this model"
        )
        return None

    if weights is None:
        weights, weights_source = measure_weights(model)
    if weights.in_bytes <= 0:
        # Without a weight measurement the peak estimate is meaningless, and
        # a meaningless estimate must not block real work.
        logger.debug("Skipping prefill budget check: weights unmeasurable")
        return None

    budget, peak = plan(
        limit=limit,
        weights=weights,
        geometry=geometry,
        prompt_tokens=prompt_tokens,
        activation_reserve=activation_reserve,
        runtime_reserve=runtime_reserve,
    )
    if peak.in_bytes <= budget.in_bytes:
        return None

    logger.error(
        f"Prefill budget exceeded: {prompt_tokens:,} tokens needs ~"
        f"{peak.in_float_mb:.0f} MB, usable budget is "
        f"{budget.in_float_mb:.0f} MB (wired limit {limit.in_float_mb:.0f} MB "
        f"x reserve {runtime_reserve})"
    )
    return PrefillBudgetError(
        prompt_tokens=prompt_tokens,
        max_prompt_tokens=max_prompt_tokens_for(
            budget=budget,
            weights=weights,
            geometry=geometry,
            activation_reserve=activation_reserve,
        ),
        estimated_peak=peak,
        budget=budget,
        weights=weights,
        weights_source=weights_source,
    )
