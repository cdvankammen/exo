"""Tests for the prefill memory budget guard (exo issue #2374).

The guard's job is to refuse a prefill that would abort the Metal process
before the abort happens, and to stay out of the way when it cannot tell.

These are deliberately pure-arithmetic tests: `limit`, `weights`, and
`kv_bits` are all injectable, so nothing here needs a second GPU or a
32 GB hub. The one place real hardware matters -- `wired_limit()` -- is
covered separately below with whatever the current machine reports.
"""

import pytest

from exo.shared.types.memory import Memory
from exo.worker.engines.mlx.generator.prefill_budget import (
    KvGeometry,
    PrefillBudgetError,
    check_prefill_budget,
    max_prompt_tokens_for,
    measure_weights,
    plan,
    wired_limit,
)

GIB = 1024**3
MEG = 1024**2


class FakeConfig:
    """Stand-in for an mlx-lm model config."""

    def __init__(
        self,
        *,
        num_hidden_layers=48,
        num_attention_heads=64,
        num_key_value_heads=8,
        hidden_size=8192,
        torch_dtype="bfloat16",
    ):
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.torch_dtype = torch_dtype


class FakeModel:
    """Stand-in for a loaded model, with the config where MLX keeps it."""

    def __init__(self, cfg=None):
        self.args = cfg if cfg is not None else FakeConfig()


# Geometry maths -------------------------------------------------------------


def test_bytes_per_token_is_k_and_v_for_every_layer():
    g = KvGeometry(kv_layers=10, kv_heads=8, head_dim=128, bytes_per_element=2)
    # 2 (K+V) * 10 layers * 8 heads * 128 dim * 2 bytes
    assert g.bytes_per_token == 2 * 10 * 8 * 128 * 2


def test_plan_scales_with_prompt_length():
    weights = Memory.from_gb(10)
    limit = Memory.from_gb(30)
    geom = KvGeometry(48, 8, 128, 2)

    _, small = plan(limit=limit, weights=weights, geometry=geom, prompt_tokens=1_000)
    _, large = plan(limit=limit, weights=weights, geometry=geom, prompt_tokens=50_000)

    assert large.in_bytes > small.in_bytes
    # Only the prompt term differs, so the gap is exactly the KV delta * reserve.
    delta = (50_000 - 1_000) * geom.bytes_per_token
    assert large.in_bytes - small.in_bytes == pytest.approx(delta * 1.20, rel=1e-6)


def test_budget_reserves_part_of_the_wired_limit():
    limit = Memory.from_gb(32)
    budget, _ = plan(
        limit=limit,
        weights=Memory.from_gb(1),
        geometry=KvGeometry(48, 8, 128, 2),
        prompt_tokens=10,
        runtime_reserve=0.90,
    )
    assert budget.in_bytes == int(limit.in_bytes * 0.90)
    assert budget.in_bytes < limit.in_bytes


def test_max_prompt_tokens_is_the_exact_refusal_boundary():
    """The advertised ceiling must be the number `plan` actually accepts.

    If these drift apart the error message lies to the user: it suggests a
    size that is immediately refused, or lets a fatal one through.
    """
    weights = Memory.from_gb(14)
    limit = Memory.from_gb(25)
    geom = KvGeometry(24, 8, 128, 2)

    budget, peak = plan(limit=limit, weights=weights, geometry=geom, prompt_tokens=10)
    ceiling = max_prompt_tokens_for(budget=budget, weights=weights, geometry=geom)

    assert ceiling > 0
    _, at_ceiling = plan(
        limit=limit, weights=weights, geometry=geom, prompt_tokens=ceiling
    )
    assert at_ceiling.in_bytes <= budget.in_bytes, "ceiling itself is refused"

    _, over_ceiling = plan(
        limit=limit, weights=weights, geometry=geom, prompt_tokens=ceiling + 1
    )
    assert over_ceiling.in_bytes > budget.in_bytes, "ceiling+1 is accepted"


def test_max_prompt_tokens_is_zero_when_weights_alone_exceed_budget():
    assert (
        max_prompt_tokens_for(
            budget=Memory.from_gb(1),
            weights=Memory.from_gb(30),
            geometry=KvGeometry(48, 8, 128, 2),
        )
        == 0
    )


# The refusal itself ---------------------------------------------------------


def _check(model, tokens, *, limit, weights, kv_layers=24, kv_bits=None):
    return check_prefill_budget(
        model=model,
        prompt_tokens=tokens,
        kv_layers=kv_layers,
        kv_bits=kv_bits,
        weights=weights,
        weights_source="test",
        limit=limit,
    )


def test_returns_none_when_the_request_fits():
    assert (
        _check(
            FakeModel(),
            512,
            limit=Memory.from_gb(30),
            weights=Memory.from_gb(20),
        )
        is None
    )


def test_refuses_a_prompt_that_cannot_fit():
    err = _check(
        FakeModel(),
        200_000,
        limit=Memory.from_gb(25),
        weights=Memory.from_gb(10),
    )
    assert isinstance(err, PrefillBudgetError)
    assert err.prompt_tokens == 200_000
    assert err.max_prompt_tokens < 200_000


def test_weights_alone_over_budget_gets_a_useful_message():
    """When the resident weights already exceed the budget there is no
    workable prompt size. The message must say *that*, not advertise
    "at most 0 tokens" -- which reads like a broken counter and gives the
    user nothing to act on."""
    err = _check(
        FakeModel(),
        512,
        limit=Memory.from_gb(25),
        weights=Memory.from_gb(30),
    )
    assert isinstance(err, PrefillBudgetError)
    assert err.max_prompt_tokens == 0
    msg = str(err)
    assert "weights alone exceed" in msg
    assert "at most" not in msg.lower()
    # Still must not be mistaken for a recoverable OOM.
    assert "oom" not in msg.lower()
    assert "out of memory" not in msg.lower()


def test_refusal_is_a_runtimeerror_so_existing_handlers_see_it():
    """`_send_error` and the generation loops catch Exception; a new base
    class would escape them and look like a crash."""
    err = _check(
        FakeModel(), 200_000, limit=Memory.from_gb(25), weights=Memory.from_gb(10)
    )
    assert isinstance(err, RuntimeError)


def test_error_message_names_the_remedy_and_the_ceiling():
    err = _check(
        FakeModel(), 200_000, limit=Memory.from_gb(25), weights=Memory.from_gb(10)
    )
    msg = str(err)
    assert "iogpu.wired_limit_mb" in msg
    assert f"{err.max_prompt_tokens:,}" in msg
    assert "200,000" in msg


def test_error_message_does_not_trip_the_oom_recovery_markers():
    """runner._is_oom() matches on substrings including the bare word "oom".

    A message containing one of those would send this deterministic,
    unfixable-by-retry rejection into the OOM retry loop -- clear the cache,
    halve concurrency, retry -- which cannot help here and hides the real
    message behind a RunnerDegraded.
    """
    markers = (
        "failed to allocate",
        "insufficient memory",
        "out of memory",
        "oom",
        "bad_alloc",
        "allocator",
    )
    err = _check(
        FakeModel(), 200_000, limit=Memory.from_gb(25), weights=Memory.from_gb(10)
    )
    low = str(err).lower()
    for marker in markers:
        assert marker not in low, f"error message trips OOM marker {marker!r}"


def test_reports_the_actual_wiring_source():
    err = _check(
        FakeModel(), 200_000, limit=Memory.from_gb(25), weights=Memory.from_gb(10)
    )
    assert "test" in err.weights_source


# Staying out of the way -----------------------------------------------------


def test_limit_none_auto_detects_rather_than_disabling():
    """`limit=None` means "ask the driver", not "no ceiling".

    Passing None must fall back to `wired_limit()`; it must not silently turn
    the guard off, which would reopen the very abort this check prevents.
    """
    import inspect

    params = inspect.signature(check_prefill_budget).parameters
    assert params["limit"].default is None
    # And on a host with no Metal device the fallback yields None -> allowed.
    if wired_limit() is None:
        assert (
            check_prefill_budget(
                model=FakeModel(),
                prompt_tokens=10_000_000,
                kv_layers=48,
                weights=Memory.from_gb(20),
                limit=None,
            )
            is None
        )


def test_unknown_geometry_means_no_check():
    """A model whose config we cannot read must not be blocked -- an
    inflated guess would refuse requests that would in fact fit."""

    class Opaque:
        pass

    assert (
        _check(
            Opaque(),
            500_000,
            limit=Memory.from_gb(1),
            weights=Memory.from_gb(20),
        )
        is None
    )


def test_zero_weight_measurement_means_no_check():
    assert (
        _check(
            FakeModel(),
            500_000,
            limit=Memory.from_gb(1),
            weights=Memory.from_bytes(0),
        )
        is None
    )


def test_zero_kv_layers_means_no_check():
    assert (
        _check(
            FakeModel(),
            500_000,
            limit=Memory.from_gb(1),
            weights=Memory.from_gb(20),
            kv_layers=0,
        )
        is None
    )


def test_malformed_config_is_survivable():
    """Bad numbers must not raise out of the guard; they must disable it."""

    class Weird:
        num_hidden_layers = "not a number"
        num_attention_heads = None
        hidden_size = -1

    assert (
        _check(Weird(), 500_000, limit=Memory.from_gb(1), weights=Memory.from_gb(20))
        is None
    )


def test_kv_heads_above_heads_is_clamped():
    """A GQA config claiming more kv heads than heads is malformed; the
    geometric value wins, otherwise the estimate can only over-count."""

    class Bad(FakeConfig):
        pass

    model = FakeModel(
        Bad(num_key_value_heads=999, num_attention_heads=8, hidden_size=4096)
    )
    err = _check(model, 200_000, limit=Memory.from_gb(25), weights=Memory.from_gb(10))
    # It must produce a sane number either way rather than an absurd one.
    if err is not None:
        assert 0 < err.max_prompt_tokens < 200_000


def test_quantized_cache_is_cheaper_than_full_precision():
    cfg = FakeConfig()
    limit = Memory.from_gb(25)
    weights = Memory.from_gb(10)
    tokens = 200_000

    full = _check(FakeModel(cfg), tokens, limit=limit, weights=weights, kv_bits=None)
    quant = _check(FakeModel(cfg), tokens, limit=limit, weights=weights, kv_bits=4)

    # bfloat16 KV at this size cannot fit; a 4-bit cache can. That asymmetry
    # is the point -- quantising the cache is the remedy the guard's own error
    # message recommends, so the recommendation has to actually hold.
    assert isinstance(full, PrefillBudgetError)
    assert quant is None


# Live wiring ---------------------------------------------------------------


def test_wired_limit_matches_what_exo_pins():
    """exo pins mx.set_wired_limit(max_recommended_working_set_size).

    If these ever diverge the guard would be measuring against a ceiling the
    process is not actually subject to.
    """
    limit = wired_limit()
    if limit is None:
        pytest.skip("no Metal device or implausible limit on this host")
    import mlx.core as mx

    reported = int(mx.device_info()["max_recommended_working_set_size"])
    assert limit.in_bytes == reported


def test_measure_weights_falls_back_when_allocator_is_empty():
    import mlx.core as mx

    if mx.get_active_memory() > 0:
        pytest.skip("allocator already holds memory in this process")
    size, source = measure_weights(object())
    assert size.in_bytes == 0
    assert source == "unknown"


def test_measure_weights_never_raises_on_exotic_objects():
    class Weird:
        def __getattr__(self, name):
            raise RuntimeError("boom")

    size, source = measure_weights(Weird())
    assert size.in_bytes >= 0
    assert source
