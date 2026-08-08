"""Tests for the TODO #7 deferred-prefill guard."""

from exo.worker.engines.mlx.generator.batch_generate import can_defer_prefill


def test_defers_when_plain_single_node() -> None:
    assert can_defer_prefill(
        group=None,
        has_prefix_cache=False,
        use_prefix_cache=True,
        has_vision=False,
        is_bench=False,
    )


def test_does_not_defer_with_distributed_group() -> None:
    # Pipeline ranks need the eager prefill's distributed sync.
    assert not can_defer_prefill(
        group=object(),  # type: ignore[arg-type]
        has_prefix_cache=False,
        use_prefix_cache=True,
        has_vision=False,
        is_bench=False,
    )


def test_does_not_defer_with_active_prefix_cache() -> None:
    assert not can_defer_prefill(
        group=None,
        has_prefix_cache=True,
        use_prefix_cache=True,
        has_vision=False,
        is_bench=False,
    )


def test_defers_when_prefix_cache_present_but_disabled() -> None:
    assert can_defer_prefill(
        group=None,
        has_prefix_cache=True,
        use_prefix_cache=False,
        has_vision=False,
        is_bench=False,
    )


def test_does_not_defer_with_vision() -> None:
    assert not can_defer_prefill(
        group=None,
        has_prefix_cache=False,
        use_prefix_cache=True,
        has_vision=True,
        is_bench=False,
    )


def test_does_not_defer_with_bench() -> None:
    assert not can_defer_prefill(
        group=None,
        has_prefix_cache=False,
        use_prefix_cache=True,
        has_vision=False,
        is_bench=True,
    )
