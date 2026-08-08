"""Tests for MemoryUsage pressure/used derivation (TODO #13)."""

from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage


def test_from_bytes_computes_used_and_pressure() -> None:
    usage = MemoryUsage.from_bytes(
        ram_total=16_000_000_000,
        ram_available=12_000_000_000,
        swap_total=2_000_000_000,
        swap_available=2_000_000_000,
    )

    assert usage.ram_used == Memory.from_bytes(4_000_000_000)
    assert usage.pressure == 0.25


def test_pressure_zero_when_nothing_used() -> None:
    usage = MemoryUsage.from_bytes(
        ram_total=8_000_000_000,
        ram_available=8_000_000_000,
        swap_total=0,
        swap_available=0,
    )

    assert usage.ram_used.in_bytes == 0
    assert usage.pressure == 0.0


def test_pressure_one_when_fully_used() -> None:
    usage = MemoryUsage.from_bytes(
        ram_total=8_000_000_000,
        ram_available=0,
        swap_total=0,
        swap_available=0,
    )

    assert usage.ram_used.in_bytes == 8_000_000_000
    assert usage.pressure == 1.0


def test_available_cannot_exceed_total() -> None:
    """psutil can transiently report available > total; clamp used at 0."""
    usage = MemoryUsage.from_bytes(
        ram_total=8_000_000_000,
        ram_available=9_000_000_000,
        swap_total=0,
        swap_available=0,
    )

    assert usage.ram_used.in_bytes == 0
    assert usage.pressure == 0.0


def test_zero_total_avoids_division_by_zero() -> None:
    usage = MemoryUsage.from_bytes(
        ram_total=0,
        ram_available=0,
        swap_total=0,
        swap_available=0,
    )

    assert usage.pressure == 0.0


def test_direct_construction_keeps_defaults() -> None:
    """Backwards compat: callers constructing MemoryUsage directly still work."""
    usage = MemoryUsage(
        ram_total=Memory.from_bytes(1000),
        ram_available=Memory.from_bytes(500),
        swap_total=Memory.from_bytes(0),
        swap_available=Memory.from_bytes(0),
    )

    assert usage.ram_used.in_bytes == 0  # default; from_bytes is the canonical path
    assert usage.pressure == 0.0
