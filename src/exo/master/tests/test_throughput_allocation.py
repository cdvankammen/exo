# pyright: reportAny=false, reportUnknownVariableType=false
"""Tests for throughput-based layer allocation and the chip bandwidth table.

Ported from PR #2229 (https://github.com/exo-explore/exo/pull/2229) by
@jgawronek — bandwidth-aware placement portion.
"""

import pytest

from exo.master.placement_utils import (
    allocate_layers_by_throughput,
    estimate_memory_bandwidth_gigabytes_per_second,
)
from exo.shared.types.profiling import NodeIdentity


def test_allocate_layers_by_throughput_fills_fastest_first() -> None:
    allocations = allocate_layers_by_throughput(
        total_layers=12,
        node_throughputs=[819.0, 273.0, 800.0],
        max_layers_per_node=[4, 4, 8],
    )
    assert allocations == [4, 1, 7]
    assert sum(allocations) == 12


def test_allocate_layers_by_throughput_keeps_one_layer_per_node() -> None:
    allocations = allocate_layers_by_throughput(
        total_layers=3,
        node_throughputs=[1000.0, 1.0, 1.0],
        max_layers_per_node=[3, 3, 3],
    )
    assert allocations == [1, 1, 1]


def test_allocate_layers_by_throughput_rejects_insufficient_capacity() -> None:
    with pytest.raises(ValueError, match="capacity"):
        _ = allocate_layers_by_throughput(
            total_layers=10,
            node_throughputs=[100.0, 100.0],
            max_layers_per_node=[4, 4],
        )


def test_estimate_memory_bandwidth_matches_known_chips() -> None:
    assert (
        estimate_memory_bandwidth_gigabytes_per_second(
            NodeIdentity(chip_id="Apple M3 Ultra")
        )
        == 819.0
    )
    assert (
        estimate_memory_bandwidth_gigabytes_per_second(
            NodeIdentity(chip_id="NVIDIA GB10")
        )
        == 273.0
    )
    assert (
        estimate_memory_bandwidth_gigabytes_per_second(
            NodeIdentity(chip_id="NVIDIA GeForce RTX 3090")
        )
        == 936.0
    )
    assert (
        estimate_memory_bandwidth_gigabytes_per_second(
            NodeIdentity(chip_id="Unknown Chip")
        )
        is None
    )
