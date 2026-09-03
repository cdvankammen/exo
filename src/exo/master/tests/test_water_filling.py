# pyright: reportUnusedFunction=false, reportAny=false
"""Tests for the water-filling pipeline layer allocator (P1 #35, GitHub #957).

Water-filling minimises the bottleneck between per-stage compute time and
inter-stage communication time, avoiding the greedy fill-fastest-first trap
where a fast GPU behind a slow link gets overloaded.
"""
import pytest

from exo.master.placement_utils import allocate_layers_by_water_filling


class TestAllocateLayersByWaterFilling:
    def test_balanced_hardware_balanced_layers(self) -> None:
        """Identical nodes with identical links get equal layer counts."""
        result = allocate_layers_by_water_filling(
            total_layers=12,
            node_throughputs=[100.0, 100.0, 100.0],
            link_bandwidths=[1000.0, 1000.0, 1000.0],
            max_layers_per_node=[12, 12, 12],
        )
        assert result == [4, 4, 4]

    def test_slower_link_gets_fewer_layers(self) -> None:
        """A stage with a slow outgoing link should be given fewer layers."""
        result = allocate_layers_by_water_filling(
            total_layers=10,
            node_throughputs=[200.0, 200.0],  # same GPU speed
            link_bandwidths=[100.0, 10.0],     # node 1 has slow link
            max_layers_per_node=[10, 10],
        )
        # Node 1's effective speed is min(200, 10) = 10, so it gets fewer layers
        assert result[0] > result[1]
        assert sum(result) == 10

    def test_memory_cap_respected(self) -> None:
        """Nodes at their memory cap don't receive more layers."""
        result = allocate_layers_by_water_filling(
            total_layers=8,
            node_throughputs=[200.0, 200.0, 200.0],
            link_bandwidths=[1000.0, 1000.0, 1000.0],
            max_layers_per_node=[2, 2, 4],  # first two capped at 2
        )
        assert result[0] <= 2
        assert result[1] <= 2
        assert result[2] >= 4  # absorbs the overflow
        assert sum(result) == 8

    def test_zero_bandwidth_link_handled(self) -> None:
        """A zero-bandwidth link degrades to greedy (not crash)."""
        result = allocate_layers_by_water_filling(
            total_layers=6,
            node_throughputs=[100.0, 100.0],
            link_bandwidths=[0.0, 1000.0],  # node 0 has unknown link
            max_layers_per_node=[6, 6],
        )
        # Both nodes still get at least 1 layer (pipeline invariant)
        assert all(layers >= 1 for layers in result)
        assert sum(result) == 6

    def test_minimum_one_layer_per_node(self) -> None:
        """Every node gets at least 1 layer even with skewed speeds."""
        result = allocate_layers_by_water_filling(
            total_layers=4,
            node_throughputs=[1000.0, 1.0, 1.0],  # 1000x speed difference
            link_bandwidths=[1000.0, 1000.0, 1000.0],
            max_layers_per_node=[4, 4, 4],
        )
        assert all(layers >= 1 for layers in result)
        assert sum(result) == 4
        # Fast node should get the most layers
        assert result[0] > result[1]
        assert result[0] > result[2]

    def test_impossible_total_capacity_raises(self) -> None:
        """Raises when total capacity < total_layers."""
        with pytest.raises(ValueError, match="only have capacity"):
            allocate_layers_by_water_filling(
                total_layers=10,
                node_throughputs=[100.0, 100.0],
                link_bandwidths=[1000.0, 1000.0],
                max_layers_per_node=[3, 3],  # 6 < 10
            )

    def test_too_few_layers_raises(self) -> None:
        """Raises when total_layers < number of nodes."""
        with pytest.raises(ValueError, match="at least 1 layer per node"):
            allocate_layers_by_water_filling(
                total_layers=2,
                node_throughputs=[100.0, 100.0, 100.0],
                link_bandwidths=[1000.0, 1000.0, 1000.0],
                max_layers_per_node=[2, 2, 2],
            )

    def test_empty_node_list_raises(self) -> None:
        with pytest.raises(ValueError, match="empty node list"):
            allocate_layers_by_water_filling(
                total_layers=10,
                node_throughputs=[],
                link_bandwidths=[],
                max_layers_per_node=[],
            )

    def test_mismatched_bandwidth_count_raises(self) -> None:
        with pytest.raises(ValueError, match="link_bandwidths must have"):
            allocate_layers_by_water_filling(
                total_layers=10,
                node_throughputs=[100.0, 100.0],
                link_bandwidths=[1000.0],  # only 1, need 2
                max_layers_per_node=[10, 10],
            )
