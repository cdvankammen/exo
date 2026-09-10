import pytest

from exo.master.placement_utils import (
    allocate_layers_proportionally,
    estimate_pipeline_activation_memory,
    estimate_ring_node_memory,
    filter_cycles_by_memory,
    filter_cycles_by_replicated_memory,
    get_mlx_jaccl_coordinators,
    get_shard_assignments,
    get_shard_assignments_for_pipeline_parallel,
    get_smallest_cycles,
    should_prefer_single_node,
)
from exo.master.tests.conftest import (
    create_node_memory,
    create_socket_connection,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import (
    NetworkInterfaceInfo,
    NodeIdentity,
    NodeNetworkInfo,
)
from exo.shared.types.topology import Connection, Cycle, SocketConnection
from exo.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    Sharding,
)


def test_filter_cycles_by_memory():
    # arrange
    node1_id = NodeId()
    node2_id = NodeId()
    connection1 = Connection(
        source=node1_id, sink=node2_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node2_id, sink=node1_id, edge=create_socket_connection(2)
    )

    node1_mem = create_node_memory(1000 * 1024)
    node2_mem = create_node_memory(1000 * 1024)
    node_memory = {node1_id: node1_mem, node2_id: node2_mem}

    topology = Topology()
    topology.add_node(node1_id)
    topology.add_node(node2_id)
    topology.add_connection(connection1)
    topology.add_connection(connection2)

    cycles = [c for c in topology.get_cycles() if len(c) != 1]
    assert len(cycles) == 1
    assert len(cycles[0]) == 2

    # act
    filtered_cycles = filter_cycles_by_memory(cycles, node_memory, Memory.from_bytes(1))

    # assert
    assert len(filtered_cycles) == 1
    assert len(filtered_cycles[0]) == 2
    assert set(n for n in filtered_cycles[0]) == {node1_id, node2_id}


def test_filter_cycles_by_insufficient_memory():
    # arrange
    node1_id = NodeId()
    node2_id = NodeId()
    connection1 = Connection(
        source=node1_id, sink=node2_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node2_id, sink=node1_id, edge=create_socket_connection(2)
    )

    node1_mem = create_node_memory(1000 * 1024)
    node2_mem = create_node_memory(1000 * 1024)
    node_memory = {node1_id: node1_mem, node2_id: node2_mem}

    topology = Topology()
    topology.add_node(node1_id)
    topology.add_node(node2_id)
    topology.add_connection(connection1)
    topology.add_connection(connection2)

    # act
    filtered_cycles = filter_cycles_by_memory(
        topology.get_cycles(), node_memory, Memory.from_kb(2001)
    )

    # assert
    assert len(filtered_cycles) == 0


def test_filter_cycles_by_memory_force_override():
    """force_override=True should keep cycles even when total available memory
    is below the required memory."""
    # arrange
    node1_id = NodeId()
    node2_id = NodeId()
    connection1 = Connection(
        source=node1_id, sink=node2_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node2_id, sink=node1_id, edge=create_socket_connection(2)
    )

    node1_mem = create_node_memory(1000 * 1024)
    node2_mem = create_node_memory(1000 * 1024)
    node_memory = {node1_id: node1_mem, node2_id: node2_mem}

    topology = Topology()
    topology.add_node(node1_id)
    topology.add_node(node2_id)
    topology.add_connection(connection1)
    topology.add_connection(connection2)

    # act
    filtered_cycles = filter_cycles_by_memory(
        topology.get_cycles(),
        node_memory,
        Memory.from_kb(2001),
        force_override=True,
    )

    # assert
    assert len(filtered_cycles) >= 1


def test_filter_multiple_cycles_by_memory():
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()
    connection1 = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node_b_id, sink=node_a_id, edge=create_socket_connection(2)
    )
    connection3 = Connection(
        source=node_a_id, sink=node_c_id, edge=create_socket_connection(3)
    )
    connection4 = Connection(
        source=node_c_id, sink=node_b_id, edge=create_socket_connection(4)
    )

    node_a_mem = create_node_memory(500 * 1024)
    node_b_mem = create_node_memory(500 * 1024)
    node_c_mem = create_node_memory(1000 * 1024)
    node_memory = {
        node_a_id: node_a_mem,
        node_b_id: node_b_mem,
        node_c_id: node_c_mem,
    }

    topology = Topology()
    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)
    topology.add_connection(connection1)
    topology.add_connection(connection2)
    topology.add_connection(connection3)
    topology.add_connection(connection4)

    cycles = topology.get_cycles()

    # act
    filtered_cycles = filter_cycles_by_memory(cycles, node_memory, Memory.from_kb(1500))

    # assert
    assert len(filtered_cycles) == 1
    assert len(filtered_cycles[0]) == 3
    assert set(n for n in filtered_cycles[0]) == {
        node_a_id,
        node_b_id,
        node_c_id,
    }


def test_get_smallest_cycles():
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()

    topology = Topology()
    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)

    connection1 = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node_b_id, sink=node_a_id, edge=create_socket_connection(2)
    )
    connection3 = Connection(
        source=node_a_id, sink=node_c_id, edge=create_socket_connection(3)
    )
    connection4 = Connection(
        source=node_c_id, sink=node_b_id, edge=create_socket_connection(4)
    )

    topology.add_connection(connection1)
    topology.add_connection(connection2)
    topology.add_connection(connection3)
    topology.add_connection(connection4)

    cycles = [c for c in topology.get_cycles() if len(c) != 1]  # ignore singletons

    # act
    smallest_cycles = get_smallest_cycles(cycles)

    # assert
    assert len(smallest_cycles) == 1
    assert len(smallest_cycles[0]) == 2
    assert set(n for n in smallest_cycles[0]) == {node_a_id, node_b_id}


@pytest.mark.parametrize(
    "available_memory,total_layers,expected_layers",
    [
        ((500, 500, 1000), 12, (3, 3, 6)),
        ((500, 500, 500), 12, (4, 4, 4)),
        ((312, 518, 1024), 12, (2, 3, 7)),
        # Edge case: one node has ~90% of memory - should not over-allocate.
        # Each node must have enough memory for at least 1 layer + 10% activation.
        # 1 layer = 1000KB/20 = 50KB, +10% = 55KB minimum.
        ((990, 55, 55), 20, (18, 1, 1)),
    ],
)
def test_get_shard_assignments(
    available_memory: tuple[int, int, int],
    total_layers: int,
    expected_layers: tuple[int, int, int],
):
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()

    # create connections (A -> B -> C -> A forms a 3-cycle, plus B -> A also exists)
    connection1 = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    connection2 = Connection(
        source=node_b_id, sink=node_c_id, edge=create_socket_connection(2)
    )
    connection3 = Connection(
        source=node_c_id, sink=node_a_id, edge=create_socket_connection(3)
    )
    connection4 = Connection(
        source=node_b_id, sink=node_a_id, edge=create_socket_connection(4)
    )

    topology = Topology()
    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)
    topology.add_connection(connection1)
    topology.add_connection(connection2)
    topology.add_connection(connection3)
    topology.add_connection(connection4)

    node_a_mem = create_node_memory(available_memory[0] * 1024)
    node_b_mem = create_node_memory(available_memory[1] * 1024)
    node_c_mem = create_node_memory(available_memory[2] * 1024)
    node_memory = {
        node_a_id: node_a_mem,
        node_b_id: node_b_mem,
        node_c_id: node_c_mem,
    }

    model_card = ModelCard(
        model_id=ModelId("test-model"),
        n_layers=total_layers,
        storage_size=Memory.from_kb(1000),
        hidden_size=1000,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )

    cycles = topology.get_cycles()

    # pick the 3-node cycle deterministically (cycle ordering can vary)
    selected_cycle = next(cycle for cycle in cycles if len(cycle) == 3)

    # act
    shard_assignments = get_shard_assignments(
        model_card, selected_cycle, Sharding.Pipeline, node_memory=node_memory
    )

    # assert
    runner_id_a = shard_assignments.node_to_runner[node_a_id]
    runner_id_b = shard_assignments.node_to_runner[node_b_id]
    runner_id_c = shard_assignments.node_to_runner[node_c_id]

    assert (
        shard_assignments.runner_to_shard[runner_id_a].end_layer
        - shard_assignments.runner_to_shard[runner_id_a].start_layer
        == expected_layers[0]
    )
    assert (
        shard_assignments.runner_to_shard[runner_id_b].end_layer
        - shard_assignments.runner_to_shard[runner_id_b].start_layer
        == expected_layers[1]
    )
    assert (
        shard_assignments.runner_to_shard[runner_id_c].end_layer
        - shard_assignments.runner_to_shard[runner_id_c].start_layer
        == expected_layers[2]
    )


def test_get_mlx_jaccl_coordinators():
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()

    # fully connected (directed) between the 3 nodes
    conn_a_b = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    conn_b_a = Connection(
        source=node_b_id, sink=node_a_id, edge=create_socket_connection(2)
    )
    conn_b_c = Connection(
        source=node_b_id, sink=node_c_id, edge=create_socket_connection(3)
    )
    conn_c_b = Connection(
        source=node_c_id, sink=node_b_id, edge=create_socket_connection(4)
    )
    conn_c_a = Connection(
        source=node_c_id, sink=node_a_id, edge=create_socket_connection(5)
    )
    conn_a_c = Connection(
        source=node_a_id, sink=node_c_id, edge=create_socket_connection(6)
    )

    network_a = NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.5"),
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.2"),
        ]
    )
    network_b = NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.1"),
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.4"),
        ]
    )
    network_c = NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.3"),
            NetworkInterfaceInfo(name="en0", ip_address="169.254.0.6"),
        ]
    )
    node_network = {
        node_a_id: network_a,
        node_b_id: network_b,
        node_c_id: network_c,
    }

    topology = Topology()
    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)

    topology.add_connection(conn_a_b)
    topology.add_connection(conn_b_a)
    topology.add_connection(conn_b_c)
    topology.add_connection(conn_c_b)
    topology.add_connection(conn_c_a)
    topology.add_connection(conn_a_c)

    # act
    coordinators = get_mlx_jaccl_coordinators(
        node_a_id,
        coordinator_port=5000,
        cycle_digraph=topology,
        node_network=node_network,
    )

    # assert
    assert len(coordinators) == 3
    assert node_a_id in coordinators
    assert node_b_id in coordinators
    assert node_c_id in coordinators

    # All coordinators should have IP:PORT format
    for node_id, coordinator in coordinators.items():
        assert ":" in coordinator, (
            f"Coordinator for {node_id} should have ':' separator"
        )

    # Verify port is correct
    for node_id, coordinator in coordinators.items():
        assert coordinator.endswith(":5000"), (
            f"Coordinator for {node_id} should use port 5000"
        )

    # Rank 0 (node_a) treats this as the listen socket so should listen on all IPs
    assert coordinators[node_a_id].startswith("0.0.0.0:"), (
        "Rank 0 node should use 0.0.0.0 as coordinator listen address"
    )

    # Non-rank-0 nodes should use the specific IP from their connection to rank 0
    # node_b uses the IP from conn_b_a (node_b -> node_a)
    assert isinstance(conn_b_a.edge, SocketConnection)
    assert (
        coordinators[node_b_id] == f"{conn_b_a.edge.sink_multiaddr.ip_address}:5000"
    ), "node_b should use the IP from conn_b_a"

    # node_c uses the IP from conn_c_a (node_c -> node_a)
    assert isinstance(conn_c_a.edge, SocketConnection)
    assert coordinators[node_c_id] == (
        f"{conn_c_a.edge.sink_multiaddr.ip_address}:5000"
    ), "node_c should use the IP from conn_c_a"


class TestAllocateLayersProportionally:
    def test_empty_node_list_raises(self):
        with pytest.raises(ValueError, match="empty node list"):
            allocate_layers_proportionally(total_layers=10, memory_fractions=[])

    def test_zero_layers_raises(self):
        with pytest.raises(ValueError, match="need at least 1 layer per node"):
            allocate_layers_proportionally(total_layers=0, memory_fractions=[0.5, 0.5])

    def test_negative_layers_raises(self):
        with pytest.raises(ValueError, match="need at least 1 layer per node"):
            allocate_layers_proportionally(total_layers=-1, memory_fractions=[0.5, 0.5])

    def test_fewer_layers_than_nodes_raises(self):
        with pytest.raises(ValueError, match="need at least 1 layer per node"):
            allocate_layers_proportionally(
                total_layers=2, memory_fractions=[0.33, 0.33, 0.34]
            )

    def test_equal_distribution(self):
        result = allocate_layers_proportionally(
            total_layers=12, memory_fractions=[0.25, 0.25, 0.25, 0.25]
        )
        assert result == [3, 3, 3, 3]
        assert sum(result) == 12

    def test_proportional_distribution(self):
        result = allocate_layers_proportionally(
            total_layers=12, memory_fractions=[0.25, 0.25, 0.50]
        )
        assert result == [3, 3, 6]
        assert sum(result) == 12

    def test_extreme_imbalance_ensures_minimum(self):
        result = allocate_layers_proportionally(
            total_layers=20, memory_fractions=[0.975, 0.0125, 0.0125]
        )
        assert all(layers >= 1 for layers in result)
        assert sum(result) == 20
        # Small nodes get minimum 1 layer
        assert result == [18, 1, 1]

    def test_single_node_gets_all_layers(self):
        result = allocate_layers_proportionally(total_layers=10, memory_fractions=[1.0])
        assert result == [10]

    def test_minimum_viable_allocation(self):
        result = allocate_layers_proportionally(
            total_layers=3, memory_fractions=[0.33, 0.33, 0.34]
        )
        assert result == [1, 1, 1]
        assert sum(result) == 3


def test_get_shard_assignments_insufficient_memory_raises():
    """Test that ValueError is raised when a node has insufficient memory for its layers."""
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()
    topology = Topology()

    # Node C has only 10 KB but would need 50 KB for 1 layer (1000 KB / 20 layers)
    node_a_mem = create_node_memory(900 * 1024)
    node_b_mem = create_node_memory(50 * 1024)
    node_c_mem = create_node_memory(10 * 1024)  # Insufficient memory

    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)

    conn_a_b = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    conn_b_c = Connection(
        source=node_b_id, sink=node_c_id, edge=create_socket_connection(2)
    )
    conn_c_a = Connection(
        source=node_c_id, sink=node_a_id, edge=create_socket_connection(3)
    )
    conn_b_a = Connection(
        source=node_b_id, sink=node_a_id, edge=create_socket_connection(3)
    )
    topology.add_connection(conn_a_b)
    topology.add_connection(conn_b_c)
    topology.add_connection(conn_c_a)
    topology.add_connection(conn_b_a)

    node_memory = {
        node_a_id: node_a_mem,
        node_b_id: node_b_mem,
        node_c_id: node_c_mem,
    }

    model_card = ModelCard(
        model_id=ModelId("test-model"),
        n_layers=20,
        storage_size=Memory.from_kb(1000),
        hidden_size=1000,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )
    cycles = topology.get_cycles()
    selected_cycle = cycles[0]

    with pytest.raises(ValueError, match="insufficient memory"):
        get_shard_assignments(
            model_card, selected_cycle, Sharding.Pipeline, node_memory
        )


def test_get_shard_assignments_insufficient_memory_force_override():
    """force_override=True should skip the per-node layer memory validation and
    still allocate layers for a model that exceeds available memory."""
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()
    topology = Topology()

    # Node C has only 10 KB but would need 50 KB for 1 layer (1000 KB / 20 layers)
    node_a_mem = create_node_memory(900 * 1024)
    node_b_mem = create_node_memory(50 * 1024)
    node_c_mem = create_node_memory(10 * 1024)  # Insufficient memory

    topology.add_node(node_a_id)
    topology.add_node(node_b_id)
    topology.add_node(node_c_id)

    conn_a_b = Connection(
        source=node_a_id, sink=node_b_id, edge=create_socket_connection(1)
    )
    conn_b_c = Connection(
        source=node_b_id, sink=node_c_id, edge=create_socket_connection(2)
    )
    conn_c_a = Connection(
        source=node_c_id, sink=node_a_id, edge=create_socket_connection(3)
    )
    conn_b_a = Connection(
        source=node_b_id, sink=node_a_id, edge=create_socket_connection(3)
    )
    topology.add_connection(conn_a_b)
    topology.add_connection(conn_b_c)
    topology.add_connection(conn_c_a)
    topology.add_connection(conn_b_a)

    node_memory = {
        node_a_id: node_a_mem,
        node_b_id: node_b_mem,
        node_c_id: node_c_mem,
    }

    model_card = ModelCard(
        model_id=ModelId("test-model"),
        n_layers=20,
        storage_size=Memory.from_kb(1000),
        hidden_size=1000,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )
    cycles = topology.get_cycles()
    selected_cycle = cycles[0]

    shard_assignments = get_shard_assignments(
        model_card,
        selected_cycle,
        Sharding.Pipeline,
        node_memory,
        force_override=True,
    )

    assert shard_assignments.model_id == "test-model"
    assert len(shard_assignments.runner_to_shard) == 3


class TestCfgParallelPlacement:
    def _create_ring_topology(self, node_ids: list[NodeId]) -> Topology:
        topology = Topology()
        for node_id in node_ids:
            topology.add_node(node_id)

        for i, node_id in enumerate(node_ids):
            next_node = node_ids[(i + 1) % len(node_ids)]
            conn = Connection(
                source=node_id,
                sink=next_node,
                edge=create_socket_connection(i + 1),
            )
            topology.add_connection(conn)

        return topology

    def test_two_nodes_cfg_model_uses_cfg_parallel(self):
        """Two nodes with CFG model should use CFG parallel (no pipeline)."""
        node_a = NodeId()
        node_b = NodeId()

        topology = self._create_ring_topology([node_a, node_b])
        cycles = [c for c in topology.get_cycles() if len(c) == 2]
        cycle = cycles[0]

        node_memory = {
            node_a: create_node_memory(1000 * 1024),
            node_b: create_node_memory(1000 * 1024),
        }

        model_card = ModelCard(
            model_id=ModelId("qwen-image-test"),
            n_layers=60,
            storage_size=Memory.from_kb(1000),
            hidden_size=1,
            supports_tensor=False,
            uses_cfg=True,
            tasks=[ModelTask.TextToImage],
            backends=[Backend.MlxMetal],
        )

        assignments = get_shard_assignments_for_pipeline_parallel(
            model_card, cycle, node_memory
        )

        shards = list(assignments.runner_to_shard.values())
        assert len(shards) == 2

        # CFG models should get CfgShardMetadata
        for shard in shards:
            assert isinstance(shard, CfgShardMetadata)
            # Both nodes should have all layers (no pipeline split)
            assert shard.start_layer == 0
            assert shard.end_layer == 60
            assert shard.cfg_world_size == 2
            # Each node is the only stage in its pipeline group
            assert shard.pipeline_world_size == 1
            assert shard.pipeline_rank == 0

        cfg_ranks = sorted(
            s.cfg_rank for s in shards if isinstance(s, CfgShardMetadata)
        )
        assert cfg_ranks == [0, 1]

    def test_four_nodes_cfg_model_uses_hybrid(self):
        """Four nodes with CFG model should use 2 CFG groups x 2 pipeline stages."""
        nodes = [NodeId() for _ in range(4)]

        topology = self._create_ring_topology(nodes)
        cycles = [c for c in topology.get_cycles() if len(c) == 4]
        cycle = cycles[0]

        node_memory = {n: create_node_memory(1000 * 1024) for n in nodes}

        model_card = ModelCard(
            model_id=ModelId("qwen-image-test"),
            n_layers=60,
            storage_size=Memory.from_kb(1000),
            hidden_size=1,
            supports_tensor=False,
            uses_cfg=True,
            tasks=[ModelTask.TextToImage],
            backends=[Backend.MlxMetal],
        )

        assignments = get_shard_assignments_for_pipeline_parallel(
            model_card, cycle, node_memory
        )

        shards = list(assignments.runner_to_shard.values())
        assert len(shards) == 4

        # CFG models should get CfgShardMetadata
        for shard in shards:
            assert isinstance(shard, CfgShardMetadata)
            assert shard.cfg_world_size == 2
            assert shard.pipeline_world_size == 2
            assert shard.pipeline_rank in [0, 1]

        # Check we have 2 nodes in each CFG group
        cfg_0_shards = [
            s for s in shards if isinstance(s, CfgShardMetadata) and s.cfg_rank == 0
        ]
        cfg_1_shards = [
            s for s in shards if isinstance(s, CfgShardMetadata) and s.cfg_rank == 1
        ]
        assert len(cfg_0_shards) == 2
        assert len(cfg_1_shards) == 2

        # Both CFG groups should have the same layer assignments
        cfg_0_layers = [(s.start_layer, s.end_layer) for s in cfg_0_shards]
        cfg_1_layers = [(s.start_layer, s.end_layer) for s in cfg_1_shards]
        assert sorted(cfg_0_layers) == sorted(cfg_1_layers)

    def test_three_nodes_cfg_model_uses_sequential_cfg(self):
        """Three nodes (odd) with CFG model should use sequential CFG (PipelineShardMetadata)."""
        nodes = [NodeId() for _ in range(3)]

        topology = self._create_ring_topology(nodes)
        cycles = [c for c in topology.get_cycles() if len(c) == 3]
        cycle = cycles[0]

        node_memory = {n: create_node_memory(1000 * 1024) for n in nodes}

        model_card = ModelCard(
            model_id=ModelId("qwen-image-test"),
            n_layers=60,
            storage_size=Memory.from_kb(1000),
            hidden_size=1,
            supports_tensor=False,
            uses_cfg=True,
            tasks=[ModelTask.TextToImage],
            backends=[Backend.MlxMetal],
        )

        assignments = get_shard_assignments_for_pipeline_parallel(
            model_card, cycle, node_memory
        )

        shards = list(assignments.runner_to_shard.values())
        assert len(shards) == 3

        # Odd node count with CFG model falls back to PipelineShardMetadata (sequential CFG)
        for shard in shards:
            assert isinstance(shard, PipelineShardMetadata)

    def test_two_nodes_non_cfg_model_uses_pipeline(self):
        """Two nodes with non-CFG model should use pure pipeline (PipelineShardMetadata)."""
        node_a = NodeId()
        node_b = NodeId()

        topology = self._create_ring_topology([node_a, node_b])
        cycles = [c for c in topology.get_cycles() if len(c) == 2]
        cycle = cycles[0]

        node_memory = {
            node_a: create_node_memory(1000 * 1024),
            node_b: create_node_memory(1000 * 1024),
        }

        model_card = ModelCard(
            model_id=ModelId("flux-test"),
            n_layers=57,
            storage_size=Memory.from_kb(1000),
            hidden_size=1,
            supports_tensor=False,
            uses_cfg=False,  # Non-CFG model
            tasks=[ModelTask.TextToImage],
            backends=[Backend.MlxMetal],
        )

        assignments = get_shard_assignments_for_pipeline_parallel(
            model_card, cycle, node_memory
        )

        shards = list(assignments.runner_to_shard.values())
        assert len(shards) == 2

        # Non-CFG models should get PipelineShardMetadata
        for shard in shards:
            assert isinstance(shard, PipelineShardMetadata)

        # Should have actual layer sharding (pipeline)
        layer_ranges = sorted(
            (s.start_layer, s.end_layer)
            for s in shards
            if isinstance(s, PipelineShardMetadata)
        )
        # First shard starts at 0, last shard ends at 57
        assert layer_ranges[0][0] == 0
        assert layer_ranges[-1][1] == 57


def test_find_ip_prioritised_prefers_measured_latency_for_ring() -> None:
    """Ring host selection should prefer the lowest measured probe latency,
    falling back to interface type and RFC1918 preference."""
    node_a = NodeId()
    node_b = NodeId()
    topology = Topology()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(
            source=node_a,
            sink=node_b,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/10.0.0.1/tcp/52415"),
                latency_ms=25.0,
            ),
        )
    )
    topology.add_connection(
        Connection(
            source=node_a,
            sink=node_b,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/192.168.1.1/tcp/52415"),
                latency_ms=2.0,
            ),
        )
    )
    node_network = {
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="en0", ip_address="10.0.0.1", interface_type="ethernet"
                ),
                NetworkInterfaceInfo(
                    name="en1", ip_address="192.168.1.1", interface_type="ethernet"
                ),
            ]
        )
    }

    from exo.master.placement_utils import find_ip_prioritised

    result = find_ip_prioritised(node_a, node_b, topology, node_network, ring=True)
    # The 192.168.1.1 address has lower measured latency -> preferred.
    assert result == "192.168.1.1"


def test_find_ip_prioritised_unmeasured_falls_back_to_type_and_rfc1918() -> None:
    """Without latency data, ring selection falls back to type priority then
    RFC1918 LAN over Tailscale CGNAT."""
    node_a = NodeId()
    node_b = NodeId()
    topology = Topology()
    topology.add_node(node_a)
    topology.add_node(node_b)
    # Same interface type (ethernet) for both; 192.168.x should win over 100.64.x
    topology.add_connection(
        Connection(
            source=node_a,
            sink=node_b,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/100.64.0.1/tcp/52415")
            ),
        )
    )
    topology.add_connection(
        Connection(
            source=node_a,
            sink=node_b,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/192.168.1.1/tcp/52415")
            ),
        )
    )
    node_network = {
        node_b: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="utun0", ip_address="100.64.0.1", interface_type="ethernet"
                ),
                NetworkInterfaceInfo(
                    name="en1", ip_address="192.168.1.1", interface_type="ethernet"
                ),
            ]
        )
    }

    from exo.master.placement_utils import find_ip_prioritised

    result = find_ip_prioritised(node_a, node_b, topology, node_network, ring=True)
    assert result == "192.168.1.1"


def test_assign_shard_backends_picks_first_preferred_backend_per_node():
    from exo.master.placement_utils import assign_shard_backends
    from exo.shared.types.worker.runners import RunnerId, ShardAssignments

    model_card = ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_mb(100),
        n_layers=4,
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal, Backend.MlxCuda, Backend.MlxCpu],
    )

    node_metal = NodeId()
    node_cuda = NodeId()
    node_unassigned = NodeId()
    node_to_runner = {
        node_metal: RunnerId(),
        node_cuda: RunnerId(),
        node_unassigned: RunnerId(),
    }
    runner_to_shard = {
        runner_id: PipelineShardMetadata(
            model_card=model_card,
            device_rank=device_rank,
            world_size=3,
            start_layer=0,
            end_layer=4,
            n_layers=4,
        )
        for device_rank, runner_id in enumerate(node_to_runner.values())
    }
    shard_assignments = ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )
    node_backends = {
        node_metal: [Backend.MlxCpu, Backend.MlxMetal],
        node_cuda: [Backend.MlxCuda, Backend.MlxCpu],
        node_unassigned: [],
    }

    result = assign_shard_backends(
        shard_assignments,
        node_backends,
        [Backend.MlxMetal, Backend.MlxCuda, Backend.MlxCpu],
    )

    def backend_of(node_id: NodeId) -> Backend | None:
        return result.runner_to_shard[result.node_to_runner[node_id]].backend

    assert backend_of(node_metal) == Backend.MlxMetal
    assert backend_of(node_cuda) == Backend.MlxCuda
    assert backend_of(node_unassigned) is None


def test_filter_cycles_by_replicated_memory_requires_each_node_to_fit():
    node1_id = NodeId()
    node2_id = NodeId()
    topology = Topology()
    topology.add_connection(
        Connection(source=node1_id, sink=node2_id, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node2_id, sink=node1_id, edge=create_socket_connection(2))
    )
    cycles = [cycle for cycle in topology.get_cycles() if len(cycle) == 2]
    node_memory = {
        node1_id: create_node_memory(8 * 1024),
        node2_id: create_node_memory(8 * 1024),
    }

    assert filter_cycles_by_memory(cycles, node_memory, Memory.from_kb(12)) == cycles
    assert (
        filter_cycles_by_replicated_memory(cycles, node_memory, Memory.from_kb(12))
        == []
    )


class TestRingMemoryAdmission:
    @staticmethod
    def _ring_card(
        *,
        context_length: int = 131072,
        num_key_value_heads: int | None = 8,
    ) -> ModelCard:
        return ModelCard(
            model_id=ModelId("ring-test"),
            n_layers=16,
            storage_size=Memory.from_mb(700),
            hidden_size=2048,
            supports_tensor=True,
            supports_ring=True,
            num_key_value_heads=num_key_value_heads,
            context_length=context_length,
            tasks=[ModelTask.TextGeneration],
            backends=[Backend.MlxMetal, Backend.MlxCuda],
        )

    def test_estimate_exceeds_weights_alone(self) -> None:
        card = self._ring_card()
        estimate = estimate_ring_node_memory(card)
        assert estimate > card.storage_size
        # 16 layers x 8 kv heads x 128 head dim x 16384 tokens x 2 (K+V)
        # x 2 bytes x 4 working-set multiplier = 4 GiB of working set.
        assert estimate - card.storage_size == Memory.from_bytes(
            2 * 2 * 16 * 8 * 128 * 16384 * 4
        )

    def test_admission_context_is_capped(self) -> None:
        huge_context = self._ring_card(context_length=1_000_000)
        normal_context = self._ring_card(context_length=8192)
        assert estimate_ring_node_memory(huge_context) == estimate_ring_node_memory(
            self._ring_card()
        )
        assert estimate_ring_node_memory(normal_context) < estimate_ring_node_memory(
            self._ring_card()
        )

    def test_missing_kv_heads_uses_hidden_size(self) -> None:
        card = self._ring_card(num_key_value_heads=None)
        estimate = estimate_ring_node_memory(card)
        assert estimate > card.storage_size


class TestShouldPreferSingleNode:
    @staticmethod
    def _model_card(storage_mb: int = 700) -> ModelCard:
        return ModelCard(
            model_id=ModelId("single-node-test"),
            n_layers=16,
            storage_size=Memory.from_mb(storage_mb),
            hidden_size=2048,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
            backends=[Backend.MlxMetal],
        )

    def test_returns_single_node_cycle_when_model_fits(self) -> None:
        node_id = NodeId()
        cycle = Cycle(node_ids=[node_id])
        node_memory = {node_id: create_node_memory(1024 * 1024 * 1024)}

        result = should_prefer_single_node(self._model_card(), [cycle], node_memory)

        assert result == cycle

    def test_returns_none_when_model_does_not_fit_any_single_node(self) -> None:
        node_id = NodeId()
        cycle = Cycle(node_ids=[node_id])
        # Far less than the 700 MB model.
        node_memory = {node_id: create_node_memory(100 * 1024)}

        result = should_prefer_single_node(self._model_card(), [cycle], node_memory)

        assert result is None

    def test_prefers_fastest_single_node(self) -> None:
        slow_id = NodeId()
        fast_id = NodeId()
        slow_cycle = Cycle(node_ids=[slow_id])
        fast_cycle = Cycle(node_ids=[fast_id])
        node_memory = {
            slow_id: create_node_memory(1024 * 1024 * 1024),
            fast_id: create_node_memory(1024 * 1024 * 1024),
        }
        identities = {
            slow_id: NodeIdentity(chip_id="Apple M1"),  # 68 GB/s
            fast_id: NodeIdentity(chip_id="Apple M3 Ultra"),  # 819 GB/s
        }

        result = should_prefer_single_node(
            self._model_card(), [slow_cycle, fast_cycle], node_memory, identities
        )

        assert result == fast_cycle

    def test_ignores_multi_node_cycles(self) -> None:
        node1_id = NodeId()
        node2_id = NodeId()
        multi_cycle = Cycle(node_ids=[node1_id, node2_id])
        node_memory = {
            node1_id: create_node_memory(1024 * 1024 * 1024),
            node2_id: create_node_memory(1024 * 1024 * 1024),
        }

        result = should_prefer_single_node(
            self._model_card(), [multi_cycle], node_memory
        )

        assert result is None

    def test_unknown_chip_is_still_viable(self) -> None:
        node_id = NodeId()
        cycle = Cycle(node_ids=[node_id])
        node_memory = {node_id: create_node_memory(1024 * 1024 * 1024)}

        result = should_prefer_single_node(
            self._model_card(),
            [cycle],
            node_memory,
            {node_id: NodeIdentity(chip_id="Unknown Chip X1")},
        )

        assert result == cycle


class TestEstimatePipelineActivationMemory:
    """Tests for estimate_pipeline_activation_memory (Phase 1.4)."""

    def test_zero_layers_returns_zero(self):
        result = estimate_pipeline_activation_memory(
            total_storage=Memory.from_kb(1000),
            total_layers=20,
            layers_on_node=0,
        )
        assert result.in_bytes == 0

    def test_single_layer_is_ten_percent_of_per_layer_weight(self):
        storage = Memory.from_kb(1000)
        total_layers = 20
        per_layer = storage.in_bytes // total_layers  # 50 KB
        expected = int(per_layer * 1 * 0.1)  # 5 KB

        result = estimate_pipeline_activation_memory(
            total_storage=storage,
            total_layers=total_layers,
            layers_on_node=1,
        )
        assert result.in_bytes == expected

    def test_all_layers_is_ten_percent_of_total(self):
        storage = Memory.from_kb(1000)
        total_layers = 20

        result = estimate_pipeline_activation_memory(
            total_storage=storage,
            total_layers=total_layers,
            layers_on_node=total_layers,
        )
        expected = int(storage.in_bytes * 0.1)
        assert result.in_bytes == expected

    def test_scales_linearly_with_layers(self):
        storage = Memory.from_kb(2000)
        total_layers = 40

        r5 = estimate_pipeline_activation_memory(storage, total_layers, 5)
        r10 = estimate_pipeline_activation_memory(storage, total_layers, 10)
        assert r10.in_bytes == 2 * r5.in_bytes


class TestPipelineActivationInValidation:
    """Verify activation memory is included in pipeline placement checks."""

    def _make_model_and_nodes(self, storage_kb: int, n_layers: int, node_rams_kb: list[int]):
        from exo.master.tests.conftest import create_socket_connection
        from exo.shared.types.topology import Connection, Cycle

        node_ids = [NodeId() for _ in node_rams_kb]
        topology = Topology()
        for nid in node_ids:
            topology.add_node(nid)
        for i in range(len(node_ids)):
            j = (i + 1) % len(node_ids)
            topology.add_connection(
                Connection(source=node_ids[i], sink=node_ids[j], edge=create_socket_connection(i + 1))
            )
            topology.add_connection(
                Connection(source=node_ids[j], sink=node_ids[i], edge=create_socket_connection(i + 10))
            )

        node_memory = {
            nid: create_node_memory(ram_kb * 1024)
            for nid, ram_kb in zip(node_ids, node_rams_kb)
        }
        model_card = ModelCard(
            model_id=ModelId("act-test"),
            n_layers=n_layers,
            storage_size=Memory.from_kb(storage_kb),
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
            backends=[Backend.MlxMetal],
        )
        cycles = [c for c in topology.get_cycles() if len(c) == len(node_ids)]
        return model_card, cycles[0], node_memory

    def test_node_tight_on_activation_raises(self):
        """A node with exactly weight-fit but no activation headroom must fail."""
        # 20 layers, 1000KB storage -> 50KB/layer, +10% = 55KB needed
        # Give node exactly 50KB (fits weight only, not activation)
        model_card, cycle, node_memory = self._make_model_and_nodes(
            storage_kb=1000, n_layers=20, node_rams_kb=[950, 50]
        )
        with pytest.raises(ValueError, match="insufficient memory"):
            get_shard_assignments(
                model_card, cycle, Sharding.Pipeline, node_memory
            )

    def test_node_with_activation_headroom_passes(self):
        """A node with weight + activation headroom must pass."""
        # 1050KB cap=19, 55KB cap=1 -> sum=20 layers fits
        model_card, cycle, node_memory = self._make_model_and_nodes(
            storage_kb=1000, n_layers=20, node_rams_kb=[1050, 55]
        )
        assignments = get_shard_assignments(
            model_card, cycle, Sharding.Pipeline, node_memory
        )
        assert assignments.model_id == "act-test"
        assert len(assignments.runner_to_shard) == 2

    def test_force_override_bypasses_activation_check(self):
        """force_override=True should skip activation validation."""
        model_card, cycle, node_memory = self._make_model_and_nodes(
            storage_kb=1000, n_layers=20, node_rams_kb=[950, 50]
        )
        assignments = get_shard_assignments(
            model_card, cycle, Sharding.Pipeline, node_memory, force_override=True
        )
        assert len(assignments.runner_to_shard) == 2
