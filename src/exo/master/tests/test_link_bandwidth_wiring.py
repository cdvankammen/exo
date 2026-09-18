# pyright: reportAny=false, reportUnknownVariableType=false
"""Tests that live topology link bandwidth reaches the pipeline layer allocator.

GitHub #957 (P1 #35): the water-filling allocator exists and is unit-tested,
but until this fix `link_bandwidths` was never passed from ``place_instance``
through ``get_shard_assignments`` — so production pipeline placement fell back
to compute-only throughput allocation and ignored slow inter-node links.

These tests prove the end-to-end wiring: per-hop ``SocketConnection.bandwidth_mbps``
flowing from the topology into ``allocate_layers_by_water_filling``.
"""

from collections.abc import Mapping

from exo.master.placement import place_instance
from exo.master.placement_utils import (
    get_link_bandwidths_for_cycle,
)
from exo.master.tests.conftest import (
    create_node_memory,
    create_node_network,
    create_socket_connection,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import NodeIdentity
from exo.shared.types.topology import Connection
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding


def _place_command(model_card: ModelCard) -> PlaceInstance:
    return PlaceInstance(
        command_id=CommandId(),
        model_card=model_card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
    )


def _metal_only(node_memory: Mapping[NodeId, object]) -> dict[NodeId, list[Backend]]:
    return {node_id: [Backend.MlxMetal] for node_id in node_memory}


def _model_for_two_nodes(total_layers: int = 16) -> ModelCard:
    """Model sized to require BOTH nodes (storage must not fit single node).

    Each node has 1000KB ram_available; storage_size 1500KB forces a 2-node
    cycle (P1 #37 single-node preference can't apply).
    """
    return ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_kb(1500),
        n_layers=total_layers,
        hidden_size=30,
        supports_tensor=True,
        supports_ring=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def test_get_link_bandwidths_for_cycle_uses_outgoing_hop_minimum() -> None:
    """Each node's effective link bandwidth is the min of its outgoing hops."""
    a = NodeId()
    b = NodeId()
    c = NodeId()
    topology = Topology()
    for node in (a, b, c):
        topology.add_node(node)
    # A→B: 100 MiB/s, B→C: 50 MiB/s, C→A: 200 MiB/s
    topology.add_connection(
        Connection(source=a, sink=b, edge=create_socket_connection(1, bandwidth_mbps=100.0))
    )
    topology.add_connection(
        Connection(source=b, sink=c, edge=create_socket_connection(2, bandwidth_mbps=50.0))
    )
    topology.add_connection(
        Connection(source=c, sink=a, edge=create_socket_connection(3, bandwidth_mbps=200.0))
    )
    # Extra slow C→B edge makes C's outgoing minimum 20 MiB/s
    topology.add_connection(
        Connection(source=c, sink=b, edge=create_socket_connection(4, bandwidth_mbps=20.0))
    )
    result = get_link_bandwidths_for_cycle(topology, [a, b, c])
    # MiB/s → GB/s conversion (÷1024): A out = 100/1024, B out = 50/1024,
    # C out = min(200,20)/1024 = 20/1024
    assert result[0] is not None and abs(result[0] - 100.0 / 1024.0) < 1e-9
    assert result[1] is not None and abs(result[1] - 50.0 / 1024.0) < 1e-9
    assert result[2] is not None and abs(result[2] - 20.0 / 1024.0) < 1e-9


def test_get_link_bandwidths_for_cycle_returns_none_when_unknown() -> None:
    """A hop with no measured bandwidth yields None, not a fabricated value."""
    a = NodeId()
    b = NodeId()
    topology = Topology()
    topology.add_node(a)
    topology.add_node(b)
    topology.add_connection(
        Connection(source=a, sink=b, edge=create_socket_connection(1))
    )  # bandwidth_mbps=None
    assert get_link_bandwidths_for_cycle(topology, [a, b]) == [None, None]


def test_place_instance_wires_topology_bandwidth_into_water_filling() -> None:
    """A GPU-fast node behind a slow link must NOT get the most layers.

    Regression for GitHub #957: before wiring, placement used compute-only
    throughput (node A = fastest GPU ⇒ most layers). With live link bandwidth
    wired through, the slow A→B hop caps A's effective speed and B gets more.
    """
    model_card = _model_for_two_nodes(total_layers=16)
    a = NodeId()
    b = NodeId()

    topology = Topology()
    topology.add_node(a)
    topology.add_node(b)
    # A→B link is SLOW (10 MiB/s); B→A is fast (1000 MiB/s)
    topology.add_connection(
        Connection(
            source=a,
            sink=b,
            edge=create_socket_connection(1, bandwidth_mbps=10.0),
        )
    )
    topology.add_connection(
        Connection(
            source=b,
            sink=a,
            edge=create_socket_connection(2, bandwidth_mbps=1000.0),
        )
    )

    node_memory = {
        a: create_node_memory(1000 * 1024),
        b: create_node_memory(1000 * 1024),
    }
    node_network = {a: create_node_network(), b: create_node_network()}
    node_identities = {
        a: NodeIdentity(chip_id="Apple M3 Max"),  # 400 GB/s GPU
        b: NodeIdentity(chip_id="Apple M3"),  # 100 GB/s GPU
    }
    cic = _place_command(model_card)

    placements = place_instance(
        cic,
        topology,
        {},
        node_memory,
        node_network,
        _metal_only(node_memory),
        node_identities=node_identities,
    )

    assert len(placements) == 1
    instance = next(iter(placements.values()))
    shard_a = instance.shard_assignments.runner_to_shard[
        instance.shard_assignments.node_to_runner[a]
    ]
    shard_b = instance.shard_assignments.runner_to_shard[
        instance.shard_assignments.node_to_runner[b]
    ]
    layers_a = shard_a.end_layer - shard_a.start_layer
    layers_b = shard_b.end_layer - shard_b.start_layer
    assert layers_a + layers_b == 16
    # With the slow A→B hop wired in, A's effective speed is min(400, 10/1024)
    # and B's is min(100, 1000/1024). B must not be starved; A must not hog.
    assert layers_b > layers_a