
import pytest

from exo.master.placement_utils import (
    get_mlx_jaccl_coordinators,
    get_mlx_ring_hosts_by_node,
)
from exo.master.tests.conftest import (
    create_rdma_connection,
    create_socket_connection,
)
from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import NetworkInterfaceInfo, NodeNetworkInfo
from exo.shared.types.topology import Connection, Cycle

# --- force_override on asymmetric topology (upstream #2077 fix) ---

# Asymmetric topology fixture:
#   A → B: SocketConnection only (dialer creates socket edges)
#   B → A: RDMAConnection only (listener never creates socket edges)
# This is the exact scenario from upstream issue #2077 (2-node Mac cluster).
# Without force_override, find_ip_prioritised() returns None for the B→A
# direction (no SocketConnection edges), causing both MlxRing and MlxJaccl
# placement to raise ValueError.


def _create_asymmetric_topology():
    """Create a 2-node asymmetric topology (upstream #2077 scenario).

    Returns (topology, node_a_id, node_b_id, node_network).
    A→B: SocketConnection only. B→A: RDMAConnection only.
    """
    node_a_id = NodeId()
    node_b_id = NodeId()

    topology = Topology()
    topology.add_node(node_a_id)
    topology.add_node(node_b_id)

    # A → B: SocketConnection only (dialer side)
    topology.add_connection(
        Connection(
            source=node_a_id,
            sink=node_b_id,
            edge=create_socket_connection(1),
        )
    )

    # B → A: RDMAConnection only (listener side — never creates socket edges)
    topology.add_connection(
        Connection(
            source=node_b_id,
            sink=node_a_id,
            edge=create_rdma_connection(1),
        )
    )

    node_network = {
        node_a_id: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(name="en0", ip_address="169.254.0.10"),
            ]
        ),
        node_b_id: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(name="en0", ip_address="169.254.0.20"),
            ]
        ),
    }

    return topology, node_a_id, node_b_id, node_network


def test_get_mlx_ring_hosts_by_node_raises_on_asymmetric_topology():
    """Without force_override, MlxRing raises on asymmetric topology.

    B→A has no SocketConnection edges, so find_ip_prioritised() returns None.
    """
    topology, node_a_id, node_b_id, node_network = _create_asymmetric_topology()
    cycle = Cycle(node_ids=[node_a_id, node_b_id])

    with pytest.raises(
        ValueError,
        match="MLX ring backend requires connectivity between neighbouring nodes",
    ):
        get_mlx_ring_hosts_by_node(
            selected_cycle=cycle,
            cycle_digraph=topology,
            ephemeral_port=50000,
            node_network=node_network,
        )


def test_get_mlx_ring_hosts_by_node_force_override_succeeds_on_asymmetric_topology():
    """force_override=True should fall back to a node_network interface IP
    when find_ip_prioritised() returns None, instead of raising."""
    topology, node_a_id, node_b_id, node_network = _create_asymmetric_topology()
    cycle = Cycle(node_ids=[node_a_id, node_b_id])

    hosts_by_node = get_mlx_ring_hosts_by_node(
        selected_cycle=cycle,
        cycle_digraph=topology,
        ephemeral_port=50000,
        node_network=node_network,
        force_override=True,
    )

    assert len(hosts_by_node) == 2

    # For each node, verify the host list structure: 2 entries (one per peer)
    for node_id, hosts in hosts_by_node.items():
        assert len(hosts) == 2

    # Node A should find the socket IP for B (normal path)
    hosts_a = hosts_by_node[node_a_id]
    peer_ips_a = {h.ip for h in hosts_a if h.ip != "0.0.0.0"}
    assert len(peer_ips_a) == 1
    # The IP should be from the socket connection (169.254.0.x)
    assert any(ip.startswith("169.254.0.") for ip in peer_ips_a)

    # Node B should fall back to node_network interface IP for A (asymmetric path)
    hosts_b = hosts_by_node[node_b_id]
    peer_ips_b = {h.ip for h in hosts_b if h.ip != "0.0.0.0"}
    assert len(peer_ips_b) == 1
    # The IP should come from A's node_network interfaces
    assert "169.254.0.10" in peer_ips_b


def test_get_mlx_jaccl_coordinators_raises_on_asymmetric_topology():
    """Without force_override, MlxJaccl raises on asymmetric topology.

    B→A has no SocketConnection edges, so find_ip_prioritised() returns None.
    """
    topology, node_a_id, node_b_id, node_network = _create_asymmetric_topology()

    with pytest.raises(
        ValueError,
        match="jaccl backend requires all participating devices",
    ):
        get_mlx_jaccl_coordinators(
            coordinator=node_a_id,
            coordinator_port=5000,
            cycle_digraph=topology,
            node_network=node_network,
        )


def test_get_mlx_jaccl_coordinators_force_override_succeeds_on_asymmetric_topology():
    """force_override=True should fall back to node_network interface IPs
    when find_ip_prioritised() returns None, instead of raising."""
    topology, node_a_id, node_b_id, node_network = _create_asymmetric_topology()

    coordinators = get_mlx_jaccl_coordinators(
        coordinator=node_a_id,
        coordinator_port=5000,
        cycle_digraph=topology,
        node_network=node_network,
        force_override=True,
    )

    assert len(coordinators) == 2

    # Coordinator (node_a) listens on all interfaces
    assert coordinators[node_a_id] == "0.0.0.0:5000"

    # Non-coordinator (node_b) should get an IP to reach coordinator A.
    # Fallback uses A's own node_network interfaces (the target's IP),
    # not B's own IP — B needs to know how to reach A.
    assert coordinators[node_b_id] == "169.254.0.10:5000"
