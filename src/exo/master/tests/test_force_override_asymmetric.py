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

# --- asymmetric topology resolution (upstream #2077 / #2310 fix) ---

# Asymmetric topology fixture:
#   A → B: SocketConnection only (dialer creates socket edges)
#   B → A: RDMAConnection only (listener never creates socket edges)
# This is the exact scenario from upstream issue #2077 (2-node Mac cluster).
#
# Since find_ip_prioritised() now includes node_network interfaces as
# candidates (fix for #2310), asymmetric topologies resolve correctly
# without requiring force_override. The "raises" tests are replaced with
# tests that verify correct fallback to node_network IPs.


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


def test_get_mlx_ring_hosts_by_node_succeeds_on_asymmetric_topology():
    """find_ip_prioritised now includes node_network IPs as candidates,
    so asymmetric topologies work without force_override (issue #2310)."""
    topology, node_a_id, node_b_id, node_network = _create_asymmetric_topology()
    cycle = Cycle(node_ids=[node_a_id, node_b_id])

    hosts_by_node = get_mlx_ring_hosts_by_node(
        selected_cycle=cycle,
        cycle_digraph=topology,
        ephemeral_port=50000,
        node_network=node_network,
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

    # Node B has no directed socket edges to A, but node_network provides
    # A's interface IP as a candidate — both ranks now resolve consistently.
    hosts_b = hosts_by_node[node_b_id]
    peer_ips_b = {h.ip for h in hosts_b if h.ip != "0.0.0.0"}
    assert len(peer_ips_b) == 1
    # The IP should come from A's node_network interfaces
    assert "169.254.0.10" in peer_ips_b


def test_get_mlx_ring_hosts_by_node_force_override_succeeds_on_asymmetric_topology():
    """force_override=True should also succeed on asymmetric topology."""
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

    for node_id, hosts in hosts_by_node.items():
        assert len(hosts) == 2

    hosts_a = hosts_by_node[node_a_id]
    peer_ips_a = {h.ip for h in hosts_a if h.ip != "0.0.0.0"}
    assert len(peer_ips_a) == 1
    assert any(ip.startswith("169.254.0.") for ip in peer_ips_a)

    hosts_b = hosts_by_node[node_b_id]
    peer_ips_b = {h.ip for h in hosts_b if h.ip != "0.0.0.0"}
    assert len(peer_ips_b) == 1
    assert "169.254.0.10" in peer_ips_b


def test_get_mlx_jaccl_coordinators_succeeds_on_asymmetric_topology():
    """find_ip_prioritised now includes node_network IPs as candidates,
    so asymmetric topologies work without force_override (issue #2310)."""
    topology, node_a_id, node_b_id, node_network = _create_asymmetric_topology()

    coordinators = get_mlx_jaccl_coordinators(
        coordinator=node_a_id,
        coordinator_port=5000,
        cycle_digraph=topology,
        node_network=node_network,
    )

    assert len(coordinators) == 2

    # Coordinator (node_a) listens on all interfaces
    assert coordinators[node_a_id] == "0.0.0.0:5000"

    # Non-coordinator (node_b) should get an IP to reach coordinator A.
    # node_network provides A's interface IP as a candidate.
    assert coordinators[node_b_id] == "169.254.0.10:5000"


def test_get_mlx_jaccl_coordinators_force_override_succeeds_on_asymmetric_topology():
    """force_override=True should also succeed on asymmetric topology."""
    topology, node_a_id, node_b_id, node_network = _create_asymmetric_topology()

    coordinators = get_mlx_jaccl_coordinators(
        coordinator=node_a_id,
        coordinator_port=5000,
        cycle_digraph=topology,
        node_network=node_network,
        force_override=True,
    )

    assert len(coordinators) == 2

    assert coordinators[node_a_id] == "0.0.0.0:5000"
    assert coordinators[node_b_id] == "169.254.0.10:5000"


def test_find_ip_prioritised_includes_node_network_as_candidates():
    """node_network interfaces are included as candidates alongside directed
    edge IPs, ensuring consistent resolution across asymmetric topologies."""
    from exo.master.placement_utils import find_ip_prioritised

    node_a_id = NodeId()
    node_b_id = NodeId()

    topology = Topology()
    topology.add_node(node_a_id)
    topology.add_node(node_b_id)

    # A→B: SocketConnection with IP 169.254.0.50
    topology.add_connection(
        Connection(
            source=node_a_id,
            sink=node_b_id,
            edge=create_socket_connection(50),
        )
    )

    # node_network reports SAME interface type IPs for B
    node_network = {
        node_a_id: NodeNetworkInfo(
            interfaces=[NetworkInterfaceInfo(name="en0", ip_address="169.254.0.1")]
        ),
        node_b_id: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(name="en0", ip_address="169.254.0.99"),
            ]
        ),
    }

    # Both edge IP (169.254.0.50) and node_network IP (169.254.0.99) are candidates
    result = find_ip_prioritised(node_a_id, node_b_id, topology, node_network, ring=True)
    assert result in ("169.254.0.50", "169.254.0.99")


def test_find_ip_prioritised_no_edges_uses_node_network():
    """When no directed edges exist, node_network provides the IP."""
    from exo.master.placement_utils import find_ip_prioritised

    node_a_id = NodeId()
    node_b_id = NodeId()

    topology = Topology()
    topology.add_node(node_a_id)
    topology.add_node(node_b_id)

    # No edges between A and B

    node_network = {
        node_a_id: NodeNetworkInfo(
            interfaces=[NetworkInterfaceInfo(name="en0", ip_address="169.254.0.1")]
        ),
        node_b_id: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(name="en0", ip_address="169.254.0.20"),
            ]
        ),
    }

    # Should resolve via node_network fallback
    result = find_ip_prioritised(node_a_id, node_b_id, topology, node_network, ring=True)
    assert result == "169.254.0.20"
