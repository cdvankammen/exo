import pytest

from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.topology import (
    Connection,
    Cycle,
    RDMAConnection,
    SocketConnection,
)


@pytest.fixture
def topology() -> Topology:
    return Topology()


@pytest.fixture
def socket_connection() -> SocketConnection:
    return SocketConnection(
        sink_multiaddr=Multiaddr(address="/ip4/127.0.0.1/tcp/1235"),
    )


# ---------------------------------------------------------------------------
# Existing tests (updated for corrected node_is_leaf semantics)
# node_is_leaf uses undirected degree: both endpoints of A->B are leaves (degree 1).
# ---------------------------------------------------------------------------


def test_add_node(topology: Topology):
    """An isolated node (0 neighbors) is NOT a leaf; it IS isolated."""
    node_id = NodeId()
    topology.add_node(node_id)
    assert not topology.node_is_leaf(node_id)
    assert topology.node_is_isolated(node_id)


def test_add_connection(topology: Topology, socket_connection: SocketConnection):
    node_a = NodeId()
    node_b = NodeId()
    connection = Connection(source=node_a, sink=node_b, edge=socket_connection)
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(connection)

    data = list(topology.list_connections())
    assert data == [connection]

    # A->B with undirected degree: both A and B have degree 1 -> both are leaves.
    assert topology.node_is_leaf(node_a)
    assert topology.node_is_leaf(node_b)


def test_remove_connection_still_connected(
    topology: Topology, socket_connection: SocketConnection
):
    node_a = NodeId()
    node_b = NodeId()
    conn = Connection(source=node_a, sink=node_b, edge=socket_connection)
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(conn)

    topology.remove_connection(conn)
    assert list(topology.get_all_connections_between(node_a, node_b)) == []


def test_remove_node_still_connected(
    topology: Topology, socket_connection: SocketConnection
):
    node_a = NodeId()
    node_b = NodeId()
    conn = Connection(source=node_a, sink=node_b, edge=socket_connection)
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(conn)
    assert list(topology.out_edges(node_a)) == [conn]

    topology.remove_node(node_b)
    assert list(topology.out_edges(node_a)) == []


def test_list_nodes(topology: Topology, socket_connection: SocketConnection):
    node_a = NodeId()
    node_b = NodeId()
    conn = Connection(source=node_a, sink=node_b, edge=socket_connection)
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(conn)
    assert list(topology.out_edges(node_a)) == [conn]

    nodes = list(topology.list_nodes())
    assert len(nodes) == 2
    assert all(isinstance(node, NodeId) for node in nodes)
    assert set(node for node in nodes) == set([node_a, node_b])


# ---------------------------------------------------------------------------
# Phase 1 - node_is_leaf / node_is_isolated edge cases
# ---------------------------------------------------------------------------


class TestNodeIsLeaf:
    """node_is_leaf uses undirected degree: degree == 1."""

    def test_single_isolated_node_is_not_leaf(self, topology: Topology):
        n = NodeId()
        topology.add_node(n)
        assert not topology.node_is_leaf(n)
        assert topology.node_is_isolated(n)

    def test_two_nodes_directed_edge_both_are_leaves(self, topology: Topology):
        """A->B: both endpoints have undirected degree 1 -> both leaves."""
        a, b = NodeId(), NodeId()
        topology.add_node(a)
        topology.add_node(b)
        topology.add_connection(
            Connection(
                source=a, sink=b, edge=SocketConnection(sink_multiaddr=Multiaddr(address="/ip4/10.0.0.2/tcp/52415"))
            )
        )
        assert topology.node_is_leaf(a)
        assert topology.node_is_leaf(b)

    def test_node_with_two_undirected_neighbors_is_not_leaf(self, topology: Topology):
        """Triangle A->B, A->C: A has degree 2 (not leaf), B/C have degree 1 (leaf)."""
        a, b, c = NodeId(), NodeId(), NodeId()
        topology.add_node(a)
        topology.add_node(b)
        topology.add_node(c)
        sc = lambda ip: SocketConnection(sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/52415"))
        topology.add_connection(Connection(source=a, sink=b, edge=sc("10.0.0.2")))
        topology.add_connection(Connection(source=a, sink=c, edge=sc("10.0.0.3")))
        assert not topology.node_is_leaf(a)  # 2 undirected neighbors
        assert topology.node_is_leaf(b)  # 1
        assert topology.node_is_leaf(c)  # 1

    def test_node_not_in_topology_returns_false(self, topology: Topology):
        assert not topology.node_is_leaf(NodeId())

    def test_sink_node_is_not_isolated(self, topology: Topology):
        """A->B: B (pure sink, 0 out-edges) is still connected (in-degree 1), so not isolated."""
        a, b = NodeId(), NodeId()
        topology.add_node(a)
        topology.add_node(b)
        topology.add_connection(
            Connection(
                source=a, sink=b, edge=SocketConnection(sink_multiaddr=Multiaddr(address="/ip4/10.0.0.2/tcp/52415"))
            )
        )
        assert not topology.node_is_isolated(a)
        assert not topology.node_is_isolated(b)

    def test_50_nodes_chain(self, topology: Topology):
        """Chain 0->1->2->...->49 undirected: endpoints degree 1 (leaf), middle degree 2 (not leaf)."""
        nodes = [NodeId() for _ in range(50)]
        for n in nodes:
            topology.add_node(n)
        for i in range(49):
            topology.add_connection(
                Connection(
                    source=nodes[i],
                    sink=nodes[i + 1],
                    edge=SocketConnection(
                        sink_multiaddr=Multiaddr(address=f"/ip4/10.0.0.{i + 2}/tcp/52415")
                    ),
                )
            )
        assert topology.node_is_leaf(nodes[0])
        assert topology.node_is_leaf(nodes[49])
        for i in range(1, 49):
            assert not topology.node_is_leaf(nodes[i])

    def test_50_nodes_star(self, topology: Topology):
        """Star center->(0..49): center has 50 undirected neighbors, leaves have 1."""
        center = NodeId()
        topology.add_node(center)
        leaves = []
        for i in range(50):
            n = NodeId()
            leaves.append(n)
            topology.add_node(n)
            topology.add_connection(
                Connection(
                    source=center,
                    sink=n,
                    edge=SocketConnection(
                        sink_multiaddr=Multiaddr(address=f"/ip4/10.0.1.{i + 1}/tcp/52415")
                    ),
                )
            )
        assert not topology.node_is_leaf(center)  # 50 neighbors
        for leaf in leaves:
            assert topology.node_is_leaf(leaf)  # 1 neighbor

    def test_three_node_chain_middle_not_leaf(self, topology: Topology):
        a, b, c = NodeId(), NodeId(), NodeId()
        topology.add_node(a)
        topology.add_node(b)
        topology.add_node(c)
        sc = lambda ip: SocketConnection(sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/52415"))
        topology.add_connection(Connection(source=a, sink=b, edge=sc("10.0.0.2")))
        topology.add_connection(Connection(source=b, sink=c, edge=sc("10.0.0.3")))
        # Undirected degrees: a=1, b=2, c=1
        assert topology.node_is_leaf(a)
        assert not topology.node_is_leaf(b)
        assert topology.node_is_leaf(c)


class TestNodeIsIsolated:
    def test_isolated_node(self, topology: Topology):
        n = NodeId()
        topology.add_node(n)
        assert topology.node_is_isolated(n)

    def test_connected_sink_not_isolated(self, topology: Topology):
        """Directed A->B: B (0 out-edges, 1 in-edge) is NOT isolated."""
        a, b = NodeId(), NodeId()
        topology.add_node(a)
        topology.add_node(b)
        topology.add_connection(
            Connection(
                source=a, sink=b, edge=SocketConnection(sink_multiaddr=Multiaddr(address="/ip4/10.0.0.2/tcp/52415"))
            )
        )
        assert not topology.node_is_isolated(a)
        assert not topology.node_is_isolated(b)

    def test_node_not_in_topology_returns_false(self, topology: Topology):
        assert not topology.node_is_isolated(NodeId())

    def test_chain_middle_not_isolated(self, topology: Topology):
        a, b, c = NodeId(), NodeId(), NodeId()
        topology.add_node(a)
        topology.add_node(b)
        topology.add_node(c)
        sc = lambda ip: SocketConnection(sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/52415"))
        topology.add_connection(Connection(source=a, sink=b, edge=sc("10.0.0.2")))
        topology.add_connection(Connection(source=b, sink=c, edge=sc("10.0.0.3")))
        for n in (a, b, c):
            assert not topology.node_is_isolated(n)


# ---------------------------------------------------------------------------
# Phase 2 - get_cycles include_singletons param
# ---------------------------------------------------------------------------


class TestGetCyclesSingletons:
    def test_empty_topology(self, topology: Topology):
        assert topology.get_cycles() == []
        assert topology.get_cycles(include_singletons=True) == []
        assert topology.get_cycles(include_singletons=False) == []

    def test_single_isolated_node_backward_compat(self, topology: Topology):
        """Default (include_singletons=True) always includes the singleton."""
        n = NodeId()
        topology.add_node(n)
        cycles = topology.get_cycles()
        assert len(cycles) == 1
        assert list(cycles[0]) == [n]

    def test_single_isolated_node_filtered(self, topology: Topology):
        """include_singletons=False excludes isolated singleton."""
        n = NodeId()
        topology.add_node(n)
        cycles = topology.get_cycles(include_singletons=False)
        assert cycles == []

    def test_two_connected_nodes_both_singletons(self, topology: Topology):
        """A->B: rx.simple_cycles returns no directed cycle; 2 singletons each. Both connected so none filtered."""
        a, b = NodeId(), NodeId()
        topology.add_node(a)
        topology.add_node(b)
        topology.add_connection(
            Connection(
                source=a, sink=b, edge=SocketConnection(sink_multiaddr=Multiaddr(address="/ip4/10.0.0.2/tcp/52415"))
            )
        )
        # Both are connected (degree 1), so even include_singletons=False keeps both singletons
        assert len(topology.get_cycles()) == 2
        assert len(topology.get_cycles(include_singletons=False)) == 2

    def test_directed_cycle_len2(self, topology: Topology):
        """A<->B forms a 2-cycle + 2 singletons."""
        a, b = NodeId(), NodeId()
        topology.add_node(a)
        topology.add_node(b)
        sc = lambda ip: SocketConnection(sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/52415"))
        topology.add_connection(Connection(source=a, sink=b, edge=sc("10.0.0.2")))
        topology.add_connection(Connection(source=b, sink=a, edge=sc("10.0.0.1")))
        cycles = topology.get_cycles()
        non_singleton = [c for c in cycles if len(c) > 1]
        assert len(non_singleton) == 1
        assert len(non_singleton[0]) == 2

    def test_directed_cycle_len2_no_isolated_to_filter(self, topology: Topology):
        """A<->B: both connected, include_singletons=False keeps the 2-cycle + 2 connected singletons."""
        a, b = NodeId(), NodeId()
        topology.add_node(a)
        topology.add_node(b)
        sc = lambda ip: SocketConnection(sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/52415"))
        topology.add_connection(Connection(source=a, sink=b, edge=sc("10.0.0.2")))
        topology.add_connection(Connection(source=b, sink=a, edge=sc("10.0.0.1")))
        cycles = topology.get_cycles(include_singletons=False)
        non_singleton = [c for c in cycles if len(c) > 1]
        assert len(non_singleton) == 1
        assert len(non_singleton[0]) == 2
        # Connected nodes' singletons are preserved even with include_singletons=False
        singletons = [c for c in cycles if len(c) == 1]
        assert len(singletons) == 2

    def test_cycle_plus_isolated_singleton_filtered(self, topology: Topology):
        """A<->B + isolated C: include_singletons=False drops only the isolated singleton."""
        a, b, c = NodeId(), NodeId(), NodeId()
        topology.add_node(a)
        topology.add_node(b)
        topology.add_node(c)
        sc = lambda ip: SocketConnection(sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/52415"))
        topology.add_connection(Connection(source=a, sink=b, edge=sc("10.0.0.2")))
        topology.add_connection(Connection(source=b, sink=a, edge=sc("10.0.0.1")))
        # C is isolated

        assert len(topology.get_cycles()) == 3 + 1  # 1 directed 2-cycle + 3 singletons
        filtered = topology.get_cycles(include_singletons=False)
        singletons = [cy for cy in filtered if len(cy) == 1]
        assert len(singletons) == 2  # only A, B (connected); C dropped
        assert any(list(cy)[0] == a for cy in singletons)
        assert any(list(cy)[0] == b for cy in singletons)
        assert not any(list(cy)[0] == c for cy in singletons)
        non_singleton = [cy for cy in filtered if len(cy) > 1]
        assert len(non_singleton) == 1

    def test_50_nodes_ring_with_isolated(self, topology: Topology):
        """Ring of 49 nodes + 1 isolated. include_singletons=False drops only the isolated singleton."""
        nodes = [NodeId() for _ in range(50)]
        for n in nodes:
            topology.add_node(n)
        sc = lambda ip: SocketConnection(sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/52415"))
        # Ring: 0->1->...->47->48->0
        for i in range(48):
            topology.add_connection(Connection(source=nodes[i], sink=nodes[i + 1], edge=sc(f"10.0.{i}.1")))
        topology.add_connection(Connection(source=nodes[48], sink=nodes[0], edge=sc("10.0.49.1")))
        # nodes[49] is isolated

        cycles_default = topology.get_cycles()
        cycles_no_singleton = topology.get_cycles(include_singletons=False)

        ring_default = [c for c in cycles_default if len(c) > 1]
        assert len(ring_default) == 1
        assert len(ring_default[0]) == 49
        assert len(cycles_default) == 1 + 50

        ring_filtered = [c for c in cycles_no_singleton if len(c) > 1]
        assert len(ring_filtered) == 1
        assert len(ring_filtered[0]) == 49
        assert len(cycles_no_singleton) == 1 + 49  # only isolated singleton dropped

    def test_thunderbolt_bridge_mixed_types(self, topology: Topology):
        """TB bridge nodes with RDMA edges: get_cycles finds the directed cycle."""
        a, b, c = NodeId(), NodeId(), NodeId()
        topology.add_node(a)
        topology.add_node(b)
        topology.add_node(c)
        sc = lambda ip: SocketConnection(sink_multiaddr=Multiaddr(address=f"/ip4/{ip}/tcp/52415"))
        rdma = lambda: RDMAConnection(source_rdma_iface="tb0", sink_rdma_iface="tb0")
        topology.add_connection(Connection(source=a, sink=b, edge=sc("10.0.0.2")))
        topology.add_connection(Connection(source=b, sink=c, edge=sc("10.0.0.3")))
        topology.add_connection(Connection(source=c, sink=a, edge=rdma()))
        cycles = topology.get_cycles()
        non_singleton = [c for c in cycles if len(c) > 1]
        assert len(non_singleton) == 1
        assert len(non_singleton[0]) == 3

    def test_1_node_backward_compat(self, topology: Topology):
        """1 node: default includes singleton, filtered does not."""
        n = NodeId()
        topology.add_node(n)
        assert len(topology.get_cycles()) == 1
        assert len(topology.get_cycles(include_singletons=False)) == 0
