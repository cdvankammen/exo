import pytest

from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.topology import (
    Connection,
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


def test_add_node(topology: Topology):
    # arrange
    node_id = NodeId()

    # act
    topology.add_node(node_id)

    # assert
    assert topology.node_is_leaf(node_id)


def test_add_connection(topology: Topology, socket_connection: SocketConnection):
    # arrange
    node_a = NodeId()
    node_b = NodeId()
    connection = Connection(source=node_a, sink=node_b, edge=socket_connection)

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(connection)

    # act
    data = list(topology.list_connections())

    # assert
    assert data == [connection]

    assert topology.node_is_leaf(node_a)
    assert topology.node_is_leaf(node_b)


def test_remove_connection_still_connected(
    topology: Topology, socket_connection: SocketConnection
):
    # arrange
    node_a = NodeId()
    node_b = NodeId()
    conn = Connection(source=node_a, sink=node_b, edge=socket_connection)

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(conn)

    # act
    topology.remove_connection(conn)

    # assert
    assert list(topology.get_all_connections_between(node_a, node_b)) == []


def test_remove_node_still_connected(
    topology: Topology, socket_connection: SocketConnection
):
    # arrange
    node_a = NodeId()
    node_b = NodeId()
    conn = Connection(source=node_a, sink=node_b, edge=socket_connection)

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(conn)
    assert list(topology.out_edges(node_a)) == [conn]

    # act
    topology.remove_node(node_b)

    # assert
    assert list(topology.out_edges(node_a)) == []


def test_list_nodes(topology: Topology, socket_connection: SocketConnection):
    # arrange
    node_a = NodeId()
    node_b = NodeId()
    conn = Connection(source=node_a, sink=node_b, edge=socket_connection)

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(conn)
    assert list(topology.out_edges(node_a)) == [conn]

    # act
    nodes = list(topology.list_nodes())

    # assert
    assert len(nodes) == 2
    assert all(isinstance(node, NodeId) for node in nodes)
    assert set(node for node in nodes) == set([node_a, node_b])


def _rdma_connection() -> RDMAConnection:
    return RDMAConnection(source_rdma_iface="rdma_en1", sink_rdma_iface="rdma_en1")


def test_replace_all_out_rdma_connections_preserves_socket_edges_under_index_holes():
    """
    The replacement must never corrupt the edge index space such that surviving
    socket edges become unreachable.

    Regression for the stale-index removal in
    ``Topology.replace_all_out_rdma_connections``: rustworkx index space is
    torn by ``remove_edge_from_index`` during iteration over a live
    ``EdgeIndices`` view. A removed index becomes a hole — a subsequent lookup
    on it raises IndexError, and on other rustworkx versions a stale index can
    resolve to a *different* edge and delete the wrong (socket) edge.
    """
    topology = Topology()
    node_a = NodeId()
    node_b = NodeId()
    node_c = NodeId()
    node_d = NodeId()
    for node in (node_a, node_b, node_c, node_d):
        topology.add_node(node)

    # Insertion order: rdma, rdma, socket, socket -> rustworkx indices 0,1,2,3.
    # The live out_edge_indices view for node_a is [3,2,1,0]; removing while
    # iterating leaves index holes that the current implementation mis-reads.
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=_rdma_connection())
    )
    topology.add_connection(
        Connection(source=node_a, sink=node_c, edge=_rdma_connection())
    )
    topology.add_connection(
        Connection(
            source=node_a,
            sink=node_d,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/10.0.0.1/tcp/1235")
            ),
        )
    )
    topology.add_connection(
        Connection(
            source=node_a,
            sink=node_b,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/10.0.0.2/tcp/1235")
            ),
        )
    )
    # An incoming edge to node_a must also survive the replacement.
    topology.add_connection(
        Connection(
            source=node_d,
            sink=node_a,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/10.0.0.3/tcp/1235")
            ),
        )
    )

    replacement = Connection(
        source=node_a, sink=node_b, edge=_rdma_connection()
    )

    # act
    topology.replace_all_out_rdma_connections(node_a, [replacement])

    # assert: exactly the replacement RDMA edge remains
    rdma_edges = [
        conn
        for conn in topology.list_connections()
        if isinstance(conn.edge, RDMAConnection)
    ]
    assert rdma_edges == [replacement]

    # and every socket edge (including the incoming one) survived and is
    # still reachable through the public API
    socket_edges = [
        conn.edge
        for conn in topology.list_connections()
        if isinstance(conn.edge, SocketConnection)
    ]
    assert len(socket_edges) == 3
    assert list(topology.out_edges(node_a))  # no IndexError, no torn index space


def test_replace_all_out_rdma_connections_no_rdma_edges_adds_replacement():
    """With zero existing RDMA edges, the replacement set is still added."""
    topology = Topology()
    node_a = NodeId()
    node_b = NodeId()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(
            source=node_a,
            sink=node_b,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/10.0.0.1/tcp/1235")
            ),
        )
    )

    replacement = Connection(
        source=node_a, sink=node_b, edge=_rdma_connection()
    )

    # act
    topology.replace_all_out_rdma_connections(node_a, [replacement])

    # assert
    rdma_edges = [
        conn
        for conn in topology.list_connections()
        if isinstance(conn.edge, RDMAConnection)
    ]
    assert rdma_edges == [replacement]
