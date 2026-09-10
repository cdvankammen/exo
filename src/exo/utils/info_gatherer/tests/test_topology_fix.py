"""Tests for the topology-zero-edges fixes (§12).

Covers:
- §12.5 P3  SocketConnection hash includes port (types/topology.py)
- §12.1 P1  Per-peer api_port in check_reachable/check_reachability
- §12.4 P2  Probe-failure diagnostics promoted to warning level
- §12.3 P2  Eager NodeNetworkInterfaces first emit
- §12.2 P1  TopologySnapshot round-trip (connections dict, no phantom edges key)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import (
    NetworkInterfaceInfo,
    NodeIdentity,
    NodeNetworkInfo,
)
from exo.shared.types.topology import (
    Connection,
    RDMAConnection,
    SocketConnection,
)


# ---------------------------------------------------------------------------
# §12.5  P3 — SocketConnection.__hash__ must include port
# ---------------------------------------------------------------------------


class TestSocketConnectionHash:
    def test_same_ip_different_ports_are_distinct_keys(self):
        """Two edges to the same IP on different ports must hash differently.

        Before the fix: __hash__ only hashes ip_address, so these collide
        on hash and dict lookup depends on __eq__ fallback — correct but
        fragile and O(bucket) on collision.
        """
        a = SocketConnection(sink_multiaddr=Multiaddr(address="/ip4/10.0.0.1/tcp/52415"))
        b = SocketConnection(sink_multiaddr=Multiaddr(address="/ip4/10.0.0.1/tcp/52416"))
        assert hash(a) != hash(b), "different ports must produce different hashes"

    def test_equal_connections_have_equal_hashes(self):
        a = SocketConnection(sink_multiaddr=Multiaddr(address="/ip4/10.0.0.1/tcp/52415"))
        b = SocketConnection(sink_multiaddr=Multiaddr(address="/ip4/10.0.0.1/tcp/52415"))
        assert a == b
        assert hash(a) == hash(b)

    def test_same_port_different_ip_differ(self):
        a = SocketConnection(sink_multiaddr=Multiaddr(address="/ip4/10.0.0.1/tcp/52415"))
        b = SocketConnection(sink_multiaddr=Multiaddr(address="/ip4/10.0.0.2/tcp/52415"))
        assert hash(a) != hash(b)

    def test_dict_lookup_returns_correct_entry_with_same_ip_different_ports(self):
        """Dict with two entries for same IP, different ports must resolve correctly."""
        a = SocketConnection(sink_multiaddr=Multiaddr(address="/ip4/10.0.0.1/tcp/52415"))
        b = SocketConnection(sink_multiaddr=Multiaddr(address="/ip4/10.0.0.1/tcp/52416"))
        d = {a: "first", b: "second"}
        assert d[a] == "first"
        assert d[b] == "second"


# ---------------------------------------------------------------------------
# §12.1  P1 — check_reachability uses per-peer api_port
# ---------------------------------------------------------------------------


class TestCheckReachabilityPerPeerPort:
    """check_reachability must use the peer's advertised port, not a caller default."""

    async def test_uses_peer_port_in_url(self):
        """When peer port differs from default, the URL must contain the peer port."""
        from exo.utils.info_gatherer.net_profile import check_reachability

        peer_id = NodeId()
        out: dict[NodeId, set[str]] = {}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = str(peer_id)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        # Peer is on port 8080, not default 52415
        result = await check_reachability(
            target_ip="10.0.0.2",
            expected_node_id=peer_id,
            out=out,
            client=mock_client,
            api_port=8080,
        )

        assert result is not None
        # Verify the URL used port 8080
        call_url = mock_client.get.call_args[0][0]
        assert ":8080/" in call_url
        assert str(peer_id) in out

    async def test_returns_latency_on_success(self):
        from exo.utils.info_gatherer.net_profile import check_reachability
        import time

        peer_id = NodeId()
        out: dict[NodeId, set[str]] = {}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = str(peer_id)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        result = await check_reachability(
            target_ip="10.0.0.2",
            expected_node_id=peer_id,
            out=out,
            client=mock_client,
            api_port=52415,
        )

        assert result is not None
        assert result >= 0.0  # latency in ms

    async def test_returns_none_on_non_200(self):
        from exo.utils.info_gatherer.net_profile import check_reachability

        peer_id = NodeId()
        out: dict[NodeId, set[str]] = {}

        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "error"

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("anyio.sleep", new_callable=AsyncMock):
            result = await check_reachability(
                target_ip="10.0.0.2",
                expected_node_id=peer_id,
                out=out,
                client=mock_client,
                api_port=52415,
            )

        assert result is None
        assert peer_id not in out

    async def test_returns_none_on_node_id_mismatch(self):
        from exo.utils.info_gatherer.net_profile import check_reachability

        peer_id = NodeId()
        wrong_id = NodeId()
        out: dict[NodeId, set[str]] = {}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = str(wrong_id)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch("anyio.sleep", new_callable=AsyncMock):
            result = await check_reachability(
                target_ip="10.0.0.2",
                expected_node_id=peer_id,
                out=out,
                client=mock_client,
                api_port=52415,
            )

        assert result is None
        assert peer_id not in out


# ---------------------------------------------------------------------------
# §12.4  P2 — Probe-failure diagnostics at warning level
# ---------------------------------------------------------------------------


class TestProbeFailureDiagnostics:
    """Node-id mismatch and connection errors must be logged at WARNING, not DEBUG."""

    async def test_node_id_mismatch_logs_warning(self):
        """§12.4: Mismatched node_id must emit a warning, not a debug log."""
        from exo.utils.info_gatherer.net_profile import check_reachability

        peer_id = NodeId()
        wrong_id = NodeId()
        out: dict[NodeId, set[str]] = {}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = str(wrong_id)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("anyio.sleep", new_callable=AsyncMock),
            patch("exo.utils.info_gatherer.net_profile.logger") as mock_logger,
        ):
            await check_reachability(
                target_ip="10.0.0.2",
                expected_node_id=peer_id,
                out=out,
                client=mock_client,
                api_port=52415,
            )
            # Must have logged at warning level, not debug
            mock_logger.warning.assert_called()
            log_msg = mock_logger.warning.call_args[0][0]
            assert "unexpected" in log_msg.lower() or "node_id" in log_msg.lower()

    async def test_connection_error_logs_warning(self):
        """§12.4: Last-attempt HTTPError must emit a warning."""
        import httpx
        from exo.utils.info_gatherer.net_profile import check_reachability

        peer_id = NodeId()
        out: dict[NodeId, set[str]] = {}

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )

        with (
            patch("anyio.sleep", new_callable=AsyncMock),
            patch("exo.utils.info_gatherer.net_profile.logger") as mock_logger,
        ):
            result = await check_reachability(
                target_ip="10.0.0.2",
                expected_node_id=peer_id,
                out=out,
                client=mock_client,
                api_port=52415,
            )
            assert result is None
            # Must have logged a warning about the connection error
            mock_logger.warning.assert_called()
            log_msg = mock_logger.warning.call_args[0][0]
            assert "connect error" in log_msg.lower() or "treating as down" in log_msg.lower()


# ---------------------------------------------------------------------------
# §12.3  P2 — Eager NodeNetworkInterfaces first emit
# ---------------------------------------------------------------------------


class TestEagerNetworkInterfacesEmit:
    """NodeNetworkInterfaces must be emitted eagerly at startup (before the sleep loop)."""

    def test_watch_system_info_emits_immediately_not_after_sleep(self):
        """The first send of NodeNetworkInterfaces must happen before any sleep.

        The old code had a `while True:` / `await anyio.sleep(interval)` loop
        meaning the first emit was delayed by the full interval (10s).
        """
        import inspect
        from exo.utils.info_gatherer.info_gatherer import InfoGatherer

        src = inspect.getsource(InfoGatherer._watch_system_info)

        # The method must NOT start with an `await anyio.sleep` before the
        # first get_network_interfaces call.  The fixed version gathers first,
        # sleeps second.
        lines = [l.strip() for l in src.splitlines() if l.strip()]
        # First meaningful non-decorator line after the `while True:` should
        # be a try/await gather, NOT a sleep.
        first_loop_line = next(
            l for l in lines
            if not l.startswith("def ") and not l.startswith("async def ")
            and not l.startswith("@") and l != "while True:" and l != ""
        )
        assert "sleep" not in first_loop_line, (
            f"_watch_system_info must gather before sleeping; first loop line: {first_loop_line}"
        )


# ---------------------------------------------------------------------------
# §12.2  P1 — TopologySnapshot round-trip (connections dict)
# ---------------------------------------------------------------------------


class TestTopologySnapshotContract:
    """TopologySnapshot must serialize connections as a nested dict, not edges."""

    def test_snapshot_has_connections_dict(self):
        """Top-level shape: {nodes, connections} — no phantom 'edges' field."""
        t = Topology()
        a, b = NodeId(), NodeId()
        edge = SocketConnection(sink_multiaddr=Multiaddr(address="/ip4/10.0.0.2/tcp/52415"))
        t.add_connection(Connection(source=a, sink=b, edge=edge))

        snap = t.to_snapshot()
        data = snap.model_dump()

        assert "nodes" in data
        assert "connections" in data
        assert "edges" not in data  # the old misparse used this key

    def test_snapshot_connections_keyed_by_source_sink(self):
        """connections[source][sink] = [edge, ...] — the dict the dashboard expects."""
        t = Topology()
        a, b = NodeId(), NodeId()
        e1 = SocketConnection(sink_multiaddr=Multiaddr(address="/ip4/10.0.0.2/tcp/52415"))
        e2 = SocketConnection(sink_multiaddr=Multiaddr(address="/ip4/10.0.0.2/tcp/52416"))
        t.add_connection(Connection(source=a, sink=b, edge=e1))
        t.add_connection(Connection(source=a, sink=b, edge=e2))

        snap = t.to_snapshot()
        conns = snap.connections

        assert a in conns
        assert b in conns[a]
        assert len(conns[a][b]) == 2

    def test_roundtrip_preserves_edges(self):
        """to_snapshot -> from_snapshot preserves all edges."""
        t = Topology()
        a, b, c = NodeId(), NodeId(), NodeId()
        e1 = SocketConnection(sink_multiaddr=Multiaddr(address="/ip4/10.0.0.2/tcp/52415"))
        e2 = RDMAConnection(source_rdma_iface="en2", sink_rdma_iface="en3")
        t.add_connection(Connection(source=a, sink=b, edge=e1))
        t.add_connection(Connection(source=b, sink=c, edge=e2))

        snap = t.to_snapshot()
        t2 = Topology.from_snapshot(snap)

        # Verify same node count and edge count
        assert len(list(t2.list_nodes())) == 3
        all_conns = list(t2.list_connections())
        assert len(all_conns) == 2

    def test_empty_topology_has_no_connections(self):
        t = Topology()
        t.add_node(NodeId())
        snap = t.to_snapshot()
        assert snap.connections == {}

    def test_total_edges_from_connections(self):
        """Correct edge count formula (replaces the broken topology['edges'] lookup)."""
        t = Topology()
        a, b, c = NodeId(), NodeId(), NodeId()
        t.add_connection(Connection(
            source=a, sink=b,
            edge=SocketConnection(sink_multiaddr=Multiaddr(address="/ip4/10.0.0.2/tcp/52415"))
        ))
        t.add_connection(Connection(
            source=b, sink=c,
            edge=SocketConnection(sink_multiaddr=Multiaddr(address="/ip4/10.0.0.3/tcp/52415"))
        ))
        t.add_connection(Connection(
            source=c, sink=a,
            edge=SocketConnection(sink_multiaddr=Multiaddr(address="/ip4/10.0.0.1/tcp/52415"))
        ))

        data = t.to_snapshot().model_dump()
        conns = data["connections"]
        total = sum(
            len(edges)
            for sink_map in conns.values()
            for edges in sink_map.values()
        )
        assert total == 3
