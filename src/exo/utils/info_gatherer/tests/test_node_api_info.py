"""Tests for periodic NodeApiInfo (API host/port) re-announcement.

NodeApiInfo is gathered exactly once at startup. If that first message is
lost (router still connecting) or the node later restarts with a different
API port, peers keep probing the wrong port and the topology edge-deletion
loop removes every edge whose sink port doesn't match the advertised port —
the node ends up with zero topology edges until every peer restarts
(observed live on tryingexo Linux containers, 2026-09-01). Re-sending on an
interval makes the announcement self-healing (same class as NodeConfig
dead4e88 / NodeBackends 9299909c).
"""

import anyio
import pytest

import exo.utils.info_gatherer.info_gatherer as ig
from exo.utils.channels import channel


class TestMonitorNodeApiInfo:
    async def test_monitor_node_api_info_reannounces_periodically(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each loop iteration re-gathers and re-sends NodeApiInfo, so a lost
        startup announcement (or a port change) is repaired within one poll
        interval."""
        sender, receiver = channel[ig.GatheredInfo]()

        calls = 0

        async def fake_gather() -> ig.NodeApiInfo | None:
            nonlocal calls
            calls += 1
            return ig.NodeApiInfo(api_host="0.0.0.0", api_port=52417)

        monkeypatch.setattr(ig.NodeApiInfo, "gather", fake_gather)

        gatherer = ig.InfoGatherer(info_sender=sender)
        with anyio.move_on_after(0.1):
            await gatherer._monitor_node_api_info(0.01)  # pyright: ignore[reportPrivateUsage]

        assert calls >= 1
        assert sender.statistics().current_buffer_used >= 1
        sent = receiver.receive_nowait()
        assert isinstance(sent, ig.NodeApiInfo)
        assert sent.api_port == 52417

    async def test_monitor_node_api_info_skips_when_gather_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed gather (None) sends nothing but keeps the loop alive."""
        sender, _receiver = channel[ig.GatheredInfo]()

        async def fake_gather() -> ig.NodeApiInfo | None:
            return None

        monkeypatch.setattr(ig.NodeApiInfo, "gather", fake_gather)

        gatherer = ig.InfoGatherer(info_sender=sender)
        with anyio.move_on_after(0.1):
            await gatherer._monitor_node_api_info(0.01)  # pyright: ignore[reportPrivateUsage]

        assert sender.statistics().current_buffer_used == 0


class TestResolveAdvertiseHost:
    async def test_explicit_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EXO_API_ADVERTISE_HOST set → used verbatim regardless of bind host."""
        assert (
            await ig.resolve_advertise_host("0.0.0.0", "10.0.0.5")
            == "10.0.0.5"
        )
        assert await ig.resolve_advertise_host("127.0.0.1", "10.0.0.5") == "10.0.0.5"

    async def test_wildcard_bind_derives_primary_ip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """0.0.0.0 is not dialable by peers → fall back to a non-loopback IP."""
        async def fake_primary() -> str:
            return "192.168.1.50"
        monkeypatch.setattr(ig, "_primary_non_loopback_ip", fake_primary)

        assert await ig.resolve_advertise_host("0.0.0.0", None) == "192.168.1.50"
        assert await ig.resolve_advertise_host("::", None) == "192.168.1.50"

    async def test_loopback_bind_derives_primary_ip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """127.0.0.1 bind (hardened) still advertises a peer-reachable address."""
        async def fake_primary() -> str:
            return "192.168.1.50"
        monkeypatch.setattr(ig, "_primary_non_loopback_ip", fake_primary)

        assert await ig.resolve_advertise_host("127.0.0.1", None) == "192.168.1.50"
        assert await ig.resolve_advertise_host("::1", None) == "192.168.1.50"

    async def test_concrete_bind_used_verbatim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A specific bind address (e.g. a fixed LAN IP) is the advertisement."""
        async def fake_primary() -> str:
            return "192.168.1.50"
        monkeypatch.setattr(ig, "_primary_non_loopback_ip", fake_primary)

        assert await ig.resolve_advertise_host("10.2.0.90", None) == "10.2.0.90"

    async def test_gather_uses_advertise_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NodeApiInfo.gather resolves through resolve_advertise_host."""
        monkeypatch.setenv("EXO_API_PORT", "52415")
        monkeypatch.setattr(ig, "resolve_advertise_host", _advertise_stub)

        info = await ig.NodeApiInfo.gather()
        assert info is not None
        assert info.api_host == "stub.example"
        assert info.api_port == 52415


async def _advertise_stub(bind_host: str, advertise_override: str | None) -> str:
    return "stub.example"