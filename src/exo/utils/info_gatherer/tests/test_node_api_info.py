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