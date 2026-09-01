"""Tests for periodic NodeConfig (identity) re-announcement.

NodeConfig is gathered exactly once at startup; if that first message is
lost (e.g. the router is still connecting when the gatherer starts), the
node joins elections and is "discovered" by peers but never appears in any
node's topology. ``_monitor_node_config`` re-sends it on an interval so the
identity announcement self-heals within one poll interval (fix dead4e88,
same class as the NodeBackends re-announce 9299909c).
"""

import anyio
import pytest

import exo.utils.info_gatherer.info_gatherer as ig
from exo.utils.channels import channel


class TestMonitorNodeConfig:
    async def test_monitor_node_config_reannounces_periodically(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each loop iteration re-gathers and re-sends NodeConfig, so a lost
        startup announcement is repaired within one poll interval."""
        sender, receiver = channel[ig.GatheredInfo]()

        calls = 0

        async def fake_gather() -> ig.NodeConfig | None:
            nonlocal calls
            calls += 1
            return ig.NodeConfig()

        monkeypatch.setattr(ig.NodeConfig, "gather", fake_gather)

        gatherer = ig.InfoGatherer(info_sender=sender)
        with anyio.move_on_after(0.1):
            await gatherer._monitor_node_config(0.01)  # pyright: ignore[reportPrivateUsage]

        assert calls >= 1
        assert sender.statistics().current_buffer_used >= 1
        sent = receiver.receive_nowait()
        assert isinstance(sent, ig.NodeConfig)

    async def test_monitor_node_config_skips_when_gather_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed gather (None) sends nothing but keeps the loop alive."""
        sender, _receiver = channel[ig.GatheredInfo]()

        async def fake_gather() -> ig.NodeConfig | None:
            return None

        monkeypatch.setattr(ig.NodeConfig, "gather", fake_gather)

        gatherer = ig.InfoGatherer(info_sender=sender)
        with anyio.move_on_after(0.1):
            await gatherer._monitor_node_config(0.01)  # pyright: ignore[reportPrivateUsage]

        assert sender.statistics().current_buffer_used == 0
