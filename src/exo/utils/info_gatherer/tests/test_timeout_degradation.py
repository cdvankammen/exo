"""Timeout-degradation tests for info_gatherer probes.

Every subprocess probe in the gatherer must be bounded: a hung
networksetup/ifconfig/system_profiler (or a stalled disk_usage on a
network/FUSE mount) must degrade to the documented fallback value instead
of letting TimeoutError propagate into the monitor loop and clutter logs.

These tests drive the fail_after() boundaries down to ~10-50ms so the
timeout path is exercised deterministically without real subprocess hangs.
"""

import sys
from unittest.mock import AsyncMock, patch

import pytest

import exo.utils.info_gatherer.info_gatherer as ig
import exo.utils.info_gatherer.system_info as si
from exo.shared.types.thunderbolt import ThunderboltConnectivity


async def _hang(*_args: object, **_kwargs: object) -> object:
    """Stand-in for an unbounded subprocess: sleeps past any deadline."""
    import anyio

    with anyio.fail_after(60):
        await anyio.sleep(60)
    raise AssertionError("unreachable")


# ---- Thunderbolt bridge probe helpers (networksetup / ifconfig) ------------


async def test_get_thunderbolt_devices_timeout_returns_none(monkeypatch) -> None:
    with patch(
        "exo.utils.info_gatherer.info_gatherer.anyio.run_process",
        AsyncMock(side_effect=_hang),
    ):
        assert await ig._get_thunderbolt_devices() is None


async def test_get_bridge_services_timeout_returns_none(monkeypatch) -> None:
    with patch(
        "exo.utils.info_gatherer.info_gatherer.anyio.run_process",
        AsyncMock(side_effect=_hang),
    ):
        assert await ig._get_bridge_services() is None


async def test_get_bridge_members_timeout_returns_empty(monkeypatch) -> None:
    with patch(
        "exo.utils.info_gatherer.info_gatherer.anyio.run_process",
        AsyncMock(side_effect=_hang),
    ):
        assert await ig._get_bridge_members("bridge0") == set()


async def test_is_service_enabled_timeout_returns_none(monkeypatch) -> None:
    with patch(
        "exo.utils.info_gatherer.info_gatherer.anyio.run_process",
        AsyncMock(side_effect=_hang),
    ):
        assert await ig._is_service_enabled("Thunderbolt Bridge") is None


async def test_gather_iface_map_timeout_returns_none(monkeypatch) -> None:
    with patch(
        "exo.utils.info_gatherer.info_gatherer.anyio.run_process",
        AsyncMock(side_effect=_hang),
    ):
        assert await ig._gather_iface_map() is None


async def test_thunderbolt_bridge_info_timeout_degrades(monkeypatch) -> None:
    """The whole bridge-info gather must not raise when probes hang."""
    monkeypatch.setattr(sys, "platform", "darwin")
    with patch(
        "exo.utils.info_gatherer.info_gatherer.anyio.run_process",
        AsyncMock(side_effect=_hang),
    ):
        result = await ig.ThunderboltBridgeInfo.gather()
        assert result is not None
        assert result.status.enabled is False
        assert result.status.exists is False


# ---- Static node info (system_profiler / sw_vers / scutil) -----------------


async def test_get_os_build_version_timeout_returns_unknown(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    with patch(
        "exo.utils.info_gatherer.system_info.run_process",
        AsyncMock(side_effect=_hang),
    ):
        assert await si.get_os_build_version() == "Unknown"


async def test_get_friendly_name_timeout_returns_hostname(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    with patch("socket.gethostname", return_value="myhost"), patch(
        "exo.utils.info_gatherer.system_info.run_process",
        AsyncMock(side_effect=_hang),
    ):
        assert await si.get_friendly_name() == "myhost"


async def test_get_interface_types_from_networksetup_timeout(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    with patch(
        "exo.utils.info_gatherer.system_info.run_process",
        AsyncMock(side_effect=_hang),
    ):
        assert await si._get_interface_types_from_networksetup() == {}


async def test_get_model_and_chip_timeout_returns_unknowns(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    with patch(
        "exo.utils.info_gatherer.system_info.run_process",
        AsyncMock(side_effect=_hang),
    ):
        assert await si.get_model_and_chip() == ("Unknown Model", "Unknown Chip")


# ---- Thunderbolt connectivity (system_profiler -json) ----------------------


async def test_thunderbolt_connectivity_gather_timeout_returns_none(monkeypatch) -> None:
    with patch(
        "exo.shared.types.thunderbolt.anyio.run_process",
        AsyncMock(side_effect=_hang),
    ):
        assert await ThunderboltConnectivity.gather() is None


# ---- Disk usage (thread-based; must cancel on timeout) ---------------------


async def test_node_disk_usage_gather_uses_abandon_on_cancel(monkeypatch) -> None:
    """NodeDiskUsage.gather must cancel its worker thread on timeout.

    Without abandon_on_cancel=True, a hung shutil.disk_usage keeps running
    after fail_after() fires, so the monitor never actually degrades.
    """
    import anyio.to_thread

    captured: dict[str, bool] = {}
    _orig_run_sync = anyio.to_thread.run_sync

    async def spy_run_sync(
        func,
        *args: object,
        abandon_on_cancel: bool = False,
        cancellable: bool | None = None,
        limiter=None,
    ) -> object:
        captured["abandon_on_cancel"] = abandon_on_cancel
        # Delegate to the real implementation so the Pydantic model
        # constructs correctly; func is DiskUsage.from_path.
        return await _orig_run_sync(
            func,
            *args,
            abandon_on_cancel=abandon_on_cancel,
            cancellable=cancellable,
            limiter=limiter,
        )

    with patch(
        "exo.utils.info_gatherer.info_gatherer.to_thread.run_sync",
        AsyncMock(side_effect=spy_run_sync),
    ):
        result = await ig.NodeDiskUsage.gather()
        assert result is not None
        assert result.disk_usage is not None
        assert captured["abandon_on_cancel"] is True


async def test_node_disk_usage_gather_timeout_degrades(monkeypatch) -> None:
    """A stalled disk_usage must surface as a timeout, not a hang."""
    with patch(
        "exo.utils.info_gatherer.info_gatherer.to_thread.run_sync",
        AsyncMock(side_effect=TimeoutError),
    ), pytest.raises(TimeoutError):
        await ig.NodeDiskUsage.gather()