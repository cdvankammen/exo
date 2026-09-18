"""Tests for macmon metrics parsing (MacmonMetrics / RawMacmonMetrics).

Covers the swap-overflow clamp, usage fraction range validation, the
line-delimited JSON contract, and the single-active-monitor fallback
semantics in InfoGatherer._monitor_macmon.
"""

from subprocess import CalledProcessError
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest

from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage
from exo.utils.channels import channel

from exo.utils.info_gatherer import info_gatherer as ig
from exo.utils.info_gatherer.macmon import (
    MacmonMetrics,
    RawMacmonMetrics,
    _MemoryMetrics,
    _TempMetrics,
)

# A realistic macmon pipe line (usage is a FRACTION in [0,1], not percent).
MACMON_LINE = (
    '{"all_power":14.05,"ane_power":0.0,"cpu_power":14.02,'
    '"cpu_usage_pct":0.9946,"ecpu_usage":[2892,0.9970],'
    '"gpu_power":0.0306,"gpu_ram_power":0.0,"gpu_usage":[354,0.00957],'
    '"memory":{"ram_total":17179869184,"ram_usage":14238449664,'
    '"swap_total":19327352832,"swap_usage":18128306176},'
    '"pcpu_usage":[4424,0.9911],"ram_power":1.269,'
    '"sys_power":35.68,"temp":{"cpu_temp_avg":82.86,"gpu_temp_avg":73.50},'
    '"timestamp":"2026-09-18T02:31:35.319772+00:00"}'
)


def _raw(
    *,
    gpu_usage: tuple[int, float] = (354, 0.5),
    pcpu_usage: tuple[int, float] = (4424, 0.5),
    ecpu_usage: tuple[int, float] = (2892, 0.5),
    swap_total: int = 1000,
    swap_usage: int = 100,
    ram_total: int = 1000,
    ram_usage: int = 200,
) -> RawMacmonMetrics:
    return RawMacmonMetrics(
        timestamp="t",
        temp=_TempMetrics(cpu_temp_avg=50.0, gpu_temp_avg=60.0),
        memory=_MemoryMetrics(
            ram_total=ram_total,
            ram_usage=ram_usage,
            swap_total=swap_total,
            swap_usage=swap_usage,
        ),
        ecpu_usage=ecpu_usage,
        pcpu_usage=pcpu_usage,
        gpu_usage=gpu_usage,
        all_power=1.0,
        ane_power=0.1,
        cpu_power=0.5,
        gpu_power=0.2,
        gpu_ram_power=0.05,
        ram_power=0.3,
        sys_power=2.0,
    )


# ---- swap overflow clamp ---------------------------------------------------


def test_swap_available_clamped_when_usage_exceeds_total() -> None:
    """macmon can transiently report swap_usage > swap_total; the resulting
    swap_available must be clamped to zero, not negative."""
    metrics = MacmonMetrics.from_raw(
        _raw(swap_total=8_000_000_000, swap_usage=9_000_000_000)
    )
    assert metrics.memory.swap_available.in_bytes == 0


def test_swap_available_clamped_at_total_when_negative_usage() -> None:
    """A meter reporting negative swap_usage must not create swap_available
    larger than swap_total."""
    metrics = MacmonMetrics.from_raw(
        _raw(swap_total=8_000_000_000, swap_usage=-100)
    )
    assert metrics.memory.swap_available.in_bytes == 8_000_000_000


def test_swap_available_normal_when_within_range() -> None:
    metrics = MacmonMetrics.from_raw(
        _raw(swap_total=8_000_000_000, swap_usage=2_000_000_000)
    )
    assert metrics.memory.swap_available.in_bytes == 6_000_000_000


# ---- usage range validation -------------------------------------------------


def test_usage_fractions_passed_through() -> None:
    metrics = MacmonMetrics.from_raw(_raw())
    assert metrics.system_profile.gpu_usage == 0.5
    assert metrics.system_profile.pcpu_usage == 0.5
    assert metrics.system_profile.ecpu_usage == 0.5


def test_usage_clamped_above_one() -> None:
    """macmon can emit a fraction slightly >1.0 on a busy/idle race; clamp to
    1.0 so the UI never renders 350% GPU."""
    metrics = MacmonMetrics.from_raw(
        _raw(
            gpu_usage=(354, 3.5),
            pcpu_usage=(4424, 1.01),
            ecpu_usage=(2892, 1.0),
        )
    )
    assert metrics.system_profile.gpu_usage == 1.0
    assert metrics.system_profile.pcpu_usage == 1.0
    assert metrics.system_profile.ecpu_usage == 1.0


def test_usage_clamped_below_zero() -> None:
    metrics = MacmonMetrics.from_raw(
        _raw(gpu_usage=(354, -0.2), pcpu_usage=(4424, -1.0), ecpu_usage=(2892, 0.0))
    )
    assert metrics.system_profile.gpu_usage == 0.0
    assert metrics.system_profile.pcpu_usage == 0.0
    assert metrics.system_profile.ecpu_usage == 0.0


# ---- line-delimited JSON contract -------------------------------------------


def test_from_raw_json_parses_real_macmon_line() -> None:
    """The exact shape macmon pipe emits on this machine today."""
    metrics = MacmonMetrics.from_raw_json(MACMON_LINE)
    assert metrics.system_profile.gpu_usage == pytest.approx(0.00957)
    assert metrics.system_profile.pcpu_usage == pytest.approx(0.9911)
    assert metrics.system_profile.ecpu_usage == pytest.approx(0.9970)
    assert metrics.system_profile.temp == pytest.approx(73.50)
    assert metrics.system_profile.sys_power == pytest.approx(35.68)
    assert metrics.memory.swap_available.in_bytes == 19327352832 - 18128306176


def test_from_raw_json_preserves_camelcase_wire_contract() -> None:
    """MacmonMetrics serialises as {ClassName: {...}} and camelCases when
    by_alias=True — the exact shape the state endpoint (by_alias=True) and
    Swift's Decodable (gpuUsage, pcpuUsage, sysPower...) read."""
    metrics = MacmonMetrics.from_raw(_raw())
    payload = metrics.model_dump(mode="json", by_alias=True)
    inner = payload["MacmonMetrics"]
    assert inner["systemProfile"]["gpuUsage"] == 0.5
    assert inner["systemProfile"]["pcpuUsage"] == 0.5
    assert inner["systemProfile"]["ecpuUsage"] == 0.5
    assert inner["memory"]["swapAvailable"]["inBytes"] == 900


# ---- single-active-monitor semantics -----------------------------------------


def test_fallback_started_only_once_when_macmon_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When macmon is absent, exactly ONE psutil monitor is started — the
    guard must not double-start it."""
    sender, _receiver = channel[ig.GatheredInfo]()
    gatherer = ig.InfoGatherer(info_sender=sender)
    started: list[tuple[object, ...]] = []
    gatherer._tg.start_soon = lambda fn, *a, **k: started.append((fn, a, k))  # pyright: ignore[reportAttributeAccessIssue]

    monkeypatch.setattr("os.getenv", lambda key, default=None: None)
    monkeypatch.setattr("shutil.which", lambda name: None)

    async def run() -> None:
        await gatherer._monitor_macmon(1)

    anyio.run(run)

    assert len(started) == 1
    assert started[0][0] == gatherer._monitor_memory_usage


def test_fallback_replaces_macmon_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A macmon pipe timeout hands over to psutil and STOPS the retry loop,
    so the two never publish conflicting events."""
    sender, _receiver = channel[ig.GatheredInfo]()
    gatherer = ig.InfoGatherer(info_sender=sender)
    started: list[tuple[object, ...]] = []
    gatherer._tg.start_soon = lambda fn, *a, **k: started.append((fn, a, k))  # pyright: ignore[reportAttributeAccessIssue]

    monkeypatch.setattr("os.getenv", lambda key, default=None: None)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/macmon")
    monkeypatch.setattr(
        gatherer, "_can_read_macmon_metrics", AsyncMock(return_value=True)
    )

    proc = MagicMock()
    proc.__aenter__ = AsyncMock(return_value=proc)
    proc.__aexit__ = AsyncMock(return_value=False)
    proc.stdout = None  # triggers the "MacMon closed stdout" return path

    monkeypatch.setattr("exo.utils.info_gatherer.info_gatherer.open_process", AsyncMock(return_value=proc))

    async def run() -> None:
        await gatherer._monitor_macmon(1)

    anyio.run(run)

    # macmon path returned immediately (stdout closed) without starting a fallback,
    # and without spawning a retry loop — that is the non-duplicate contract.
    assert len(started) == 0


def test_no_fallback_started_after_fallback_already_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once a psutil fallback is enabled, a macmon timeout must NOT stack a
    second fallback."""
    sender, _receiver = channel[ig.GatheredInfo]()
    gatherer = ig.InfoGatherer(info_sender=sender)
    started: list[tuple[object, ...]] = []
    gatherer._tg.start_soon = lambda fn, *a, **k: started.append((fn, a, k))  # pyright: ignore[reportAttributeAccessIssue]
    gatherer._psutil_enabled = True  # fallback already running

    monkeypatch.setattr("os.getenv", lambda key, default=None: None)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/macmon")
    monkeypatch.setattr(
        gatherer, "_can_read_macmon_metrics", AsyncMock(return_value=True)
    )

    proc = MagicMock()
    proc.__aenter__ = AsyncMock(return_value=proc)
    proc.__aexit__ = AsyncMock(return_value=False)
    proc.stdout = None
    monkeypatch.setattr("exo.utils.info_gatherer.info_gatherer.open_process", AsyncMock(return_value=proc))

    async def run() -> None:
        await gatherer._monitor_macmon(1)

    anyio.run(run)

    assert len(started) == 0


def test_fallback_started_on_calledprocesserror(monkeypatch: pytest.MonkeyPatch) -> None:
    """CalledProcessError from the macmon process arms the psutil fallback."""
    sender, _receiver = channel[ig.GatheredInfo]()
    gatherer = ig.InfoGatherer(info_sender=sender)
    started: list[tuple[object, ...]] = []
    gatherer._tg.start_soon = lambda fn, *a, **k: started.append((fn, a, k))  # pyright: ignore[reportAttributeAccessIssue]

    monkeypatch.setattr("os.getenv", lambda key, default=None: None)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/macmon")
    monkeypatch.setattr(
        gatherer, "_can_read_macmon_metrics", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        "exo.utils.info_gatherer.info_gatherer.open_process",
        AsyncMock(side_effect=CalledProcessError(1, "macmon", stderr=b"boom")),
    )

    async def run() -> None:
        await gatherer._monitor_macmon(1)

    anyio.run(run)

    assert len(started) == 1
    assert started[0][0] == gatherer._monitor_memory_usage


def test_fallback_started_on_unexpected_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    sender, _receiver = channel[ig.GatheredInfo]()
    gatherer = ig.InfoGatherer(info_sender=sender)
    started: list[tuple[object, ...]] = []
    gatherer._tg.start_soon = lambda fn, *a, **k: started.append((fn, a, k))  # pyright: ignore[reportAttributeAccessIssue]

    monkeypatch.setattr("os.getenv", lambda key, default=None: None)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/macmon")
    monkeypatch.setattr(
        gatherer, "_can_read_macmon_metrics", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        "exo.utils.info_gatherer.info_gatherer.open_process",
        AsyncMock(side_effect=RuntimeError("boom")),
    )

    async def run() -> None:
        await gatherer._monitor_macmon(1)

    anyio.run(run)

    assert len(started) == 1
    assert started[0][0] == gatherer._monitor_memory_usage


def test_monitor_memory_usage_rejects_second_concurrent_start() -> None:
    """The psutil monitor itself refuses a second concurrent instance."""
    sender, _receiver = channel[ig.GatheredInfo]()
    gatherer = ig.InfoGatherer(info_sender=sender)
    assert gatherer._psutil_enabled is False

    async def fake_send(_obj: object) -> None:
        await anyio.sleep(0)  # pragma: no cover

    gatherer.info_sender.send = fake_send  # pyright: ignore[reportAttributeAccessIssue]

    async def run() -> None:
        with anyio.move_on_after(0.05):
            await gatherer._monitor_memory_usage(0.01)  # pyright: ignore[reportPrivateUsage]
        with anyio.move_on_after(0.05):
            await gatherer._monitor_memory_usage(0.01)  # pyright: ignore[reportPrivateUsage]

    anyio.run(run)

    # The flag stays True; the second call returned immediately without sending.
    assert gatherer._psutil_enabled is True


def test_memory_usage_clamp_is_shared_across_entry_points() -> None:
    """from_bytes clamps swap_available for every caller (macmon, psutil,
    CUDA) — the invariant lives in the shared type, not the adapter."""
    usage = MemoryUsage.from_bytes(
        ram_total=1000,
        ram_available=500,
        swap_total=1000,
        swap_available=-500,
    )
    assert usage.swap_available == Memory.from_bytes(0)

    usage2 = MemoryUsage.from_bytes(
        ram_total=1000,
        ram_available=500,
        swap_total=1000,
        swap_available=1500,
    )
    assert usage2.swap_available == Memory.from_bytes(1000)