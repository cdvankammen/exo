"""Liveness tests for the macmon pipe reader.

Regression cover for exo-explore/exo#2551: the macmon monitor awaited
``info_sender.send(metrics)`` *outside* any timeout, so a full downstream
channel blocked the pipe reader forever.  macmon kept writing, the stdout
pipe filled (64 KB), the child blocked in ``write(2)``, and neither the
"MacMon produced no output" timeout nor the psutil fallback ever fired.
``node_memory``/``node_system`` then vanished from ``/state`` for good and
every placement failed.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

from exo.utils.info_gatherer import info_gatherer as ig_mod
from exo.utils.info_gatherer.info_gatherer import InfoGatherer


def _macmon_line() -> bytes:
    """One valid macmon metrics line, as macmon pipe would emit it."""
    return (
        json.dumps(
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "temp": {"cpu_temp_avg": 40.0, "gpu_temp_avg": 35.0},
                "memory": {
                    "ram_total": 64 * 1024**3,
                    "ram_usage": 32 * 1024**3,
                    "swap_total": 0,
                    "swap_usage": 0,
                },
                "ecpu_usage": [1000, 0.1],
                "pcpu_usage": [3000, 0.2],
                "gpu_usage": [1000, 0.0],
                "all_power": 10.0,
                "ane_power": 0.0,
                "cpu_power": 5.0,
                "gpu_power": 1.0,
                "gpu_ram_power": 0.0,
                "ram_power": 0.5,
                "sys_power": 12.0,
            }
        )
        + "\n"
    ).encode("utf-8")


class _FakeStdout:
    """Endless supply of valid macmon lines (the child never stops writing)."""

    async def receive(self, max_bytes: int = 8192) -> bytes:
        return _macmon_line()


class _FakeProcess:
    def __init__(self) -> None:
        self.stdout = _FakeStdout()
        self.terminated = False

    async def __aenter__(self) -> "_FakeProcess":
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        self.terminated = True
        return False


class _DummyTaskGroup:
    """Stand-in for the gatherer's TaskGroup (it is never entered here)."""

    def __init__(self) -> None:
        self.started: list[str] = []

    def start_soon(self, fn, *args) -> None:  # noqa: ANN001
        self.started.append(getattr(fn, "__name__", repr(fn)))


class _BlockingSender:
    """Sender whose consumer has gone away: every send blocks forever."""

    def __init__(self) -> None:
        self.attempts = 0

    async def send(self, item: object) -> None:
        self.attempts += 1
        await asyncio.Event().wait()


async def test_blocked_send_restarts_the_macmon_stream() -> None:
    """A send that never completes must not wedge the reader forever.

    With the fix, the send is bounded: it times out, the stream is torn down
    and reopened (so ``open_process`` is called again) and the psutil
    fallback is started.  Without the fix this coroutine never returns and
    the surrounding ``asyncio.timeout`` is the only thing that ends it.
    """
    opens = 0

    async def fake_open_process(_cmd):  # noqa: ANN001, ANN202
        nonlocal opens
        opens += 1
        return _FakeProcess()

    sender = _BlockingSender()
    gatherer = InfoGatherer(info_sender=sender)  # type: ignore[arg-type]
    tg = _DummyTaskGroup()
    gatherer._tg = tg  # type: ignore[assignment]

    with (
        patch.object(ig_mod.shutil, "which", return_value="/fake/macmon"),
        patch.object(
            InfoGatherer,
            "_can_read_macmon_metrics",
            new=AsyncMock(return_value=True),
        ),
        patch.object(ig_mod, "open_process", fake_open_process),
        patch.object(ig_mod, "_MACMON_SEND_TIMEOUT", 0.05),
    ):
        try:
            async with asyncio.timeout(1.5):
                await gatherer._monitor_macmon(0.01)
        except TimeoutError:
            pass

    assert sender.attempts >= 1, "the send should have been attempted"
    assert opens >= 2, (
        "the stream was never restarted after the send blocked "
        f"(open_process calls: {opens})"
    )
    assert "_monitor_memory_usage" in tg.started, (
        "the psutil memory fallback should start once the send is detected "
        f"as blocked (started: {tg.started})"
    )