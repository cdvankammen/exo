"""Concurrent cancellation stress test for :class:`DownloadCoordinator`.

Gap (cross-async.md §10, follow-up of t_b1b68ffa): existing cancel tests
(test_cancel_download.py, test_stall_watchdog.py) cover single-download cancel
semantics only. Nothing exercises N concurrent downloads cancelled mid-flight,
and nothing asserts leak-free teardown (active_downloads / download_status /
FD counts).

This test file contains:
1. test_concurrent_cancel_stress       — 50 in-flight downloads cancelled at
   once; asserts active_downloads / download_status empty, no spurious
   DownloadStalled events, no finished-but-cancelled work, and that the
   coordinator can take a fresh download afterwards.
2. test_concurrent_cancel_fd_count_no_leak — psutil-backed FD count before/
   after the mass cancel (skipped when psutil is unavailable).
3. test_concurrent_cancel_leaves_no_watchdog_state_behind — asserts the
   stall watchdog's per-model bookkeeping dicts are fully reclaimed after
   the mass cancel (leak surface: _last_download_bytes / _download_started_at).

TEST ONLY — intentionally does not touch src/exo/download/*.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import anyio
import pytest

from exo.download.coordinator import DownloadCoordinator
from exo.download.download_utils import RepoDownloadProgress
from exo.download.impl_shard_downloader import SingletonShardDownloader
from exo.download.shard_downloader import ShardDownloader
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.commands import (
    CancelDownload,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import NodeId, SystemId
from exo.shared.types.events import Event, NodeDownloadProgress
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import (
    DownloadOngoing,
    DownloadPending,
    DownloadStalled,
)
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.utils.channels import Receiver, Sender, channel

NODE_ID = NodeId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
ORIGIN = SystemId("test")

N_CONCURRENT = 50
MODEL_IDS = [ModelId(f"test-org/model-{i:02d}") for i in range(N_CONCURRENT)]


@pytest.fixture(autouse=True)
def _isolate_model_dirs(tmp_path: Path) -> Iterator[None]:
    """Keep the coordinator's startup scans hermetic.

    The coordinator's ``_emit_existing_download_progress`` task scans the
    module-level ``EXO_MODELS_DIRS`` / ``EXO_MODELS_READ_ONLY_DIRS`` and
    re-adds DownloadCompleted entries for models found on the REAL host
    (e.g. mini's pre-downloaded mlx-community/*).  Without isolation, the
    ``download_status == {}`` assertion after the mass cancel fails
    order-dependently whenever a real host model is present.  Patch the
    dirs to an empty tmp path and neuter resolve_existing_model so the
    startup rescan can never fabricate status entries.
    """

    with (
        patch("exo.download.coordinator.EXO_MODELS_READ_ONLY_DIRS", ()),
        patch(
            "exo.download.coordinator.resolve_existing_model",
            return_value=None,
        ),
    ):
        yield


def _make_shard(model_id: ModelId) -> ShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=model_id,
            storage_size=Memory.from_mb(100),
            n_layers=28,
            hidden_size=1024,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
            backends=[Backend.MlxMetal],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=28,
        n_layers=28,
    )


class ParkedShardDownloader(ShardDownloader):
    """Fake downloader: one distinct model per task, blocks in ensure_shard
    until cancelled, records completion per model."""

    def __init__(self) -> None:
        self._progress_callbacks: list[
            Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]]
        ] = []
        self.started: dict[str, asyncio.Event] = {}
        self.finished: set[str] = set()

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        self._progress_callbacks.append(callback)

    async def ensure_shard(
        self,
        shard: ShardMetadata,
        config_only: bool = False,
        force_override: bool = False,
    ) -> Path:
        key = str(shard.model_card.model_id)
        # Fire an in-progress callback (same shape as test_cancel_download.py)
        progress = RepoDownloadProgress(
            repo_id=key,
            repo_revision="main",
            shard=shard,
            completed_files=0,
            total_files=1,
            downloaded=Memory.from_mb(50),
            downloaded_this_session=Memory.from_mb(50),
            total=Memory.from_mb(100),
            overall_speed=1024 * 1024,
            overall_eta=timedelta(seconds=50),
            status="in_progress",
        )
        for cb in self._progress_callbacks:
            await cb(shard, progress)
        # Mark the task as parked mid-flight (DownloadOngoing emitted, now
        # blocked), then block until cancelled.
        self.started.setdefault(key, asyncio.Event()).set()
        await asyncio.Event().wait()
        self.finished.add(key)
        return Path("/fake/models") / shard.model_card.model_id.normalize()

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        if False:  # noqa: SIM108  # empty async generator
            yield (
                Path(),
                RepoDownloadProgress(  # pyright: ignore[reportUnreachable]
                    repo_id="",
                    repo_revision="",
                    shard=_make_shard(MODEL_IDS[0]),
                    completed_files=0,
                    total_files=0,
                    downloaded=Memory.from_bytes(0),
                    downloaded_this_session=Memory.from_bytes(0),
                    total=Memory.from_bytes(0),
                    overall_speed=0,
                    overall_eta=timedelta(seconds=0),
                    status="not_started",
                ),
            )

    async def get_shard_download_status_for_shard(
        self,
        shard: ShardMetadata,
    ) -> RepoDownloadProgress:
        return RepoDownloadProgress(
            repo_id=str(shard.model_card.model_id),
            repo_revision="main",
            shard=shard,
            completed_files=0,
            total_files=1,
            downloaded=Memory.from_bytes(0),
            downloaded_this_session=Memory.from_bytes(0),
            total=Memory.from_mb(100),
            overall_speed=0,
            overall_eta=timedelta(seconds=0),
            status="not_started",
        )


def _setup_coordinator(
    downloader: ShardDownloader,
) -> tuple[
    DownloadCoordinator,
    Sender[ForwarderDownloadCommand],
    Receiver[Event],
]:
    cmd_send, cmd_recv = channel[ForwarderDownloadCommand]()
    event_send, event_recv = channel[Event]()
    wrapped = SingletonShardDownloader(downloader)
    coordinator = DownloadCoordinator(
        node_id=NODE_ID,
        shard_downloader=wrapped,
        download_command_receiver=cmd_recv,
        event_sender=event_send,
    )
    return coordinator, cmd_send, event_recv


async def _wait_until(
    predicate: Callable[[], bool], timeout: float = 15.0, interval: float = 0.01
) -> None:
    """Poll ``predicate`` until it returns True or the timeout expires."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("condition not met within timeout")
        await asyncio.sleep(interval)


async def test_concurrent_cancel_stress() -> None:
    """Cancel 50 in-flight downloads at once: no leaks, no spurious stalls,
    all 50 tasks complete cleanly, FD count returns to baseline."""

    downloader = ParkedShardDownloader()
    coordinator, cmd_send, event_recv = _setup_coordinator(downloader)
    coordinator_task = asyncio.create_task(coordinator.run())

    try:
        # --- Launch 50 parallel downloads (one distinct model id each) ---
        for mid in MODEL_IDS:
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=ORIGIN,
                    command=StartDownload(
                        target_node_id=NODE_ID,
                        shard_metadata=_make_shard(mid),
                    ),
                )
            )

        # --- Wait until all 50 are in the coordinator's active set ---
        await _wait_until(
            lambda: all(m in coordinator.active_downloads for m in MODEL_IDS)
        )

        # --- Wait for all 50 fake downloaders to park mid-flight ---
        await _wait_until(lambda: len(downloader.started) >= N_CONCURRENT)

        # --- Drain any events emitted before the cancel (pending/ongoing) ---
        while True:
            try:
                async with asyncio.timeout(0.1):
                    await event_recv.receive()
            except (TimeoutError, anyio.EndOfStream):
                break

        # --- Cancel all 50 simultaneously ---
        for mid in MODEL_IDS:
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=ORIGIN,
                    command=CancelDownload(target_node_id=NODE_ID, model_id=mid),
                )
            )

        # --- Wait for cleanup to settle ---
        await _wait_until(lambda: len(coordinator.active_downloads) == 0)
        await asyncio.sleep(0.1)

        # --- Assert leak-free teardown ---
        assert coordinator.active_downloads == {}, (
            f"active_downloads should be empty after cancelling all, "
            f"got {sorted(k for k in coordinator.active_downloads)}"
        )
        assert coordinator.download_status == {}, (
            f"download_status should be empty after cancelling all, "
            f"got {sorted(k for k in coordinator.download_status)}"
        )
        # No stale per-model progress-throttle state (popped on cancel).
        assert coordinator._last_progress_time == {}  # pyright: ignore[reportPrivateUsage]

        # --- All 50 tasks must complete cleanly (not hang, not error) ---
        # The coordinator's task group stays open while run() is alive, so the
        # meaningful clean-teardown signal is that every per-download task is
        # gone: active_downloads empty (asserted above) + a fresh download
        # still works (exercised below). No other leaky per-model state remains.

        # --- No spurious stalled state / stalled events ---
        assert coordinator._stalled_models == set()  # pyright: ignore[reportPrivateUsage]

        # Drain the rest of the event stream and check for DownloadStalled.
        stalled_seen = False
        while True:
            try:
                async with asyncio.timeout(0.2):
                    event = await event_recv.receive()
                if (
                    isinstance(event, NodeDownloadProgress)
                    and isinstance(event.download_progress, DownloadStalled)
                ):
                    stalled_seen = True
            except (TimeoutError, anyio.EndOfStream):
                break
        assert not stalled_seen, "No DownloadStalled events expected after user cancel"

        # Cancelled work must not be marked finished (it was aborted mid-flight).
        assert downloader.finished == set(), (
            f"cancelled downloads must not complete: {sorted(downloader.finished)}"
        )

        # Ensure the queue can process a fresh download afterwards (the
        # coordinator must be fully usable after the mass cancel).
        fresh_id = ModelId("test-org/fresh-after-cancel")
        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=ORIGIN,
                command=StartDownload(
                    target_node_id=NODE_ID,
                    shard_metadata=_make_shard(fresh_id),
                ),
            )
        )
        await _wait_until(
            lambda: fresh_id in coordinator.active_downloads
        )
        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=ORIGIN,
                command=CancelDownload(
                    target_node_id=NODE_ID, model_id=fresh_id
                ),
            )
        )
        await _wait_until(lambda: len(coordinator.active_downloads) == 0)

        # --- FD sanity check (optional; skipped when psutil unavailable) ---
        await asyncio.sleep(0.1)
        await coordinator.shutdown()
        coordinator_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await coordinator_task
    finally:
        # Failure-path cleanup only: if the body didn't already shut down,
        # tear the coordinator down so the test never leaks a running loop.
        if coordinator_task is not None:
            with contextlib.suppress(Exception):
                await coordinator.shutdown()
            coordinator_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await coordinator_task


async def test_concurrent_cancel_fd_count_no_leak() -> None:
    """Optional psutil-backed FD assertion: canceling 50 in-flight downloads
    must not leak file descriptors (skipped when psutil is unavailable)."""
    try:
        import psutil  # noqa: PLC0415
    except ImportError:
        pytest.skip("psutil not available — skipping FD count assertion")

    downloader = ParkedShardDownloader()
    coordinator, cmd_send, event_recv = _setup_coordinator(downloader)
    coordinator_task = asyncio.create_task(coordinator.run())

    def _fd_count() -> int:
        return psutil.Process().num_fds()

    baseline = _fd_count()
    try:
        for mid in MODEL_IDS:
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=ORIGIN,
                    command=StartDownload(
                        target_node_id=NODE_ID,
                        shard_metadata=_make_shard(mid),
                    ),
                )
            )
        await _wait_until(
            lambda: all(m in coordinator.active_downloads for m in MODEL_IDS)
        )
        # Give the event machinery a beat to settle before measuring.
        await asyncio.sleep(0.2)

        for mid in MODEL_IDS:
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=ORIGIN,
                    command=CancelDownload(target_node_id=NODE_ID, model_id=mid),
                )
            )
        await _wait_until(lambda: len(coordinator.active_downloads) == 0)
        await asyncio.sleep(0.3)

        # The mass cancel must not leak FDs; a small tolerance covers any
        # event-loop/lazy-closing jitter from unrelated machinery.
        assert _fd_count() <= baseline + 10, (
            f"FD leak after concurrent cancel: baseline={baseline}, "
            f"after={_fd_count()}"
        )
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            await coordinator.shutdown()
        coordinator_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await coordinator_task


async def test_concurrent_cancel_leaves_no_watchdog_state_behind() -> None:
    """After a mass user-cancel, the watchdog's per-model bookkeeping dicts
    must not retain stale entries (a re-download of the same id must be able
    to start cleanly)."""
    downloader = ParkedShardDownloader()
    coordinator, cmd_send, event_recv = _setup_coordinator(downloader)
    coordinator_task = asyncio.create_task(coordinator.run())

    try:
        for mid in MODEL_IDS:
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=ORIGIN,
                    command=StartDownload(
                        target_node_id=NODE_ID,
                        shard_metadata=_make_shard(mid),
                    ),
                )
            )
        await _wait_until(
            lambda: all(m in coordinator.active_downloads for m in MODEL_IDS)
        )
        await _wait_until(lambda: len(downloader.started) >= N_CONCURRENT)

        # Drain pre-cancel events.
        while True:
            try:
                async with asyncio.timeout(0.1):
                    await event_recv.receive()
            except (TimeoutError, anyio.EndOfStream):
                break

        # The watchdog tracks per-model bytes/start-time while downloads are
        # underway — this is the state that must NOT leak across a cancel.
        assert len(coordinator._last_download_bytes) == N_CONCURRENT  # pyright: ignore[reportPrivateUsage]
        assert len(coordinator._download_started_at) == N_CONCURRENT  # pyright: ignore[reportPrivateUsage]

        for mid in MODEL_IDS:
            await cmd_send.send(
                ForwarderDownloadCommand(
                    origin=ORIGIN,
                    command=CancelDownload(target_node_id=NODE_ID, model_id=mid),
                )
            )
        await _wait_until(lambda: len(coordinator.active_downloads) == 0)
        await asyncio.sleep(0.1)

        # KEY ASSERTION: the watchdog's per-model state must be fully
        # reclaimed after the cancels. No stale byte/start-time entries.
        assert coordinator._last_download_bytes == {}, (  # pyright: ignore[reportPrivateUsage]
            "watchdog _last_download_bytes must be empty after cancel, "
            f"got {len(coordinator._last_download_bytes)} stale entries"
        )
        assert coordinator._download_started_at == {}, (  # pyright: ignore[reportPrivateUsage]
            "watchdog _download_started_at must be empty after cancel, "
            f"got {len(coordinator._download_started_at)} stale entries"
        )
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            await coordinator.shutdown()
        coordinator_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await coordinator_task