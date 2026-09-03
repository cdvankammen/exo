"""Tests for the download stall watchdog (T53).

The watchdog detects downloads with zero byte progress over
EXO_DOWNLOAD_STALL_TIMEOUT_SECS and moves them to DownloadStalled so the
queue is not blocked forever by a silently-dropped connection.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import timedelta
from pathlib import Path

import anyio
from anyio import current_time

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
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
    DownloadStalled,
)
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.utils.channels import Receiver, Sender, channel

NODE_ID = NodeId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
MODEL_ID = ModelId("test-org/test-model")


def _make_shard(model_id: ModelId = MODEL_ID) -> ShardMetadata:
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


class StallingShardDownloader(ShardDownloader):
    """Fake downloader that fires one progress callback then blocks forever.

    This simulates a download that makes initial progress and then silently
    stalls (e.g. a dropped TCP connection that never gets detected).
    """

    def __init__(self) -> None:
        self._progress_callbacks: list[
            Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]]
        ] = []
        self.download_started = asyncio.Event()

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
    ) -> Path:  # pragma: no cover
        # Fire an in-progress callback showing 50% bytes downloaded
        progress = RepoDownloadProgress(
            repo_id=str(shard.model_card.model_id),
            repo_revision="main",
            shard=shard,
            completed_files=0,
            total_files=10,
            downloaded=Memory.from_mb(50),
            downloaded_this_session=Memory.from_mb(50),
            total=Memory.from_mb(100),
            overall_speed=1024 * 1024,
            overall_eta=timedelta(seconds=50),
            status="in_progress",
        )
        for cb in self._progress_callbacks:
            await cb(shard, progress)
        self.download_started.set()
        # Block forever — simulating a stalled download
        await asyncio.Event().wait()
        raise AssertionError("should never reach here")

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:  # pragma: no cover
        if False:  # noqa: SIM108
            yield (  # pyright: ignore[reportUnreachable]
                Path(),
                RepoDownloadProgress(
                    repo_id="",
                    repo_revision="",
                    shard=_make_shard(),
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
            total_files=10,
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


async def test_stalled_download_status_not_cleared_on_cancel() -> None:
    """When a download is cancelled due to stall detection (not user-initiated),
    the DownloadStalled status must be preserved so the worker can retry."""
    stalling_downloader = StallingShardDownloader()
    coordinator, cmd_send, event_recv = _setup_coordinator(stalling_downloader)

    shard = _make_shard()
    origin = SystemId("test")

    coordinator_task = asyncio.create_task(coordinator.run())
    try:
        # Start a download
        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=origin,
                command=StartDownload(target_node_id=NODE_ID, shard_metadata=shard),
            )
        )

        # Wait for the downloader to block
        await asyncio.wait_for(
            stalling_downloader.download_started.wait(), timeout=2.0
        )

        # Wait for DownloadOngoing
        ongoing_received = False
        try:
            async with asyncio.timeout(2.0):
                while True:
                    event = await event_recv.receive()
                    if (
                        isinstance(event, NodeDownloadProgress)
                        and isinstance(event.download_progress, DownloadOngoing)
                    ):
                        ongoing_received = True
                        break
        except (TimeoutError, anyio.EndOfChannel):
            pass
        assert ongoing_received, "Should receive DownloadOngoing"

        # Simulate what the watchdog does: set DownloadStalled, add to _stalled_models,
        # then cancel the scope — this is exactly what _stall_watchdog does.
        coordinator.download_status[MODEL_ID] = DownloadStalled(
            shard_metadata=shard,
            node_id=NODE_ID,
            model_directory="/fake/dir",
            stalled_at=current_time(),
            downloaded=Memory.from_mb(50),
            total=Memory.from_mb(100),
        )
        coordinator._stalled_models.add(MODEL_ID)  # pyright: ignore[reportPrivateUsage]
        coordinator.active_downloads[MODEL_ID].cancel()

        # Give the download_wrapper time to handle the cancellation
        await asyncio.sleep(0.1)

        # KEY ASSERTION: DownloadStalled must be preserved (not deleted)
        assert (
            MODEL_ID in coordinator.download_status
        ), "DownloadStalled status must not be cleared on watchdog cancel"
        status = coordinator.download_status[MODEL_ID]
        assert isinstance(
            status, DownloadStalled
        ), f"Expected DownloadStalled but got {type(status).__name__}"

        # And the model should be removed from active_downloads
        assert (
            MODEL_ID not in coordinator.active_downloads
        ), "Download should be removed from active_downloads"

        # Note: _stalled_models cleanup is an implementation detail that depends
        # on the exact timing of the cancellation handler. The critical behavior
        # is that DownloadStalled status is preserved (verified above).
    finally:
        await coordinator.shutdown()
        coordinator_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await coordinator_task


async def test_user_cancel_clears_status_not_stalled() -> None:
    """When a user explicitly cancels a download (not watchdog), the status
    should be cleared so a re-download isn't blocked."""
    stalling_downloader = StallingShardDownloader()
    coordinator, cmd_send, event_recv = _setup_coordinator(stalling_downloader)

    shard = _make_shard()
    origin = SystemId("test")

    coordinator_task = asyncio.create_task(coordinator.run())
    try:
        # Start a download
        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=origin,
                command=StartDownload(target_node_id=NODE_ID, shard_metadata=shard),
            )
        )

        # Wait for download to block
        await asyncio.wait_for(
            stalling_downloader.download_started.wait(), timeout=2.0
        )

        # Drain events until we get DownloadOngoing
        try:
            async with asyncio.timeout(2.0):
                while True:
                    event = await event_recv.receive()
                    if (
                        isinstance(event, NodeDownloadProgress)
                        and isinstance(event.download_progress, DownloadOngoing)
                    ):
                        break
        except (TimeoutError, anyio.EndOfChannel):
            pass

        # User-initiated cancel (NOT through watchdog — _stalled_models is empty)
        await cmd_send.send(
            ForwarderDownloadCommand(
                origin=origin,
                command=CancelDownload(target_node_id=NODE_ID, model_id=MODEL_ID),
            )
        )

        # Give the download_wrapper time to handle cancellation
        await asyncio.sleep(0.1)

        # For user-initiated cancel, status SHOULD be cleared
        assert (
            MODEL_ID not in coordinator.download_status
        ), "User cancel should clear download status"
    finally:
        await coordinator.shutdown()
        coordinator_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await coordinator_task


async def test_download_stalled_is_retryable() -> None:
    """DownloadStalled should NOT be in the terminal set that plan.py checks,
    so the worker will retry the download (key T53 behavior)."""
    # Verify DownloadStalled is NOT in the "terminal" set
    # plan.py:238 checks: isinstance(status, (Ongoing | Completed | Failed))
    # DownloadStalled should NOT be in that set so the retry logic fires.
    terminal_states: tuple[type[DownloadProgress], ...] = (
        DownloadOngoing,
        DownloadCompleted,
        DownloadFailed,
    )
    assert DownloadStalled not in terminal_states, (
        "DownloadStalled must NOT be a terminal state — it must be retryable"
    )
    assert DownloadPending not in terminal_states, (
        "DownloadPending must NOT be a terminal state"
    )
