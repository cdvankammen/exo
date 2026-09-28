"""E2E: a node that joins late catches up from a state snapshot, not a replay.

Exercises the real Master, the real EventRouter and the real apply(); the
master->node hop goes through the real STATE_SNAPSHOTS topic serialization,
so a passing test means the snapshot survives the wire format, not just the
objects in memory.
"""

import anyio
import pytest
from loguru import logger

from exo.master.main import REPLAYABLE_EVENTS, Master
from exo.master.tests.e2e_snapshot_fixtures import N, event_for
from exo.routing.event_router import EventRouter
from exo.routing.router import get_node_zid
from exo.routing.topics import GLOBAL_EVENTS, STATE_SNAPSHOTS
from exo.shared.apply import apply
from exo.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    RequestEventLog,
)
from exo.shared.types.common import SessionId, SystemId
from exo.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    LocalForwarderEvent,
    StateSnapshot,
)
from exo.shared.types.state import State
from exo.utils.channels import Sender, channel

logger.remove()

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_late_node_catches_up_from_snapshot():
    master_node = get_node_zid()
    session = SessionId(master_node_id=master_node, election_clock=0)

    # Master side.
    le_send, le_recv = channel[LocalForwarderEvent]()
    cmd_send, cmd_recv = channel[ForwarderCommand]()
    ev_send, _ = channel[Event]()
    dl_send, _ = channel[ForwarderDownloadCommand]()
    ge_send, ge_recv = channel[GlobalForwarderEvent]()
    snap_send, snap_recv = channel[StateSnapshot]()

    # Node side: the node's inbound only ever sees what arrives after it
    # attaches, which is why it needs a snapshot rather than a replay.
    late_ge_send, late_ge_recv = channel[GlobalForwarderEvent | StateSnapshot]()
    late_snap_send, late_snap_recv = channel[StateSnapshot]()
    late_cmd_send, _late_cmd_recv = channel[ForwarderCommand]()
    late_out, _late_out_recv = channel[LocalForwarderEvent]()

    master = Master(
        master_node,
        session,
        event_sender=ev_send,
        global_event_sender=ge_send,
        snapshot_sender=snap_send,
        local_event_receiver=le_recv,
        command_receiver=cmd_recv,
        download_command_sender=dl_send,
    )
    router = EventRouter(
        session_id=session,
        command_sender=late_cmd_send,
        external_outbound=late_out,
        external_inbound=late_ge_recv,  # type: ignore[arg-type]
        snapshot_inbound=late_snap_recv,
    )
    # receiver() must be called before the router's task group starts, so no
    # is_running() probe is needed here: calling it either raises or doesn't.
    updates = router.receiver()
    # SystemId is generated per instance and not settable, so a snapshot is
    # only accepted by the router that asked for it. Read the router's own id
    # once, so the ask we forge below is addressed to this router.
    node_id = router._system_id  # pyright: ignore[reportPrivateUsage]

    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)
        tg.start_soon(router.run)

        # One producer for every event: MultiSourceBuffer sequences per source,
        # so a fresh SystemId per event would stall the master at index 0.
        producer = SystemId()
        for i in range(N):
            await le_send.send(
                LocalForwarderEvent(
                    origin_idx=i,
                    origin=producer,
                    session=session,
                    event=event_for(i, master_node),
                )
            )
        with anyio.fail_after(60):
            while master.state.last_event_applied_idx < N - 1:
                await anyio.sleep(0.01)
        assert master.state.last_event_applied_idx == N - 1

        # The node attaches now: it asks for the whole log and nacks until
        # caught up. It has already missed more than REPLAYABLE_EVENTS.
        observed: list[str] = []
        nacks: list[int] = []
        box: dict[str, StateSnapshot] = {}

        async def relay_events():
            with ge_recv as src, late_ge_send as dst:
                async for ev in src:
                    # Only post-attach traffic reaches the node in reality.
                    if ev.origin_idx >= N:
                        await dst.send(
                            GLOBAL_EVENTS.deserialize(GLOBAL_EVENTS.serialize(ev))
                        )

        async def relay_snapshots():
            with snap_recv as src, late_snap_send as dst:
                async for snap in src:
                    await dst.send(
                        STATE_SNAPSHOTS.deserialize(STATE_SNAPSHOTS.serialize(snap))
                    )

        async def watch_nacks():
            # Forward the node's RequestEventLog to the master, and record it:
            # this is the node's only way to ask for what it missed.
            with _late_cmd_recv as cmds:
                async for c in cmds:
                    if isinstance(c.command, RequestEventLog):
                        nacks.append(c.command.since_idx)
                        await cmd_send.send(c)

        async def consume():
            state: State = State()
            with updates as items:
                async for item in items:
                    if isinstance(item, StateSnapshot):
                        state = item.state
                        box["snapshot"] = item
                        observed.append(f"snapshot@{state.last_event_applied_idx}")
                    else:
                        state = apply(state, item)
                        observed.append(f"event@{state.last_event_applied_idx}")

        tg.start_soon(consume)
        tg.start_soon(watch_nacks)
        tg.start_soon(relay_snapshots)
        tg.start_soon(relay_events)

        # A late node's first ask is for the index it is actually missing, which
        # is outside the master's replay window — that is what makes the master
        # answer with a snapshot instead of a replay.
        await cmd_send.send(
            ForwarderCommand(
                origin=node_id,
                command=RequestEventLog(since_idx=N - REPLAYABLE_EVENTS - 1),
            )
        )

        # Wait for the snapshot before any more traffic: the master answers
        # with whatever its state is at that moment, so sending live events
        # first would make the snapshot index non-deterministic.
        with anyio.fail_after(30):
            while "snapshot" not in box:
                await anyio.sleep(0.01)
        assert box["snapshot"].state.last_event_applied_idx == N - 1

        # Now live events flow, and they must continue contiguously from the
        # snapshot rather than triggering another catch-up request.
        for i in range(N, N + 3):
            await le_send.send(
                LocalForwarderEvent(
                    origin_idx=i,
                    origin=producer,
                    session=session,
                    event=event_for(i, master_node),
                )
            )

        with anyio.fail_after(30):
            while len(observed) < 4:
                await anyio.sleep(0.01)

        # Exactly one snapshot, taken at the node's requested position, and it
        # is what the node had been missing.
        assert sum(o.startswith("snapshot@") for o in observed) == 1, observed
        assert observed[0] == f"snapshot@{N - 1}", observed
        # After the catch-up the node is contiguous, so it must not ask again.
        assert nacks == [], nacks
        # And the live events after it stay contiguous: no gap, no re-request.
        assert [o for o in observed if o.startswith("event@")] == [
            f"event@{N}",
            f"event@{N + 1}",
            f"event@{N + 2}",
        ], observed

        # The snapshot survived the real topic serialization, and apply()'s
        # index continuity check holds on top of it.
        received = box["snapshot"]
        assert received.requester == node_id
        assert received.session == session
        restored = STATE_SNAPSHOTS.deserialize(STATE_SNAPSHOTS.serialize(received))
        assert restored.state.last_event_applied_idx == N - 1
        assert restored.state.model_dump_json() == received.state.model_dump_json()
        nxt = apply(
            restored.state, IndexedEvent(idx=N, event=event_for(N, master_node))
        )
        assert nxt.last_event_applied_idx == N

        router.shutdown()
        await master.shutdown()
        tg.cancel_scope.cancel()


async def test_in_window_request_is_replayed_not_snapshot():
    """A node only slightly behind still gets events, never a snapshot."""
    master_node = get_node_zid()
    session = SessionId(master_node_id=master_node, election_clock=0)

    le_send, le_recv = channel[LocalForwarderEvent]()
    cmd_send, cmd_recv = channel[ForwarderCommand]()
    ev_send, _ = channel[Event]()
    dl_send, _ = channel[ForwarderDownloadCommand]()
    ge_send, ge_recv = channel[GlobalForwarderEvent]()
    snap_send, snap_recv = channel[StateSnapshot]()
    # Live broadcasts go nowhere; we only observe replay replies.
    live_send, _live_drain = channel[GlobalForwarderEvent]()

    class _ReplayTee:
        """Splits the master's one global sender into live vs replay traffic.

        A replay reply re-sends an index the master already broadcast live, so
        the second sighting of an index is the replay.
        """

        def __init__(
            self,
            live: Sender[GlobalForwarderEvent],
            replay: Sender[GlobalForwarderEvent],
        ) -> None:
            self._live, self._replay = live, replay
            self.broadcast_idxs: set[int] = set()

        async def send(self, ev: GlobalForwarderEvent) -> None:
            if ev.origin_idx in self.broadcast_idxs:
                await self._replay.send(ev)
            else:
                self.broadcast_idxs.add(ev.origin_idx)
                await self._live.send(ev)

        def close(self) -> None:
            self._live.close()
            self._replay.close()

    master = Master(
        master_node,
        session,
        event_sender=ev_send,
        global_event_sender=_ReplayTee(live_send, ge_send),  # type: ignore[arg-type]
        snapshot_sender=snap_send,
        local_event_receiver=le_recv,
        command_receiver=cmd_recv,
        download_command_sender=dl_send,
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)

        producer = SystemId()
        for i in range(N):
            await le_send.send(
                LocalForwarderEvent(
                    origin_idx=i,
                    origin=producer,
                    session=session,
                    event=event_for(i, master_node),
                )
            )
        with anyio.fail_after(60):
            while master.state.last_event_applied_idx < N - 1:
                await anyio.sleep(0.01)
        assert master.state.last_event_applied_idx == N - 1
        # The log is exactly at the retention window, so "one before the
        # oldest retained" is a real index a node could still be missing.
        retained = len(master._recent_events)  # pyright: ignore[reportPrivateUsage]
        assert retained == REPLAYABLE_EVENTS

        async def ask(since: int) -> None:
            await cmd_send.send(
                ForwarderCommand(
                    origin=SystemId(), command=RequestEventLog(since_idx=since)
                )
            )

        with ge_recv as src, snap_recv as snaps:
            # Three behind: the smallest in-window case, replayed verbatim.
            await ask(N - 3)
            got: list[int] = []
            with anyio.fail_after(30):
                while len(got) < 3:
                    got.append((await src.receive()).origin_idx)
            assert got == [N - 3, N - 2, N - 1], got
            assert snaps.collect() == [], "snapshotted an in-window request"

            # One below the oldest retained event: a snapshot instead.
            await ask(N - REPLAYABLE_EVENTS - 1)
            with anyio.fail_after(30):
                snaps_seen = []
                while not snaps_seen:
                    await anyio.sleep(0.01)
                    snaps_seen = snaps.collect()
            assert len(snaps_seen) == 1, snaps_seen
            assert snaps_seen[0].state.last_event_applied_idx == N - 1

            # Past the end of the log: ignored, and in particular no snapshot.
            await ask(N + 50)
            await anyio.sleep(0.5)
            assert snaps.collect() == [], "answered a request from the future"

        await master.shutdown()
        tg.cancel_scope.cancel()
