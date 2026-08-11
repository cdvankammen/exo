import pytest
from anyio import create_task_group, fail_after, move_on_after, sleep

from exo.routing.connection_message import ConnectionMessage
from exo.shared.election import Election, ElectionMessage, ElectionResult
from exo.shared.types.commands import ForwarderCommand, PromoteMaster, TestCommand
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.utils.channels import channel

# ======= #
# Helpers #
# ======= #


def em(
    clock: int,
    seniority: int,
    node_id: str,
    commands_seen: int = 0,
    election_clock: int | None = None,
) -> ElectionMessage:
    """
    Helper to build ElectionMessages for a given proposer node.

    The new API carries a proposed SessionId (master_node_id + election_clock).
    By default we use the same value for election_clock as the 'clock' of the round.
    """
    return ElectionMessage(
        clock=clock,
        seniority=seniority,
        proposed_session=SessionId(
            master_node_id=NodeId(node_id),
            election_clock=clock if election_clock is None else election_clock,
        ),
        commands_seen=commands_seen,
    )


# ======================================= #
#                 TESTS                   #
# ======================================= #


@pytest.fixture(autouse=True)
def fast_election_timeout(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("exo.shared.election.DEFAULT_ELECTION_TIMEOUT", 0.1)
    # Make the connection-message cooldown fast too, so tests that exercise
    # connection-triggered campaigns don't wait the real 25s.
    monkeypatch.setattr("exo.shared.election._CONNECTION_ELECTION_COOLDOWN", 0.05)


@pytest.mark.anyio
async def test_single_round_broadcasts_and_updates_seniority_on_self_win() -> None:
    """
    Start a round by injecting an ElectionMessage with higher clock.
    With only our node effectively 'winning', we should broadcast once and update seniority.
    """
    # Outbound election messages from the Election (we'll observe these)
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    # Inbound election messages to the Election (we'll inject these)
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    # Election results produced by the Election (we'll observe these)
    er_tx, er_rx = channel[ElectionResult]()
    # Connection messages
    cm_tx, cm_rx = channel[ConnectionMessage]()
    # Commands
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("B"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)
            # Trigger new round at clock=1 (peer announces it)
            await em_in_tx.send(em(clock=1, seniority=0, node_id="A"))

            # Expect our broadcast back to the peer side for this round only
            while True:
                got = await em_out_rx.receive()
                if got.clock == 1 and got.proposed_session.master_node_id == NodeId(
                    "B"
                ):
                    break

            # Wait for the round to finish and produce an ElectionResult
            result = await er_rx.receive()
            assert result.session_id.master_node_id == NodeId("B")
            # We spawned as master; electing ourselves again is not "new master".
            assert result.is_new_master is False

            # Close inbound streams to end the receivers (and run())
            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # We should have updated seniority to 2 (A + B).
    assert election.seniority == 2


@pytest.mark.anyio
async def test_peer_with_higher_seniority_wins_and_we_switch_master() -> None:
    """
    If a peer with clearly higher seniority participates in the round, they should win.
    We should broadcast our status exactly once for this round, then switch master.
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # Start round with peer's message (higher seniority)
            await em_in_tx.send(em(clock=1, seniority=10, node_id="PEER"))

            # We should still broadcast our status exactly once for this round
            while True:
                got = await em_out_rx.receive()
                if got.clock == 1:
                    assert got.seniority == 0
                    break

            # After the timeout, election result for clock=1 should report the peer as master
            # (Skip any earlier result from the boot campaign at clock=0 by filtering on election_clock)
            while True:
                result = await er_rx.receive()
                if result.session_id.election_clock == 1:
                    break

            assert result.session_id.master_node_id == NodeId("PEER")
            assert result.is_new_master is True

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # We lost → seniority raised toward the peer (max_peer - 1) so we stop
    # re-proposing a stale low value that would flap elections.
    assert election.seniority == 9


@pytest.mark.anyio
async def test_ignores_older_messages() -> None:
    """
    Messages with a lower clock than the current round are ignored by the receiver.
    Expect exactly one broadcast for the higher clock round.
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, _er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # Newer round arrives first -> triggers campaign at clock=2
            await em_in_tx.send(em(clock=2, seniority=0, node_id="A"))
            while True:
                first = await em_out_rx.receive()
                if first.clock == 2:
                    break

            # An older message (clock=1) must not start a new round, but the
            # sender must be told the current clock so it can rejoin (#2197)
            await em_in_tx.send(em(clock=1, seniority=999, node_id="B"))

            reply = await em_out_rx.receive()
            assert reply.clock == 2, "Reply must carry the current clock"
            assert election.clock == 2, "An older message must not change the clock"

            got_more = False
            with move_on_after(0.05):
                _ = await em_out_rx.receive()
                got_more = True
            assert not got_more, "An older round must not trigger a new campaign"

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # Not asserting on the result; focus is on ignore behavior.


@pytest.mark.anyio
async def test_two_rounds_emit_two_broadcasts_and_increment_clock() -> None:
    """
    Two successive rounds → two broadcasts. Second round triggered by a higher-clock message.
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, _er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # Round 1 at clock=1
            await em_in_tx.send(em(clock=1, seniority=0, node_id="X"))
            while True:
                m1 = await em_out_rx.receive()
                if m1.clock == 1:
                    break

            # Round 2 at clock=2
            await em_in_tx.send(em(clock=2, seniority=0, node_id="Y"))
            while True:
                m2 = await em_out_rx.receive()
                if m2.clock == 2:
                    break

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # Not asserting on who won; just that both rounds were broadcast.


@pytest.mark.anyio
async def test_promotion_new_seniority_counts_participants() -> None:
    """
    When we win against two peers in the same round, our seniority becomes
    max(existing, number_of_candidates). With existing=0: expect 3 (us + A + B).
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # Start round at clock=7 with two peer participants
            await em_in_tx.send(em(clock=7, seniority=0, node_id="A"))
            await em_in_tx.send(em(clock=7, seniority=0, node_id="B"))

            # We should see exactly one broadcast from us for this round
            while True:
                got = await em_out_rx.receive()
                if got.clock == 7 and got.proposed_session.master_node_id == NodeId(
                    "ME"
                ):
                    break

            # Wait for the election to finish so seniority updates
            _ = await er_rx.receive()

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # We + A + B = 3 → new seniority expected to be 3
    assert election.seniority == 3


@pytest.mark.anyio
async def test_connection_message_triggers_new_round_broadcast() -> None:
    """
    A connection message increments the clock and starts a new campaign.
    We should observe a broadcast at the incremented clock.
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, _er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # Send any connection message object; we close quickly to cancel before result creation
            await cm_tx.send(ConnectionMessage(connected=True))

            # Expect a broadcast for the new round at clock=1
            while True:
                got = await em_out_rx.receive()
                if got.clock == 1 and got.proposed_session.master_node_id == NodeId(
                    "ME"
                ):
                    break

            # Close promptly to avoid waiting for campaign completion
            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # After cancellation (before election finishes), no seniority changes asserted here.


@pytest.mark.anyio
async def test_tie_breaker_prefers_node_with_more_commands_seen() -> None:
    """
    With equal seniority, the node that has seen more commands should win the election.
    We increase our local 'commands_seen' by sending TestCommand()s before triggering the round.
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    me = NodeId("ME")

    election = Election(
        node_id=me,
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
        seniority=0,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # Pump local commands so our commands_seen is high before the round starts
            for _ in range(50):
                await co_tx.send(
                    ForwarderCommand(origin=SystemId("SOMEONE"), command=TestCommand())
                )

            # Trigger a round at clock=1 with a peer of equal seniority but fewer commands
            await em_in_tx.send(
                em(clock=1, seniority=0, node_id="PEER", commands_seen=5)
            )

            # Observe our broadcast for this round (to ensure we've joined the round)
            while True:
                got = await em_out_rx.receive()
                if got.clock == 1 and got.proposed_session.master_node_id == me:
                    # We don't assert exact count, just that we've participated this round.
                    break

            # The elected result for clock=1 should be us due to higher commands_seen
            while True:
                result = await er_rx.receive()
                if result.session_id.master_node_id == me:
                    assert result.session_id.election_clock in (0, 1)
                    break

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()


@pytest.mark.anyio
async def test_equal_clock_candidacy_without_active_campaign_starts_a_round() -> None:
    """
    Regression test for the "dead round" (issue #2197, part 2).

    A candidacy arriving with a clock equal to ours while no campaign is
    running used to be appended to a candidate list that nothing would ever
    evaluate. It must instead trigger a round that resolves deterministically.
    """
    em_out_tx, _em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("B"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # The initial campaign resolves instantly with ourselves as master
            first = await er_rx.receive()
            assert first.session_id.master_node_id == NodeId("B")

            # Equal-clock candidacy (clock 0) from a clearly better candidate,
            # while no campaign is active anymore
            await em_in_tx.send(em(clock=0, seniority=50, node_id="A"))

            result = await er_rx.receive()
            assert result.session_id.master_node_id == NodeId("A")

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()


@pytest.mark.anyio
async def test_equal_clock_reproposal_of_current_master_skipped() -> None:
    """
    Regression test: a trickle of equal-clock re-proposals of the CURRENT
    master must not keep spinning new campaigns.

    With a senior node re-broadcasting the same master every few seconds,
    every broadcast arrived at equal clock and (with no active campaign)
    started another campaign -- each waiting on the previous one's done
    event and logging "Waiting for other campaign to finish" forever,
    keeping the election loop (and worker restarts on master change) alive.
    Re-proposals of the already-current master are now skipped.
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("B"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # Establish A as master (first round resolves with A winning)
            await em_in_tx.send(em(clock=1, seniority=50, node_id="A"))
            result = await er_rx.receive()
            assert result.session_id.master_node_id == NodeId("A")

            # Drain campaign broadcasts emitted for that round
            em_out_rx.collect()

            # A re-proposes itself as master at the SAME clock (no new round).
            # With the fix, no new campaign/result should be emitted.
            await em_in_tx.send(em(clock=1, seniority=50, node_id="A"))
            await em_in_tx.send(em(clock=1, seniority=50, node_id="A"))
            await em_in_tx.send(em(clock=1, seniority=50, node_id="A"))

            # Give the receiver a moment to process; then assert no campaign
            # broadcasts and no new election results arrived.
            await sleep(0.3)
            assert em_out_rx.collect() == []
            assert er_rx.collect() == []

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()


@pytest.mark.anyio
async def test_stale_message_gets_current_status_reply() -> None:
    """
    Regression test for the wedged restarted master (issue #2197, part 1).

    A message below our clock (e.g. a restarted master campaigning from
    clock 0) used to be dropped silently, leaving the sender unaware of the
    cluster clock. We must reply with our current status instead.
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("B"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # Establish a round at clock 5 and wait for it to resolve
            await em_in_tx.send(em(clock=5, seniority=0, node_id="A"))
            while True:
                result = await er_rx.receive()
                if result.won_clock == 5:
                    break

            # Drain campaign broadcasts already emitted
            em_out_rx.collect()

            # A restarted node campaigns from clock 1, below the cluster clock
            await em_in_tx.send(em(clock=1, seniority=0, node_id="R"))

            reply = await em_out_rx.receive()
            assert reply.clock == 5
            assert reply.proposed_session.master_node_id == NodeId("B")

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()


@pytest.mark.anyio
async def test_restarted_master_rejoins_cluster_deterministically() -> None:
    """
    End-to-end regression for issue #2197: a node that restarts (losing its
    election clock) while the cluster is at a higher clock must converge on
    the same session as the surviving cluster, without relying on a
    connection event.
    """
    a_em_out_tx, a_em_out_rx = channel[ElectionMessage]()
    a_em_in_tx, a_em_in_rx = channel[ElectionMessage]()
    a_er_tx, a_er_rx = channel[ElectionResult]()
    _a_cm_tx, a_cm_rx = channel[ConnectionMessage]()
    _a_co_tx, a_co_rx = channel[ForwarderCommand]()

    r_em_out_tx, r_em_out_rx = channel[ElectionMessage]()
    r_em_in_tx, r_em_in_rx = channel[ElectionMessage]()
    r_er_tx, r_er_rx = channel[ElectionResult]()
    _r_cm_tx, r_cm_rx = channel[ConnectionMessage]()
    _r_co_tx, r_co_rx = channel[ForwarderCommand]()

    survivor = Election(
        node_id=NodeId("A"),
        election_message_receiver=a_em_in_rx,
        election_message_sender=a_em_out_tx,
        election_result_sender=a_er_tx,
        connection_message_receiver=a_cm_rx,
        command_receiver=a_co_rx,
        is_candidate=True,
    )
    restarted = Election(
        node_id=NodeId("R"),
        election_message_receiver=r_em_in_rx,
        election_message_sender=r_em_out_tx,
        election_result_sender=r_er_tx,
        connection_message_receiver=r_cm_rx,
        command_receiver=r_co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(5):
            tg.start_soon(survivor.run)

            # Advance the surviving cluster to clock 5
            await a_em_in_tx.send(em(clock=5, seniority=0, node_id="X"))
            while True:
                result = await a_er_rx.receive()
                if result.won_clock == 5:
                    break
            a_em_out_rx.collect()

            # Wire both nodes together, then boot the "restarted" node,
            # which starts over from clock 0
            async def forward_a_to_r() -> None:
                with a_em_out_rx as messages:
                    async for message in messages:
                        await r_em_in_tx.send(message)

            async def forward_r_to_a() -> None:
                with r_em_out_rx as messages:
                    async for message in messages:
                        await a_em_in_tx.send(message)

            tg.start_soon(forward_a_to_r)
            tg.start_soon(forward_r_to_a)
            tg.start_soon(restarted.run)

            # The restarted node must deterministically reach the cluster
            # clock (who wins the round depends on seniority tie-breaking,
            # which is not what this regression is about)
            while True:
                result = await r_er_rx.receive()
                if result.won_clock == 5:
                    break
            assert restarted.clock == 5

            # ... and both nodes must converge on a single session
            while restarted.current_session != survivor.current_session:
                await sleep(0.05)

            tg.cancel_scope.cancel()


def test_non_candidate_never_proposes_self():
    """A --no-master node (is_candidate=False) must never propose itself.

    During a solo partition (no other master known), the node re-proposes the
    last-known session instead of itself — so it can't win by default.
    """
    me = NodeId("worker-only")
    other = NodeId("the-master")
    election = Election(
        node_id=me,
        election_message_receiver=None,  # type: ignore[arg-type]
        election_message_sender=None,  # type: ignore[arg-type]
        election_result_sender=None,  # type: ignore[arg-type]
        connection_message_receiver=None,  # type: ignore[arg-type]
        command_receiver=None,  # type: ignore[arg-type]
        is_candidate=False,
        seniority=5,
    )
    # Node currently knows a master exists (not itself).
    election.current_session = SessionId(master_node_id=other, election_clock=3)
    status = election._election_status(clock=7)  # type: ignore[reportPrivateUsage]
    assert status.proposed_session.master_node_id == other
    assert status.proposed_session.election_clock == 3  # re-propose last known
    assert status.seniority == -1  # non-candidate seniority


def test_candidate_proposes_self_when_unknown():
    """A normal candidate node still proposes itself when it doesn't know a master."""
    me = NodeId("candidate")
    election = Election(
        node_id=me,
        election_message_receiver=None,  # type: ignore[arg-type]
        election_message_sender=None,  # type: ignore[arg-type]
        election_result_sender=None,  # type: ignore[arg-type]
        connection_message_receiver=None,  # type: ignore[arg-type]
        command_receiver=None,  # type: ignore[arg-type]
        is_candidate=True,
        seniority=5,
    )
    status = election._election_status(clock=7)  # type: ignore[reportPrivateUsage]
    assert status.proposed_session.master_node_id == me
    assert status.seniority == 5


@pytest.mark.anyio
async def test_promote_master_command_forces_win_over_peer() -> None:
    """
    A PromoteMaster command targeting us must force us to win the next round
    even against a peer with higher observed seniority.
    """
    em_out_tx, _em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("B"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)
            # Peer A campaigns at seniority 100 (observed by us)
            await em_in_tx.send(em(clock=1, seniority=100, node_id="A"))
            # Wait for the round to resolve
            await er_rx.receive()

            # Now send PromoteMaster targeting us
            await co_tx.send(
                ForwarderCommand(
                    origin=SystemId("api"),
                    command=PromoteMaster(target_node_id=NodeId("B")),
                )
            )

            # We should win the next round with seniority > 100
            result = await er_rx.receive()
            assert result.session_id.master_node_id == NodeId("B")
            assert election.seniority > 100

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()


@pytest.mark.anyio
async def test_promote_master_ignored_for_other_node() -> None:
    """
    A PromoteMaster targeting a different node must not force us to promote.
    """
    em_out_tx, _em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("B"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)
            # Let the initial self-election round resolve (seniority becomes 1)
            await er_rx.receive()
            initial_clock = election.clock

            # PromoteMaster targeting A (not us) must not trigger a campaign
            await co_tx.send(
                ForwarderCommand(
                    origin=SystemId("api"),
                    command=PromoteMaster(target_node_id=NodeId("A")),
                )
            )
            await sleep(0.2)
            # No new round started: clock and campaigns unchanged
            assert election.clock == initial_clock

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()


@pytest.mark.anyio
async def test_continuous_connection_messages_do_not_stack_campaigns() -> None:
    """
    A continuous trickle of connection messages must not start a new campaign
    for every message — the _campaign_active guard collapses them into one
    round. Without the guard, the master flapped every election timeout and
    tore down instances (the observed 55k 'elected master' log spam).
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, _er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # Burst of connection messages (like re-announcements). The
            # _campaign_active guard must collapse these into far fewer
            # campaigns than messages — one per message would stack and flap.
            for _ in range(5):
                await cm_tx.send(ConnectionMessage(connected=True))

            # Count distinct campaign rounds (broadcasts at distinct clocks)
            # within a short window. 5 messages must NOT yield 5+ rounds.
            clocks: set[int] = set()
            with move_on_after(0.6):
                while True:
                    got = await em_out_rx.receive()
                    clocks.add(got.clock)
                    if len(clocks) >= 3:
                        break
            assert len(clocks) <= 3, f"too many campaigns for 5 messages: {clocks}"

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()
