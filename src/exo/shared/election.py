from typing import Self

import anyio
from anyio import (
    CancelScope,
    Event,
    get_cancelled_exc_class,
)
from loguru import logger

from exo.routing.connection_message import ConnectionMessage
from exo.shared.types.commands import ForwarderCommand, PromoteMaster
from exo.shared.types.common import NodeId, SessionId
from exo.utils.channels import Receiver, Sender
from exo.utils.pydantic_ext import FrozenModel
from exo.utils.task_group import TaskGroup

DEFAULT_ELECTION_TIMEOUT = 3.0

# Seniority high enough that no node can realistically out-grow it through
# normal re-election (seniority only ever grows to at most the number of
# candidates seen in a round). Shared by --force-master at startup and by
# a runtime PromoteMaster command.
FORCE_MASTER_SENIORITY = 1_000_000
# After a connection-triggered campaign, further connection messages are
# absorbed for this long and coalesced into at most one follow-up round.
# Drains topology-formation bursts so a burst triggers a bounded number
# of re-elections instead of one per message.
_CONNECTION_ELECTION_COOLDOWN = 25.0


class ElectionMessage(FrozenModel):
    clock: int
    seniority: int
    proposed_session: SessionId
    commands_seen: int

    # Could eventually include a list of neighbour nodes for centrality
    def __lt__(self, other: Self) -> bool:
        if self.clock != other.clock:
            return self.clock < other.clock
        if self.seniority != other.seniority:
            return self.seniority < other.seniority
        elif self.commands_seen != other.commands_seen:
            return self.commands_seen < other.commands_seen
        else:
            return (
                self.proposed_session.master_node_id
                < other.proposed_session.master_node_id
            )


class ElectionResult(FrozenModel):
    session_id: SessionId
    won_clock: int
    is_new_master: bool


class Election:
    def __init__(
        self,
        node_id: NodeId,
        *,
        election_message_receiver: Receiver[ElectionMessage],
        election_message_sender: Sender[ElectionMessage],
        election_result_sender: Sender[ElectionResult],
        connection_message_receiver: Receiver[ConnectionMessage],
        command_receiver: Receiver[ForwarderCommand],
        is_candidate: bool = True,
        seniority: int = 0,
    ):
        # If we aren't a candidate, simply don't increment seniority.
        # For reference: This node can be elected master if all nodes are not master candidates
        # Any master candidate will automatically win out over this node.
        self.is_candidate = is_candidate
        self.seniority = seniority if is_candidate else -1
        self.clock = 0
        self.node_id = node_id
        self.commands_seen = 0
        # Every node spawns as master
        self.current_session: SessionId = SessionId(
            master_node_id=node_id, election_clock=0
        )

        # Senders/Receivers
        self._em_sender = election_message_sender
        self._em_receiver = election_message_receiver
        self._er_sender = election_result_sender
        self._cm_receiver = connection_message_receiver
        self._co_receiver = command_receiver

        # Campaign state
        self._candidates: list[ElectionMessage] = []
        self._campaign_cancel_scope: CancelScope | None = None
        self._campaign_done: Event | None = None
        self._campaigns_started = 0
        self._campaigns_finished = 0
        self._tg = TaskGroup()

        # Highest seniority ever observed from a peer, used by _force_promote
        # so repeated PromoteMaster commands always outrank whatever the
        # cluster has seen so far, instead of tying against each other.
        self._max_peer_seniority_seen = 0

    @property
    def _campaign_active(self) -> bool:
        return self._campaigns_finished < self._campaigns_started

    def _start_campaign(
        self, candidates: list[ElectionMessage], campaign_timeout: float
    ) -> None:
        self._campaigns_started += 1
        self._tg.start_soon(self._campaign, candidates, campaign_timeout)

    async def run(self):
        logger.info("Starting Election")
        try:
            async with self._tg as tg:
                tg.start_soon(self._election_receiver)
                tg.start_soon(self._connection_receiver)
                tg.start_soon(self._command_counter)

                # And start an election immediately, that instantly resolves
                candidates: list[ElectionMessage] = []
                logger.debug("Starting initial campaign")
                self._candidates = candidates
                self._campaigns_started += 1
                await self._campaign(candidates, campaign_timeout=0.0)
                logger.debug("Initial campaign finished")
        finally:
            # Cancel and wait for the last election to end
            if self._campaign_cancel_scope is not None:
                logger.debug("Cancelling campaign")
                self._campaign_cancel_scope.cancel()
            if self._campaign_done is not None:
                logger.debug("Waiting for campaign to finish")
                await self._campaign_done.wait()
            logger.debug("Campaign cancelled and finished")
            logger.info("Election shutdown")

    async def elect(self, em: ElectionMessage) -> None:
        logger.debug(f"Electing: {em}")
        is_new_master = em.proposed_session != self.current_session
        self.current_session = em.proposed_session
        logger.debug(f"Current session: {self.current_session}")
        await self._er_sender.send(
            ElectionResult(
                won_clock=em.clock,
                session_id=em.proposed_session,
                is_new_master=is_new_master,
            )
        )

    async def shutdown(self) -> None:
        self._tg.cancel_tasks()

    async def _election_receiver(self) -> None:
        with self._em_receiver as election_messages:
            async for message in election_messages:
                logger.debug(f"Election message received: {message}")
                if message.proposed_session.master_node_id == self.node_id:
                    logger.debug("Dropping message from ourselves")
                    # Drop messages from us (See exo.routing.router)
                    continue
                self._max_peer_seniority_seen = max(
                    self._max_peer_seniority_seen, message.seniority
                )
                # If a new round is starting, we participate
                if message.clock > self.clock:
                    self.clock = message.clock
                    logger.debug(f"New clock: {self.clock}")
                    logger.debug("Starting new campaign")
                    candidates: list[ElectionMessage] = [message]
                    logger.debug(f"Candidates: {candidates}")
                    logger.debug(f"Current candidates: {self._candidates}")
                    self._candidates = candidates
                    logger.debug(f"New candidates: {self._candidates}")
                    logger.debug("Starting new campaign")
                    self._start_campaign(candidates, DEFAULT_ELECTION_TIMEOUT)
                    logger.debug("Campaign started")
                    continue
                # Reply to stale messages with our status so that a node
                # campaigning below the cluster clock (e.g. a restarted
                # master starting over from clock 0) deterministically
                # learns the current clock and rejoins the election.
                if message.clock < self.clock:
                    logger.debug(f"Replying to stale message: {message}")
                    await self._em_sender.send(self._election_status())
                    continue
                logger.debug(f"Election added candidate {message}")
                # Now we are processing this rounds messages - including the message that triggered this round.
                self._candidates.append(message)
                # If an equal-clock message re-proposes the node that is
                # ALREADY our current master, the round is already resolved —
                # starting another campaign here would just re-broadcast the
                # same status and (with a trickle of such re-proposals from a
                # senior node) keep the election loop spinning forever,
                # re-entering _campaign and logging "Waiting for other
                # campaign" every few seconds. Skip it; the master stays.
                if (
                    self.current_session.master_node_id
                    == message.proposed_session.master_node_id
                ):
                    logger.debug(
                        "Equal-clock re-proposal of current master; skipping"
                    )
                    continue
                # An equal-clock candidacy with no campaign running would
                # otherwise sit in a candidate list that nothing evaluates;
                # start a campaign so the round deterministically resolves.
                if not self._campaign_active:
                    logger.debug(
                        "Equal-clock candidacy with no active campaign; starting one"
                    )
                    self._start_campaign(self._candidates, DEFAULT_ELECTION_TIMEOUT)

    async def _connection_receiver(self) -> None:
        with self._cm_receiver as connection_messages:
            async for first in connection_messages:
                # Delay after connection message for time to symmetrically setup
                await anyio.sleep(0.2)
                rest = connection_messages.collect()

                logger.debug(
                    f"Connection messages received: {first} followed by {rest}"
                )
                while True:
                    # Campaign on the first message immediately (a node joining
                    # shouldn't wait for a master). Guard against stacking
                    # campaigns: if one is already active (e.g. a continuous
                    # trickle of re-announcements), skip starting another —
                    # that trickle was what caused the master to flap every
                    # DEFAULT_ELECTION_TIMEOUT and tear down instances.
                    if self._campaign_active:
                        logger.debug(
                            "Campaign already active; skipping follow-up round"
                        )
                    else:
                        logger.debug(f"Current clock: {self.clock}")
                        # These messages are strictly peer to peer
                        self.clock += 1
                        logger.debug(f"New clock: {self.clock}")
                        candidates: list[ElectionMessage] = []
                        self._candidates = candidates
                        logger.debug("Starting new campaign")
                        self._start_campaign(candidates, DEFAULT_ELECTION_TIMEOUT)
                        logger.debug("Campaign started")

                    # Cooldown: absorb the connection-message burst that
                    # accompanies topology formation instead of campaigning
                    # once per message. Anything that arrived during the
                    # window (another join, the master disconnecting) is
                    # coalesced into one follow-up round for the settled
                    # topology. A closed channel ends iteration immediately,
                    # so shutdown is not delayed by the cooldown.
                    absorbed = 0
                    with anyio.move_on_after(_CONNECTION_ELECTION_COOLDOWN):
                        async for _ in connection_messages:
                            absorbed += 1
                    if absorbed == 0:
                        break
                    logger.debug(
                        f"Absorbed {absorbed} connection messages during cooldown, "
                        "starting follow-up election round"
                    )

    async def _command_counter(self) -> None:
        with self._co_receiver as commands:
            async for forwarder_command in commands:
                self.commands_seen += 1
                command = forwarder_command.command
                if (
                    isinstance(command, PromoteMaster)
                    and command.target_node_id == self.node_id
                ):
                    self._force_promote()

    def _force_promote(self) -> None:
        """Guarantee this node wins the next election round, then trigger one.

        Sets seniority to one more than the highest value this node or any
        peer has ever been observed at (floored at FORCE_MASTER_SENIORITY, the
        same baseline --force-master uses at startup). Using a fixed constant
        here would let a *second* PromoteMaster tie the first: both nodes
        would sit at the same seniority and the round would fall through to
        the commands_seen tiebreak, which favours whichever node has been
        master longest -- silently no-opping the newer promotion. Always
        going one higher than anything seen so far keeps repeated
        promotions, and promotions away from a --force-master node, working.
        """
        logger.info("Forcing this node to win the next master election")
        self.seniority = (
            max(
                self.seniority,
                self._max_peer_seniority_seen,
                FORCE_MASTER_SENIORITY - 1,
            )
            + 1
        )
        self.clock += 1
        candidates: list[ElectionMessage] = []
        self._candidates = candidates
        self._start_campaign(candidates, DEFAULT_ELECTION_TIMEOUT)

    async def _campaign(
        self, candidates: list[ElectionMessage], campaign_timeout: float
    ) -> None:
        clock = self.clock

        # Kill the old campaign
        if self._campaign_cancel_scope:
            logger.info("Cancelling other campaign")
            self._campaign_cancel_scope.cancel()
        if self._campaign_done:
            logger.info("Waiting for other campaign to finish")
            await self._campaign_done.wait()

        done = Event()
        self._campaign_done = done
        scope = CancelScope()
        self._campaign_cancel_scope = scope

        try:
            with scope:
                logger.debug(f"Election {clock} started")

                status = self._election_status(clock)
                candidates.append(status)
                await self._em_sender.send(status)

                logger.debug(f"Sleeping for {campaign_timeout} seconds")
                await anyio.sleep(campaign_timeout)
                # minor hack - rebroadcast status in case anyone has missed it.
                await self._em_sender.send(status)
                logger.debug("Woke up from sleep")
                # add an anyio checkpoint - anyio.lowlevel.chekpoint() or checkpoint_if_cancelled() is preferred, but wasn't typechecking last I checked
                await anyio.sleep(0)

                # Election finished!
                elected = max(candidates)
                logger.debug(f"Election queue {candidates}")
                logger.debug(f"Elected: {elected}")
                if (
                    self.node_id == elected.proposed_session.master_node_id
                    and self.seniority >= 0
                ):
                    logger.debug(
                        f"Node is a candidate and seniority is {self.seniority}"
                    )
                    # Winning must give a DURABLE seniority edge, not just
                    # len(candidates). With a 2-node cluster both nodes end up
                    # at the same seniority after enough rounds (winner:
                    # max(sen, len(candidates)); loser: peer-1 -> converge),
                    # so every subsequent campaign can flip the master
                    # (commands_seen tiebreak alternates) -> ping-pong. Go one
                    # above the highest peer seniority ever seen so the winner
                    # keeps winning subsequent rounds and the master stays
                    # stable (max+1 is also what explicit PromoteMaster uses).
                    self.seniority = max(
                        self.seniority,
                        self._max_peer_seniority_seen + 1,
                        len(candidates),
                    )
                    logger.debug(f"New seniority: {self.seniority}")
                else:
                    logger.debug(
                        f"Node is not a candidate or seniority is not {self.seniority}"
                    )
                # A node that did NOT win must still raise its seniority toward
                # the highest peer seniority ever observed. Without this, a
                # restarted node stays at a low seniority (e.g. 1) and keeps
                # re-proposing itself every round, ping-ponging elections with
                # the established master (which wins at its much higher
                # seniority) and flapping the cluster. We stay just below the
                # peer (max+1 is only for explicit PromoteMaster) so the
                # higher-seniority peer always wins and there is never a tie.
                if self._max_peer_seniority_seen > self.seniority:
                    target = self._max_peer_seniority_seen - 1
                    logger.debug(
                        f"Raising seniority {self.seniority} -> {target} "
                        "(observed from peer)"
                    )
                    self.seniority = max(self.seniority, target)
                logger.debug(
                    f"Election finished, new SessionId({elected.proposed_session}) with queue {candidates}"
                )
                logger.debug("Sending election result")
                await self.elect(elected)
                logger.debug("Election result sent")
        except get_cancelled_exc_class():
            logger.debug(f"Election {clock} cancelled")
        finally:
            logger.debug(f"Election {clock} finally")
            self._campaigns_finished += 1
            if self._campaign_cancel_scope is scope:
                self._campaign_cancel_scope = None
            logger.debug("Setting done event")
            done.set()
            logger.debug("Done event set")

    def _election_status(self, clock: int | None = None) -> ElectionMessage:
        c = self.clock if clock is None else clock
        # Non-candidate nodes must never propose themselves as master.
        # Re-propose the last known master instead. During a solo partition
        # this prevents the node from winning by default.
        if (
            not self.is_candidate
            and self.current_session.master_node_id != self.node_id
        ):
            return ElectionMessage(
                proposed_session=self.current_session,
                clock=c,
                seniority=self.seniority,
                commands_seen=self.commands_seen,
            )
        return ElectionMessage(
            proposed_session=(
                self.current_session
                if self.current_session.master_node_id == self.node_id
                else SessionId(master_node_id=self.node_id, election_clock=c)
            ),
            clock=c,
            seniority=self.seniority,
            commands_seen=self.commands_seen,
        )
