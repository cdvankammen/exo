"""Unit tests for Election campaign-owner funnel (RC-01 fix).

Verifies that:
1. Campaign state is owned by a single task (_campaign_owner)
2. Concurrent _StartCampaign requests are serialized, not stacked
3. _AddCandidate messages during collection are appended to the running round
4. A _StartCampaign arriving mid-collection supersedes the current round
5. _campaign_active is consistent across concurrent readers
6. Counter accounting (_active_campaigns) never drifts
"""

from __future__ import annotations

import anyio
import pytest

from exo.shared.election import (
    DEFAULT_ELECTION_TIMEOUT,
    Election,
    ElectionMessage,
    ElectionResult,
    _AddCandidate,
    _StartCampaign,
)
from exo.shared.types.common import NodeId, SessionId
from exo.utils.channels import channel


def _make_em(
    node_id: str = "node-a",
    clock: int = 1,
    seniority: int = 0,
    commands_seen: int = 0,
) -> ElectionMessage:
    return ElectionMessage(
        clock=clock,
        seniority=seniority,
        proposed_session=SessionId(master_node_id=node_id, election_clock=clock),
        commands_seen=commands_seen,
    )


class _ElectionHarness:
    """Manages an Election + result capture for a test."""

    def __init__(self, node_id: str = "node-a"):
        em_tx, em_rx = channel(max_buffer_size=64)
        er_tx, er_rx = channel(max_buffer_size=64)
        cm_tx, cm_rx = channel(max_buffer_size=64)
        co_tx, co_rx = channel(max_buffer_size=64)

        # Clone BEFORE constructing Election so we see all results
        self.results_rx = er_rx.clone()

        self.election = Election(
            node_id=NodeId(node_id),
            election_message_receiver=em_rx,
            election_message_sender=em_tx,
            election_result_sender=er_tx,
            connection_message_receiver=cm_rx,
            command_receiver=co_rx,
        )

    async def wait_result(
        self, timeout: float = 2.0, clock: int | None = None
    ) -> ElectionResult:
        """Block until an ElectionResult arrives matching *clock* (if given)."""
        with anyio.move_on_after(timeout):
            async for result in self.results_rx:
                if clock is None or result.won_clock == clock:
                    return result
        pytest.fail(f"No election result received within {timeout}s (clock={clock})")

    async def drain_results(self, count: int, timeout: float = 5.0) -> list[ElectionResult]:
        results: list[ElectionResult] = []
        with anyio.move_on_after(timeout):
            async for result in self.results_rx:
                results.append(result)
                if len(results) >= count:
                    break
        return results


@pytest.mark.anyio
async def test_initial_campaign_completes():
    """The initial campaign (timeout=0) resolves immediately and emits a result."""
    h = _ElectionHarness("node-a")

    async with anyio.create_task_group() as outer:
        outer.start_soon(h.election.run)
        result = await h.wait_result(clock=0)
        assert result.session_id.master_node_id == "node-a"
        assert result.won_clock == 0
        await h.election.shutdown()

    assert h.election._active_campaigns == 0


@pytest.mark.anyio
async def test_campaign_requests_are_serialized():
    """Multiple _StartCampaign requests serialize; _active_campaigns never > 1."""
    h = _ElectionHarness("node-a")
    max_active = 0
    stop = False

    async def monitor_active():
        nonlocal max_active, stop
        while not stop:
            cur = h.election._active_campaigns
            if cur > max_active:
                max_active = cur
            await anyio.sleep(0.01)

    async with anyio.create_task_group() as tg:
        tg.start_soon(h.election.run)
        tg.start_soon(monitor_active)

        await h.wait_result(clock=0)

        for i in range(5):
            h.election._request_campaign(
                [_make_em(clock=i + 10)], timeout=0.05
            )

        await anyio.sleep(0.5)
        stop = True
        await h.election.shutdown()

    assert max_active <= 1, f"Campaigns stacked: max_active={max_active}"
    assert h.election._active_campaigns == 0


@pytest.mark.anyio
async def test_add_candidate_during_collection():
    """_AddCandidate messages during collection are included in the round."""
    h = _ElectionHarness("node-a")
    collected_candidates: list[list[ElectionMessage]] = []

    orig_elect = h.election.elect

    async def capturing_elect(em: ElectionMessage) -> None:
        collected_candidates.append(list(h.election._candidates))
        await orig_elect(em)

    h.election.elect = capturing_elect  # type: ignore[method-assign]

    async with anyio.create_task_group() as tg:
        tg.start_soon(h.election.run)
        await h.wait_result(clock=0)

        h.election._request_campaign([_make_em(clock=100)], timeout=0.5)
        await anyio.sleep(0.05)

        h.election._request_add_candidate(_make_em(node_id="node-b", clock=100))
        h.election._request_add_candidate(_make_em(node_id="node-c", clock=100))

        result = await h.wait_result(clock=100, timeout=2.0)
        assert result is not None
        await h.election.shutdown()

    node_ids = {c.proposed_session.master_node_id for c in collected_candidates[-1]}
    assert "node-a" in node_ids
    assert "node-b" in node_ids
    assert "node-c" in node_ids


@pytest.mark.anyio
async def test_superseding_campaign_abandons_current():
    """A _StartCampaign arriving mid-collection cancels the current round."""
    h = _ElectionHarness("node-a")
    elected_clocks: list[int] = []

    orig_elect = h.election.elect

    async def tracking_elect(em: ElectionMessage) -> None:
        elected_clocks.append(em.clock)
        await orig_elect(em)

    h.election.elect = tracking_elect  # type: ignore[method-assign]

    async with anyio.create_task_group() as tg:
        tg.start_soon(h.election.run)
        await h.wait_result(clock=0)

        h.election._request_campaign([_make_em(clock=50)], timeout=5.0)
        await anyio.sleep(0.05)

        h.election._request_campaign([_make_em(clock=99)], timeout=0.05)
        await h.wait_result(clock=99, timeout=2.0)

        await h.election.shutdown()

    assert elected_clocks.count(50) == 0, "Superseded campaign should not elect"
    assert 99 in elected_clocks


@pytest.mark.anyio
async def test_active_campaigns_counter_never_drifts():
    """After many campaigns, _active_campaigns returns to 0."""
    h = _ElectionHarness("node-a")

    async with anyio.create_task_group() as tg:
        tg.start_soon(h.election.run)
        await h.wait_result(clock=0)

        for i in range(20):
            h.election._request_campaign(
                [_make_em(clock=i + 100)], timeout=0.02
            )
            await anyio.sleep(0.03)

        await anyio.sleep(0.5)
        await h.election.shutdown()

    assert h.election._active_campaigns == 0


@pytest.mark.anyio
async def test_campaign_active_property_consistent():
    """_campaign_active reflects actual state from outside the owner task."""
    h = _ElectionHarness("node-a")
    observations: list[bool] = []
    stop = False

    async def observer():
        while not stop:
            observations.append(h.election._campaign_active)
            await anyio.sleep(0.005)

    async with anyio.create_task_group() as tg:
        tg.start_soon(h.election.run)
        tg.start_soon(observer)
        await h.wait_result(clock=0)

        h.election._request_campaign([_make_em(clock=200)], timeout=0.3)
        await anyio.sleep(0.15)

        stop = True
        await h.election.shutdown()

    assert any(o for o in observations), "Should have observed _campaign_active=True"
    assert h.election._active_campaigns == 0
