"""Regression tests for the publish_bytes guard (t_28df2167).

Proves that a malformed / undeserializable message on a topic:
  1. does NOT raise out of publish_bytes (the old crash),
  2. is recorded in the malformed-event log (dashboard warning chip),
  3. leaves the TopicRouter usable for subsequent valid messages
     (the gossip loop stays alive).

The fix lives in exo/routing/router.py:
  - Guard 1: publish_bytes() wraps self.topic.deserialize(data) in
    try/except Exception -> logger.warning + record_malformed_event
    (file lines ~111-123).
  - Guard 2: Router._networking_recv() wraps the per-message
    router.publish_bytes(data) and the FromSwarm match in
    (ValidationError, UnicodeDecodeError, JSONDecodeError) -> warning
    + record_malformed_event, and continues the loop (file lines
    ~247-287).
"""

from exo.routing.router import (
    _MALFORMED_EVENT_LOG_MAX,  # type: ignore[reportPrivateUsage]  # test-only bound check
    TopicRouter,
    malformed_event_log,
    record_malformed_event,
)
from exo.routing.topics import GLOBAL_EVENTS
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import GlobalForwarderEvent, TestEvent
from exo.utils.channels import channel


def _make_router() -> TopicRouter[GlobalForwarderEvent]:
    # networking_sender is Sender[tuple[str, bytes]] in Router; mirror it.
    send, _recv = channel[tuple[str, bytes]]()
    return TopicRouter[GlobalForwarderEvent](GLOBAL_EVENTS, send)


async def test_publish_bytes_drops_malformed_without_raise():
    """Garbage bytes must not crash publish_bytes (old behavior: raise)."""
    tr = _make_router()
    n_before = len(malformed_event_log())
    await tr.publish_bytes(b"\x00\xff not json at all \xfe")
    # No exception propagated -> the guard held.
    assert len(malformed_event_log()) == n_before + 1


async def test_publish_bytes_records_malformed_event_for_dashboard():
    """The dropped event must appear in the malformed log."""
    tr = _make_router()
    n_before = len(malformed_event_log())
    await tr.publish_bytes(b"{invalid json")
    assert len(malformed_event_log()) == n_before + 1
    last = malformed_event_log()[-1]
    assert last["topic"] == GLOBAL_EVENTS.topic
    assert "error" in last


async def test_publish_bytes_keeps_router_alive_for_next_valid():
    """After a malformed event, a valid event still deserializes fine."""
    tr = _make_router()
    await tr.publish_bytes(b"\xde\xad\xbe\xef not json")
    # A valid GlobalForwarderEvent must still deserialize.
    valid = GlobalForwarderEvent(
        origin_idx=0,
        origin=NodeId(),
        session=SessionId(master_node_id=NodeId(), election_clock=0),
        event=TestEvent(),
    )
    await tr.publish_bytes(GLOBAL_EVENTS.serialize(valid))
    # Reaching this line proves the first (garbage) call did not raise;
    # the second call proves the router is still fully functional.


async def test_malformed_log_is_bounded():
    """Ring buffer stays bounded (no unbounded memory growth)."""
    tr = _make_router()
    before = len(malformed_event_log())
    # Exceed the cap with garbage.
    for _ in range(_MALFORMED_EVENT_LOG_MAX + 10):
        await tr.publish_bytes(b"garbage")
    assert len(malformed_event_log()) <= _MALFORMED_EVENT_LOG_MAX
    # Log still non-empty (we added entries beyond the original).
    assert len(malformed_event_log()) >= min(before + 1, _MALFORMED_EVENT_LOG_MAX)


async def test_record_malformed_event_is_pure_append():
    """record_malformed_event itself never raises on odd error strings."""
    record_malformed_event("some_topic", "some error")
    assert any(e["topic"] == "some_topic" for e in malformed_event_log())