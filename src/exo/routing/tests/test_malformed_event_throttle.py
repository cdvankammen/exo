"""Regression: malformed-event logging must be throttled per topic.

A version-skewed peer emits one malformed payload per message, forever. Before
the throttle this produced one full log line (plus a multi-line pydantic block)
per event: 407,046 error blocks / 254 MB of log on one node in a single day.

The throttle must:
  1. log the first _MALFORMED_LOG_FIRST events on a topic,
  2. then log only every _MALFORMED_LOG_EVERY-th,
  3. stay per-topic (a second noisy topic still gets its own first N),
  4. never drop the UI warning-chip record (bounded ring buffer unchanged).
"""

from collections.abc import Iterator

import pytest

from exo.routing import router as _router
from exo.routing.router import (
    _MALFORMED_LOG_EVERY,
    _MALFORMED_LOG_FIRST,
    _should_log_malformed,
    malformed_event_log,
    record_malformed_event,
)


@pytest.fixture(autouse=True)
def _isolate_router_globals() -> Iterator[None]:
    """Snapshot/restore the module-global throttle + ring-buffer state.

    Both are process-global singletons. Without this, the 300 events this file
    records would evict every entry earlier tests added and break them.
    """
    saved_log = list(_router._MALFORMED_EVENT_LOG)  # type: ignore[reportPrivateUsage]
    saved_seen = dict(_router._malformed_seen)  # type: ignore[reportPrivateUsage]
    try:
        yield
    finally:
        _router._MALFORMED_EVENT_LOG[:] = saved_log  # type: ignore[reportPrivateUsage]
        _router._malformed_seen.clear()  # type: ignore[reportPrivateUsage]
        _router._malformed_seen.update(saved_seen)  # type: ignore[reportPrivateUsage]


def test_logs_first_n_then_throttles():
    assert _MALFORMED_LOG_FIRST == 5
    assert _MALFORMED_LOG_EVERY == 1000


def test_throttle_decision_sequence():
    topic = "pytest-throttle-sequence"
    decisions = [_should_log_malformed(topic) for _ in range(2500)]

    # first N always logged
    assert all(decisions[:_MALFORMED_LOG_FIRST])
    # nothing between N and the first multiple of EVERY
    # (index i is event i+1, so event _MALFORMED_LOG_EVERY sits at index
    # _MALFORMED_LOG_EVERY-1 and must be excluded from the quiet window)
    assert not any(decisions[_MALFORMED_LOG_FIRST : _MALFORMED_LOG_EVERY - 1])
    # the 1000th and 2000th are logged again
    assert decisions[999] is True
    assert decisions[1999] is True
    # total log lines over 2500 events is tiny, not 2500
    assert sum(decisions) == _MALFORMED_LOG_FIRST + 2


def test_throttle_is_per_topic():
    a = "pytest-topic-a"
    b = "pytest-topic-b"
    assert _should_log_malformed(a) is True  # first for a
    assert _should_log_malformed(b) is True  # first for b, unaffected by a


def test_warning_chip_still_records_every_event():
    """Throttling affects the log line only, never the UI chip record."""
    topic = "pytest-chip-record"
    for _ in range(300):
        record_malformed_event(topic, "boom")

    entries = [e for e in malformed_event_log() if e["topic"] == topic]
    assert len(entries) == 20  # ring buffer still bounded at 20
