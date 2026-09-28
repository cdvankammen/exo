# pyright: reportUnusedFunction=false, reportAny=false
"""Tests for the GET /events endpoint cursor support.

The handler at exo/api/main.py:1616 supports two query parameters:
  - since: int (default 0) — start index, inclusive
  - limit: int | None (default None) — max events to return

The response sets `X-EXO-Last-Idx` to the upper bound consumed, allowing
clients to chain reads without a separate /state round-trip.

The events come from a bounded in-memory window (`API._recent_events`, a
deque of at most RECENT_EVENTS entries) rather than an unbounded on-disk log,
so indexes are relative to the window, not to the whole session.
"""

from collections import deque
from typing import Any
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.shared.types.events import Event, TestEvent


def _make_api(n_events: int) -> Any:
    """Create a minimal API whose bounded window holds n_events records
    and only the GET /events route mounted."""
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api._send = AsyncMock()  # pyright: ignore[reportPrivateUsage]
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]

    api._recent_events = deque[Event](  # pyright: ignore[reportPrivateUsage]
        [TestEvent() for _ in range(n_events)]
    )

    app.get("/events")(api.stream_events)
    return api


def test_stream_events_full_dump_backward_compatible() -> None:
    """No params -> every event in the window; X-EXO-Last-Idx equals count."""
    api = _make_api(n_events=5)
    client = TestClient(api.app)

    resp = client.get("/events")
    assert resp.status_code == 200
    data: list[dict[str, Any]] = resp.json()
    assert len(data) == 5
    assert resp.headers["X-EXO-Last-Idx"] == "5"


def test_stream_events_with_since_and_limit() -> None:
    """since=N&limit=M returns events in [since, since+M); header reflects bound."""
    api = _make_api(n_events=10)
    client = TestClient(api.app)

    resp = client.get("/events", params={"since": 3, "limit": 4})
    assert resp.status_code == 200
    data: list[dict[str, Any]] = resp.json()
    assert len(data) == 4
    assert resp.headers["X-EXO-Last-Idx"] == "7"


def test_stream_events_since_only_reads_to_end() -> None:
    """since with no limit returns [since, count); header equals count."""
    api = _make_api(n_events=8)
    client = TestClient(api.app)

    resp = client.get("/events", params={"since": 5})
    assert resp.status_code == 200
    data: list[dict[str, Any]] = resp.json()
    assert len(data) == 3
    assert resp.headers["X-EXO-Last-Idx"] == "8"


def test_stream_events_since_beyond_count_returns_empty() -> None:
    """since past end yields []; header reflects clamped end."""
    api = _make_api(n_events=4)
    client = TestClient(api.app)

    resp = client.get("/events", params={"since": 99})
    assert resp.status_code == 200
    assert resp.json() == []
    # end is clamped to the window size (4) since limit is None and since > count
    assert resp.headers["X-EXO-Last-Idx"] == "4"


def test_stream_events_limit_larger_than_remaining() -> None:
    """limit > remaining is clamped to the window size; no error."""
    api = _make_api(n_events=10)
    client = TestClient(api.app)

    resp = client.get("/events", params={"since": 7, "limit": 100})
    assert resp.status_code == 200
    data: list[dict[str, Any]] = resp.json()
    assert len(data) == 3
    assert resp.headers["X-EXO-Last-Idx"] == "10"


def test_stream_events_negative_since_rejected() -> None:
    """FastAPI Query(ge=0) rejects negative since with 422."""
    api = _make_api(n_events=3)
    client = TestClient(api.app)

    resp = client.get("/events", params={"since": -1})
    assert resp.status_code == 422


def test_stream_events_chained_cursor_reads() -> None:
    """Two sequential reads using returned cursor cover the window with no overlap."""
    api = _make_api(n_events=6)
    client = TestClient(api.app)

    first = client.get("/events", params={"since": 0, "limit": 4})
    assert first.status_code == 200
    cursor = int(first.headers["X-EXO-Last-Idx"])
    assert cursor == 4
    first_data: list[dict[str, Any]] = first.json()
    assert len(first_data) == 4

    second = client.get("/events", params={"since": cursor})
    assert second.status_code == 200
    second_data: list[dict[str, Any]] = second.json()
    assert len(second_data) == 2
    assert second.headers["X-EXO-Last-Idx"] == "6"

    # Two reads cover the full window with no gap and no overlap.
    assert len(first_data) + len(second_data) == 6


def test_stream_events_is_capped_by_the_window() -> None:
    """The window is bounded, so /events never returns more than RECENT_EVENTS."""
    from exo.api.main import RECENT_EVENTS

    # Push past the maxlen so the deque evicts its oldest entries, as it would in production.
    api = _make_api(n_events=0)
    api._recent_events = deque[Event](maxlen=RECENT_EVENTS)
    for _ in range(RECENT_EVENTS + 50):
        api._recent_events.append(TestEvent())
    client = TestClient(api.app)

    resp = client.get("/events")
    assert resp.status_code == 200
    data: list[dict[str, Any]] = resp.json()
    assert len(data) == RECENT_EVENTS
    assert resp.headers["X-EXO-Last-Idx"] == str(RECENT_EVENTS)
