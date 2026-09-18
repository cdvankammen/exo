"""Per-node circuit breaker for graceful degradation.

Production hosting providers trip a node (or shard) out of the placement pool
when it starts failing mid-inference. Without a breaker, exo keeps selecting
the failing node for every new placement: ``NodeTimedOut`` removes a node only
after a full last-seen timeout (5s), and a node that fails a runner then
re-announces (``NodeGatheredInfo`` re-adds it) is instantly eligible again —
so a node crashing under load loops: selected → runner dies → re-announces →
selected again.

This module tracks, per node, a small failure window:

* ``record_failure`` — a runner died on this node (``RunnerFailed``). After
  ``threshold`` failures inside the window the breaker *trips*: the node is
  marked degraded and excluded from new placements.
* ``record_success`` — the node reported healthy telemetry again
  (``NodeGatheredInfo``), which opens the circuit and clears the failure
  window. A node that recovered is immediately usable again; the *cooldown*
  only applies between a trip and the next healthy heartbeat, so a node that
  keeps failing keeps its breaker closed (each failure re-trips instantly
  because the window is still full).

The state is deliberately cheap and pure: a mapping of node id → failure
count + window timestamps, no network, no locks, no process coupling. The
master applies it as part of :class:`~exo.shared.types.state.State` so it
survives snapshots and is visible via ``/state``.

Constants mirror the master's ``node_inactivity_timeout`` (5s): the failure
window and the cooldown are short so a transient blip does not strand a node
for long, while still catching a node that fails repeatedly.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from exo.shared.types.common import NodeId
from exo.utils.pydantic_ext import FrozenModel

# How far back (seconds) a recorded failure still counts toward tripping.
EXO_CIRCUIT_FAILURE_WINDOW_SECS = 30.0
# How many failures inside the window trip the breaker (CLOSED → OPEN).
EXO_CIRCUIT_FAILURE_THRESHOLD = 2
# How long (seconds) a tripped breaker stays OPEN without a healthy heartbeat.
# Any successful heartbeat (NodeGatheredInfo) opens it immediately; the
# cooldown only bounds the fallback when telemetry has gone silent.
EXO_CIRCUIT_COOLDOWN_SECS = 10.0

# Sentinel returned by NodeCircuitBreaker.healthy_nodes() when no state exists
# for a node — callers (placement filters) treat missing as healthy.
_UNKNOWN = object()


class NodeCircuitState(FrozenModel):
    """Serialisable per-node circuit state stored in :class:`State`."""

    failures: int = 0
    window_started_at: float = 0.0
    tripped_at: float | None = None


@dataclass
class NodeCircuitBreaker:
    """Tracks per-node failure windows and exposes placement health."""

    _failures: dict[NodeId, NodeCircuitState] = field(default_factory=dict)
    failure_window_secs: float = EXO_CIRCUIT_FAILURE_WINDOW_SECS
    failure_threshold: int = EXO_CIRCUIT_FAILURE_THRESHOLD
    cooldown_secs: float = EXO_CIRCUIT_COOLDOWN_SECS
    _now: float | None = None  # test hook

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.failure_window_secs <= 0 or self.cooldown_secs < 0:
            raise ValueError("failure_window_secs must be > 0; cooldown_secs >= 0")

    def _time(self) -> float:
        return self._now if self._now is not None else time.monotonic()

    def record_failure(self, node_id: NodeId) -> None:
        """Record a runner failure on ``node_id`` and trip at threshold."""
        now = self._time()
        state = self._failures.get(node_id)
        if state is None:
            state = NodeCircuitState(failures=0, window_started_at=now)
        # Reset the window if the previous failure is too old to count.
        if now - state.window_started_at > self.failure_window_secs:
            state = NodeCircuitState(failures=0, window_started_at=now)
        state = state.model_copy(
            update={"failures": state.failures + 1, "window_started_at": state.window_started_at}
        )
        if state.failures >= self.failure_threshold:
            state = state.model_copy(update={"tripped_at": now})
        self._failures[node_id] = state

    def record_success(self, node_id: NodeId) -> None:
        """Clear the failure window for ``node_id`` (healthy heartbeat)."""
        if node_id in self._failures:
            del self._failures[node_id]

    def is_tripped(self, node_id: NodeId) -> bool:
        """True when ``node_id`` is currently excluded from placement."""
        state = self._failures.get(node_id)
        if state is None or state.tripped_at is None:
            return False
        return self._time() - state.tripped_at <= self.cooldown_secs

    def healthy_nodes(self, node_ids: Iterable[NodeId]) -> set[NodeId]:
        """Filter ``node_ids`` down to nodes not currently tripped."""
        return {node_id for node_id in node_ids if not self.is_tripped(node_id)}

    def state_for(self, node_id: NodeId) -> NodeCircuitState | None:
        return self._failures.get(node_id)

    def snapshot(self) -> dict[NodeId, NodeCircuitState]:
        """Return the current failure map for persistence into :class:`State`."""
        return dict(self._failures)

    @classmethod
    def from_mapping(cls, mapping: Mapping[NodeId, NodeCircuitState]) -> "NodeCircuitBreaker":
        """Rehydrate a breaker from a persisted failure map.

        ``mapping`` is the value previously stored in ``State.node_circuits``.
        Failure timestamps are monotonic-clock values from the recording
        process, so a breaker rehydrated from a snapshot on the same process
        keeps correct windows; on a different process the windows are best
        effort (worst case: an extra trip that a healthy heartbeat clears).
        """
        return cls(_failures=dict(mapping))

    def __len__(self) -> int:
        return len(self._failures)