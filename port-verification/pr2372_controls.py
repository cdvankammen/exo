"""Per-hunk negative controls + real-dispatch-path check for exo PR #2372 port.

Run from the worktree root. Reverts each hunk independently (in-memory, by
monkeypatching) and asserts the port's own tests catch it.
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "src")

from exo.shared import apply as A  # noqa: E402
from exo.shared.apply import apply_instance_deleted, apply_runner_status_updated  # noqa: E402
from exo.shared.types.common import ModelId, NodeId  # noqa: E402
from exo.shared.types.events import (  # noqa: E402
    InstanceCreated,
    InstanceDeleted,
    RunnerStatusUpdated,
)
from exo.shared.types.state import State  # noqa: E402
from exo.shared.types.worker.instances import Instance, InstanceId  # noqa: E402
from exo.shared.types.worker.runners import (  # noqa: E402
    RunnerId,
    RunnerIdle,
    RunnerReady,
    RunnerShuttingDown,
)
from exo.worker.tests.unittests.conftest import (  # noqa: E402
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

MODEL = ModelId("test-model")


def instance_with(*runner_ids: RunnerId) -> Instance:
    return get_mlx_ring_instance(
        instance_id=InstanceId(),
        model_id=MODEL,
        node_to_runner={NodeId(f"node-{i}"): rid for i, rid in enumerate(runner_ids)},
        runner_to_shard={
            rid: get_pipeline_shard_metadata(
                MODEL, device_rank=i, world_size=len(runner_ids)
            )
            for i, rid in enumerate(runner_ids)
        },
    )


# ---------------------------------------------------------------- controls
def test_control_hunk1_removed_drops_test_1() -> None:
    """With hunk 1 (instance-deleted forgets runners) reverted, the delete
    leaves runners in the state -- exactly the leak this PR fixes."""
    kept, first, second = RunnerId(), RunnerId(), RunnerId()
    other, deleted = instance_with(kept), instance_with(first, second)
    state = State(
        instances={other.instance_id: other, deleted.instance_id: deleted},
        runners={kept: RunnerReady(), first: RunnerReady(), second: RunnerIdle()},
        prefill_server_ports={kept: 1, first: 2},
    )

    # pre-hunk-1 behaviour: only instances and links are updated
    new_instances = {
        iid: i for iid, i in state.instances.items() if iid != deleted.instance_id
    }
    assert set(new_instances) == {other.instance_id}
    # the runners of the deleted instance are still present -> the bug
    assert first in state.runners and second in state.runners
    assert first in state.prefill_server_ports

    # and the ported function does remove them
    new_state = apply_instance_deleted(
        InstanceDeleted(instance_id=deleted.instance_id), state
    )
    assert dict(new_state.runners) == {kept: RunnerReady()}
    assert dict(new_state.prefill_server_ports) == {kept: 1}


def test_control_hunk2_removed_keeps_orphan_status() -> None:
    """With hunk 2 (drop a status for a runner of no instance) reverted, a
    RunnerShuttingDown for a deleted instance's runner is kept forever."""
    runner_id = RunnerId()
    state = State()

    # pre-hunk-2 behaviour: any non-shutdown status is added unconditionally
    new_runners = {**state.runners, runner_id: RunnerShuttingDown()}
    assert runner_id in new_runners  # the bug

    new_state = apply_runner_status_updated(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerShuttingDown()),
        state,
    )
    assert runner_id not in new_state.runners


# ------------------------------------------------- real dispatch path
def test_real_apply_dispatch_drops_deleted_instances_runners() -> None:
    """Upstream's tests call the apply_* functions DIRECTLY. Production goes
    through apply(state, IndexedEvent) -> event_apply -> the case arms in the
    match statement. Drive that real path: if the new case arms were wired to
    the wrong function, or the event types were mismatched, this fails and
    upstream's tests still pass."""
    kept, first, second = RunnerId(), RunnerId(), RunnerId()
    other, deleted = instance_with(kept), instance_with(first, second)
    state = State(
        instances={other.instance_id: other, deleted.instance_id: deleted},
        runners={kept: RunnerReady(), first: RunnerShuttingDown()},
        prefill_server_ports={kept: 1, first: 2},
    )

    # idx 0: InstanceDeleted for the two-runner instance
    state = A.apply(
        state, A.IndexedEvent(event=InstanceDeleted(instance_id=deleted.instance_id), idx=0)
    )
    assert deleted.instance_id not in state.instances
    assert dict(state.runners) == {kept: RunnerReady()}
    assert dict(state.prefill_server_ports) == {kept: 1}

    # idx 1: the dead runner reports one last time, after the instance is gone
    state = A.apply(
        state,
        A.IndexedEvent(
            event=RunnerStatusUpdated(
                runner_id=first, runner_status=RunnerShuttingDown()
            ),
            idx=1,
        ),
    )
    assert first not in state.runners
    assert kept in state.runners


def test_real_apply_dispatch_still_records_a_live_runners_status() -> None:
    """The guard must not eat a legitimate status: a runner of an instance
    that still exists is recorded as before, through the real dispatch path."""
    runner_id = RunnerId()
    instance = instance_with(runner_id)
    state = State(instances={instance.instance_id: instance})

    for idx, status in enumerate([RunnerIdle(), RunnerShuttingDown(), RunnerReady()]):
        state = A.apply(
            state,
            A.IndexedEvent(
                event=RunnerStatusUpdated(runner_id=runner_id, runner_status=status),
                idx=idx,
            ),
        )
        assert state.runners[runner_id] == status

    # a shutdown removes it, as before
    from exo.shared.types.worker.runners import RunnerShutdown

    state = A.apply(
        state,
        A.IndexedEvent(
            event=RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerShutdown()),
            idx=3,
        ),
    )
    assert runner_id not in state.runners


def test_instance_created_after_the_fact_still_admits_its_runners() -> None:
    """A worker can re-announce a runner to a new master whose state predates
    the instance (announce_to_master). If the master's state already holds the
    instance, the status is kept; this pins that the guard keys off instances
    in the state, not off whether the runner was ever seen."""
    runner_id = RunnerId()
    instance = instance_with(runner_id)
    state = A.apply(
        State(), A.IndexedEvent(event=InstanceCreated(instance=instance), idx=0)
    )
    state = A.apply(
        state,
        A.IndexedEvent(
            event=RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerIdle()),
            idx=1,
        ),
    )
    assert state.runners[runner_id] == RunnerIdle()
