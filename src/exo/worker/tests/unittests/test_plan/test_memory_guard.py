"""Test: memory pressure guard in _create_runner (auto-restart VRAM check, #1872).

_create_runner() must skip spawning a replacement runner when
virtual memory is > 90% used, and resume creation once pressure
drops back below the threshold.

We drive this by monkeypatching exo.worker.plan.virtual_memory_statistics
(the import location that _create_runner actually reads from).
"""

from dataclasses import dataclass
from unittest.mock import patch

import exo.worker.plan as plan_mod
from exo.shared.types.tasks import CreateRunner
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import RunnerFailed
from exo.utils.keyed_backoff import KeyedBackoff
from exo.utils.virtual_memory import VirtualMemoryStatistics
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
)
from exo.worker.tests.unittests.conftest import (
    FakeRunnerSupervisor,
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)


def _make_instance():
    shard = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=0, world_size=2)
    shard2 = get_pipeline_shard_metadata(model_id=MODEL_A_ID, device_rank=1, world_size=2)
    # Node A owns runner 1, Node B owns runner 2
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={RUNNER_1_ID: shard, RUNNER_2_ID: shard2},
    )
    return instance, shard


def _base_kwargs():
    instance, shard = _make_instance()
    instances = {INSTANCE_1_ID: instance}
    all_runners = {
        # Node B's runner is healthy, so the "failed remote waits for fix" gate
        # does not block.  Node A's entry is a prior RunnerFailed, so the
        # "we_have_failed_before" bypass allows creation again.
        RUNNER_2_ID: FakeRunnerSupervisor(
            bound_instance=BoundInstance(instance=instance, bound_runner_id=RUNNER_2_ID, bound_node_id=NODE_B),
            status=__import__("exo.shared.types.worker.runners", fromlist=["RunnerConnected"]).RunnerConnected(),
        ).status,
        # pretend NODE_A had a prior RunnerFailed observed by the cluster:
        RUNNER_1_ID: RunnerFailed(error_message="OOM", diagnostics=[]),
    }
    # type: ignore — FakeRunnerSupervisor not needed here; RunnerId absent from runners means _create_runner will try
    return dict(
        node_id=NODE_A,
        runners={},  # type: ignore
        global_download_status={},
        instances=instances,
        all_runners=all_runners,
        tasks={},
        input_chunk_buffer={},
        image_cache={},
        instance_backoff=KeyedBackoff(),
        download_backoff=KeyedBackoff(),
    )


def test_create_runner_skipped_when_memory_above_90_percent():
    """When virtual memory is >90% in use, _create_runner must return None (skip restart)."""
    kwargs = _base_kwargs()
    high = VirtualMemoryStatistics(total_bytes=32 * 1024**3, available_bytes=1 * 1024**3)  # ~97% used
    assert high.used_fraction > 0.90
    with patch.object(plan_mod, "virtual_memory_statistics", return_value=high):
        result = plan_mod.plan(**kwargs)  # type: ignore
    assert result is None, "Expected skip when memory > 90%, but got CreateRunner"


def test_create_runner_proceeds_when_memory_below_threshold():
    """When memory pressure is normal (<90%), runner creation must proceed."""
    kwargs = _base_kwargs()
    low = VirtualMemoryStatistics(total_bytes=32 * 1024**3, available_bytes=10 * 1024**3)  # ~69% used
    assert low.used_fraction < 0.90
    with patch.object(plan_mod, "virtual_memory_statistics", return_value=low):
        result = plan_mod.plan(**kwargs)  # type: ignore
    assert isinstance(result, CreateRunner)
    assert result.instance_id == INSTANCE_1_ID


def test_create_runner_at_exactly_90_is_allowed():
    """Exactly 90.0% used must not be treated as 'over threshold'."""
    kwargs = _base_kwargs()
    boundary = VirtualMemoryStatistics(total_bytes=100 * 1024**3, available_bytes=10 * 1024**3)  # 90% used, not >90
    assert boundary.used_fraction == 0.90
    with patch.object(plan_mod, "virtual_memory_statistics", return_value=boundary):
        result = plan_mod.plan(**kwargs)  # type: ignore
    # == 90% must NOT block — only >90 does
    assert isinstance(result, CreateRunner)


def test_create_runner_fail_open_when_memory_check_raises():
    """If virtual_memory_statistics raises, _create_runner must not block — it fails open."""
    kwargs = _base_kwargs()

    def boom():
        raise RuntimeError("psutil broken on this OS")

    with patch.object(plan_mod, "virtual_memory_statistics", side_effect=boom):
        result = plan_mod.plan(**kwargs)  # type: ignore
    assert isinstance(result, CreateRunner), "Guard must fail open on exception"


def test_used_fraction_property_matches_expected():
    """Smoke: VirtualMemoryStatistics.used_fraction behaves as expected."""
    # Drive directly to ensure the guard threshold interpretation is right
    for avail, used_frac in [
        (1 * 1024**3, 0.96875 if 32 * 1024**3 else None),  # 31/32
        (3 * 1024**3, 0.90625),  # 29/32
    ]:
        stats = VirtualMemoryStatistics(total_bytes=32 * 1024**3, available_bytes=avail)
        assert abs(stats.used_fraction - (1 - avail / (32 * 1024**3))) < 1e-9
