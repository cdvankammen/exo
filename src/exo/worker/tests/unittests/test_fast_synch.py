"""MLX fast synch is used where it helps (RDMA instances), unless overridden."""

import os
import sys

import pytest

from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.worker.instances import (
    BoundInstance,
    Instance,
    InstanceId,
    MlxJacclInstance,
    TinygradInstance,
)
from exo.shared.types.worker.runners import RunnerId
from exo.worker.runner import bootstrap
from exo.worker.runner.bootstrap import entrypoint, use_fast_synch
from exo.worker.tests.constants import MODEL_A_ID
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
    get_shard_assignments,
)

RUNNER = RunnerId()
NODE = NodeId("node")


def ring() -> Instance:
    return get_mlx_ring_instance(
        instance_id=InstanceId(),
        model_id=MODEL_A_ID,
        node_to_runner={NODE: RUNNER},
        runner_to_shard={
            RUNNER: get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0)
        },
    )


def rdma() -> Instance:
    return MlxJacclInstance(
        instance_id=InstanceId(),
        shard_assignments=get_shard_assignments(
            MODEL_A_ID,
            {NODE: RUNNER},
            {RUNNER: get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0)},
        ),
        # One rank, one peer slot, no physical link: on this fork jaccl_devices is
        # a matrix of per-link interface-name lists, so an empty inner list is the
        # type-correct stand-in for upstream's `[[None]]`.
        jaccl_devices=[[[]]],
        jaccl_coordinators={NODE: "127.0.0.1:5000"},
    )


def tinygrad() -> Instance:
    return TinygradInstance(
        instance_id=InstanceId(),
        shard_assignments=get_shard_assignments(
            MODEL_A_ID,
            {NODE: RUNNER},
            {RUNNER: get_pipeline_shard_metadata(MODEL_A_ID, device_rank=0)},
        ),
        device_backend_by_node={NODE: Backend.TinygradCpu},
    )


@pytest.mark.parametrize(
    ("instance", "override", "expected"),
    [
        (ring(), None, False),
        (rdma(), None, True),
        (tinygrad(), None, False),
        (ring(), "true", True),
        (rdma(), "false", False),
        (ring(), "false", False),
        (rdma(), "true", True),
    ],
)
def test_fast_synch_choice(
    instance: Instance, override: str | None, expected: bool
) -> None:
    assert use_fast_synch(instance, override) is expected


class _DeadSender:
    """Stands in for the MpSender; the runner must never be reached."""

    def send(self, event: object) -> None:
        raise AssertionError("entrypoint reached the runner body")

    def close(self) -> None:
        pass

    def join(self) -> None:
        pass


class _DeadReceiver:
    def close(self) -> None:
        pass

    def join(self) -> None:
        pass


class _StopAfterEnvWrite(BaseException):
    """End the entrypoint right after the fast-synch write.

    Deliberately a BaseException: entrypoint's own `except Exception` handler
    converts an ordinary exception into a RunnerTerminationError and re-raises
    SystemExit, so an Exception subclass never reaches the caller.
    """


def _run_entrypoint(instance: Instance, override: str | None) -> str | None:
    """Drive entrypoint far enough to write the env var, then read it back.

    entrypoint goes on to import and run a real Runner, so stop it immediately
    after the fast-synch write: `apply_shard_backend` is called next and before
    `resolve_builder` reaches mlx_lm (which cannot import on this machine - no
    `mlx_lm.models.deepseek_v4`). Patching it to raise is deterministic, and the
    env var has already been written by then, so what we read back is exactly
    what a real runner process would inherit.
    """
    bound_instance = BoundInstance(
        instance=instance,
        bound_runner_id=RUNNER,
        bound_node_id=NODE,
    )

    saved = {
        key: os.environ.get(key)
        for key in (
            "MLX_METAL_FAST_SYNCH",
            "EXO_FAST_SYNCH",
            "EXO_DISABLE_ORPHAN_WATCHDOG",
        )
    }
    os.environ["EXO_DISABLE_ORPHAN_WATCHDOG"] = "1"
    if override is None:
        os.environ.pop("EXO_FAST_SYNCH", None)
    else:
        os.environ["EXO_FAST_SYNCH"] = override
    os.environ.pop("MLX_METAL_FAST_SYNCH", None)

    def _stop_after_env_write(backend: object) -> None:
        raise _StopAfterEnvWrite

    original = bootstrap.apply_shard_backend
    bootstrap.apply_shard_backend = _stop_after_env_write  # type: ignore[assignment]
    try:
        with pytest.raises(_StopAfterEnvWrite):
            entrypoint(
                bound_instance=bound_instance,
                event_sender=_DeadSender(),  # type: ignore[arg-type]
                task_receiver=_DeadReceiver(),  # type: ignore[arg-type]
                cancel_receiver=_DeadReceiver(),  # type: ignore[arg-type]
                _logger=bootstrap.logger,  # type: ignore[arg-type]
            )
        return os.environ.get("MLX_METAL_FAST_SYNCH")
    finally:
        bootstrap.apply_shard_backend = original
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="the fast synch env var is only written on Darwin",
)
@pytest.mark.parametrize(
    ("instance", "override", "expected"),
    [
        (ring(), None, "0"),
        (rdma(), None, "1"),
        (tinygrad(), None, "0"),
        (ring(), "true", "1"),
        (rdma(), "false", "0"),
    ],
)
def test_entrypoint_writes_fast_synch_env(
    instance: Instance, override: str | None, expected: str
) -> None:
    """entrypoint must export the decision, not merely compute it."""
    assert _run_entrypoint(instance, override) == expected