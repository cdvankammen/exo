"""Runner status types.

:class:`RunnerId`, :class:`RunnerError`, the full runner state family, and :class:`ShardAssignments`."""

from collections.abc import Mapping

from pydantic import model_validator

from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import Id, NodeId
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.pydantic_ext import FrozenModel, TaggedModel
from exo.worker.runner.diagnostics import KnownRunnerDiagnostic


class RunnerId(Id):
    """Identifier of a runner."""


class RunnerError(Exception):
    """Raised for runner-level errors."""


class BaseRunnerStatus(TaggedModel):
    """Base class for runner status variants (tagged by status type)."""

    def is_running(self):
        """Whether the runner is currently executing a task."""
        return isinstance(self, RunnerRunning)


class RunnerIdle(BaseRunnerStatus):
    """Runner is idle and available."""


class RunnerConnecting(BaseRunnerStatus):
    """Runner is establishing its connection."""


class RunnerConnected(BaseRunnerStatus):
    """Runner has connected to the swarm."""


class RunnerLoading(BaseRunnerStatus):
    """Runner is loading model layers (progress tracked by counts)."""

    layers_loaded: int = 0
    total_layers: int = 0


class RunnerLoaded(BaseRunnerStatus):
    """Runner has finished loading the model into memory."""


class RunnerWarmingUp(BaseRunnerStatus):
    """Runner is warming the model up (prefill pass)."""


class RunnerReady(BaseRunnerStatus):
    """Runner is ready to serve; prefill server port when applicable."""

    prefill_server_port: int | None = None


class RunnerRunning(BaseRunnerStatus):
    """Runner is actively executing a task."""


class RunnerDegraded(RunnerRunning):
    """Runner is still executing but has degraded after an OOM: the KV cache
    was cleared, the admission limit was halved, and active tasks were
    restarted. Inherits from RunnerRunning so every isinstance-based consumer
    (circuit breaker, dashboard, plan) treats the node as alive."""

    max_batch_size: int | None = None
    reason: str | None = None


class RunnerShuttingDown(BaseRunnerStatus):
    """Runner is shutting down."""


class RunnerShutdown(BaseRunnerStatus):
    """Runner has shut down."""


class RunnerFailed(BaseRunnerStatus):
    """Runner failed, with message and known diagnostics."""

    error_message: str | None = None
    diagnostics: list[KnownRunnerDiagnostic]


RunnerStatus = (
    RunnerIdle
    | RunnerConnecting
    | RunnerConnected
    | RunnerLoading
    | RunnerLoaded
    | RunnerWarmingUp
    | RunnerReady
    | RunnerRunning
    | RunnerDegraded
    | RunnerShuttingDown
    | RunnerShutdown
    | RunnerFailed
)


class ShardAssignments(FrozenModel):
    """Runner↔shard and node↔runner mapping for a model instance."""

    model_id: ModelId
    runner_to_shard: Mapping[RunnerId, ShardMetadata]
    node_to_runner: Mapping[NodeId, RunnerId]

    @model_validator(mode="after")
    def validate_runners_exist(self) -> "ShardAssignments":
        """Require every node-mapped runner to exist in runner_to_shard."""
        for runner_id in self.node_to_runner.values():
            if runner_id not in self.runner_to_shard:
                raise ValueError(
                    f"Runner {runner_id} in node_to_runner does not exist in runner_to_shard"
                )
        return self
