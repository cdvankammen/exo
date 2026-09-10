from collections.abc import Mapping

from pydantic import model_validator

from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import Id, NodeId
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.pydantic_ext import FrozenModel, TaggedModel
from exo.worker.runner.diagnostics import KnownRunnerDiagnostic


class RunnerId(Id):
    pass


class RunnerError(Exception):
    pass


class BaseRunnerStatus(TaggedModel):
    def is_running(self):
        return isinstance(self, RunnerRunning)


class RunnerIdle(BaseRunnerStatus):
    pass


class RunnerConnecting(BaseRunnerStatus):
    pass


class RunnerConnected(BaseRunnerStatus):
    pass


class RunnerLoading(BaseRunnerStatus):
    layers_loaded: int = 0
    total_layers: int = 0


class RunnerLoaded(BaseRunnerStatus):
    pass


class RunnerWarmingUp(BaseRunnerStatus):
    pass


class RunnerReady(BaseRunnerStatus):
    prefill_server_port: int | None = None


class RunnerDegraded(BaseRunnerStatus):
    """Runner is operational but at reduced capacity after OOM recovery.
    
    Emitted when the runner catches an OOM, clears KV cache, reduces
    concurrency, and resumes. Downstream consumers can use this to
    adjust load balancing or alert operators.
    """
    reason: str = ""
    original_concurrency: int = 0
    current_concurrency: int = 0
    # Carried so the apply.py prefill_server_ports capture still works after
    # an OOM recovery — the prefill server itself survives clear_memory().
    prefill_server_port: int | None = None


class RunnerRunning(BaseRunnerStatus):
    pass


class RunnerShuttingDown(BaseRunnerStatus):
    pass


class RunnerShutdown(BaseRunnerStatus):
    pass


class RunnerFailed(BaseRunnerStatus):
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
    | RunnerDegraded
    | RunnerRunning
    | RunnerShuttingDown
    | RunnerShutdown
    | RunnerFailed
)


class ShardAssignments(FrozenModel):
    model_id: ModelId
    runner_to_shard: Mapping[RunnerId, ShardMetadata]
    node_to_runner: Mapping[NodeId, RunnerId]

    @model_validator(mode="after")
    def validate_runners_exist(self) -> "ShardAssignments":
        for runner_id in self.node_to_runner.values():
            if runner_id not in self.runner_to_shard:
                raise ValueError(
                    f"Runner {runner_id} in node_to_runner does not exist in runner_to_shard"
                )
        return self
