"""Task envelope types executed by nodes.

:class:`BaseTask` and subclasses for generation, downloads, model loading, group connect, warmup, cancellation, and shutdown."""

from enum import Enum

from pydantic import Field

from exo.api.types import (
    ImageEditsTaskParams,
    ImageGenerationTaskParams,
)
from exo.shared.types.common import CommandId, Id
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.pydantic_ext import TaggedModel


class TaskId(Id):
    """Identifier of a task."""


CANCEL_ALL_TASKS = TaskId("CANCEL_ALL_TASKS")


class TaskStatus(str, Enum):
    """Lifecycle status of a task."""

    Pending = "Pending"
    Running = "Running"
    Complete = "Complete"
    TimedOut = "TimedOut"
    Failed = "Failed"
    Cancelled = "Cancelled"


class BaseTask(TaggedModel):
    """Base class for all tasks executed by nodes (tagged by task type)."""

    task_id: TaskId = Field(default_factory=TaskId)
    task_status: TaskStatus = Field(default=TaskStatus.Pending)
    instance_id: InstanceId


class CreateRunner(BaseTask):  # emitted by Worker
    """Request to spawn a runner bound to an instance, emitted by the worker."""

    bound_instance: BoundInstance


class DownloadModel(BaseTask):  # emitted by Worker
    """Request to download a model shard, emitted by the worker."""

    shard_metadata: ShardMetadata


class LoadModel(BaseTask):  # emitted by Worker
    """Request to load a model into memory, emitted by the worker."""



class ConnectToGroup(BaseTask):  # emitted by Worker
    """Request to join a group, emitted by the worker."""



class StartWarmup(BaseTask):  # emitted by Worker
    """Request to begin model warmup, emitted by the worker."""



class TextGeneration(BaseTask):  # emitted by Master
    """Text-generation task issued by the master; carries error fields."""

    command_id: CommandId
    task_params: TextGenerationTaskParams

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class CancelTask(BaseTask):
    """Request to cancel a task and shut down its runner."""

    cancelled_task_id: TaskId
    runner_id: RunnerId


class ImageGeneration(BaseTask):  # emitted by Master
    """Image-generation task issued by the master; carries error fields."""

    command_id: CommandId
    task_params: ImageGenerationTaskParams

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class ImageEdits(BaseTask):  # emitted by Master
    """Image-editing task issued by the master; carries error fields."""

    command_id: CommandId
    task_params: ImageEditsTaskParams

    error_type: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class Shutdown(BaseTask):  # emitted by Worker
    """Request to shut down a runner, emitted by the worker."""

    runner_id: RunnerId


Task = (
    CreateRunner
    | DownloadModel
    | ConnectToGroup
    | LoadModel
    | StartWarmup
    | TextGeneration
    | CancelTask
    | ImageGeneration
    | ImageEdits
    | Shutdown
)
TextTask = TextGeneration
ImageTask = ImageGeneration | ImageEdits
GenerationTask = TextTask | ImageTask
