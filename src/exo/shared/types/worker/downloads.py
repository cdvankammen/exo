from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import ShardMetadata
from exo.utils.pydantic_ext import FrozenModel, TaggedModel


class DownloadProgressData(FrozenModel):
    total: Memory
    downloaded: Memory
    downloaded_this_session: Memory

    completed_files: int
    total_files: int

    speed: float
    eta_ms: int

    files: dict[str, "DownloadProgressData"]


class BaseDownloadProgress(TaggedModel):
    node_id: NodeId
    shard_metadata: ShardMetadata
    model_directory: str = ""


class DownloadPending(BaseDownloadProgress):
    downloaded: Memory = Memory()
    total: Memory = Memory()


class DownloadCompleted(BaseDownloadProgress):
    total: Memory
    read_only: bool = False


class DownloadFailed(BaseDownloadProgress):
    error_message: str


class DownloadOngoing(BaseDownloadProgress):
    download_progress: DownloadProgressData


class DownloadStalled(BaseDownloadProgress):
    """A download that has made zero progress for longer than the stall timeout.

    Unlike DownloadFailed (which is terminal), a stalled download can be
    retried. The coordinator moves downloads here when its watchdog detects
    no byte progress for EXO_DOWNLOAD_STALL_TIMEOUT_SECS seconds, then cancels
    the underlying task. The dashboard renders these with a warning so the
    user can retry instead of the queue being silently blocked.
    """

    error_message: str = "download stalled (no progress for extended period)"
    stalled_at: float = 0.0
    downloaded: Memory = Memory()
    total: Memory = Memory()


DownloadProgress = (
    DownloadPending
    | DownloadCompleted
    | DownloadFailed
    | DownloadOngoing
    | DownloadStalled
)


class ModelSafetensorsIndexMetadata(BaseModel):
    total_size: PositiveInt | None = None


class ModelSafetensorsIndex(BaseModel):
    metadata: ModelSafetensorsIndexMetadata | None
    weight_map: dict[str, str]


class FileListEntry(BaseModel):
    type: Literal["file", "directory"]
    path: str
    size: int | None = None


class RepoFileDownloadProgress(BaseModel):
    repo_id: str
    repo_revision: str
    file_path: str
    downloaded: Memory
    downloaded_this_session: Memory
    total: Memory
    speed: float
    eta: timedelta
    status: Literal["not_started", "in_progress", "complete"]
    start_time: float

    model_config = ConfigDict(frozen=True)


class RepoDownloadProgress(BaseModel):
    repo_id: str
    repo_revision: str
    shard: ShardMetadata
    completed_files: int
    total_files: int
    downloaded: Memory
    downloaded_this_session: Memory
    total: Memory
    overall_speed: float
    overall_eta: timedelta
    status: Literal["not_started", "in_progress", "complete"]
    file_progress: dict[str, RepoFileDownloadProgress] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)
