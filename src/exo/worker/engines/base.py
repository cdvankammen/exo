from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from typing import BinaryIO

from exo.shared.types.chunks import Chunk
from exo.shared.types.tasks import CANCEL_ALL_TASKS, GenerationTask, TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    ModelLoadingResponse,
)
from exo.worker.disaggregated.server import PrefillRequest


class Engine(ABC):
    _cancelled_tasks: set[TaskId]

    def should_cancel(self, task_id: TaskId) -> bool:
        return (
            task_id in self._cancelled_tasks
            or CANCEL_ALL_TASKS in self._cancelled_tasks
        )

    @abstractmethod
    def warmup(self) -> None: ...

    @abstractmethod
    def submit(
        self,
        task: GenerationTask,
    ) -> None: ...

    @abstractmethod
    def step(
        self,
    ) -> Iterable[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]]: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None: ...

    def clear_memory(self) -> None:
        """Release cached state (KV prefix cache, activation buffers, etc.) to
        recover from out-of-memory conditions. Implementations should:
        
        1. Clear the KV prefix cache (drop all cached prefixes)
        2. Call mx.clear_cache() to release GPU memory pools
        3. Run gc.collect() to reclaim CPU-side reference cycles
        
        This does NOT touch the loaded model weights. After calling this,
        the engine remains functional but will need to re-prefill all
        prompts from scratch.
        
        The default implementation handles this for all engines that store
        state on the ``kv_prefix_cache`` attribute. Subclasses that store
        additional GPU state should override and call super().
        """
        kv = getattr(self, "kv_prefix_cache", None)
        if kv is not None:
            kv.clear()
        import gc
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except ImportError:
            pass


class Builder(ABC):
    @abstractmethod
    def connect(self, bound_instance: BoundInstance) -> None: ...

    @abstractmethod
    def load(
        self,
        bound_instance: BoundInstance,
    ) -> Generator[ModelLoadingResponse]: ...

    @abstractmethod
    def build(self) -> Engine: ...

    @abstractmethod
    def close(self) -> None: ...
