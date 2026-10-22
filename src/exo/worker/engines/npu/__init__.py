"""Heterogeneous NPU and GPU pipelining engine for low-power edge acceleration."""

from __future__ import annotations

from .builder import NpuBuilder
from .device import NpuArchitecture, NpuDeviceProfile
from .engine import NpuEngine
from .pipeline_coordinator import PipelinedInferenceCoordinator
from .pipelined_scheduler import HeterogeneousPipelinedScheduler
from .runtime import NpuExecutionSession

__all__ = [
    "HeterogeneousPipelinedScheduler",
    "NpuArchitecture",
    "NpuBuilder",
    "NpuDeviceProfile",
    "NpuEngine",
    "NpuExecutionSession",
    "PipelinedInferenceCoordinator",
]
