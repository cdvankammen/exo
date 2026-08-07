import ctypes
import glob
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self, cast

from exo.shared.types.memory import Memory
from exo.shared.types.thunderbolt import ThunderboltIdentifier
from exo.utils.pydantic_ext import FrozenModel
from exo.utils.virtual_memory import swap_memory_statistics, virtual_memory_statistics


class MemoryUsage(FrozenModel):
    ram_total: Memory
    ram_available: Memory
    swap_total: Memory
    swap_available: Memory

    @classmethod
    def from_bytes(
        cls, *, ram_total: int, ram_available: int, swap_total: int, swap_available: int
    ) -> Self:
        return cls(
            ram_total=Memory.from_bytes(ram_total),
            ram_available=Memory.from_bytes(ram_available),
            swap_total=Memory.from_bytes(swap_total),
            swap_available=Memory.from_bytes(swap_available),
        )

    @classmethod
    def from_psutil(cls, *, override_memory: int | None) -> Self:
        virtual_memory = virtual_memory_statistics()
        swap_memory = swap_memory_statistics()

        return cls.from_bytes(
            ram_total=virtual_memory.total_bytes,
            ram_available=virtual_memory.available_bytes
            if override_memory is None
            else override_memory,
            swap_total=swap_memory.total_bytes,
            swap_available=swap_memory.free_bytes,
        )

    @classmethod
    def from_system(cls, *, override_memory: int | None) -> Self:
        """Report system memory via the psutil-compatible robust helpers.

        On macOS 26 (Darwin 27) psutil's host_statistics64 calls can fail with
        "array not large enough" (kernel grew vm_statistics64). The helpers in
        exo.utils.virtual_memory fall back to direct kernel queries on Darwin.
        """
        virtual_memory = virtual_memory_statistics()
        swap_memory = swap_memory_statistics()

        return cls.from_bytes(
            ram_total=virtual_memory.total_bytes,
            ram_available=virtual_memory.available_bytes
            if override_memory is None
            else override_memory,
            swap_total=swap_memory.total_bytes,
            swap_available=swap_memory.free_bytes,
        )

    @classmethod
    def from_cuda(cls, *, override_memory: int | None) -> Self | None:
        """Report a CUDA GPU's VRAM as the node's memory.

        On a discrete NVIDIA GPU the memory that actually bounds MLX inference is
        the GPU's VRAM, not system RAM (unlike Apple Silicon's unified memory).
        Returns None when no GPU/VRAM can be queried so the caller can fall back
        to :meth:`from_psutil`.
        """
        vram = _query_cuda_vram_bytes()
        if vram is None:
            return None
        total_vram, free_vram = vram
        sm = swap_memory_statistics()
        return cls.from_bytes(
            ram_total=total_vram,
            ram_available=free_vram if override_memory is None else override_memory,
            swap_total=sm.total_bytes,
            swap_available=sm.free_bytes,
        )


def _find_nvidia_smi_binary() -> str | None:
    """Locate the ``nvidia-smi`` binary, tolerating version-suffixed names.

    Standard NVIDIA driver installs put ``nvidia-smi`` on PATH. Manual
    driver installs (e.g. extracting NVIDIA's ``.run`` to match a kernel
    module version) can leave version-suffixed binaries such as
    ``nvidia-smi-595.58`` — which ``shutil.which`` would miss. As a
    fallback, scan PATH directories for executables matching
    ``nvidia-smi*`` (sorted, so plain ``nvidia-smi`` wins when both exist).
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        return nvidia_smi
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if not directory:
            continue
        try:
            candidates = sorted(glob.glob(os.path.join(directory, "nvidia-smi*")))
        except OSError:
            continue
        for candidate in candidates:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


def _query_cuda_vram_bytes() -> tuple[int, int] | None:
    """Total and free VRAM in bytes for the first CUDA GPU.

    Tries ``nvidia-smi`` first (tolerating version-suffixed binary names),
    then the NVML driver library directly. Returns None when neither is
    available or the output cannot be parsed, so callers fall back to
    system RAM.
    """
    nvidia_smi = _find_nvidia_smi_binary()
    if nvidia_smi is not None:
        try:
            completed = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
        except (subprocess.SubprocessError, OSError):
            pass
        else:
            lines = completed.stdout.strip().splitlines()
            if lines:
                fields = lines[0].split(",")
                if len(fields) == 2:
                    try:
                        total_mebibytes = int(fields[0].strip())
                        free_mebibytes = int(fields[1].strip())
                    except ValueError:
                        pass
                    else:
                        mebibyte = 1024 * 1024
                        return (
                            total_mebibytes * mebibyte,
                            free_mebibytes * mebibyte,
                        )
    return _query_cuda_vram_bytes_nvml()


def _query_cuda_vram_bytes_nvml() -> tuple[int, int] | None:
    """Query VRAM via the NVML driver library (``libnvidia-ml.so.1``).

    Fallback used when no ``nvidia-smi`` binary is on PATH. The driver
    library ships with every NVIDIA driver (including minimal/container
    installs), so this removes the binary-name dependency entirely.
    Returns None on any error — never raises.
    """
    try:
        lib = ctypes.CDLL("libnvidia-ml.so.1")
    except OSError:
        return None
    try:
        if lib.nvmlInit_v2() != 0:
            return None
        try:
            device_count = ctypes.c_uint()
            if (
                lib.nvmlDeviceGetCount_v2(ctypes.byref(device_count)) != 0
                or device_count.value == 0
            ):
                return None
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
                return None
            memory = _NvmlMemory()
            if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0:
                return None
            return cast(int, memory.total), cast(int, memory.free)
        finally:
            lib.nvmlShutdown()
    except (AttributeError, OSError):
        return None


class _NvmlMemory(ctypes.Structure):
    """``nvmlMemory_t``: total/free/used bytes for a device."""

    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


class DiskUsage(FrozenModel):
    """Disk space usage for the models directory."""

    total: Memory
    available: Memory

    @classmethod
    def from_path(cls, path: Path) -> Self:
        """Get disk usage stats for the partition containing path."""
        total, _used, free = shutil.disk_usage(path)
        return cls(
            total=Memory.from_bytes(total),
            available=Memory.from_bytes(free),
        )


class SystemPerformanceProfile(FrozenModel):
    # TODO: flops_fp16: float

    gpu_usage: float = 0.0
    temp: float = 0.0
    sys_power: float = 0.0
    pcpu_usage: float = 0.0
    ecpu_usage: float = 0.0


InterfaceType = Literal["wifi", "ethernet", "maybe_ethernet", "thunderbolt", "unknown"]


class NetworkInterfaceInfo(FrozenModel):
    name: str
    ip_address: str
    interface_type: InterfaceType = "unknown"


class NodeIdentity(FrozenModel):
    """Static and slow-changing node identification data."""

    model_id: str = "Unknown"
    chip_id: str = "Unknown"
    friendly_name: str = "Unknown"
    os_version: str = "Unknown"
    os_build_version: str = "Unknown"


class NodeNetworkInfo(FrozenModel):
    """Network interface information for a node."""

    interfaces: Sequence[NetworkInterfaceInfo] = []


class NodeThunderboltInfo(FrozenModel):
    """Thunderbolt interface identifiers for a node."""

    interfaces: Sequence[ThunderboltIdentifier] = []


class NodeRdmaCtlStatus(FrozenModel):
    """Whether RDMA is enabled on this node (via rdma_ctl)."""

    enabled: bool


class ThunderboltBridgeStatus(FrozenModel):
    """Whether the Thunderbolt Bridge network service is enabled on this node."""

    enabled: bool
    exists: bool
    service_name: str | None = None
