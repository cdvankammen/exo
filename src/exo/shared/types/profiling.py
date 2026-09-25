"""Performance profiling types and CUDA VRAM query helpers.

Memory/disk/system/network profile models plus nvidia-smi/NVML-backed VRAM queries."""

import ctypes
import glob
import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self, cast

from loguru import logger

from exo.shared.types.memory import Memory
from exo.shared.types.thunderbolt import ThunderboltIdentifier
from exo.utils.pydantic_ext import FrozenModel
from exo.utils.virtual_memory import swap_memory_statistics, virtual_memory_statistics


class GpuMemoryInfo(FrozenModel):
    """Per-device accelerator (GPU) memory report.

    Carries one entry per physical CUDA device so placement can reason about
    individual GPUs instead of a single total/max aggregate. ``index`` is the
    NVML/nvidia-smi device index (0-based), ``total``/``free`` are in bytes.
    """

    index: int
    total: Memory
    free: Memory

    @classmethod
    def from_bytes(cls, *, index: int, total: int, free: int) -> Self:
        return cls(
            index=index,
            total=Memory.from_bytes(total),
            free=Memory.from_bytes(free),
        )

    @property
    def used(self) -> Memory:
        return Memory.from_bytes(max(self.total.in_bytes - self.free.in_bytes, 0))


class MemoryUsage(FrozenModel):
    ram_total: Memory
    ram_available: Memory
    swap_total: Memory
    swap_available: Memory
    # TODO #13: memory pressure instead of memory used — derived at construction.
    ram_used: Memory = Memory()
    pressure: float = 0.0
    # Dedicated accelerator memory (e.g. CUDA VRAM). None on unified-memory
    # platforms, where system RAM is the accelerator memory.
    accelerator_total: Memory | None = None
    accelerator_available: Memory | None = None
    # Per-device accelerator memory (one entry per physical GPU). Empty on
    # unified-memory platforms or when per-device queries are unavailable.
    # Placement uses this for accurate per-GPU admission (a multi-GPU box may
    # run MLX CUDA on ONE device, so summed VRAM over-admits).
    accelerator_devices: Sequence[GpuMemoryInfo] = []

    @property
    def inference_available(self) -> Memory:
        """Memory actually available to hold model state on this node.

        On discrete-GPU nodes the accelerator's free memory bounds what
        inference can use, regardless of how much system RAM is free.
        """
        if self.accelerator_available is None:
            return self.ram_available
        return min(self.ram_available, self.accelerator_available)

    @property
    def _reportable_devices(self) -> Sequence[GpuMemoryInfo]:
        """Per-device entries that describe real VRAM.

        A device reporting a zero total is a disabled/unsupported GPU that
        enumerates without usable memory. It must not contribute: including it
        would drag both per-device figures to 0 even though the node has RAM.
        Filters on ``total`` rather than ``free`` so a genuinely full GPU is
        still reported (a full GPU cannot hold a model either, and that is
        information placement needs).
        """
        return tuple(
            device for device in self.accelerator_devices if device.total.in_bytes > 0
        )

    @property
    def largest_device_available(self) -> Memory:
        """Free memory of the largest single accelerator device.

        MLX CUDA uses exactly ONE GPU per process, so summed VRAM across
        devices would over-admit models that only fit a single device.
        Placement (e.g. single-node Pipeline preference) should consult this
        when a node reports per-device GPU memory. Falls back to
        ``inference_available`` when no usable per-device report exists.
        """
        devices = self._reportable_devices
        if not devices:
            return self.inference_available
        return max((device.free for device in devices), key=lambda m: m.in_bytes)

    @property
    def summed_device_available(self) -> Memory:
        """Sum of free memory across ALL accelerator devices.

        Useful for sharding modes that CAN span devices (Tensor parallelism,
        multi-GPU MLX via config), where the aggregate free VRAM is the real
        budget. Falls back to ``inference_available`` when no usable
        per-device report exists.
        """
        devices = self._reportable_devices
        if not devices:
            return self.inference_available
        return sum((device.free for device in devices), start=Memory())

    @classmethod
    def from_bytes(
        cls,
        *,
        ram_total: int,
        ram_available: int,
        swap_total: int,
        swap_available: int,
        accelerator_total: int | None = None,
        accelerator_available: int | None = None,
        accelerator_devices: Sequence[GpuMemoryInfo] = (),
    ) -> Self:
        used_bytes = max(ram_total - ram_available, 0)
        pressure = used_bytes / ram_total if ram_total > 0 else 0.0
        # swap_available must never go negative or exceed total when a meter
        # transiently reports swap_usage > swap_total (macmon can); clamp to
        # [0, swap_total] at the conversion boundary like the ram side.
        swap_available = max(0, min(swap_available, swap_total))
        return cls(
            ram_total=Memory.from_bytes(ram_total),
            ram_available=Memory.from_bytes(ram_available),
            swap_total=Memory.from_bytes(swap_total),
            swap_available=Memory.from_bytes(swap_available),
            ram_used=Memory.from_bytes(used_bytes),
            pressure=round(min(max(pressure, 0.0), 1.0), 4),
            accelerator_total=(
                Memory.from_bytes(accelerator_total)
                if accelerator_total is not None
                else None
            ),
            accelerator_available=(
                Memory.from_bytes(accelerator_available)
                if accelerator_available is not None
                else None
            ),
            accelerator_devices=list(accelerator_devices),
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
        accelerator = _query_cuda_vram_bytes()
        devices = _query_cuda_vram_devices() or []
        used_bytes = max(
            virtual_memory.total_bytes
            - (
                virtual_memory.available_bytes
                if override_memory is None
                else override_memory
            ),
            0,
        )
        pressure = used_bytes / virtual_memory.total_bytes if virtual_memory.total_bytes > 0 else 0.0

        return cls(
            ram_total=Memory.from_bytes(virtual_memory.total_bytes),
            ram_available=Memory.from_bytes(
                virtual_memory.available_bytes
                if override_memory is None
                else override_memory
            ),
            swap_total=Memory.from_bytes(swap_memory.total_bytes),
            swap_available=Memory.from_bytes(swap_memory.free_bytes),
            ram_used=Memory.from_bytes(used_bytes),
            pressure=round(min(max(pressure, 0.0), 1.0), 4),
            accelerator_total=(
                Memory.from_bytes(accelerator[0]) if accelerator is not None else None
            ),
            accelerator_available=(
                Memory.from_bytes(accelerator[1]) if accelerator is not None else None
            ),
            accelerator_devices=devices,
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
        devices = _query_cuda_vram_devices() or []
        sm = swap_memory_statistics()
        used_bytes = max(total_vram - (free_vram if override_memory is None else override_memory), 0)
        pressure = used_bytes / total_vram if total_vram > 0 else 0.0
        return cls(
            ram_total=Memory.from_bytes(total_vram),
            ram_available=Memory.from_bytes(free_vram if override_memory is None else override_memory),
            swap_total=Memory.from_bytes(sm.total_bytes),
            swap_available=Memory.from_bytes(sm.free_bytes),
            ram_used=Memory.from_bytes(used_bytes),
            pressure=round(min(max(pressure, 0.0), 1.0), 4),
            accelerator_total=Memory.from_bytes(total_vram),
            accelerator_available=Memory.from_bytes(free_vram),
            accelerator_devices=devices,
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


def _query_cuda_vram_devices() -> list[GpuMemoryInfo] | None:
    """Per-device VRAM report (one entry per physical CUDA GPU).

    Parses ``nvidia-smi --query-gpu=index,memory.total,memory.free`` (with the
    same version-suffixed binary tolerance as :func:`_query_cuda_vram_bytes`),
    falling back to NVML iteration. Returns None when neither is available or
    the output cannot be parsed, so callers can treat it as "no per-device
    data" and continue with aggregate values.
    """
    nvidia_smi = _find_nvidia_smi_binary()
    if nvidia_smi is not None:
        try:
            completed = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=index,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "nvidia-smi timed out querying per-device VRAM — falling back to NVML"
            )
            return _query_cuda_vram_devices_nvml()
        except (subprocess.SubprocessError, OSError):
            pass
        else:
            devices: list[GpuMemoryInfo] = []
            for line in completed.stdout.strip().splitlines():
                fields = line.split(",")
                if len(fields) != 3:
                    return _query_cuda_vram_devices_nvml()
                try:
                    index = int(fields[0].strip())
                    total_mebibytes = int(fields[1].strip())
                    free_mebibytes = int(fields[2].strip())
                except ValueError:
                    return _query_cuda_vram_devices_nvml()
                mebibyte = 1024 * 1024
                total_bytes = total_mebibytes * mebibyte
                # Skip devices reporting zero total (a disabled/unsupported GPU
                # enumerates but has no VRAM). The scalar query drops the whole
                # reading to None in this situation, so keeping the device here
                # would make the two queries disagree on the same host and let a
                # phantom 0/0 entry drive largest_device_available to 0.
                if total_bytes <= 0:
                    continue
                devices.append(
                    GpuMemoryInfo.from_bytes(
                        index=index,
                        total=total_bytes,
                        free=free_mebibytes * mebibyte,
                    )
                )
            if devices:
                return devices
    return _query_cuda_vram_devices_nvml()


def _query_cuda_vram_devices_nvml() -> list[GpuMemoryInfo] | None:
    """Per-device VRAM via the NVML driver library (``libnvidia-ml.so.1``).

    Fallback used when no ``nvidia-smi`` binary is on PATH. Returns one entry
    per device, preserving NVML device indices. Returns None on any error —
    never raises.
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
            devices: list[GpuMemoryInfo] = []
            for index in range(device_count.value):
                handle = ctypes.c_void_p()
                if (
                    lib.nvmlDeviceGetHandleByIndex_v2(
                        index, ctypes.byref(handle)
                    )
                    != 0
                ):
                    continue
                memory = _NvmlMemory()
                if lib.nvmlDeviceGetMemoryInfo(
                    handle, ctypes.byref(memory)
                ) != 0:
                    continue
                # Same zero-total rule as the nvidia-smi branch above: a
                # disabled GPU enumerates with no VRAM and must not become a
                # phantom device entry.
                if cast(int, memory.total) <= 0:
                    continue
                devices.append(
                    GpuMemoryInfo.from_bytes(
                        index=index,
                        total=cast(int, memory.total),
                        free=cast(int, memory.free),
                    )
                )
            return devices or None
        finally:
            lib.nvmlShutdown()
    except (AttributeError, OSError):
        return None


def _query_cuda_vram_bytes() -> tuple[int, int] | None:
    """Total and free VRAM in bytes for the LARGEST single CUDA GPU.

    MLX CUDA (0.32) uses ONE device per process — ``mx.default_device()`` is
    ``Device(gpu, 0)`` regardless of how many GPUs a box has. Summing every
    GPU's VRAM (the previous behaviour) advertised 2×16GB as 32GB, so
    placement accepted models like Qwen3.6-35B-A3B (19.5GB) that then OOM'd
    loading into a single 16GB device: ``cudaMallocAsync ... out of memory``.
    Reporting the largest single device keeps placement honest: a model is
    only placed when it fits the one GPU MLX will actually use. Tries
    ``nvidia-smi`` first (tolerating version-suffixed binary names), then the
    NVML driver library directly. Returns None when neither is available or
    the output cannot be parsed, so callers fall back to system RAM.
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
                timeout=3,
                check=True,
            )
        except subprocess.TimeoutExpired:
            # A hung nvidia-smi (GPU driver in D-state) must NEVER propagate —
            # it previously bubbled up through from_cuda() and, combined with a
            # SIGTERM, hard-killed the whole EXO process. Fall back to NVML or None.
            logger.warning("nvidia-smi timed out querying VRAM — falling back to NVML")
            return _query_cuda_vram_bytes_nvml()
        except (subprocess.SubprocessError, OSError):
            pass
        else:
            lines = completed.stdout.strip().splitlines()
            max_total_mebibytes = 0
            max_free_mebibytes = 0
            for line in lines:
                fields = line.split(",")
                if len(fields) != 2:
                    return _query_cuda_vram_bytes_nvml()
                try:
                    total_mebibytes = int(fields[0].strip())
                    free_mebibytes = int(fields[1].strip())
                except ValueError:
                    return _query_cuda_vram_bytes_nvml()
                if total_mebibytes > max_total_mebibytes:
                    max_total_mebibytes = total_mebibytes
                    max_free_mebibytes = free_mebibytes
            if max_total_mebibytes > 0:
                mebibyte = 1024 * 1024
                return (
                    max_total_mebibytes * mebibyte,
                    max_free_mebibytes * mebibyte,
                )
    return _query_cuda_vram_bytes_nvml()


def _query_cuda_vram_bytes_nvml() -> tuple[int, int] | None:
    """Query VRAM via the NVML driver library (``libnvidia-ml.so.1``).

    Fallback used when no ``nvidia-smi`` binary is on PATH. The driver
    library ships with every NVIDIA driver (including minimal/container
    installs), so this removes the binary-name dependency entirely.
    Returns the LARGEST single device's VRAM (MLX CUDA uses one GPU per
    process), NOT the sum across devices.
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
            max_total_bytes = 0
            max_free_bytes = 0
            for index in range(device_count.value):
                handle = ctypes.c_void_p()
                if (
                    lib.nvmlDeviceGetHandleByIndex_v2(
                        index, ctypes.byref(handle)
                    )
                    != 0
                ):
                    continue
                memory = _NvmlMemory()
                if lib.nvmlDeviceGetMemoryInfo(
                    handle, ctypes.byref(memory)
                ) != 0:
                    continue
                total = cast(int, memory.total)
                if total > max_total_bytes:
                    max_total_bytes = total
                    max_free_bytes = cast(int, memory.free)
            if max_total_bytes == 0:
                return None
            return max_total_bytes, max_free_bytes
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
    # Negotiated link speed in megabits per second (None when the OS cannot
    # report it, e.g. Wi-Fi on some platforms or virtual interfaces).
    link_speed_megabits: int | None = None
    # Cumulative bytes sent/received on this interface since boot (from
    # psutil.net_io_counters).  The dashboard computes a live rate from the
    # delta between consecutive polls.  None when the platform cannot report.
    rx_bytes: int | None = None
    tx_bytes: int | None = None
    # Monotonic timestamp (ns) when rx/tx_bytes were sampled, for rate math.
    timestamp_ns: int | None = None


class NodeIdentity(FrozenModel):
    """Static and slow-changing node identification data."""

    model_id: str = "Unknown"
    chip_id: str = "Unknown"
    friendly_name: str = "Unknown"
    os_version: str = "Unknown"
    os_build_version: str = "Unknown"
    # Where this node's HTTP API listens, so cluster-wide views (logs, errors)
    # can reach every node. Populated from the node's own launch args.
    api_host: str = ""
    api_port: int = 0


class NodeNetworkInfo(FrozenModel):
    """Network interface information for a node."""

    interfaces: Sequence[NetworkInterfaceInfo] = []


class NodeThunderboltInfo(FrozenModel):
    """Thunderbolt interface identifiers for a node."""

    interfaces: Sequence[ThunderboltIdentifier] = []


class NodeRdmaCtlStatus(FrozenModel):
    """Whether RDMA is enabled on this node (via rdma_ctl).

    ``has_verbs_device`` additionally requires that an RDMA verbs device is
    actually enumerated (``ibv_devices``). ``rdma_ctl`` can report "enabled"
    while no verbs device exists (e.g. Thunderbolt RDMA not provisioned), which
    makes jaccl crash with a NULL protection-domain dereference at init time
    (ml-explore/mlx#3777). Treating that state as RDMA-incapable turns the
    segfault into a clean placement error.
    """

    enabled: bool
    has_verbs_device: bool = True


class ThunderboltBridgeStatus(FrozenModel):
    """Whether the Thunderbolt Bridge network service is enabled on this node."""

    enabled: bool
    exists: bool
    service_name: str | None = None
