import ctypes
import glob
import os
import shutil
import subprocess
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self, cast

from loguru import logger

from exo.shared.types.memory import Memory
from exo.shared.types.thunderbolt import ThunderboltIdentifier
from exo.utils.pydantic_ext import FrozenModel
from exo.utils.virtual_memory import swap_memory_statistics, virtual_memory_statistics


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

    @property
    def inference_available(self) -> Memory:
        """Memory actually available to hold model state on this node.

        On discrete-GPU nodes the accelerator's free memory bounds what
        inference can use, regardless of how much system RAM is free.
        """
        if self.accelerator_available is None:
            return self.ram_available
        return min(self.ram_available, self.accelerator_available)

    @classmethod
    def from_bytes(
        cls, *, ram_total: int, ram_available: int, swap_total: int, swap_available: int
    ) -> Self:
        used_bytes = max(ram_total - ram_available, 0)
        pressure = used_bytes / ram_total if ram_total > 0 else 0.0
        return cls(
            ram_total=Memory.from_bytes(ram_total),
            ram_available=Memory.from_bytes(ram_available),
            swap_total=Memory.from_bytes(swap_total),
            swap_available=Memory.from_bytes(swap_available),
            ram_used=Memory.from_bytes(used_bytes),
            pressure=round(min(max(pressure, 0.0), 1.0), 4),
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
    """Total and free VRAM in bytes across ALL CUDA GPUs.

    Sums every GPU's VRAM so multi-GPU boxes (2x, 4x, 8x..) advertise their
    full capacity -- MLX CUDA can span all devices.  Tries ``nvidia-smi``
    first (tolerating version-suffixed binary names), then the NVML driver
    library directly.  Returns None when neither is available or the output
    cannot be parsed, so callers fall back to system RAM.

    .. important::

       ``nvidia-smi`` can hang indefinitely when the GPU driver is in
       D-state (uninterruptible sleep).  ``subprocess.run(timeout=N)``
       only guards the *initial* ``waitpid`` -- if the process is
       unkillable, the post-``kill()`` ``proc.wait()`` inside CPython's
       ``communicate`` blocks forever, making the timeout ineffective and
       allowing a SIGTERM to hard-kill the node via ``sys.exit(1)``.

       We therefore use ``Popen`` with a **watchdog thread** that
       enforces a *hard* deadline: if the process hasn't exited within
       ``_CUDA_SMI_DEADLINE_S`` seconds, it is SIGKILL'd and the
       function returns ``None`` (falling back to NVML or system RAM).
    """
    nvidia_smi = _find_nvidia_smi_binary()
    if nvidia_smi is not None:
        try:
            stdout = _run_nvidia_smi_with_hard_deadline(nvidia_smi)
            if stdout is None:
                return _query_cuda_vram_bytes_nvml()
            lines = stdout.strip().splitlines()
            total_mebibytes = 0
            free_mebibytes = 0
            for line in lines:
                fields = line.split(",")
                if len(fields) != 2:
                    return _query_cuda_vram_bytes_nvml()
                try:
                    total_mebibytes += int(fields[0].strip())
                    free_mebibytes += int(fields[1].strip())
                except ValueError:
                    return _query_cuda_vram_bytes_nvml()
            if total_mebibytes > 0:
                mebibyte = 1024 * 1024
                return (
                    total_mebibytes * mebibyte,
                    free_mebibytes * mebibyte,
                )
        except OSError:
            pass
    return _query_cuda_vram_bytes_nvml()


# Deadline for the nvidia-smi watchdog (seconds).
# Must be shorter than info_gatherer's fail_after(10) so the caller
# still has headroom for NVML fallback.
_CUDA_SMI_DEADLINE_S = 5


def _run_nvidia_smi_with_hard_deadline(nvidia_smi: str) -> str | None:
    """Run ``nvidia-smi`` with a hard deadline enforced by a watchdog thread.

    Returns the process's decoded stdout on success, or ``None`` if the
    process times out, crashes, or produces unparseable output.  The
    watchdog sends SIGKILL if the deadline is exceeded -- this is the
    *only* way to break a D-state ``nvidia-smi`` that resists normal
    termination.
    """
    cmd = [
        nvidia_smi,
        "--query-gpu=memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    timed_out = threading.Event()

    def _watchdog() -> None:
        """Kill the process if the deadline is exceeded."""
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass  # already dead
            timed_out.set()

    timer = threading.Timer(_CUDA_SMI_DEADLINE_S, _watchdog)
    timer.daemon = True
    timer.start()
    try:
        stdout, _stderr = proc.communicate()
    except Exception:
        # Unexpected error during communicate -- treat as failure.
        try:
            proc.kill()
        except OSError:
            pass
        logger.warning("nvidia-smi communicate failed -- falling back to NVML")
        return None
    finally:
        timer.cancel()

    if timed_out.is_set():
        logger.warning(
            f"nvidia-smi exceeded {_CUDA_SMI_DEADLINE_S}s hard deadline "
            "(GPU driver in D-state?) -- falling back to NVML"
        )
        return None

    if proc.returncode != 0:
        return None

    return stdout


def _query_cuda_vram_bytes_nvml() -> tuple[int, int] | None:
    """Query VRAM via the NVML driver library (``libnvidia-ml.so.1``).

    Fallback used when no ``nvidia-smi`` binary is on PATH. The driver
    library ships with every NVIDIA driver (including minimal/container
    installs), so this removes the binary-name dependency entirely.
    Sums VRAM across ALL devices (adaptive to GPU count).
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
            total_bytes = 0
            free_bytes = 0
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
                total_bytes += cast(int, memory.total)
                free_bytes += cast(int, memory.free)
            if total_bytes == 0:
                return None
            return total_bytes, free_bytes
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
