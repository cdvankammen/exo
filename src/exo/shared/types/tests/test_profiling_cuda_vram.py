# type: ignore[reportPrivateUsage]
"""Tests for CUDA VRAM querying robustness (T2 fix).

Verifies the nvidia-smi binary lookup tolerates version-suffixed names
(e.g. ``nvidia-smi-595.58`` from manual driver installs) and that the
NVML ctypes fallback works when no binary is on PATH at all.
"""

import shutil
from pathlib import Path

import pytest

import exo.shared.types.profiling as profiling


class TestFindNvidiaSmiBinary:
    def test_plain_name_found_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Plain ``nvidia-smi`` on PATH wins (standard installs)."""
        fake = Path("/fake/bin/nvidia-smi")
        monkeypatch.setattr(shutil, "which", lambda _name: str(fake))
        assert profiling._find_nvidia_smi_binary() == str(fake)

    def test_version_suffixed_name_found_via_glob(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No plain name but ``nvidia-smi-595.58`` exists → found via glob."""
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        suffixed = bin_dir / "nvidia-smi-595.58"
        suffixed.write_text("#!/bin/sh\n")
        suffixed.chmod(0o755)
        monkeypatch.setenv("PATH", str(bin_dir))

        assert profiling._find_nvidia_smi_binary() == str(suffixed)

    def test_no_binary_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty PATH → None (caller falls back to system RAM)."""
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        monkeypatch.setenv("PATH", "")
        assert profiling._find_nvidia_smi_binary() is None


class TestQueryCudaVramBytes:
    def test_nvidia_smi_path_returns_bytes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MiB output from nvidia-smi is converted to bytes (single GPU)."""
        fake = "/fake/bin/nvidia-smi"

        class _FakeCompleted:
            stdout = "16311, 12345\n"

        monkeypatch.setattr(profiling, "_find_nvidia_smi_binary", lambda: fake)
        monkeypatch.setattr(
            profiling.subprocess,
            "run",
            lambda *_a, **_k: _FakeCompleted(),
        )

        total, free = profiling._query_cuda_vram_bytes()
        assert total == 16311 * 1024 * 1024
        assert free == 12345 * 1024 * 1024

    def test_nvidia_smi_reports_largest_gpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multi-GPU boxes report the largest single GPU's VRAM, not the sum.

        MLX CUDA uses one device per process, so placement must only see
        what a single GPU actually offers.
        """
        fake = "/fake/bin/nvidia-smi"

        class _FakeCompleted:
            # Two GPUs: 16GB + 16GB total, 14GB + 15GB free
            stdout = "16311, 14276\n16311, 15697\n"

        monkeypatch.setattr(profiling, "_find_nvidia_smi_binary", lambda: fake)
        monkeypatch.setattr(
            profiling.subprocess,
            "run",
            lambda *_a, **_k: _FakeCompleted(),
        )

        total, free = profiling._query_cuda_vram_bytes()
        assert total == 16311 * 1024 * 1024
        assert free == 14276 * 1024 * 1024  # free for the first device (tie on total)

    def test_nvidia_smi_four_gpus_reports_largest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """4-GPU boxes (DGX-style) report the largest single GPU too."""
        fake = "/fake/bin/nvidia-smi"

        class _FakeCompleted:
            stdout = "8192, 7000\n8192, 7000\n8192, 7000\n8192, 7000\n"

        monkeypatch.setattr(profiling, "_find_nvidia_smi_binary", lambda: fake)
        monkeypatch.setattr(
            profiling.subprocess,
            "run",
            lambda *_a, **_k: _FakeCompleted(),
        )

        total, free = profiling._query_cuda_vram_bytes()
        assert total == 8192 * 1024 * 1024
        assert free == 7000 * 1024 * 1024

    def test_nvml_fallback_when_no_binary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No nvidia-smi anywhere → NVML ctypes fallback returns bytes (2 GPUs)."""
        monkeypatch.setattr(profiling, "_find_nvidia_smi_binary", lambda: None)

        class _FakeLib:
            def __init__(self) -> None:
                self._count = 2
                self._totals = [8_500_000_000, 6_000_000_000]
                self._frees = [4_000_000_000, 5_000_000_000]

            def nvmlInit_v2(self) -> int:  # noqa: N802
                return 0

            def nvmlDeviceGetCount_v2(self, out) -> int:  # noqa: N802
                out._obj.value = self._count
                return 0

            def nvmlDeviceGetHandleByIndex_v2(self, i, _handle) -> int:  # noqa: N802
                self._current_index = i
                return 0

            def nvmlDeviceGetMemoryInfo(self, _handle, mem) -> int:  # noqa: N802
                mem._obj.total = self._totals[self._current_index]
                mem._obj.free = self._frees[self._current_index]
                mem._obj.used = mem._obj.total - mem._obj.free
                return 0

            def nvmlShutdown(self) -> int:  # noqa: N802
                return 0

        monkeypatch.setattr(profiling.ctypes, "CDLL", lambda _name: _FakeLib())

        total, free = profiling._query_cuda_vram_bytes()
        assert total == 8_500_000_000  # largest single device, not the sum
        assert free == 4_000_000_000  # free for that same device

    def test_nvml_one_bad_device_skips_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A device that fails its memory query is skipped when finding the largest."""
        monkeypatch.setattr(profiling, "_find_nvidia_smi_binary", lambda: None)

        class _FakeLib:
            def __init__(self) -> None:
                self._count = 2
                self._totals = [8_000_000_000, 8_000_000_000]
                self._frees = [4_000_000_000, 4_000_000_000]

            def nvmlInit_v2(self) -> int:  # noqa: N802
                return 0

            def nvmlDeviceGetCount_v2(self, out) -> int:  # noqa: N802
                out._obj.value = self._count
                return 0

            def nvmlDeviceGetHandleByIndex_v2(self, i, _handle) -> int:  # noqa: N802
                self._current_index = i
                return 0

            def nvmlDeviceGetMemoryInfo(self, _handle, mem) -> int:  # noqa: N802
                if self._current_index == 1:
                    return 1  # fail device 1
                mem._obj.total = self._totals[0]
                mem._obj.free = self._frees[0]
                mem._obj.used = mem._obj.total - mem._obj.free
                return 0

            def nvmlShutdown(self) -> int:  # noqa: N802
                return 0

        monkeypatch.setattr(profiling.ctypes, "CDLL", lambda _name: _FakeLib())

        total, free = profiling._query_cuda_vram_bytes()
        assert total == 8_000_000_000  # only device 0 counted
        assert free == 4_000_000_000

    def test_all_paths_fail_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No binary AND no NVML lib → None (safe system-RAM fallback)."""
        monkeypatch.setattr(profiling, "_find_nvidia_smi_binary", lambda: None)
        monkeypatch.setattr(
            profiling.ctypes, "CDLL", lambda _name: (_ for _ in ()).throw(OSError())
        )

        assert profiling._query_cuda_vram_bytes() is None


class TestQueryCudaVramDevices:
    def test_nvidia_smi_returns_all_devices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """nvidia-smi index,total,free parsing returns one entry per GPU."""
        fake = "/fake/bin/nvidia-smi"

        class _FakeCompleted:
            stdout = "0, 16311, 14276\n1, 16311, 15697\n"

        monkeypatch.setattr(profiling, "_find_nvidia_smi_binary", lambda: fake)
        monkeypatch.setattr(
            profiling.subprocess,
            "run",
            lambda *_a, **_k: _FakeCompleted(),
        )

        devices = profiling._query_cuda_vram_devices()
        assert devices is not None
        assert len(devices) == 2
        assert devices[0].index == 0
        assert devices[0].total == profiling.Memory.from_bytes(16311 * 1024 * 1024)
        assert devices[0].free == profiling.Memory.from_bytes(14276 * 1024 * 1024)
        assert devices[1].index == 1
        assert devices[1].total == profiling.Memory.from_bytes(16311 * 1024 * 1024)
        assert devices[1].free == profiling.Memory.from_bytes(15697 * 1024 * 1024)

    def test_nvml_returns_all_devices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No binary → NVML fallback returns one entry per device."""
        monkeypatch.setattr(profiling, "_find_nvidia_smi_binary", lambda: None)

        class _FakeLib:
            def __init__(self) -> None:
                self._count = 2
                self._totals = [8_500_000_000, 6_000_000_000]
                self._frees = [4_000_000_000, 5_000_000_000]

            def nvmlInit_v2(self) -> int:  # noqa: N802
                return 0

            def nvmlDeviceGetCount_v2(self, out) -> int:  # noqa: N802
                out._obj.value = self._count
                return 0

            def nvmlDeviceGetHandleByIndex_v2(self, i, _handle) -> int:  # noqa: N802
                self._current_index = i
                return 0

            def nvmlDeviceGetMemoryInfo(self, _handle, mem) -> int:  # noqa: N802
                mem._obj.total = self._totals[self._current_index]
                mem._obj.free = self._frees[self._current_index]
                mem._obj.used = mem._obj.total - mem._obj.free
                return 0

            def nvmlShutdown(self) -> int:  # noqa: N802
                return 0

        monkeypatch.setattr(profiling.ctypes, "CDLL", lambda _name: _FakeLib())

        devices = profiling._query_cuda_vram_devices()
        assert devices is not None
        assert len(devices) == 2
        assert devices[0].index == 0
        assert devices[0].total == profiling.Memory.from_bytes(8_500_000_000)
        assert devices[0].free == profiling.Memory.from_bytes(4_000_000_000)
        assert devices[1].index == 1
        assert devices[1].total == profiling.Memory.from_bytes(6_000_000_000)
        assert devices[1].free == profiling.Memory.from_bytes(5_000_000_000)


class TestMemoryUsageDevices:
    def _usage_with_devices(self) -> profiling.MemoryUsage:
        return profiling.MemoryUsage.from_bytes(
            ram_total=64_000_000_000,
            ram_available=32_000_000_000,
            swap_total=0,
            swap_available=0,
            accelerator_total=16_000_000_000,
            accelerator_available=14_000_000_000,
            accelerator_devices=[
                profiling.GpuMemoryInfo.from_bytes(
                    index=0, total=16_000_000_000, free=14_000_000_000
                ),
                profiling.GpuMemoryInfo.from_bytes(
                    index=1, total=16_000_000_000, free=15_000_000_000
                ),
            ],
        )

    def test_largest_device_available_uses_per_device(self) -> None:
        usage = self._usage_with_devices()
        assert (
            usage.largest_device_available
            == profiling.Memory.from_bytes(15_000_000_000)
        )

    def test_summed_device_available_sums_all_devices(self) -> None:
        usage = self._usage_with_devices()
        assert (
            usage.summed_device_available
            == profiling.Memory.from_bytes(29_000_000_000)
        )

    def test_largest_device_falls_back_to_inference_available(self) -> None:
        usage = profiling.MemoryUsage.from_bytes(
            ram_total=64_000_000_000,
            ram_available=32_000_000_000,
            swap_total=0,
            swap_available=0,
        )
        assert usage.largest_device_available == usage.inference_available
        assert usage.summed_device_available == usage.inference_available


class TestZeroAcceleratorContract:
    """A node with no usable accelerator budgets by RAM, not by zero.

    Regression cover for t_9918d5c8. The card claimed a host with no NVIDIA
    GPU would collapse ``inference_available`` to 0; execution disproved that
    for the scalar path (``_query_cuda_vram_bytes`` returns None, never 0).
    The equivalent defect DOES exist in the per-device path, which kept a
    device reporting 0 total and let a lone phantom 0/0 entry drive
    ``largest_device_available`` to 0 while the node had 32 GB of RAM.
    """

    _MIB = 1024 * 1024

    def _usage(self, **kwargs: object) -> profiling.MemoryUsage:
        return profiling.MemoryUsage.from_bytes(
            ram_total=64_000_000_000,
            ram_available=32_000_000_000,
            swap_total=0,
            swap_available=0,
            **kwargs,  # type: ignore[arg-type]
        )

    def _drive_smi(
        self, monkeypatch: pytest.MonkeyPatch, stdout: str, devices: bool
    ) -> object:
        """Run the nvidia-smi branch with the NVML fallback poisoned to None.

        Poisoning the fallback means a "fell through to NVML" path surfaces as
        a clean None instead of being mistaken for a genuine reading.
        """

        class _FakeCompleted:
            pass

        completed = _FakeCompleted()
        completed.stdout = stdout  # type: ignore[attr-defined]
        monkeypatch.setattr(
            profiling, "_find_nvidia_smi_binary", lambda: "/fake/bin/nvidia-smi"
        )
        monkeypatch.setattr(
            profiling.subprocess, "run", lambda *_a, **_k: completed
        )
        if devices:
            monkeypatch.setattr(
                profiling, "_query_cuda_vram_devices_nvml", lambda: None
            )
            return profiling._query_cuda_vram_devices()
        monkeypatch.setattr(profiling, "_query_cuda_vram_bytes_nvml", lambda: None)
        return profiling._query_cuda_vram_bytes()

    def test_no_accelerator_budgets_by_ram(self) -> None:
        """Absent accelerator => inference_available == ram_available, not 0."""
        usage = self._usage()
        assert usage.accelerator_total is None
        assert usage.accelerator_available is None
        assert usage.inference_available.in_bytes == 32_000_000_000
        assert usage.inference_available.in_bytes != 0

    def test_query_returns_none_without_gpu(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No binary and no NVML library => None, engaging the RAM fallback."""
        monkeypatch.setattr(profiling, "_find_nvidia_smi_binary", lambda: None)
        monkeypatch.setattr(
            profiling.ctypes,
            "CDLL",
            lambda _name: (_ for _ in ()).throw(OSError()),
        )
        assert profiling._query_cuda_vram_bytes() is None

    @pytest.mark.parametrize(
        "stdout",
        [
            pytest.param("", id="empty"),
            pytest.param("0, 0\n", id="zero-reported"),
            pytest.param("N/A, N/A\n", id="unparseable"),
        ],
    )
    def test_zero_readings_never_become_a_zero_total(
        self, monkeypatch: pytest.MonkeyPatch, stdout: str
    ) -> None:
        """A zero/absent reading becomes None, never ``(0, 0)``.

        This is what makes the RAM fallback safe: were the query to return
        ``(0, 0)``, ``inference_available`` would collapse to 0.
        """
        assert self._drive_smi(monkeypatch, stdout, False) is None

    def test_gpu_node_still_bounded_by_free_vram(self) -> None:
        """The dGPU path keeps min(ram, vram_free) — not regressed."""
        usage = self._usage(
            accelerator_total=24_000_000_000, accelerator_available=12_000_000_000
        )
        assert usage.inference_available.in_bytes == 12_000_000_000

    def test_fully_occupied_gpu_reports_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A real GPU with no free VRAM legitimately yields 0, not 'absent'.

        Guards against a "treat 0 as no accelerator" refactor: falling back to
        RAM here would admit a model that then fails to load with
        ``cudaMallocAsync out of memory``.
        """
        result = self._drive_smi(monkeypatch, "16311, 0\n", False)
        assert result is not None
        total, free = result  # type: ignore[misc]
        assert total == 16311 * self._MIB
        assert free == 0

        usage = self._usage(accelerator_total=total, accelerator_available=free)
        assert usage.inference_available.in_bytes == 0

    def test_zero_total_device_is_dropped_by_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 0/0 GPU must not survive into ``accelerator_devices``."""
        assert self._drive_smi(monkeypatch, "0, 0, 0\n", True) is None

    def test_zero_total_device_dropped_among_healthy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        devices = self._drive_smi(monkeypatch, "0, 16311, 14276\n1, 0, 0\n", True)
        assert devices is not None
        assert [d.index for d in devices] == [0]

    def test_lone_zero_device_falls_back_to_ram(self) -> None:
        """Defence in depth: the property ignores a 0/0 device.

        ``MemoryUsage`` is network-deserialized, so a remote or older node can
        still send one even though the local query now filters it.
        """
        usage = self._usage(
            accelerator_devices=[
                profiling.GpuMemoryInfo.from_bytes(index=0, total=0, free=0)
            ]
        )
        assert usage.largest_device_available.in_bytes == 32_000_000_000
        assert usage.summed_device_available.in_bytes == 32_000_000_000
