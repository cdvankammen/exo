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

        monkeypatch.setattr(profiling, "_find_nvidia_smi_binary", lambda: fake)
        monkeypatch.setattr(
            profiling,
            "_run_nvidia_smi_with_hard_deadline",
            lambda _bin: "16311, 12345\n",
        )

        total, free = profiling._query_cuda_vram_bytes()
        assert total == 16311 * 1024 * 1024
        assert free == 12345 * 1024 * 1024

    def test_nvidia_smi_sums_all_gpus(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multi-GPU boxes sum every GPU's VRAM (adaptive to GPU count)."""
        fake = "/fake/bin/nvidia-smi"

        monkeypatch.setattr(profiling, "_find_nvidia_smi_binary", lambda: fake)
        monkeypatch.setattr(
            profiling,
            "_run_nvidia_smi_with_hard_deadline",
            lambda _bin: "16311, 14276\n16311, 15697\n",
        )

        total, free = profiling._query_cuda_vram_bytes()
        assert total == 32622 * 1024 * 1024  # 16311 + 16311
        assert free == 29973 * 1024 * 1024  # 14276 + 15697

    def test_nvidia_smi_sums_four_gpus(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """4-GPU boxes (DGX-style) sum correctly too."""
        fake = "/fake/bin/nvidia-smi"

        monkeypatch.setattr(profiling, "_find_nvidia_smi_binary", lambda: fake)
        monkeypatch.setattr(
            profiling,
            "_run_nvidia_smi_with_hard_deadline",
            lambda _bin: "8192, 7000\n8192, 7000\n8192, 7000\n8192, 7000\n",
        )

        total, free = profiling._query_cuda_vram_bytes()
        assert total == 32768 * 1024 * 1024  # 8192 x 4
        assert free == 28000 * 1024 * 1024  # 7000 x 4

    def test_nvml_fallback_when_no_binary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No nvidia-smi anywhere → NVML ctypes fallback returns bytes (2 GPUs)."""
        monkeypatch.setattr(profiling, "_find_nvidia_smi_binary", lambda: None)

        class _FakeLib:
            def __init__(self) -> None:
                self._count = 2
                # Per-device totals (2 × 8.5GB) = 17GB total
                self._totals = [8_500_000_000, 8_500_000_000]
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
        assert total == 17_000_000_000  # 8.5e9 + 8.5e9
        assert free == 9_000_000_000  # 4e9 + 5e9

    def test_nvml_one_bad_device_skips_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A device that fails its memory query is skipped, others still sum."""
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
        """No binary AND no NVML lib -> None (safe system-RAM fallback)."""
        monkeypatch.setattr(profiling, "_find_nvidia_smi_binary", lambda: None)
        monkeypatch.setattr(
            profiling.ctypes, "CDLL", lambda _name: (_ for _ in ()).throw(OSError())
        )

        assert profiling._query_cuda_vram_bytes() is None

    def test_nvidia_smi_timeout_kills_hung_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Watchdog enforces hard deadline on hung nvidia-smi (D-state)."""
        import subprocess as sp
        import time

        monkeypatch.setattr(profiling, "_find_nvidia_smi_binary", lambda: "/fake/nvidia-smi")

        # Simulate a hung nvidia-smi by making Popen return a process that
        # never exits.  The watchdog should SIGKILL it within the deadline.
        class _FakeProc:
            returncode = None

            def communicate(self):
                # Sleep longer than the watchdog deadline
                time.sleep(profiling._CUDA_SMI_DEADLINE_S + 2)
                return ("", "")

            def poll(self):
                return self.returncode

            def kill(self):
                self.returncode = -9

            def __del__(self):
                pass

        fake_proc = _FakeProc()

        def _fake_popen(*_args, **_kwargs):
            return fake_proc

        monkeypatch.setattr(sp, "Popen", _fake_popen)
        # Use a very short deadline so the test completes quickly
        monkeypatch.setattr(profiling, "_CUDA_SMI_DEADLINE_S", 0.1)

        result = profiling._query_cuda_vram_bytes()
        # The watchdog should have killed the hung process and returned None
        assert result is None

    def test_nvidia_smi_watchdog_does_not_interfere_with_fast_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Normal nvidia-smi completes before watchdog fires; no interference."""
        fake = "/fake/bin/nvidia-smi"

        monkeypatch.setattr(profiling, "_find_nvidia_smi_binary", lambda: fake)

        # Monkeypatch _run_nvidia_smi_with_hard_deadline to return good data
        monkeypatch.setattr(
            profiling,
            "_run_nvidia_smi_with_hard_deadline",
            lambda _bin: "16311, 12345\n",
        )

        total, free = profiling._query_cuda_vram_bytes()
        assert total == 16311 * 1024 * 1024
        assert free == 12345 * 1024 * 1024
