"""Tests for NodeBackends.gather() CUDA advertisement (T1 fix).

Verifies a CUDA node never advertises as CPU-only, even when the
optional pynvml package is absent — the _mlx_uses_cuda_gpu() fallback
must catch it (fresh Linux+NVIDIA installs using the documented
``uv sync --extra mlx`` command don't install nvidia-ml-py).
"""

import pytest

import exo.utils.info_gatherer.info_gatherer as ig


class TestNodeBackendsGather:
    async def test_cuda_via_nvml(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """pynvml path: NVML detects a GPU → MlxCuda + Vllm advertised."""
        monkeypatch.setattr(ig, "IS_DARWIN", False)
        monkeypatch.setattr(ig, "_has_nvml_cuda", lambda: True)
        monkeypatch.setattr(ig, "_mlx_uses_cuda_gpu", lambda: False)

        backends = (await ig.NodeBackends.gather()).backends
        names = [b.value for b in backends]

        assert "MlxCpu" in names
        assert "MlxCuda" in names
        assert "Vllm" in names

    async def test_cuda_via_mlx_fallback_without_pynvml(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """THE FALLBACK: pynvml absent (returns False) but MLX runs on CUDA →
        node must still advertise MlxCuda (never CPU-only)."""
        monkeypatch.setattr(ig, "IS_DARWIN", False)
        monkeypatch.setattr(ig, "_has_nvml_cuda", lambda: False)  # pynvml missing
        monkeypatch.setattr(ig, "_mlx_uses_cuda_gpu", lambda: True)  # real CUDA

        backends = (await ig.NodeBackends.gather()).backends
        names = [b.value for b in backends]

        assert "MlxCpu" in names
        assert "MlxCuda" in names
        assert "Vllm" in names

    async def test_cpu_only_node(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No GPU anywhere → MlxCpu only (correct for CPU boxes)."""
        monkeypatch.setattr(ig, "IS_DARWIN", False)
        monkeypatch.setattr(ig, "_has_nvml_cuda", lambda: False)
        monkeypatch.setattr(ig, "_mlx_uses_cuda_gpu", lambda: False)

        backends = (await ig.NodeBackends.gather()).backends
        names = [b.value for b in backends]

        assert names == ["MlxCpu"]

    async def test_darwin_advertises_metal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """macOS: MlxCpu + MlxMetal, and no CUDA on modern Macs (no NVIDIA GPU).

        Note: the CUDA branch is intentionally NOT gated by IS_DARWIN (a Mac
        with an NVIDIA eGPU could legitimately report CUDA via pynvml) — this
        test uses the realistic case where NVML finds no GPU.
        """
        monkeypatch.setattr(ig, "IS_DARWIN", True)
        monkeypatch.setattr(ig, "_has_nvml_cuda", lambda: False)  # no NVIDIA GPU
        monkeypatch.setattr(ig, "_mlx_uses_cuda_gpu", lambda: False)

        backends = (await ig.NodeBackends.gather()).backends
        names = [b.value for b in backends]

        assert "MlxCpu" in names
        assert "MlxMetal" in names
        assert "MlxCuda" not in names
        assert "Vllm" not in names
