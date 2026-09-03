# pyright: reportUnusedFunction=false, reportAny=false
"""Tests for the runner diagnostic classifier — CUDA OOM detection.

The classifier reads runner stderr lines and tags known failure modes so the
dashboard can show a specific reason instead of a generic "model didn't load".
"""
from exo.worker.runner.diagnostics import (
    RunnerCudaOutOfMemory,
    RunnerDiagnosticCollector,
    RunnerUnknown,
)


class TestCudaOomClassifier:
    def test_cuda_malloc_async_oom(self) -> None:
        collector = RunnerDiagnosticCollector()
        collector.record_line(
            "cudaMallocAsync failed with out of memory error: tried to allocate "
            "5.50 GiB on device 0"
        )
        diags = collector.diagnostics()
        assert len(diags) == 1
        assert isinstance(diags[0], RunnerCudaOutOfMemory)
        assert "CUDA out of memory" in diags[0].message

    def test_cuda_out_of_memory_phrase(self) -> None:
        collector = RunnerDiagnosticCollector()
        collector.record_line(
            "CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 15.78 GiB "
            "total capacity; 4.12 GiB already allocated)"
        )
        diags = collector.diagnostics()
        assert len(diags) == 1
        assert isinstance(diags[0], RunnerCudaOutOfMemory)

    def test_mlx_cuda_oom(self) -> None:
        """MLX CUDA wraps errors with 'out of memory' + 'device' context."""
        collector = RunnerDiagnosticCollector()
        collector.record_line(
            "[METAL] Error: Failed to allocate 4096 MiB out of memory on device "
            "(GPU, 0)"
        )
        diags = collector.diagnostics()
        assert len(diags) == 1
        assert isinstance(diags[0], RunnerCudaOutOfMemory)

    def test_plain_oom_keyword(self) -> None:
        collector = RunnerDiagnosticCollector()
        collector.record_line("RuntimeError: OOM error when allocating memory for activation")
        diags = collector.diagnostics()
        assert len(diags) == 1
        assert isinstance(diags[0], RunnerCudaOutOfMemory)

    def test_unrelated_line_classified_unknown(self) -> None:
        collector = RunnerDiagnosticCollector()
        collector.record_line("Some random stderr message without OOM keywords")
        diags = collector.diagnostics()
        assert len(diags) == 1
        assert isinstance(diags[0], RunnerUnknown)

    def test_evidence_captured(self) -> None:
        collector = RunnerDiagnosticCollector()
        collector.record_line("previous line")
        collector.record_line("cudaMallocAsync failed with out of memory: tried to allocate 1 GiB")
        diags = collector.diagnostics()
        assert len(diags) == 2
        oom = diags[1]
        assert isinstance(oom, RunnerCudaOutOfMemory)
        # Evidence should include the last 4 lines
        assert len(oom.evidence) == 2
        assert oom.evidence[-1].startswith("cudaMallocAsync")

    def test_empty_line_ignored(self) -> None:
        collector = RunnerDiagnosticCollector()
        collector.record_line("   ")
        collector.record_line("cudaMallocAsync failed with out of memory on device 0")
        diags = collector.diagnostics()
        assert len(diags) == 1
        assert isinstance(diags[0], RunnerCudaOutOfMemory)
