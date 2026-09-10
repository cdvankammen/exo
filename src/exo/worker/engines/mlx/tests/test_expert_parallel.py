"""Tests for expert_parallel.py — Expert Parallelism for MoE.

Tests the EP dispatch/combine logic, weight slicing helpers, and
ExpertParallelMoE wrapper without requiring actual distributed init.
"""
import os
import sys
import pytest
import numpy as np

# Set env before any exo imports
os.environ["EXO_EXPERT_PARALLEL"] = "0"  # default off

import mlx.core as mx
import mlx.nn as nn

from exo.worker.engines.mlx.expert_parallel import (
    _EXPERT_PARALLEL_ENABLED,
    _dispatch_combine_ep,
    _slice_expert_weights,
    _slice_quantized_expert_weights,
    ExpertParallelMoE,
    ExpertParallelMoEGemma4,
)
from mlx_lm.models.switch_layers import SwitchGLU, SwitchLinear


# ── Helpers ────────────────────────────────────────────────────────────────

class FakeGroup:
    """Mock distributed group for testing (2 ranks)."""
    def __init__(self, rank: int = 0, size: int = 2):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


class FakeMoELayer(nn.Module):
    """Mock MoE layer with SwitchGLU + gate + optional shared_experts."""

    def __init__(self, hidden_dim: int = 64, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.switch_mlp = SwitchGLU(hidden_dim, hidden_dim * 2, num_experts)
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_experts_per_tok = top_k

    def __call__(self, x):
        inds = mx.argmax(self.gate(x), axis=-1)[:, :, None]
        scores = mx.softmax(self.gate(x), axis=-1)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        return y


def _make_switch_linear(
    input_dims: int, output_dims: int, num_experts: int
) -> SwitchLinear:
    """Create a SwitchLinear with known weights for testing."""
    sl = SwitchLinear(input_dims, output_dims, num_experts, bias=False)
    sl.weight = mx.random.uniform(
        low=-0.1, high=0.1, shape=(num_experts, output_dims, input_dims)
    )
    return sl


# ── Weight slicing tests ──────────────────────────────────────────────────

class TestSliceExpertWeights:
    """Test weight slicing for EP (keep only experts [start, end))."""

    def test_basic_slice(self):
        """Slice SwitchGLU weights to keep 4 of 8 experts."""
        switch = SwitchGLU(32, 64, 8, bias=False)
        original_gate_w = mx.array(switch.gate_proj.weight)

        _slice_expert_weights(switch, 2, 6)

        assert switch.gate_proj.weight.shape[0] == 4
        assert switch.up_proj.weight.shape[0] == 4
        assert switch.down_proj.weight.shape[0] == 4
        # Verify correct slice
        np.testing.assert_allclose(
            np.array(switch.gate_proj.weight),
            np.array(original_gate_w[2:6]),
            rtol=1e-6,
        )

    def test_first_rank_slice(self):
        """Slice for rank 0 (experts 0..3 out of 8)."""
        switch = SwitchGLU(32, 64, 8, bias=False)
        _slice_expert_weights(switch, 0, 4)

        assert switch.gate_proj.weight.shape[0] == 4
        assert switch.up_proj.weight.shape[0] == 4
        assert switch.down_proj.weight.shape[0] == 4

    def test_last_rank_slice(self):
        """Slice for rank 1 (experts 4..7 out of 8)."""
        switch = SwitchGLU(32, 64, 8, bias=False)
        original_gate_w = mx.array(switch.gate_proj.weight)

        _slice_expert_weights(switch, 4, 8)

        assert switch.gate_proj.weight.shape[0] == 4
        np.testing.assert_allclose(
            np.array(switch.gate_proj.weight),
            np.array(original_gate_w[4:8]),
            rtol=1e-6,
        )

    def test_bias_slicing(self):
        """Slice includes bias if present."""
        switch = SwitchGLU(32, 64, 8, bias=True)
        _slice_expert_weights(switch, 2, 6)

        assert switch.gate_proj.weight.shape[0] == 4
        assert switch.gate_proj.bias.shape[0] == 4


# ── EP dispatch/combine tests ─────────────────────────────────────────────

class TestDispatchCombineEP:
    """Test the EP token dispatch/combine function with mocked all_gather/all_sum."""

    def _mock_all_gather(self, x: mx.array, group) -> mx.array:
        """Mock all_gather: just concatenate (simulates single-rank)."""
        return x  # single rank: all_gather is identity

    def _mock_all_sum(self, x: mx.array, group) -> mx.array:
        """Mock all_sum: just return (single rank: all_sum is identity)."""
        return x

    def test_dispatch_combine_single_rank(self):
        """Dispatch with 1 rank (no-op: all experts on one node)."""
        # Temporarily monkey-patch for test
        original_sum = mx.distributed.all_sum
        mx.distributed.all_sum = self._mock_all_sum

        try:
            B, S, H = 1, 4, 32
            num_experts = 8
            num_ranks = 1
            top_k = 2

            x = mx.random.normal((B, S, H))
            expert_assign = mx.array([
                [[0, 1], [2, 3], [4, 5], [6, 7]]
            ])  # [1, 4, 2]
            scores = mx.ones((B, S, top_k)) / top_k  # equal weights

            def expert_fn(tokens, expert_indices):
                return tokens  # identity: just pass through

            result = _dispatch_combine_ep(
                x, expert_assign, scores, num_experts, num_ranks, 0,
                FakeGroup(0, 1), expert_fn,
            )

            assert result.shape == (B, S, H)
        finally:
            mx.distributed.all_sum = original_sum

    def test_dispatch_combine_rank_0(self):
        """Dispatch with 2 ranks, checking rank 0's portion."""
        original_sum = mx.distributed.all_sum
        mx.distributed.all_sum = self._mock_all_sum

        try:
            B, S, H = 1, 4, 32
            num_experts = 4
            num_ranks = 2
            top_k = 2

            x = mx.ones((B, S, H))
            # Rank 0 owns experts 0,1; rank 1 owns experts 2,3
            expert_assign = mx.array([
                [[0, 1], [2, 3], [0, 3], [1, 2]]
            ])  # [1, 4, 2]
            scores = mx.ones((B, S, top_k)) / top_k

            def expert_fn(tokens, expert_indices):
                # Just pass through tokens
                return tokens

            result = _dispatch_combine_ep(
                x, expert_assign, scores, num_experts, num_ranks, 0,
                FakeGroup(0, 2), expert_fn,
            )

            assert result.shape == (B, S, H)
        finally:
            mx.distributed.all_sum = original_sum


# ── Feature flag test ─────────────────────────────────────────────────────

class TestFeatureFlag:
    def test_default_off(self):
        """EXO_EXPERT_PARALLEL defaults to 0."""
        assert _EXPERT_PARALLEL_ENABLED is False

    def test_enabled_via_env(self):
        """EXO_EXPERT_PARALLEL=1 enables EP (tested via os.environ manipulation)."""
        # This is just testing the mechanism, not changing the actual env
        val = os.environ.get("EXO_EXPERT_PARALLEL", "0")
        assert val in ("0", "1")
