"""
Expert Parallelism for Mixture-of-Experts models in exo.

Instead of Tensor Parallelism (shards each expert's weight matrix across all
nodes), Expert Parallelism assigns different experts to different nodes:

    Node 0: experts 0..63     Node 1: experts 64..127
    Node 2: experts 128..191  Node 3: experts 192..255

Per-node memory: 1/N of MoE weights (vs TP which keeps ALL experts).
Communication: all-gather (for token dispatch/combine) + all_sum (for shared
experts / downstream reduction).

Opt-in via ``EXO_EXPERT_PARALLEL=1``. Default (off) = current tensor-parallel
behavior for MoE layers.

Supported MoE architectures:
    - DeepSeek V3 / V3.2 / Kimi K2.5 (SwitchGLU + MoEGate + shared_experts)
    - DeepSeek V4 (SwitchGLU + shared_experts)
    - GLM4-MoE / GLM4-MoE-Lite (SwitchGLU, optional shared_experts)
    - MiniMax (SwitchGLU)
    - GptOSS (SwitchGLU with fc1/fc2 naming)
    - Step3.5 (SwitchGLU + share_expert)
    - NemotronH (SwitchMLP with fc1/fc2 + shared_experts)
    - Qwen3-MoE / Qwen3-Next / Qwen3.5-MoE (SwitchGLU + shared_expert)
    - Gemma4 (SwitchGLU)
"""
from __future__ import annotations

import os
from collections.abc import Generator
from typing import TYPE_CHECKING, cast

import mlx.core as mx
import mlx.nn as nn

from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from mlx_lm.models.cache import Cache

# ── Feature flag ──────────────────────────────────────────────────────────
_EXPERT_PARALLEL_ENABLED: bool = os.environ.get("EXO_EXPERT_PARALLEL", "0") == "1"


# ══════════════════════════════════════════════════════════════════════════
#  EP token dispatch / combine (using all_gather)
# ══════════════════════════════════════════════════════════════════════════

def _dispatch_combine_ep(
    x: mx.array,
    expert_assign: mx.array,
    scores: mx.array,
    num_experts: int,
    num_ranks: int,
    rank: int,
    group: mx.distributed.Group,
    expert_fn,
) -> mx.array:
    """Dispatch tokens to expert-owning ranks via all_gather, compute locally,
    and combine back to token order via all_gather + permutation.

    This is a simpler alternative to native all-to-all (not available in mlx).
    Each rank temporarily holds N*T_local tokens during all_gather.

    Args:
        x: input [B, S, hidden]
        expert_assign: gate routing indices [B, S, top_k]
        scores: per-expert weights [B, S, top_k] (before sum)
        expert_fn: callable(local_tokens, local_expert_indices) -> local_output
    Returns:
        output: [B, S, hidden] (weighted MoE output for all tokens)
    """
    B, S, H = x.shape
    top_k = int(expert_assign.shape[-1])
    flat_experts = expert_assign.flatten()  # [B*S*top_k]
    flat_scores = scores.flatten()  # [B*S*top_k]
    flat_tokens_idx = mx.arange(B * S * top_k) // top_k  # which token each entry belongs to

    # Each rank knows which (token, expert) pairs it owns
    num_local_experts = num_experts // num_ranks
    global_eid = flat_experts.astype(mx.int32)
    token_rank = global_eid // num_local_experts  # which rank owns each assignment

    # Build local indices using argsort compaction (mlx has no boolean indexing)
    is_local = token_rank == rank  # [B*S*top_k] bool mask
    T_local = int(is_local.sum().item())

    if T_local == 0:
        return mx.zeros((B, S, H), dtype=x.dtype)

    # Compact: sort local entries to the front using argsort trick
    sort_key = mx.where(is_local, mx.array(-1, dtype=mx.int32), mx.array(0, dtype=mx.int32))
    order = mx.argsort(sort_key)
    compact_idx = order[:T_local]  # global positions of this rank's assignments

    # Extract global flat indices and local expert ids for this rank's slots
    perm = flat_tokens_idx[compact_idx]  # [T_local] global flat-token indices
    local_expert_indices = (global_eid[compact_idx] - rank * num_local_experts).astype(mx.int32)

    # Gather input tokens for this rank's experts
    x_flat = x.reshape(B * S, H)
    x_repeated = mx.repeat(x_flat, top_k, axis=0)  # [B*S*top_k, H]
    local_tokens = x_repeated[perm]  # [T_local, H]

    # Run local expert computation (only this rank's experts)
    local_out = expert_fn(local_tokens, local_expert_indices)  # [T_local, H]

    # Scatter local_out to full positions using one-hot matmul (no native scatter in mlx)
    full = mx.zeros((B * S * top_k, H), dtype=local_out.dtype)
    perm_oh = mx.eye(B * S * top_k, dtype=local_out.dtype)[perm]  # [T_local, B*S*top_k]
    full = perm_oh.T @ local_out  # [B*S*top_k, H]

    # Also scatter and gather scores for correct per-expert weighting
    local_scores = flat_scores[compact_idx]  # [T_local] per-token-expert score
    score_full = perm_oh.T @ local_scores.reshape(T_local, 1)  # [B*S*top_k, 1]

    # All-reduce: each rank fills different positions → sum = full result
    full = mx.distributed.all_sum(full, group=group)
    score_full = mx.distributed.all_sum(score_full, group=group)

    # Apply per-expert scores: [B*S*top_k, H] * [B*S*top_k, 1]
    full = full * score_full

    # Sum over top_k duplicates: [B*S*top_k, H] → [B*S, H]
    full = full.reshape(B * S, top_k, H).sum(axis=1)

    return full.reshape(B, S, H)


# ══════════════════════════════════════════════════════════════════════════
#  Expert-parallel MoE wrapper
# ══════════════════════════════════════════════════════════════════════════

class ExpertParallelMoE(nn.Module):
    """Wraps a MoE layer for expert-parallel execution.

    In forward:
    1. Replicate gate routing (same on all ranks)
    2. Dispatch tokens to expert-owning ranks via all_gather
    3. Compute local SwitchGLU with only this rank's expert weights
    4. Combine via all_gather + permutation
    """

    def __init__(self, original_layer: nn.Module, group: mx.distributed.Group):
        super().__init__()
        self.original_layer = original_layer
        self._group = group
        self._num_experts = original_layer.num_experts
        self._top_k = original_layer.top_k
        self._gate = original_layer.gate
        self._switch_mlp = original_layer.switch_mlp

    @property
    def num_experts(self) -> int:
        return self._num_experts

    @property
    def top_k(self) -> int:
        return self._top_k

    def _local_expert_fn(self, local_tokens: mx.array, local_expert_indices: mx.array) -> mx.array:
        """Run SwitchGLU on local expert subset only."""
        # local_tokens: [T_local, H]
        # local_expert_indices: [T_local] (indices into local weight slice [0..E_local-1])
        local_in = local_tokens.reshape(local_tokens.shape[0], 1, -1)  # [T_local, 1, H]
        local_out = self._switch_mlp(local_in, local_expert_indices)  # [T_local, 1, H]
        return local_out.reshape(local_tokens.shape[0], -1)  # [T_local, H]

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        if self._group is None:
            return self.original_layer.__call__(x, *args, **kwargs)

        # 1. Compute gate routing (identical on all ranks)
        inds, scores = self._gate(x)  # [B, S, top_k], [B, S, top_k]

        # 2. Dispatch + compute + combine (scores applied per-expert inside)
        routed_output = _dispatch_combine_ep(
            x, inds, scores, self._num_experts, self._group.size(),
            self._group.rank(), self._group, self._local_expert_fn,
        )  # [B, S, hidden] already scored
        if getattr(self.original_layer, "shared_experts", None) is not None:
            routed_output = routed_output + self.original_layer.shared_experts(x)

        return routed_output


class ExpertParallelMoEGemma4(nn.Module):
    """EP wrapper for Gemma4 MoE (slightly different interface: top_k_indices + top_k_weights)."""

    def __init__(self, original_layer: nn.Module, group: mx.distributed.Group):
        super().__init__()
        self.original_layer = original_layer
        self._group = group
        self._num_experts = original_layer.num_experts
        self._top_k = original_layer.num_experts_per_tok
        self._gate = original_layer.gate
        self._switch_glu = original_layer.switch_glu

    @property
    def num_experts(self) -> int:
        return self._num_experts

    @property
    def top_k(self) -> int:
        return self._top_k

    def _local_expert_fn(self, local_tokens: mx.array, local_expert_indices: mx.array) -> mx.array:
        local_in = local_tokens.reshape(local_tokens.shape[0], 1, -1)
        local_out = self._switch_glu(local_in, local_expert_indices)
        return local_out.reshape(local_tokens.shape[0], -1)

    def __call__(self, x: mx.array, top_k_indices: mx.array, top_k_weights: mx.array) -> mx.array:
        if self._group is None:
            return self.original_layer.__call__(x, top_k_indices, top_k_weights)

        routed_output = _dispatch_combine_ep(
            x, top_k_indices, top_k_weights, self._num_experts, self._group.size(),
            self._group.rank(), self._group, self._local_expert_fn,
        )
        return routed_output


# ══════════════════════════════════════════════════════════════════════════
#  Expert-parallel weight slicing helpers
# ══════════════════════════════════════════════════════════════════════════

def _slice_expert_weights(
    switch_layer: nn.Module,
    start: int,
    end: int,
) -> None:
    """Slice SwitchLinear weights to keep only experts [start, end).

    Mutates gate_proj, up_proj, down_proj (SwitchGLU) or fc1/fc2 (SwitchMLP)
    in-place.  Called during model loading before forward pass.
    """
    for attr_name in ("gate_proj", "up_proj", "fc1"):
        proj = getattr(switch_layer, attr_name, None)
        if proj is None:
            continue
        w = proj.weight
        if w.ndim == 3:
            proj.weight = w[start:end]
        if hasattr(proj, "bias") and proj.bias is not None and proj.bias.ndim == 2:
            proj.bias = proj.bias[start:end]

    for attr_name in ("down_proj", "fc2"):
        proj = getattr(switch_layer, attr_name, None)
        if proj is None:
            continue
        w = proj.weight
        if w.ndim == 3:
            proj.weight = w[start:end]
        if hasattr(proj, "bias") and proj.bias is not None and proj.bias.ndim == 2:
            proj.bias = proj.bias[start:end]


def _slice_quantized_expert_weights(
    switch_layer: nn.Module,
    start: int,
    end: int,
) -> None:
    """Slice QuantizedSwitchLinear weights to keep only experts [start, end).

    For quantized weights: weight [E, out, in], scales [E, out, in//gs],
    biases [E, out] (if present).
    """
    for attr_name in ("gate_proj", "up_proj", "fc1", "down_proj", "fc2"):
        proj = getattr(switch_layer, attr_name, None)
        if proj is None:
            continue
        if hasattr(proj, "weight") and proj.weight.ndim == 3:
            proj.weight = proj.weight[start:end]
        if hasattr(proj, "scales") and proj.scales.ndim == 3:
            proj.scales = proj.scales[start:end]
        if hasattr(proj, "biases") and proj.biases is not None and proj.biases.ndim == 2:
            proj.biases = proj.biases[start:end]


# ══════════════════════════════════════════════════════════════════════════
#  EP weight slicing + wrap helper
# ══════════════════════════════════════════════════════════════════════════

def _ep_slice_and_wrap(
    moe_layer: nn.Module,
    group: mx.distributed.Group,
    *,
    switch_mlp_attr: str = "switch_mlp",
    has_gate: bool = True,
    wrap_cls: type = ExpertParallelMoE,
) -> nn.Module:
    """Slice MoE weights for this rank and wrap with EP layer."""
    N = group.size()
    rank = group.rank()

    switch_mlp = getattr(moe_layer, switch_mlp_attr)
    n_routed = int(switch_mlp.gate_proj.weight.shape[0])
    E_local = n_routed // N
    e_start = rank * E_local
    e_end = (rank + 1) * E_local

    _slice_expert_weights(switch_mlp, e_start, e_end)
    if hasattr(switch_mlp.gate_proj, "bits"):
        _slice_quantized_expert_weights(switch_mlp, e_start, e_end)

    ep_layer = wrap_cls(moe_layer, group)
    return ep_layer


# ══════════════════════════════════════════════════════════════════════════
#  Expert-parallel sharding strategies
# ══════════════════════════════════════════════════════════════════════════

from functools import partial

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.worker.engines.mlx.auto_parallel import (
    TensorParallelShardingStrategy,
)


class DeepSeekExpertParallelStrategy(TensorParallelShardingStrategy):
    """Expert Parallelism for DeepSeek V3 / V3.2 / Kimi K2.5.

    Attention: tensor-parallel (same as existing TP path).
    MoE: expert-parallel — different experts on different nodes.
    Shared experts: replicated (all nodes compute).
    """

    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        from mlx_lm.models.deepseek_v3 import (
            DeepseekV3MLP,
            DeepseekV3Model,
            DeepseekV32MLP,
            DeepseekV32Model,
        )

        model = cast(DeepseekV3Model | DeepseekV32Model, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            # ── Attention (tensor-parallel, same as existing) ──
            if layer.self_attn.q_lora_rank is None:
                layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            else:
                layer.self_attn.q_b_proj = self.all_to_sharded_linear(layer.self_attn.q_b_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
            layer.self_attn.num_heads //= self.N

            num_heads = layer.self_attn.num_heads
            sh = self.group.rank() * num_heads
            eh = sh + num_heads

            def shard_heads(w: mx.array, sh: int = sh, eh: int = eh) -> mx.array:
                return w[sh:eh]

            layer.self_attn.embed_q.apply(shard_heads)
            layer.self_attn.unembed_out.apply(shard_heads)

            # ── MoE (expert-parallel) ──
            if isinstance(layer.mlp, (DeepseekV3MLP, DeepseekV32MLP)):
                # Dense MLP: tensor-parallel
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)
            else:
                # Shared experts: replicated (all nodes compute, no slicing)
                if getattr(layer.mlp, "shared_experts", None) is not None:
                    self.all_to_sharded_linear_in_place(layer.mlp.shared_experts.gate_proj)
                    self.sharded_to_all_linear_in_place(layer.mlp.shared_experts.down_proj)
                    self.all_to_sharded_linear_in_place(layer.mlp.shared_experts.up_proj)

                layer.mlp = _ep_slice_and_wrap(layer.mlp, self.group)

            mx.eval(layer)
            yield ModelLoadingResponse(layers_loaded=i, total=total)

        return model


class DeepseekV4ExpertParallelStrategy(TensorParallelShardingStrategy):
    """Expert Parallelism for DeepSeek V4 (MoE-only, attention replicated)."""

    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        from mlx_lm.models.deepseek_v4 import DeepseekV4Model
        from exo.worker.engines.mlx.auto_parallel import ShardedMoEV4

        model = cast(DeepseekV4Model, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            ffn = layer.ffn

            # Shared experts: replicated
            if getattr(ffn, "shared_experts", None) is not None:
                self.all_to_sharded_linear_in_place(ffn.shared_experts.gate_proj)
                self.sharded_to_all_linear_in_place(ffn.shared_experts.down_proj)
                self.all_to_sharded_linear_in_place(ffn.shared_experts.up_proj)

            layer.ffn = _ep_slice_and_wrap(ffn, self.group)

            mx.eval(layer)
            mx.clear_cache()
            yield ModelLoadingResponse(layers_loaded=i, total=total)

        return model


class QwenExpertParallelStrategy(TensorParallelShardingStrategy):
    """Expert Parallelism for Qwen3-MoE / Qwen3-Next / Qwen3.5-MoE."""

    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        from mlx_lm.models.qwen3_moe import (
            Qwen3MoeDecoderLayer,
            Qwen3MoeSparseMoeBlock,
        )
        from mlx_lm.models.qwen3_next import (
            Qwen3NextDecoderLayer,
            Qwen3NextSparseMoeBlock,
        )
        from mlx_lm.models.qwen3_5 import (
            Qwen3_5DecoderLayer,
            SparseMoeBlock as Qwen3_5SparseMoeBlock,
        )
        from mlx_lm.models.qwen3 import TransformerBlock as Qwen3TransformerBlock
        from mlx_lm.models.qwen3 import Model as Qwen3Model
        from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel
        from mlx_lm.models.qwen3_next import Model as Qwen3NextModel
        from mlx_lm.models.qwen3_5 import Model as Qwen3_5TextModel
        from mlx_lm.models.qwen3_5_moe import Model as Qwen3_5MoeModel
        from mlx_lm.models.qwen3_vl import Model as Qwen3VLModel

        model = cast(
            Qwen3Model | Qwen3MoeModel | Qwen3NextModel
            | Qwen3_5TextModel | Qwen3_5MoeModel | Qwen3VLModel,
            model,
        )
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            # ── Attention (tensor-parallel) ──
            if isinstance(layer, (Qwen3MoeDecoderLayer, Qwen3TransformerBlock)):
                layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
                layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
                layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
                layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
                layer.self_attn.n_heads //= self.N
                layer.self_attn.n_kv_heads //= self.N
            else:
                # Qwen3NextDecoderLayer / Qwen3_5DecoderLayer
                if hasattr(layer, "linear_attn"):
                    linear_attn = layer.linear_attn
                    rank_r = self.group.rank()
                    N = self.N

                    if hasattr(linear_attn, "in_proj_qkvz"):
                        linear_attn.in_proj_qkvz = self.all_to_sharded_linear(linear_attn.in_proj_qkvz)
                        linear_attn.in_proj_ba = self.all_to_sharded_linear(linear_attn.in_proj_ba)
                    else:
                        linear_attn.in_proj_qkv = self.all_to_sharded_linear(linear_attn.in_proj_qkv)
                        linear_attn.in_proj_z = self.all_to_sharded_linear(linear_attn.in_proj_z)
                        linear_attn.in_proj_b = self.all_to_sharded_linear(linear_attn.in_proj_b)
                        linear_attn.in_proj_a = self.all_to_sharded_linear(linear_attn.in_proj_a)
                    linear_attn.out_proj = self.sharded_to_all_linear(linear_attn.out_proj)

                    # Conv1d and per-head params
                    key_dim = linear_attn.key_dim
                    value_dim = linear_attn.value_dim
                    key_dim_shard = key_dim // N
                    value_dim_shard = value_dim // N
                    q_idx = mx.arange(rank_r * key_dim_shard, (rank_r + 1) * key_dim_shard)
                    k_idx = mx.arange(key_dim + rank_r * key_dim_shard, key_dim + (rank_r + 1) * key_dim_shard)
                    v_idx = mx.arange(2 * key_dim + rank_r * value_dim_shard, 2 * key_dim + (rank_r + 1) * value_dim_shard)
                    conv_indices = mx.concatenate([q_idx, k_idx, v_idx])
                    linear_attn.conv1d.weight = linear_attn.conv1d.weight[conv_indices]
                    new_conv_dim = key_dim_shard * 2 + value_dim_shard
                    linear_attn.conv1d.groups = new_conv_dim
                    num_v_shard = linear_attn.num_v_heads // N
                    v_start = rank_r * num_v_shard
                    linear_attn.A_log = linear_attn.A_log[v_start:v_start + num_v_shard]
                    linear_attn.dt_bias = linear_attn.dt_bias[v_start:v_start + num_v_shard]
                    linear_attn.num_k_heads //= N
                    linear_attn.num_v_heads //= N
                    linear_attn.key_dim = linear_attn.head_k_dim * linear_attn.num_k_heads
                    linear_attn.value_dim = linear_attn.head_v_dim * linear_attn.num_v_heads
                    linear_attn.conv_dim = linear_attn.key_dim * 2 + linear_attn.value_dim
                else:
                    layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
                    layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
                    layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
                    layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
                    layer.self_attn.num_attention_heads //= self.N
                    layer.self_attn.num_key_value_heads //= self.N

            # ── MoE (expert-parallel) ──
            if isinstance(
                layer.mlp,
                (Qwen3MoeSparseMoeBlock, Qwen3NextSparseMoeBlock, Qwen3_5SparseMoeBlock),
            ):
                # Shared expert: replicated
                if hasattr(layer.mlp, "shared_expert") and layer.mlp.shared_expert is not None:
                    self.all_to_sharded_linear_in_place(layer.mlp.shared_expert.gate_proj)
                    self.sharded_to_all_linear_in_place(layer.mlp.shared_expert.down_proj)
                    self.all_to_sharded_linear_in_place(layer.mlp.shared_expert.up_proj)
                layer.mlp = _ep_slice_and_wrap(layer.mlp, self.group)
            else:
                # Dense MLP: tensor-parallel
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            mx.eval(layer)
            mx.clear_cache()
            yield ModelLoadingResponse(layers_loaded=i, total=total)

        return model


class GLM4MoeExpertParallelStrategy(TensorParallelShardingStrategy):
    """Expert Parallelism for GLM4-MoE / GLM4-MoE-Lite."""

    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        from mlx_lm.models.glm4_moe import Glm4MoeModel, MoE
        from mlx_lm.models.glm4_moe_lite import (
            GLM4MoeLiteModel,
            Glm4MoeLiteDecoderLayer,
            Glm4MoeLiteMLP,
        )

        if isinstance(model, GLM4MoeLiteModel):
            model = cast(GLM4MoeLiteModel, model)
            total = len(model.layers)
            for i, layer in enumerate(model.layers):
                layer = cast(Glm4MoeLiteDecoderLayer, layer)
                mx.eval(layer.parameters())

                if layer.self_attn.q_lora_rank is None:
                    layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
                else:
                    layer.self_attn.q_b_proj = self.all_to_sharded_linear(layer.self_attn.q_b_proj)
                layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
                layer.self_attn.num_heads //= self.N

                num_heads = layer.self_attn.num_heads
                sh = self.group.rank() * num_heads
                eh = sh + num_heads

                def shard_heads(w: mx.array, sh: int = sh, eh: int = eh) -> mx.array:
                    return w[sh:eh]

                layer.self_attn.embed_q.apply(shard_heads)
                layer.self_attn.unembed_out.apply(shard_heads)

                if isinstance(layer.mlp, Glm4MoeLiteMLP):
                    layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                    layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                    layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)
                else:
                    if getattr(layer.mlp, "shared_experts", None) is not None:
                        self.all_to_sharded_linear_in_place(layer.mlp.shared_experts.gate_proj)
                        self.sharded_to_all_linear_in_place(layer.mlp.shared_experts.down_proj)
                        self.all_to_sharded_linear_in_place(layer.mlp.shared_experts.up_proj)
                    layer.mlp = _ep_slice_and_wrap(layer.mlp, self.group)

                mx.eval(layer)
                mx.clear_cache()
                yield ModelLoadingResponse(layers_loaded=i, total=total)
            return model

        # Glm4MoeModel path
        model = cast(Glm4MoeModel, model)
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
            layer.self_attn.n_heads //= self.N
            layer.self_attn.n_kv_heads //= self.N

            if isinstance(layer.mlp, MoE):
                if getattr(layer.mlp, "shared_experts", None) is not None:
                    self.all_to_sharded_linear_in_place(layer.mlp.shared_experts.gate_proj)
                    self.sharded_to_all_linear_in_place(layer.mlp.shared_experts.down_proj)
                    self.all_to_sharded_linear_in_place(layer.mlp.shared_experts.up_proj)
                layer.mlp = _ep_slice_and_wrap(layer.mlp, self.group)
            else:
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            mx.eval(layer)
            mx.clear_cache()
            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


class MiniMaxExpertParallelStrategy(TensorParallelShardingStrategy):
    """Expert Parallelism for MiniMax."""

    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        from mlx_lm.models.minimax import MiniMaxModel
        from exo.worker.engines.mlx.auto_parallel import WrappedMiniMaxAttention

        model = cast(MiniMaxModel, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
            layer.self_attn.num_attention_heads //= self.N
            layer.self_attn.num_key_value_heads //= self.N
            layer.self_attn = WrappedMiniMaxAttention(layer.self_attn, self.group)

            layer.block_sparse_moe = _ep_slice_and_wrap(
                layer.block_sparse_moe, self.group, switch_mlp_attr="switch_mlp"
            )

            mx.eval(layer)
            mx.clear_cache()
            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


class GptOssExpertParallelStrategy(TensorParallelShardingStrategy):
    """Expert Parallelism for GptOSS MoE."""

    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        from mlx_lm.models.gpt_oss import GptOssMoeModel

        model = cast(GptOssMoeModel, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
            layer.self_attn.num_attention_heads //= self.N
            layer.self_attn.num_key_value_heads //= self.N
            layer.self_attn.num_key_value_groups = (
                layer.self_attn.num_attention_heads // layer.self_attn.num_key_value_heads
            )
            rank = self.group.rank()
            layer.self_attn.sinks = layer.self_attn.sinks[
                layer.self_attn.num_attention_heads * rank
                : layer.self_attn.num_attention_heads * (rank + 1)
            ]

            # GptOSS uses `experts` (SwitchGLU), not `switch_mlp`
            layer.mlp = _ep_slice_and_wrap(
                layer.mlp, self.group, switch_mlp_attr="experts"
            )

            mx.eval(layer)
            mx.clear_cache()
            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


class Step35ExpertParallelStrategy(TensorParallelShardingStrategy):
    """Expert Parallelism for Step3.5 MoE."""

    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        from mlx_lm.models.step3p5 import Step35MLP, Step35Model

        model = cast(Step35Model, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
            layer.self_attn.num_heads //= self.N
            layer.self_attn.num_kv_heads //= self.N
            if getattr(layer.self_attn, "use_head_wise_attn_gate", False):
                layer.self_attn.g_proj = self.all_to_sharded_linear(layer.self_attn.g_proj)

            if isinstance(layer.mlp, Step35MLP):
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
            else:
                # share_expert: replicated
                self.all_to_sharded_linear_in_place(layer.mlp.share_expert.gate_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.share_expert.up_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.share_expert.down_proj)
                layer.mlp = _ep_slice_and_wrap(layer.mlp, self.group)

            mx.eval(layer)
            mx.clear_cache()
            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


class NemotronHExpertParallelStrategy(TensorParallelShardingStrategy):
    """Expert Parallelism for NemotronH MoE layers."""

    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        from mlx_lm.models.nemotron_h import (
            NemotronHAttention,
            NemotronHMamba2Mixer,
            NemotronHMoE,
            NemotronHModel,
        )

        model = cast(NemotronHModel, model)
        total = len(model.layers)
        rank = self.group.rank()

        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())
            mixer = layer.mixer

            if isinstance(mixer, NemotronHAttention):
                mixer.q_proj = self.all_to_sharded_linear(mixer.q_proj)
                mixer.k_proj = self.all_to_sharded_linear(mixer.k_proj)
                mixer.v_proj = self.all_to_sharded_linear(mixer.v_proj)
                mixer.o_proj = self.sharded_to_all_linear(mixer.o_proj)
                mixer.num_heads //= self.N
                mixer.num_key_value_heads //= self.N

            elif isinstance(mixer, NemotronHMamba2Mixer):
                # Mamba2: tensor-parallel (no MoE)
                from exo.worker.engines.mlx.auto_parallel import NemotronHShardingStrategy
                _tp = NemotronHShardingStrategy(
                    self.group, self.all_to_sharded_linear, self.sharded_to_all_linear,
                    self.all_to_sharded_linear_in_place, self.sharded_to_all_linear_in_place,
                )
                _tp._shard_mamba2_mixer(mixer, rank)

            elif isinstance(mixer, NemotronHMoE):
                N = self.N
                n_routed = mixer.switch_mlp.fc1.weight.shape[0]
                E_local = n_routed // N
                e_start = rank * E_local
                e_end = (rank + 1) * E_local
                _slice_expert_weights(mixer.switch_mlp, e_start, e_end)
                if hasattr(mixer.switch_mlp.fc1, "bits"):
                    _slice_quantized_expert_weights(mixer.switch_mlp, e_start, e_end)
                if hasattr(mixer, "shared_experts"):
                    self.all_to_sharded_linear_in_place(mixer.shared_experts.up_proj)
                    self.sharded_to_all_linear_in_place(mixer.shared_experts.down_proj)
                layer.mixer = _ep_slice_and_wrap(
                    mixer, self.group, switch_mlp_attr="switch_mlp"
                )

            mx.eval(layer)
            mx.clear_cache()
            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


class Gemma4ExpertParallelStrategy(TensorParallelShardingStrategy):
    """Expert Parallelism for Gemma4 MoE layers."""

    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        from mlx_lm.models.gemma4 import Gemma4Model

        model = cast(Gemma4Model, model)
        N = self.N
        layers = model.language_model.model.layers
        total = len(layers)

        for i, layer in enumerate(layers):
            mx.eval(layer.parameters())

            attn = layer.self_attn
            attn.q_proj = self.all_to_sharded_linear(attn.q_proj)
            if attn.has_kv:
                attn.k_proj = self.all_to_sharded_linear(attn.k_proj)
                if not attn.use_k_eq_v:
                    attn.v_proj = self.all_to_sharded_linear(attn.v_proj)
            attn.o_proj = self.sharded_to_all_linear(attn.o_proj)
            attn.n_heads //= N
            attn.n_kv_heads //= N

            layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
            layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
            layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            if layer.enable_moe:
                layer.experts = _ep_slice_and_wrap(
                    layer.experts, self.group, switch_mlp_attr="switch_glu",
                    wrap_cls=ExpertParallelMoEGemma4,
                )

            mx.eval(layer)
            mx.clear_cache()
            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


# ══════════════════════════════════════════════════════════════════════════
#  Public entry point
# ══════════════════════════════════════════════════════════════════════════

def expert_auto_parallel(
    model: nn.Module,
    group: mx.distributed.Group,
) -> Generator[ModelLoadingResponse, None, nn.Module]:
    """Expert-parallel auto-parallel: tensor-parallel attention + EP MoE.

    Called as alternative to tensor_auto_parallel when EXO_EXPERT_PARALLEL=1.
    """
    from mlx_lm.models.deepseek_v3 import DeepseekV3Model
    from mlx_lm.models.deepseek_v32 import Model as DeepseekV32Model
    from mlx_lm.models.kimi_k25 import Model as KimiK25Model
    from mlx_lm.models.deepseek_v4 import Model as DeepseekV4Model
    from mlx_lm.models.minimax import Model as MiniMaxModel
    from mlx_lm.models.glm4_moe import Model as Glm4MoeModel
    from mlx_lm.models.glm4_moe_lite import Model as GLM4MoeLiteModel
    from mlx_lm.models.qwen3 import Model as Qwen3Model
    from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel
    from mlx_lm.models.qwen3_next import Model as Qwen3NextModel
    from mlx_lm.models.qwen3_5 import Model as Qwen3_5TextModel
    from mlx_lm.models.qwen3_5_moe import Model as Qwen3_5MoeModel
    from mlx_lm.models.qwen3_vl import Model as Qwen3VLModel
    from mlx_lm.models.gpt_oss import GptOssModel
    from mlx_lm.models.step3p5 import Model as Step35Model
    from mlx_lm.models.nemotron_h import Model as NemotronHModel
    from mlx_lm.models.gemma4 import Model as Gemma4Model

    all_to_sharded_linear = partial(mx.nn.shard_linear, sharding="all-to-sharded", group=group)
    sharded_to_all_linear = partial(mx.nn.shard_linear, sharding="sharded-to-all", group=group)

    def _all_to_sharded(path: str, weight: mx.array):
        segments = 1
        if path.endswith("bias"):
            return weight.ndim - 1, segments
        return max(weight.ndim - 2, 0), segments

    all_to_sharded_linear_in_place = partial(mx.nn.shard_inplace, sharding=_all_to_sharded, group=group)

    def _sharded_to_all(path: str, weight: mx.array):
        segments = 1
        if path.endswith("bias"):
            weight /= group.size()
            return None
        return -1, segments

    sharded_to_all_linear_in_place = partial(mx.nn.shard_inplace, sharding=_sharded_to_all, group=group)

    if isinstance(model, (DeepseekV3Model, DeepseekV32Model, KimiK25Model)):
        strategy = DeepSeekExpertParallelStrategy(group, all_to_sharded_linear, sharded_to_all_linear, all_to_sharded_linear_in_place, sharded_to_all_linear_in_place)
    elif isinstance(model, DeepseekV4Model):
        strategy = DeepseekV4ExpertParallelStrategy(group, all_to_sharded_linear, sharded_to_all_linear, all_to_sharded_linear_in_place, sharded_to_all_linear_in_place)
    elif isinstance(model, MiniMaxModel):
        strategy = MiniMaxExpertParallelStrategy(group, all_to_sharded_linear, sharded_to_all_linear, all_to_sharded_linear_in_place, sharded_to_all_linear_in_place)
    elif isinstance(model, (GLM4MoeLiteModel, Glm4MoeModel)):
        strategy = GLM4MoeExpertParallelStrategy(group, all_to_sharded_linear, sharded_to_all_linear, all_to_sharded_linear_in_place, sharded_to_all_linear_in_place)
    elif isinstance(model, (Qwen3Model, Qwen3MoeModel, Qwen3NextModel, Qwen3_5TextModel, Qwen3_5MoeModel, Qwen3VLModel)):
        strategy = QwenExpertParallelStrategy(group, all_to_sharded_linear, sharded_to_all_linear, all_to_sharded_linear_in_place, sharded_to_all_linear_in_place)
    elif isinstance(model, GptOssModel):
        strategy = GptOssExpertParallelStrategy(group, all_to_sharded_linear, sharded_to_all_linear, all_to_sharded_linear_in_place, sharded_to_all_linear_in_place)
    elif isinstance(model, Step35Model):
        strategy = Step35ExpertParallelStrategy(group, all_to_sharded_linear, sharded_to_all_linear, all_to_sharded_linear_in_place, sharded_to_all_linear_in_place)
    elif isinstance(model, NemotronHModel):
        strategy = NemotronHExpertParallelStrategy(group, all_to_sharded_linear, sharded_to_all_linear, all_to_sharded_linear_in_place, sharded_to_all_linear_in_place)
    elif isinstance(model, Gemma4Model):
        strategy = Gemma4ExpertParallelStrategy(group, all_to_sharded_linear, sharded_to_all_linear, all_to_sharded_linear_in_place, sharded_to_all_linear_in_place)
    else:
        raise ValueError(f"Expert parallel not supported for model type: {type(model)}")

    logger.info(f"Expert parallel: using {strategy.__class__.__name__}")
    model = yield from strategy.shard_model(model)
    return model
