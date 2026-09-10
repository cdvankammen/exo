import os
from collections.abc import Generator, Mapping, Sequence

from loguru import logger

from exo.shared.models.model_cards import ModelCard
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.common import Host, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage, NodeIdentity, NodeNetworkInfo
from exo.shared.types.topology import Cycle, RDMAConnection, SocketConnection
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    RingShardMetadata,
    Sharding,
    ShardMetadata,
    TensorShardMetadata,
)


# --- Audit logging for force_override bypass (RISK-1 from commit review) ---


def _log_force_override_audit(
    *,
    bypass_point: str,
    node_ids: list[str],
    model_id: str = "",
    sharding: str = "",
    cycle_total_mem_bytes: int = 0,
    required_mem_bytes: int = 0,
    extra: dict | None = None,
) -> None:
    """Emit a structured WARNING audit entry when force_override bypasses a guard rail.

    Every call produces a single loguru WARNING line with a [FORCE_OVERRIDE_AUDIT] tag
    so it is trivially greppable and filterable in the logs panel.

    The ``bypass_point`` names the specific guard rail that was skipped:
      GR1_MEMORY_CAPACITY  — filter_cycles_by_memory accepted a cycle despite memory shortfall
      GR2_ZERO_MEMORY      — _compute_total_memory proceeded with zero total memory
      GR3_LAYER_CAPS       — per-node layer allocation caps disabled
      GR4_BANDWIDTH_ALLOC  — bandwidth-based water-filling skipped
      GR5_PER_NODE_MEMORY  — per-node memory validation skipped
    """
    logger.warning(
        f"[FORCE_OVERRIDE_AUDIT] guard_rail={bypass_point} "
        f"model={model_id} nodes={[n[:12] for n in node_ids]} "
        f"sharding={sharding} "
        f"cycle_mem={cycle_total_mem_bytes} required={required_mem_bytes} "
        f"shortfall={required_mem_bytes - cycle_total_mem_bytes} "
        f"| {extra or {}}"
    )


def filter_cycles_by_memory(
    cycles: list[Cycle],
    node_memory: Mapping[NodeId, MemoryUsage],
    required_memory: Memory,
    force_override: bool = False,
) -> list[Cycle]:
    filtered_cycles: list[Cycle] = []
    for cycle in cycles:
        if not all(node in node_memory for node in cycle):
            continue

        total_mem = sum(
            (node_memory[node_id].ram_available for node_id in cycle.node_ids),
            start=Memory(),
        )
        if force_override or total_mem >= required_memory:
            if force_override and total_mem < required_memory:
                _log_force_override_audit(
                    bypass_point="GR1_MEMORY_CAPACITY",
                    node_ids=list(cycle.node_ids),
                    cycle_total_mem_bytes=total_mem.in_bytes,
                    required_mem_bytes=required_memory.in_bytes,
                )
            filtered_cycles.append(cycle)
    return filtered_cycles


def filter_cycles_by_replicated_memory(
    cycles: list[Cycle],
    node_memory: Mapping[NodeId, MemoryUsage],
    required_memory: Memory,
) -> list[Cycle]:
    """Keep cycles where every node can hold a fully replicated model."""
    return [
        cycle
        for cycle in cycles
        if all(
            node_id in node_memory
            and node_memory[node_id].inference_available >= required_memory
            for node_id in cycle.node_ids
        )
    ]


# K and V cache entries are stored unquantized at 2 bytes each during Ring
# prefill; the working-set multiplier covers transient attention buffers and
# allocator caching observed in practice (Metal peaked at ~3x the steady KV
# size, CUDA at ~5x, for the same prefill).
_KV_CACHE_BYTES_PER_ELEMENT = 2
_RING_KV_WORKING_SET_MULTIPLIER = 4
# When tiered KV offload is active (EXO_TIERED_KV=1), the KV cache can spill
# to CPU RAM during prefill, so the GPU working-set headroom needed is lower.
# 2× covers attention transients while the spill handles the steady-state KV.
_RING_KV_WORKING_SET_MULTIPLIER_TIERED = int(
    os.environ.get("EXO_RING_KV_WORKING_SET_MULTIPLIER_TIERED", "2")
)
# Upper bound on head_dim across the verified Ring model families (Llama,
# Qwen3), used when the card does not pin the exact geometry.
_RING_HEAD_DIM_BOUND = 128
# Ring exists for long-context prefill; admit against at least this context
# even if requests may be shorter, and no more than this even for cards that
# advertise 128K+ so admission stays achievable.
_RING_ADMISSION_CONTEXT_TOKENS = int(os.environ.get("EXO_RING_ADMISSION_CONTEXT", "16384"))


def estimate_ring_node_memory(model_card: ModelCard) -> Memory:
    """Per-node memory a Ring rank needs: replicated weights plus the working
    set of a long-context prefill (full KV cache and attention transients).

    Weight-only admission demonstrably over-admits: a 4 GB-VRAM rank passed
    the old check for a model whose 16K prefill peaks well past 4 GB.

    When tiered KV offload is active (EXO_TIERED_KV=1), the KV cache can
    spill to CPU RAM during prefill, so the GPU-resident working-set
    multiplier drops from 4× to 2× — the spill handles steady-state KV while
    the reduced headroom still covers attention transients.
    """
    admission_context = (
        min(model_card.context_length, _RING_ADMISSION_CONTEXT_TOKENS)
        if model_card.context_length > 0
        else _RING_ADMISSION_CONTEXT_TOKENS
    )
    if model_card.num_key_value_heads is not None:
        kv_width = model_card.num_key_value_heads * _RING_HEAD_DIM_BOUND
    else:
        kv_width = model_card.hidden_size
    kv_cache_bytes = (
        2  # keys and values
        * _KV_CACHE_BYTES_PER_ELEMENT
        * model_card.n_layers
        * kv_width
        * admission_context
    )
    # Tier-aware multiplier: with tiered KV spill, the GPU working set is
    # smaller because excess KV moves to CPU RAM rather than staying on GPU.
    _tiered = os.environ.get("EXO_TIERED_KV", "0") == "1"
    multiplier = (
        _RING_KV_WORKING_SET_MULTIPLIER_TIERED
        if _tiered
        else _RING_KV_WORKING_SET_MULTIPLIER
    )
    working_set = Memory.from_bytes(kv_cache_bytes * multiplier)
    return model_card.storage_size + working_set


def get_smallest_cycles(
    cycles: list[Cycle],
) -> list[Cycle]:
    min_nodes = min(len(cycle) for cycle in cycles)
    return [cycle for cycle in cycles if len(cycle) == min_nodes]


# Pipeline parallel stages need transient activation memory in addition to
# their weight shard. Empirically this is ~10% of the layer weights resident
# on the node (intermediate tensors, attention scores, residual stream).
_PIPELINE_ACTIVATION_FRACTION = 0.1


def estimate_pipeline_activation_memory(
    total_storage: Memory,
    total_layers: int,
    layers_on_node: int,
) -> Memory:
    """Estimated activation memory for a pipeline stage.

    Formula: per_layer_bytes * layers_on_node * _PIPELINE_ACTIVATION_FRACTION
    (10% of the weight bytes on the node).
    """
    per_layer_bytes = total_storage.in_bytes // total_layers
    activation_bytes = int(per_layer_bytes * layers_on_node * _PIPELINE_ACTIVATION_FRACTION)
    return Memory.from_bytes(activation_bytes)


# Approximate GPU memory bandwidth (GB/s) by chip/GPU name substring. Decode
# is memory-bandwidth bound, so pipeline stage time is proportional to the
# bytes of weights a node reads per token divided by its bandwidth. Substring
# order matters: more specific names (e.g. "M3 Ultra") must precede less
# specific ones (e.g. "M3").
_MEMORY_BANDWIDTH_GBPS_BY_CHIP_SUBSTRING: tuple[tuple[str, float], ...] = (
    ("M4 Ultra", 1092.0),
    ("M3 Ultra", 819.0),
    ("M2 Ultra", 800.0),
    ("M1 Ultra", 800.0),
    ("M4 Max", 546.0),
    ("M3 Max", 400.0),
    ("M2 Max", 400.0),
    ("M1 Max", 400.0),
    ("M4 Pro", 273.0),
    ("M3 Pro", 150.0),
    ("M2 Pro", 200.0),
    ("M1 Pro", 200.0),
    ("M4", 120.0),
    ("M3", 100.0),
    ("M2", 100.0),
    ("M1", 68.0),
    ("GB10", 273.0),  # NVIDIA DGX Spark
    ("RTX 5090", 1792.0),
    ("RTX 5080", 960.0),
    ("RTX 5060 Ti", 448.0),
    ("RTX 4090", 1008.0),
    ("RTX 4080", 717.0),
    ("RTX 3090", 936.0),
    ("RTX 3080", 760.0),
)


def estimate_memory_bandwidth_gigabytes_per_second(
    node_identity: NodeIdentity,
) -> float | None:
    """Estimated GPU memory bandwidth for a node, or None when unrecognised."""
    chip_name = node_identity.chip_id.lower()
    for chip_substring, bandwidth in _MEMORY_BANDWIDTH_GBPS_BY_CHIP_SUBSTRING:
        if chip_substring.lower() in chip_name:
            return bandwidth
    return None


def should_prefer_single_node(
    model_card: ModelCard,
    cycles: list[Cycle],
    node_memory: Mapping[NodeId, MemoryUsage],
    node_identities: Mapping[NodeId, NodeIdentity] | None = None,
) -> Cycle | None:
    """Return a single-node cycle when TP would be slower than running locally.

    Tensor parallelism splits weights across nodes but requires an all-reduce
    after every attention and MLP block. On slow interconnects (WiFi, 1 GbE)
    the communication cost dominates decode latency for models that already
    fit in one node's memory. This heuristic picks the fastest single node
    whose available RAM can hold the full model, returning its cycle so the
    caller can downgrade to Pipeline sharding.

    Returns None when no single node can hold the model or when multi-node
    is genuinely faster (e.g. model exceeds any single node's capacity).
    """
    identities = node_identities or {}

    # Find single-node cycles where the model fits entirely in RAM
    viable_singles: list[tuple[Cycle, float]] = []
    for cycle in cycles:
        if len(cycle) != 1:
            continue
        node_id = cycle.node_ids[0]
        mem = node_memory.get(node_id)
        if mem is None:
            continue
        if mem.ram_available < model_card.storage_size:
            continue
        bandwidth = estimate_memory_bandwidth_gigabytes_per_second(
            identities.get(node_id, NodeIdentity())
        )
        # Use bandwidth as proxy for compute speed; unknown hardware gets 0
        viable_singles.append((cycle, bandwidth or 0.0))

    if not viable_singles:
        return None

    # Pick the fastest single node
    best_cycle, _ = max(viable_singles, key=lambda pair: pair[1])
    return best_cycle


def allocate_layers_by_throughput(
    total_layers: int,
    node_throughputs: list[float],
    max_layers_per_node: list[int],
) -> list[int]:
    """Split layers to minimise summed per-stage decode time.

    Per-token pipeline decode latency is the sum of every stage's compute
    time, and stage time is (layers on node) / (node throughput), so the sum
    is minimised by loading the fastest nodes to their memory capacity first.
    Every node keeps at least one layer (a pipeline stage cannot be empty).
    Raises ValueError when allocation is impossible; handled by the placement
    caller (``place_instance``) which surfaces it to the API.
    """
    n = len(node_throughputs)
    if n == 0:
        raise ValueError("Cannot allocate layers to an empty node list")
    if total_layers < n:
        raise ValueError(
            f"Cannot distribute {total_layers} layers across {n} nodes "
            "(need at least 1 layer per node)"
        )
    if any(cap < 1 for cap in max_layers_per_node):
        raise ValueError(
            "Every pipeline node must have memory capacity for at least one layer"
        )
    if sum(max_layers_per_node) < total_layers:
        raise ValueError(
            f"Selected nodes only have capacity for {sum(max_layers_per_node)} of "
            f"{total_layers} layers"
        )

    result = [1] * n
    remaining = total_layers - n
    for i in sorted(range(n), key=lambda i: node_throughputs[i], reverse=True):
        take = min(max_layers_per_node[i] - result[i], remaining)
        result[i] += take
        remaining -= take
        if remaining == 0:
            break
    assert remaining == 0
    return result


def allocate_layers_by_bandwidth_waterfill(
    total_layers: int,
    node_bandwidths: list[float],
    max_layers_per_node: list[int],
) -> list[int]:
    """Split layers via water-filling to equalize per-stage decode time.

    Greedy fill-fastest-first minimizes summed stage time (latency) but
    creates a bottleneck stage, capping pipeline throughput at max(L_i/B_i).
    Water-filling instead assigns each layer to the node with the lowest
    current L_i/B_i ratio, which minimizes max(L_i/B_i) and maximizes
    decode throughput.

    Example: 20 layers, M1 Max (400 GB/s) + M3 Ultra (819 GB/s)
      Greedy: [19, 1] -> max(19/400, 1/819) = 0.0475
      Water:  [13, 7]  -> max(13/400, 7/819) = 0.0325  (1.46x better)
    """
    n = len(node_bandwidths)
    if n == 0:
        raise ValueError("Cannot allocate layers to an empty node list")
    if total_layers < n:
        raise ValueError(
            f"Cannot distribute {total_layers} layers across {n} nodes "
            "(need at least 1 layer per node)"
        )
    if any(cap < 1 for cap in max_layers_per_node):
        raise ValueError(
            "Every pipeline node must have memory capacity for at least one layer"
        )
    if sum(max_layers_per_node) < total_layers:
        raise ValueError(
            f"Selected nodes only have capacity for {sum(max_layers_per_node)} of "
            f"{total_layers} layers"
        )

    # Start with 1 layer per node (a pipeline stage cannot be empty).
    allocations = [1] * n
    current_loads = [1.0 / bw for bw in node_bandwidths]
    remaining = total_layers - n

    for _ in range(remaining):
        # Each new layer goes to the node with the least accumulated load
        # relative to its bandwidth, skipping nodes already at capacity.
        best = 0
        for i in range(1, n):
            if allocations[i] < max_layers_per_node[i] and (
                current_loads[i] < current_loads[best]
                or allocations[best] >= max_layers_per_node[best]
            ):
                best = i
        if allocations[best] >= max_layers_per_node[best]:
            for i in range(n):
                if allocations[i] < max_layers_per_node[i]:
                    best = i
                    break
        allocations[best] += 1
        current_loads[best] = allocations[best] / node_bandwidths[best]

    return allocations



def cycle_bandwidth_score(
    cycle: Cycle,
    node_identities: Mapping[NodeId, NodeIdentity] | None,
) -> float:
    """Pipeline throughput score: minimum node bandwidth in cycle.

    Higher minimum means no slow bottleneck stage.
    Returns 0.0 when hardware is unknown.
    """
    if not node_identities:
        return 0.0
    bws: list[float] = []
    for nid in cycle.node_ids:
        bw = estimate_memory_bandwidth_gigabytes_per_second(
            node_identities.get(nid, NodeIdentity())
        )
        if bw is None:
            return 0.0
        bws.append(bw)
    return min(bws)


def allocate_layers_proportionally(
    total_layers: int,
    memory_fractions: list[float],
    max_layers_per_node: list[int] | None = None,
) -> list[int]:
    """Split layers across nodes proportionally to their memory fractions.

    ``max_layers_per_node`` caps how many layers each node may receive (how
    many fit in its available memory). Without caps, largest-remainder
    rounding can hand a leftover layer to a node that has no memory slack
    for it. Raises ValueError when allocation is impossible; handled by the
    placement caller (``place_instance``) which surfaces it to the API.
    """
    n = len(memory_fractions)
    if n == 0:
        raise ValueError("Cannot allocate layers to an empty node list")
    if total_layers < n:
        raise ValueError(
            f"Cannot distribute {total_layers} layers across {n} nodes "
            "(need at least 1 layer per node)"
        )
    caps = (
        max_layers_per_node if max_layers_per_node is not None else [total_layers] * n
    )
    assert len(caps) == n
    if any(cap < 1 for cap in caps):
        raise ValueError(
            "A selected pipeline node has insufficient memory to hold even one layer"
        )
    if sum(caps) < total_layers:
        raise ValueError(
            f"Selected nodes only have capacity for {sum(caps)} of "
            f"{total_layers} layers"
        )

    # Largest remainder: floor each (capped), then hand out the remaining
    # layers by fractional part, skipping nodes that are at capacity.
    raw = [fraction * total_layers for fraction in memory_fractions]
    result = [min(int(r), cap) for r, cap in zip(raw, caps, strict=True)]
    by_remainder = sorted(range(n), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    remaining = total_layers - sum(result)
    while remaining > 0:
        for i in by_remainder:
            if remaining == 0:
                break
            if result[i] < caps[i]:
                result[i] += 1
                remaining -= 1

    # Ensure minimum 1 per node by taking from the largest
    for i in range(n):
        if result[i] == 0:
            max_idx = max(range(n), key=lambda j: result[j])
            assert result[max_idx] > 1
            result[max_idx] -= 1
            result[i] = 1

    return result


def _validate_cycle(cycle: Cycle) -> None:
    if not cycle.node_ids:
        raise ValueError("Cannot create shard assignments for empty node cycle")


def _compute_total_memory(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
    force_override: bool = False,
) -> Memory:
    total_memory = sum(
        (node_memory[node_id].ram_available for node_id in node_ids),
        start=Memory(),
    )
    if not force_override and total_memory.in_bytes == 0:
        raise ValueError("Cannot create shard assignments: total available memory is 0")
    if force_override and total_memory.in_bytes == 0:
        _log_force_override_audit(
            bypass_point="GR2_ZERO_MEMORY",
            node_ids=node_ids,
            cycle_total_mem_bytes=0,
            required_mem_bytes=0,
        )
    return total_memory


def _allocate_and_validate_layers(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
    total_memory: Memory,
    model_card: ModelCard,
    force_override: bool = False,
    node_identities: Mapping[NodeId, NodeIdentity] | None = None,
    include_activation: bool = True,
) -> list[int]:
    # NOTE (memory-override feature, coordinated with RDMA multi-link commit):
    # The RDMA commit added `max_layers_per_node` caps (per-node memory limits
    # that raise if a node can't hold even 1 layer). When force_override=True
    # (user chose "load the model anyway"), we pass caps=None to bypass those
    # caps so an oversized model can still be allocated. Without this, the caps
    # ValueError fires BEFORE the per-node check below, blocking the override.
    # Caps must account for the 10% activation headroom (pipeline only):
    # each layer costs storage_size/n_layers * 1.1, so max =
    # ram * n_layers * 10 // (storage * 11).  When include_activation=False
    # (CFG parallel), use the simpler weight-only cap.
    if include_activation:
        caps = (
            None
            if force_override
            else [
                (
                    node_memory[node_id].ram_available.in_bytes
                    * model_card.n_layers
                    * 10
                )
                // (model_card.storage_size.in_bytes * 11)
                for node_id in node_ids
            ]
        )
    else:
        caps = (
            None
            if force_override
            else [
                (
                    node_memory[node_id].ram_available.in_bytes
                    * model_card.n_layers
                )
                // model_card.storage_size.in_bytes
                for node_id in node_ids
            ]
        )

    node_bandwidths = [
        estimate_memory_bandwidth_gigabytes_per_second(
            (node_identities or {}).get(node_id, NodeIdentity())
        )
        for node_id in node_ids
    ]

    if not force_override and all(
        bandwidth is not None for bandwidth in node_bandwidths
    ):
        # Water-filling equalizes per-stage decode time (L_i/B_i) across
        # nodes, minimising the bottleneck stage time and maximising
        # pipeline decode throughput.
        assert caps is not None  # force_override is False in this branch
        layer_allocations = allocate_layers_by_bandwidth_waterfill(
            total_layers=model_card.n_layers,
            node_bandwidths=[
                bandwidth for bandwidth in node_bandwidths if bandwidth is not None
            ],
            max_layers_per_node=caps,
        )
    else:
        if force_override:
            _log_force_override_audit(
                bypass_point="GR3_LAYER_CAPS",
                node_ids=node_ids,
                model_id=model_card.model_id,
                cycle_total_mem_bytes=total_memory.in_bytes,
            )
        # Unknown hardware (or force_override): fall back to memory-proportional
        # allocation.
        layer_allocations = allocate_layers_proportionally(
            total_layers=model_card.n_layers,
            memory_fractions=[
                node_memory[node_id].ram_available / total_memory
                for node_id in node_ids
            ],
            max_layers_per_node=caps,
        )

    total_storage = model_card.storage_size
    total_layers = model_card.n_layers

    for i, node_id in enumerate(node_ids):
        node_layers = layer_allocations[i]
        weight_memory = (total_storage * node_layers) // total_layers
        if include_activation:
            activation_memory = estimate_pipeline_activation_memory(
                total_storage, total_layers, node_layers
            )
        else:
            activation_memory = Memory.from_bytes(0)
        required_memory = weight_memory + activation_memory
        available_memory = node_memory[node_id].ram_available
        if not force_override and required_memory > available_memory:
            raise ValueError(
                f"Node {i} ({node_id}) has insufficient memory: "
                f"requires {required_memory.in_gb:.2f} GB for {node_layers} layers "
                f"(weights {weight_memory.in_gb:.2f} GB + activations "
                f"{activation_memory.in_gb:.2f} GB), "
                f"but only has {available_memory.in_gb:.2f} GB available"
            )
        if force_override and required_memory > available_memory:
            _log_force_override_audit(
                bypass_point="GR5_PER_NODE_MEMORY",
                node_ids=[node_id],
                model_id=model_card.model_id,
                cycle_total_mem_bytes=available_memory.in_bytes,
                required_mem_bytes=required_memory.in_bytes,
                extra={
                    "node_layers": node_layers,
                    "weight_gb": f"{weight_memory.in_gb:.2f}",
                    "activation_gb": f"{activation_memory.in_gb:.2f}",
                },
            )

    return layer_allocations


def _validate_manual_layer_allocations(
    node_ids: list[NodeId],
    node_memory: Mapping[NodeId, MemoryUsage],
    model_card: ModelCard,
    node_layers: Mapping[NodeId, int],
) -> list[int]:
    if set(node_layers) != set(node_ids):
        raise ValueError(
            "Manual layer allocation must specify exactly the selected pipeline nodes"
        )
    if any(layer_count < 1 for layer_count in node_layers.values()):
        raise ValueError(
            "Manual layer allocations must assign at least one layer per node"
        )
    if sum(node_layers.values()) != model_card.n_layers:
        raise ValueError(
            f"Manual layer allocations must sum to {model_card.n_layers} layers"
        )

    allocations = [node_layers[node_id] for node_id in node_ids]
    for index, (node_id, layer_count) in enumerate(
        zip(node_ids, allocations, strict=True)
    ):
        weight_memory = (
            (model_card.storage_size * layer_count) // model_card.n_layers
        )
        activation_memory = estimate_pipeline_activation_memory(
            model_card.storage_size, model_card.n_layers, layer_count
        )
        required_memory = weight_memory + activation_memory
        available_memory = node_memory[node_id].ram_available
        if required_memory > available_memory:
            raise ValueError(
                f"Node {index} ({node_id}) has insufficient memory: "
                f"requires {required_memory.in_gb:.2f} GB for {layer_count} layers "
                f"(weights {weight_memory.in_gb:.2f} GB + activations "
                f"{activation_memory.in_gb:.2f} GB), "
                f"but only has {available_memory.in_gb:.2f} GB available"
            )
    return allocations


def get_shard_assignments_for_pipeline_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
    force_override: bool = False,
    node_identities: Mapping[NodeId, NodeIdentity] | None = None,
    node_layers: Mapping[NodeId, int] | None = None,
) -> ShardAssignments:
    """Create shard assignments for pipeline parallel execution."""
    world_size = len(cycle)
    use_cfg_parallel = model_card.uses_cfg and world_size >= 2 and world_size % 2 == 0

    if use_cfg_parallel:
        if node_layers is not None:
            raise ValueError(
                "Manual layer allocation is not supported for CFG-parallel models"
            )
        return _get_shard_assignments_for_cfg_parallel(
            model_card, cycle, node_memory, force_override
        )
    else:
        return _get_shard_assignments_for_pure_pipeline(
            model_card, cycle, node_memory, force_override, node_identities, node_layers
        )


def _get_shard_assignments_for_cfg_parallel(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
    force_override: bool = False,
) -> ShardAssignments:
    """Create shard assignments for CFG parallel execution.

    CFG parallel runs two independent pipelines. Group 0 processes the positive
    prompt, group 1 processes the negative prompt. The ring topology places
    group 1's ranks in reverse order so both "last stages" are neighbors for
    efficient CFG exchange.
    """
    _validate_cycle(cycle)

    world_size = len(cycle)
    cfg_world_size = 2
    pipeline_world_size = world_size // cfg_world_size

    # Allocate layers for one pipeline group (both groups run the same layers)
    pipeline_node_ids = cycle.node_ids[:pipeline_world_size]
    pipeline_memory = _compute_total_memory(
        pipeline_node_ids, node_memory, force_override
    )
    layer_allocations = _allocate_and_validate_layers(
        pipeline_node_ids,
        node_memory,
        pipeline_memory,
        model_card,
        force_override,
        include_activation=False,
    )

    # Ring topology: group 0 ascending [0,1,2,...], group 1 descending [...,2,1,0]
    # This places both last stages as neighbors for CFG exchange.
    position_to_cfg_pipeline = [(0, r) for r in range(pipeline_world_size)] + [
        (1, r) for r in reversed(range(pipeline_world_size))
    ]

    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for device_rank, node_id in enumerate(cycle.node_ids):
        cfg_rank, pipeline_rank = position_to_cfg_pipeline[device_rank]
        layers_before = sum(layer_allocations[:pipeline_rank])
        node_layers = layer_allocations[pipeline_rank]

        shard = CfgShardMetadata(
            model_card=model_card,
            device_rank=device_rank,
            world_size=world_size,
            start_layer=layers_before,
            end_layer=layers_before + node_layers,
            n_layers=model_card.n_layers,
            cfg_rank=cfg_rank,
            cfg_world_size=cfg_world_size,
            pipeline_rank=pipeline_rank,
            pipeline_world_size=pipeline_world_size,
        )

        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def _get_shard_assignments_for_pure_pipeline(
    model_card: ModelCard,
    cycle: Cycle,
    node_memory: Mapping[NodeId, MemoryUsage],
    force_override: bool = False,
    node_identities: Mapping[NodeId, NodeIdentity] | None = None,
    node_layers: Mapping[NodeId, int] | None = None,
) -> ShardAssignments:
    """Create shard assignments for pure pipeline execution."""
    _validate_cycle(cycle)
    total_memory = _compute_total_memory(cycle.node_ids, node_memory, force_override)

    layer_allocations = (
        _allocate_and_validate_layers(
            cycle.node_ids,
            node_memory,
            total_memory,
            model_card,
            force_override,
            node_identities,
        )
        if node_layers is None
        else _validate_manual_layer_allocations(
            cycle.node_ids, node_memory, model_card, node_layers
        )
    )

    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for pipeline_rank, node_id in enumerate(cycle.node_ids):
        layers_before = sum(layer_allocations[:pipeline_rank])
        layer_count = layer_allocations[pipeline_rank]

        shard = PipelineShardMetadata(
            model_card=model_card,
            device_rank=pipeline_rank,
            world_size=len(cycle),
            start_layer=layers_before,
            end_layer=layers_before + layer_count,
            n_layers=model_card.n_layers,
        )

        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def get_shard_assignments_for_tensor_parallel(
    model_card: ModelCard,
    cycle: Cycle,
):
    total_layers = model_card.n_layers
    world_size = len(cycle)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for i, node_id in enumerate(cycle):
        shard = TensorShardMetadata(
            model_card=model_card,
            device_rank=i,
            world_size=world_size,
            start_layer=0,
            end_layer=total_layers,
            n_layers=total_layers,
        )

        runner_id = RunnerId()

        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id

    shard_assignments = ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    return shard_assignments


def get_shard_assignments_for_ring_attention(
    model_card: ModelCard,
    cycle: Cycle,
) -> ShardAssignments:
    total_layers = model_card.n_layers
    world_size = len(cycle)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}
    for i, node_id in enumerate(cycle):
        shard = RingShardMetadata(
            model_card=model_card,
            device_rank=i,
            world_size=world_size,
            start_layer=0,
            end_layer=total_layers,
            n_layers=total_layers,
        )
        runner_id = RunnerId()
        runner_to_shard[runner_id] = shard
        node_to_runner[node_id] = runner_id
    return ShardAssignments(
        model_id=model_card.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )


def get_shard_assignments(
    model_card: ModelCard,
    cycle: Cycle,
    sharding: Sharding,
    node_memory: Mapping[NodeId, MemoryUsage],
    force_override: bool = False,
    node_identities: Mapping[NodeId, NodeIdentity] | None = None,
    node_layers: Mapping[NodeId, int] | None = None,
) -> ShardAssignments:
    match sharding:
        case Sharding.Pipeline:
            return get_shard_assignments_for_pipeline_parallel(
                model_card=model_card,
                cycle=cycle,
                node_memory=node_memory,
                force_override=force_override,
                node_identities=node_identities,
                node_layers=node_layers,
            )
        case Sharding.Tensor:
            return get_shard_assignments_for_tensor_parallel(
                model_card=model_card,
                cycle=cycle,
            )
        case Sharding.Ring:
            return get_shard_assignments_for_ring_attention(
                model_card=model_card,
                cycle=cycle,
            )


def get_mlx_jaccl_devices_matrix(
    selected_cycle: list[NodeId],
    cycle_digraph: Topology,
) -> list[list[list[str]]]:
    """Build connectivity matrix mapping device i to device j via RDMA interface names.

    The matrix element [i][j] contains the interface names on device i that
    connect to device j, one per physical link (e.g. one per Thunderbolt
    cable). Diagonal elements are always empty.

    Element k of [i][j] and element k of [j][i] are the two endpoints of the
    same physical link. The jaccl backend pairs queue pairs between ranks by
    index, so both cells must enumerate links in the same order; deriving both
    directions from a single enumeration guarantees this even when more than
    one link connects the same pair of nodes.
    """
    num_nodes = len(selected_cycle)
    matrix: list[list[list[str]]] = [
        [[] for _ in range(num_nodes)] for _ in range(num_nodes)
    ]

    for i, node_i in enumerate(selected_cycle):
        for j in range(i + 1, num_nodes):
            node_j = selected_cycle[j]

            # Each directed edge carries the interface names of both
            # endpoints, so edges from either direction describe the same
            # physical link and can be merged into one set of endpoint pairs.
            links: set[tuple[str, str]] = set()
            for conn in cycle_digraph.get_all_connections_between(node_i, node_j):
                if isinstance(conn, RDMAConnection):
                    links.add((conn.source_rdma_iface, conn.sink_rdma_iface))
            for conn in cycle_digraph.get_all_connections_between(node_j, node_i):
                if isinstance(conn, RDMAConnection):
                    links.add((conn.sink_rdma_iface, conn.source_rdma_iface))

            if not links:
                raise ValueError(
                    "Current jaccl backend requires all-to-all RDMA connections"
                )

            ordered_links = sorted(links)
            matrix[i][j] = [iface_i for iface_i, _ in ordered_links]
            matrix[j][i] = [iface_j for _, iface_j in ordered_links]

    return matrix


def _find_connection_ip(
    node_i: NodeId,
    node_j: NodeId,
    cycle_digraph: Topology,
) -> Generator[SocketConnection, None, None]:
    """Find all socket connections from node i to node j."""
    for connection in cycle_digraph.get_all_connections_between(node_i, node_j):
        if isinstance(connection, SocketConnection):
            yield connection


# Nominal link speeds (Mb/s) used when the OS does not report a negotiated
# speed. Ordering preserves the previous ring preference:
# thunderbolt > maybe_ethernet > ethernet > wifi > unknown.
_NOMINAL_LINK_SPEED_MEGABITS: dict[str, int] = {
    "thunderbolt": 40_000,
    "maybe_ethernet": 10_000,
    "ethernet": 1_000,
    "wifi": 300,
    "unknown": 100,
}


def find_ip_prioritised(
    node_id: NodeId,
    other_node_id: NodeId,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    ring: bool,
) -> str | None:
    """Find an IP address between nodes with prioritization.

    Ring links prefer the fastest interface: lowest measured probe latency
    first, then the negotiated link speed when the node reports one (Linux
    sysfs), otherwise a nominal per-type speed, then RFC1918 LAN as a final
    tiebreak. RDMA coordinators prefer ethernet, then RFC1918 LAN.
    """
    connections = list(_find_connection_ip(node_id, other_node_id, cycle_digraph))
    if not connections:
        return None

    latency_by_ip: dict[str, float] = {}
    for connection in connections:
        ip_address = connection.sink_multiaddr.ip_address
        if connection.latency_ms is None:
            continue
        known = latency_by_ip.get(ip_address)
        if known is None or connection.latency_ms < known:
            latency_by_ip[ip_address] = connection.latency_ms

    other_network = node_network.get(other_node_id, NodeNetworkInfo())
    ip_to_interface = {iface.ip_address: iface for iface in other_network.interfaces}

    if ring:
        def effective_link_speed_megabits(ip: str) -> int:
            interface = ip_to_interface.get(ip)
            if interface is None:
                return _NOMINAL_LINK_SPEED_MEGABITS["unknown"]
            if interface.link_speed_megabits is not None:
                return interface.link_speed_megabits
            return _NOMINAL_LINK_SPEED_MEGABITS.get(
                interface.interface_type, _NOMINAL_LINK_SPEED_MEGABITS["unknown"]
            )

        return max(
            {connection.sink_multiaddr.ip_address for connection in connections},
            key=lambda ip: (
                -latency_by_ip.get(ip, float("inf")),
                effective_link_speed_megabits(ip),
                -_address_priority(ip),
            ),
        )

    # RDMA prefers ethernet coordinator
    type_priority = {
        "ethernet": 0,
        "maybe_ethernet": 1,
        "wifi": 2,
        "unknown": 3,
        "thunderbolt": 4,
    }

    def interface_type_for_ip(ip: str) -> str:
        interface = ip_to_interface.get(ip)
        return interface.interface_type if interface is not None else "unknown"

    return min(
        {connection.sink_multiaddr.ip_address for connection in connections},
        key=lambda ip: (
            type_priority.get(interface_type_for_ip(ip), 5),
            _address_priority(ip),
        ),
    )


def _address_priority(ip: str) -> int:
    """RFC1918 LAN addresses are preferred; everything else (CGNAT/Tailscale,
    link-local, public) is treated identically and only ranked by interface
    type."""
    if ip.startswith(("192.168.", "10.")):
        return 0
    if ip.startswith("172."):
        try:
            second_octet = int(ip.split(".")[1])
        except (IndexError, ValueError):
            return 1
        if 16 <= second_octet <= 31:
            return 0
    return 1


def get_mlx_ring_hosts_by_node(
    selected_cycle: Cycle,
    cycle_digraph: Topology,
    ephemeral_port: int,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> dict[NodeId, list[Host]]:
    """Generate per-node host lists for MLX ring backend.

    Each node gets a list where:
    - Self position: Host(ip="0.0.0.0", port=ephemeral_port)
    - Left/right neighbors: actual connection IPs
    - Non-neighbors: Host(ip="198.51.100.1", port=0) placeholder (RFC 5737 TEST-NET-2)
    """
    world_size = len(selected_cycle)
    if world_size == 0:
        return {}

    hosts_by_node: dict[NodeId, list[Host]] = {}

    for rank, node_id in enumerate(selected_cycle):
        left_rank = (rank - 1) % world_size
        right_rank = (rank + 1) % world_size

        hosts_for_node: list[Host] = []

        for idx, other_node_id in enumerate(selected_cycle):
            if idx == rank:
                hosts_for_node.append(Host(ip="0.0.0.0", port=ephemeral_port))
                continue

            if idx not in {left_rank, right_rank}:
                # Placeholder IP from RFC 5737 TEST-NET-2
                hosts_for_node.append(Host(ip="198.51.100.1", port=0))
                continue

            connection_ip = find_ip_prioritised(
                node_id, other_node_id, cycle_digraph, node_network, ring=True
            )
            if connection_ip is None:
                raise ValueError(
                    "MLX ring backend requires connectivity between neighbouring nodes"
                )

            hosts_for_node.append(Host(ip=connection_ip, port=ephemeral_port))

        hosts_by_node[node_id] = hosts_for_node

    return hosts_by_node


def get_mlx_jaccl_coordinators(
    coordinator: NodeId,
    coordinator_port: int,
    cycle_digraph: Topology,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> dict[NodeId, str]:
    """Get the coordinator addresses for MLX JACCL (rank 0 device).

    Select an IP address that each node can reach for the rank 0 node. Returns
    address in format "X.X.X.X:PORT" per node.
    """
    logger.debug(f"Selecting coordinator: {coordinator}")

    def get_ip_for_node(n: NodeId) -> str:
        if n == coordinator:
            return "0.0.0.0"

        ip = find_ip_prioritised(
            n, coordinator, cycle_digraph, node_network, ring=False
        )
        if ip is not None:
            return ip

        raise ValueError(
            "Current jaccl backend requires all participating devices to be able to communicate"
        )

    return {
        n: f"{get_ip_for_node(n)}:{coordinator_port}"
        for n in cycle_digraph.list_nodes()
    }


def assign_shard_backends(
    shard_assignments: ShardAssignments,
    node_backends: Mapping[NodeId, list[Backend]],
    preferred_backends: Sequence[Backend],
) -> ShardAssignments:
    """Pick, for every runner, the first preferred backend its node supports.

    Runners on nodes with no matching backend keep `backend=None`, which
    preserves the engine default device selection.
    """
    runner_to_shard = dict(shard_assignments.runner_to_shard)
    for node_id, runner_id in shard_assignments.node_to_runner.items():
        shard = runner_to_shard.get(runner_id)
        if shard is None:
            continue
        node_supported = set(node_backends.get(node_id, []))
        backend = next(
            (
                candidate
                for candidate in preferred_backends
                if candidate in node_supported
            ),
            None,
        )
        if backend is not None:
            runner_to_shard[runner_id] = shard.model_copy(update={"backend": backend})
    return shard_assignments.model_copy(update={"runner_to_shard": runner_to_shard})
