# RDMA / Multi-Node Investigation — "Ghost Edge" Follow-up

> **Purpose:** Full, evidence-based account of the multi-node RDMA (JACCL) failures seen on
> 2026-08-06, including a **correction** of an earlier hypothesis, the code-level truth,
> and what (if anything) needs fixing.
> **Date:** 2026-08-06 | **Branch:** `fix-memory-error`

---

## 1. Correction of the record ⚠️

Earlier in the session I stated there was an "RDMA ghost edge" between the Macs and the
Linux box, claiming the master built a JACCL ring over a fake Mac↔Linux RDMA link. That
was **wrong**. The evidence:

1. The failing 2-node instance's `jaccl_coordinators` were:
   ```
   {'3ad22f85...': '0.0.0.0:63256',     ← mini2 (rank 0)
    '9faabaa5...': '10.2.0.11:63256'}   ← this Mac (rank 1, connecting to mini2's LAN IP)
   ```
   **Both endpoints were Macs** (10.2.0.11 = mini2's LAN IP per SSH config). The Linux
   box was never in a 2-node instance.
2. The code (`apply.py`, `MacThunderboltConnections` case) **cannot create an RDMA edge
   to a Linux node** — both endpoints must have reported `MacThunderboltIdentifiers`
   (macOS-only `system_profiler SPThunderboltDataType`). Linux reports none, so it can
   never appear as an RDMA sink. **A Mac↔Linux RDMA edge is structurally impossible.**

## 2. What actually happened (timeline, all times local)

| Time | Event |
|---|---|
| 18:03 | Qwen3.5 placed 2-node **Mac↔Mac** (mini2 + this Mac) — model loaded, chat worked on the Mac |
| 18:04:46 | Model files **deleted** ("Deleting model files ... Successfully deleted model") — while the instance was still running |
| 18:05:01 | Runner segfaults (`Fatal Python error: Segmentation fault`, `signal=11`) during `mlx.distributed.init(backend="jaccl")` |
| 18:09–18:10 | Retry: master creates another 2-node **Mac↔Mac** `MlxJacclInstance` → runner segfaults again during jaccl init |
| 18:14 | Single-node placement succeeds (model now only on this Mac) — chat works at 17.6 tok/s |
| 19:14 | mini2 starts **flapping** out of topology ("node ... is no longer in topology" ×4) |
| 19:36 | Explicit `min_nodes=2, MlxJaccl` placement → **refused** with `ValueError: No cycle where every node supports ...` — the guard did its job |

## 3. Why the Mac↔Mac jaccl segfault happened (likely)

Three compounding factors, in order of likelihood:

1. **Stale app bundles.** Both Macs were running EXO.app builds from 15:16 (before our
   Wave-4 commits: #2129 TcpRelay, #2254 latency selection, etc.). The running bundle's
   mlx is the fork's **darwin git branch** (`address-rdma-gpu-locks`) — a custom build.
   The segfault happened inside `mx.distributed.init(backend="jaccl")`, which is
   **mlx-level code we don't control**.
2. **Model deleted mid-run** (18:04:46). The instance was live when its weights vanished;
   the runner's jaccl re-init had nothing to work with and crashed.
3. **mini2 instability** — the node kept leaving topology (19:14+), consistent with the
   crash-looping stale bundle seen earlier in the session (the `event_id` issue).

**Conclusion: not proven to be a code bug in exo.** The correct next step is to rebuild
both Mac apps from the current branch and re-test 2-node jaccl. The guard refused the
recent attempt cleanly, which is the behavior we want.

> **UPDATE (08-07 12:31-12:40, see `tasks/2026-08-07-wave5-t1-t3.md`):** The retest
> WAS DONE on fresh 76-commit builds — 2-node `MlxJaccl` STILL segfaults (signal 11 in
> `mx.distributed.init(backend="jaccl")`, rdma_ctl enabled + TB 40Gb/s on both Macs),
> while 2-node `MlxRing` works ("Hello, how are you?" across both Macs). **Conclusion
> updated: this IS an upstream bug in the rltakashige mlx darwin fork** (mlx-internal,
> not our code). Workaround = MlxRing (default). Action: file upstream issue (draft
> ready, awaiting user go — decision D36).

## 4. The guard that protects us — verified working

`src/exo/master/placement.py`:

```
smallest_rdma_cycles = [
    cycle for cycle in smallest_cycles
    if topology.is_rdma_cycle(cycle)          # ← edges in the cycle are real RDMA
    and _all_rdma_ctl_enabled(cycle)          # ← EVERY node reports rdma_ctl enabled
]
...
if command.instance_meta == InstanceMeta.MlxJaccl:
    if not smallest_rdma_cycles:
        raise ValueError("Requested RDMA (MlxJaccl) but no RDMA-connected cycles available")
```

- `is_rdma_cycle` requires the topology to contain `RDMAConnection` edges between the
  cycle's nodes — and per §1 those edges only form between two macOS Thunderbolt-reporting
  nodes with both `rdma_ctl` enabled (`apply.py` checks `source_is_rdma_enabled AND
  _is_rdma_ctl_enabled(sink)`).
- `_all_rdma_ctl_enabled` requires every node's `NodeRdmaCtlStatus.enabled == True`.
  On Linux `RdmaCtlStatus.gather()` returns `None` (non-Darwin) — so a Linux node in a
  cycle makes this False. Belt and suspenders.
- **Verified live:** the `min_nodes=2, MlxJaccl` request at 19:36 was refused with a
  clean `ValueError` instead of a segfault.

## 5. So — does anything need fixing?

| Question | Answer |
|---|---|
| Is there a Mac↔Linux ghost edge? | **No — structurally impossible** (both endpoints must be macOS TB-reporting nodes). |
| Does the Linux box need a guard fix? | **No.** It can only ever be in socket (RING) or single-node instances. |
| Does the guard work? | **Yes** — verified live (clean refusal instead of segfault). |
| Is Mac↔Mac jaccl broken? | **✅ CONFIRMED BROKEN (08-07)** — retest on fresh 76-commit builds: segfault signal 11 in jaccl init on BOTH Macs despite rdma_ctl enabled + TB 40Gb/s. **Upstream mlx darwin fork bug** (not our code). Workaround: MlxRing works. Action: file upstream (see UPDATE note §3). |
| Mini2 flapping? | Separate issue — the app on mini2 was the stale bundle. Rebuild both Mac apps from `fix-memory-error` (with all Wave-4 commits) and re-observe. |

**Recommended follow-up (not a code change):**
1. ✅ DONE (08-07): Rebuild + reinstall EXO.app on this Mac and mini2 from current source.
2. ✅ DONE (08-07): Re-test `place_instance { model, sharding: Pipeline, instance_meta: MlxJaccl, min_nodes: 2 }` — segfault confirmed.
3. ✅ DONE (2026-09-09, user authorized via kanban card t_867d449d = D36 go):
   filed upstream as **ml-explore/mlx#4485** (the rltakashige fork has issues
   disabled; parent repo is where JACCL bugs live). `mx.distributed.init(backend="jaccl")`
   is mlx-internal. Evidence in `tasks/2026-08-07-wave5-t1-t3.md`.

## 6. For multi-node with the Linux box (what IS supported)

```
 Mac (Metal)  ◄──socket (RING)──►  Linux (CUDA)
     │                                  │
  pipeline shard 0                  pipeline shard 1
```
- Use `instance_meta: MlxRing` + `min_nodes: 2` — plain TCP sockets, no RDMA needed.
- This is exactly what our #2129 port (commit `72cf86bb`) enables: `TcpRelay` for any
  CUDA↔CUDA hops + `_rotate_metal_to_middle` for 3+ node cycles.
- Expect slower inter-node transfers than Mac↔Mac TB4 (the Linux link is the virtio
  bridge, ~1G/10G depending on host), but it works.
