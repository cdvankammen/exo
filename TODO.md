1. ~~EXO_BOOTSTRAP_PEERS is currently broken~~ → **FIXED (T21)** — parsed from env in `src/exo/main.py` (`--bootstrap-peers`/`EXO_BOOTSTRAP_PEERS`); `EXO_ZENOH_CONNECT` unicast dial (`d300410f`, PR #2243) covers cross-subnet without multicast

4. ~~I'd like to see profiled network latency / bandwidth.~~ → **DONE** — topology links show measured latency (reachability probes) + bandwidth (HTTP probe): `graph-link-latency` labels (`234c322a`).
5. ~~I'd like to see how much bandwidth each link is using.~~ → **DONE** — per-link `bandwidth_mbps` shown on topology edges (`234c322a`).
7. ~~In continuous batching, a new prompt blocks decode of the current batch until prefill completes~~ → **DONE (`24ec252c`)** — `batch_generate.py` defers prefill to the chunked decode-first path when possible (`can_defer_prefill()` guard + 6 tests in `test_deferred_prefill.py`)
8. ~~Offline model copy detection~~ → **DONE** — non-empty local model dir with no .partial files detected as fully-downloaded (`download_utils.py:429`).
13. ~~Memory pressure instead of memory used~~ → **DONE** — `profiling.py` computes `pressure` (used/total) on RAM + VRAM, surfaced in node cards.
14. ~~Show the type of each connection (TB5, Ethernet, etc.) in the UI~~ → **DONE** — `interface_type` + `link_speed_megabits` in `NodeNetworkInfo`; TB4/TB5 distinction via link speed.
16. Dynamically switch to higher priority connection when it becomes available. Probably bring back InstanceReplacedAtomically.
17. Faster model loads by streaming model from other devices in cluster.
18. ~~Add support for specifying the type of network connection to use in a test~~ → **DONE** — `@pytest.mark.cluster(..., thunderbolt='a2a')` in `tests/framework.py`, used by test_2node/resilience.
27. ~~Log cleanup - per-module log filters and default to DEBUG log levels~~ → **DONE** — per-module filters via `EXO_LOG_LEVELS` (`91d60a01`) + rotation for `.exo/exo.log` (`009b43c6`)
28. ~~Validate RDMA connections with ibv_devinfo in the info gatherer~~ → **DONE (`383ac419`)** — `_gather_has_verbs_device()` runs `ibv_devices` in `utils/info_gatherer/info_gatherer.py`; placement/apply gate on `has_verbs_device` (`test_rdma_verbs_validation.py`)
