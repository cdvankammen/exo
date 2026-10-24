# Work Queue Chart — 2026-08-07

> Purpose: one chart showing every open item, whether it needs user input, effort, and the recommended order for autonomous work.
> Branch state: 78 commits ahead of main (`9ea20961` = T24 message queue landed). EXO.apps are 2 commits stale (Settings page + queue UI).

## 📋 Full queue

| ID | Item | What it is | Effort | Needs your input? | Recommended |
|---|---|---|---|---|---|
| T17 | Rebuild EXO.app (2 commits stale) | Routine rebuild + deploy so both Macs have Settings page + message queue | Small | No | 🔴 Do now |
| T29 | 2-node RDMA retest | Mac↔Mac `MlxJaccl min_nodes=2` with fresh bundles; if segfaults → file upstream at rltakashige | Small (ops) | No | 🔴 Do now |
| T25 | Queue reorder/priority | Drag or ↑↓ reorder of queued messages | Small | No | 🟢 Next |
| T27 | Sampling controls panel | temp/seed/top_p/stop/logit_bias UI | Small–Med | No | 🟢 Next |
| T4 | #2200 backend-per-shard | Deterministic backend selection; foundation for other work | Medium | No | 🟢 Next |
| T19 | Settings manager env-var editor | Editable ~50 `EXO_*` vars UI (Settings page is display-only today) | Medium | No | 🟡 Soon |
| T26 | Parallel drain | Fire N requests at once (backend already batches 8) | Medium | No | 🟡 Soon |
| T21 | EXO_BOOTSTRAP_PEERS fix | Cross-subnet cluster bootstrap | Medium | No | 🟡 Soon |
| T22 | Stable node identity | Keypair persistence so nodes keep IDs across restarts | Medium | No | 🟡 Soon |
| T8 | #2255 Ring Attention | 65K+ token contexts | Large | No (big) | 🟠 Big |
| T23 | API auth layer | Auth for API endpoints | Medium–Large | ⚠️ Approach choice | 🟠 Big |
| T28 | Guided/constrained decoding | Generator sampling hook | Large | No (big) | 🟠 Big |
| T6 | #2064 asymmetric TP | >200B models on mixed memory (Qwen3.5 only) | Large | ✅ Yes | ⏸ Waiting |
| T9 | #1860 RotatingKVCache | KV cache rotation | Small–Med | ✅ Yes (defaults) | ⏸ Waiting |
| T10 | PromoteMaster command (#2246) | Manual master promotion | Small | ✅ Yes | ⏸ Waiting |
| T11 | `--no-master` flag (#2135) | Run node without master role | Small | ✅ Yes | ⏸ Waiting |
| T12 | API host binding (#2190) | Bind API to configurable host | Small | ✅ Yes | ⏸ Waiting |
| T13 | Disable server-side tools (#1864) | Security toggle | Small | ✅ Yes | ⏸ Waiting |
| T14 | Non-MLX models (GGUF/PyTorch) | New subsystem, no PR exists | Large | ✅ Yes | ⏸ Waiting |
| T18 | Branch cleanup (track-a/b) | Delete empty branches | Trivial | ✅ Yes (confirm) | ⏸ Waiting |
| T15/T16 | Recurring maintenance | VS Code patch re-apply; NVIDIA userspace after kernel upgrade | — | — | 🔁 When needed |
| — | Housekeeping | Delete stray `default.profraw` at repo root | Trivial | No | 🧹 Do now |

## 🎯 Recommended autonomous order (no input needed)

1. **🔴 T17 rebuild** — both Macs get the Settings page + queue (2 quick wins ship)
2. **🧹 delete `default.profraw`** — stray artifact
3. **🔴 T29 RDMA retest** — 15 min; resolves the open upstream question
4. **🟢 T25 + T27** — small UX wins, research doc already written
5. **🟢 T4 backend-per-shard** — foundation that de-risks T6/T8 later
6. **🟡 T19 env-var editor** — high usability value, design doc exists
7. **🟡 T26 parallel drain / T21 bootstrap / T22 node identity** — medium, independent
8. **🟠 Large items (T8, T28, T23)** — only after small/medium queue cleared

## ⏸ Waiting on user (do not start)

T6 (asymmetric TP), T9 (KVCache defaults), T10–T14 (Tier-4), T18 (branch deletion confirm)
