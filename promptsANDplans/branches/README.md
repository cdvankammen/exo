# Branches Log

> **Purpose:** Every branch in the repo, what it's for, what's on it, and its status. Prevents "which branch has this?" confusion.

## Historical records (dated)
- `2026-08-06.md` — Wave 4/5 updates (now 65 commits, #2129 port commit, f48bdf86 torch fix)
- `2026-08-07.md` — Wave 5-7 (78 commits: #2229 4-part, error-UX, Linux UX, Settings page, message queue)

---

## Current Branch: `fix-memory-error` (ACTIVE)

- **Status:** 78 commits ahead of main (snapshot 08-07)
- **Purpose:** All PR ports + session work
- **Composition:** 56+ ported PRs + Wave 4-7 session commits (#2129, #2229×4, error-UX×6, Linux UX, `db3736e2` Settings, `9ea20961` message queue)
- **Latest:** `9ea20961` (message queue T24)

### Session commits on this branch (2026-08-06)
| Commit | Author | What |
|---|---|---|
| `ca8474c7` | this chat | inline EJECT split-button |
| `72cf86bb` | other agent | port of PR #2129 (heterogeneous Metal+CUDA) |
| `d6fcef17` | other agent | loaded models at top of picker |
| `4f850371` | other agent | keep failed instances visible |
| `85f1b024` | other agent | menu bar startup errors |
| `5acd6d6c` | other agent | machine-readable error codes |
| `3d1a675c` | this chat | eject no longer auto-deletes weights |

## Empty Bookmark Branches (created then abandoned)

| Branch | Created for | Commits on it | Status |
|---|---|---|---|
| `fix-memory-error/track-a-asymmetric-tp` | #2063 + #2064 | **0** | Empty pointer; work went to main (868088f7) |
| `fix-memory-error/track-b-model-discovery` | #1598 | **0** | Empty pointer; work went to main (157a4c24) |

**Why abandoned:** User pivoted from per-PR branches to single-branch merging. The branches were never checked out or committed to. Safe to delete.

## Other branches (upstream refs, not ours)
- `main` — upstream tracking
- `remotes/origin/*` — upstream PR branches (JakeHillion/*, alexcheema/*, etc.)

---

## Branch history: how `fix-memory-error` was built

```
6b8d3a78  ← WIP memory problem (pre-campaign)
a6b99564  ← WIP memory/download override
014a23ee  ← baseline (connect_peer groundwork)
│
├── Wave 1: 7dd12e9f ... 086787dd  (23 ports, 08-05)
├── Wave 2: 868088f7 ... 165b614e  (21 ports, 08-05/06)
├── Wave 3: 3abc8623 ... 91173df0  (8 ports, 08-06)
├── Session: ca8474c7 (eject UI), 3d1a675c (no auto-delete)
└── Other agent: 72cf86bb, d6fcef17, 4f850371, 85f1b024, 5acd6d6c
```
