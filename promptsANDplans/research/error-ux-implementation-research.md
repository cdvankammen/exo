# Error UX — Implementation Research (Making Error Details Available)

**Date:** 2026-08-06 (part 2 — implementation-focused)
**Branch:** fix-memory-error
**Scope:** Research only. Covers (A) how to make rich error data available at each layer, and (B) item #1 — the `publish_bytes` guard.

---

## A. Making Error Details Available — Concrete Mechanisms

The backend ALREADY produces rich errors. The work is **surfacing** them. Three layers need changes, each with a clear mechanism:

### A1. Dashboard: render runner `error_message` + diagnostics on FAILED cards

**Data availability (already proven):**
- `dashboard/src/routes/+page.svelte` has `runnersData` (map of runner_id → tagged status)
- `deriveInstanceStatus()` (line ~1960) already iterates runner ids and reads `runnersData[rid]`
- The `RunnerFailed` payload contains `error_message: str | None` and `diagnostics: KnownRunnerDiagnostic[]`

**Current behavior:** `RunnerFailed → "Failed"` (line 2003) — the payload is **read but discarded**; only the label "FAILED" is shown.

**Proposed change (pattern already exists — copy the downloads card):**
- Downloads already render errors (line 5713-5717):
  ```svelte
  {#if downloadInfo.isFailed && downloadInfo.errorMessage}
    <div class="text-xs text-red-400/80 font-mono mt-1 break-words">
      {downloadInfo.errorMessage}
    </div>
  {/if}
  ```
- Extend `deriveInstanceStatus` to return `errorMessage` (join all runner `error_message`s) + `diagnostics`
- Render them in the instance card's FAILED branch, same red-mono style
- Optionally: a small expandable "details" toggle for the diagnostics evidence lines

**No backend change needed — the data is already in the state the dashboard fetches.**

### A2. App shell (Swift): stop discarding stderr, show a reason

**Current behavior:**
- `ExoProcessController.swift:188-189` — `standardOutput = FileHandle.nullDevice`, `standardError = FileHandle.nullDevice` → **stderr discarded**
- Line 202-204 — `status = .failed(message: "Exited with code \(proc.terminationStatus)")` → bare code
- `ContentView.swift:385` — shows `lastError` string in the UI

**Proposed change:**
1. Capture stderr into a ring buffer (e.g. `Pipe()` + read into last N lines) OR read the exo log file tail. exo already writes a log — `~/.exo/exo_log/exo.log` (`exoDirectoryURL` = `~/.exo` is already known at ExoProcessController.swift:54).
2. In the `terminationHandler`, instead of "Exited with code 1":
   - Read last ~10 lines of `~/.exo/exo_log/exo.log`
   - Grep for known signatures: `Address already in use` (port conflict), `ModuleNotFoundError` (missing dep), `ValidationError`/`extra_forbidden` (schema mismatch)
   - Show: `"Exited with code 1 — port 52415 already in use (another exo instance is running)"` etc.
3. Keep the raw code + log path in `lastError` for power users.

### A3. API: stable error codes instead of HTTP status phrases

**Current:** `ErrorInfo.type = HTTPStatus(status_code).phrase` (main.py:361) — e.g. "Not Found". Message strings are informative but no stable machine code.

**Proposed:** add an optional `code: str` (stable string enum) to `ErrorInfo` (keep `code: int` as HTTP status or rename carefully — it's used by clients). Suggested codes:
- `INSUFFICIENT_MEMORY` (memory gate, main.py:493)
- `MODEL_CARD_NOT_FOUND` (main.py:565)
- `PLACEMENT_FAILED` (main.py:609,630)
- `INSTANCE_NOT_FOUND` (main.py:678,725)
- `PEER_UNREACHABLE` (main.py:1982)
- `RUNNER_CRASHED` (streaming ErrorChunk, main.py:1197)

Mechanism: introduce `ExoError(Exception)` with `code` field, or a small helper `raise_api(code, status, message)`. Then `http_exception_handler` passes `code` through. The dashboard can map `code → hint text` (e.g. show "Try loading anyway" for INSUFFICIENT_MEMORY).

---

## B. Item #1 — `publish_bytes` Guard (crash → warning)

### The crash path (verified, exact lines)

```
src/exo/routing/router.py:197  _networking_recv()
  ├─ line 211:  await router.publish_bytes(data)      ← per-message
  │    └─ line 86-87:  publish_bytes
  │         └─ self.topic.deserialize(data)            ← THROWS on bad data
  │              └─ topics.py:36: model_validate_json  ← pydantic ValidationError / UnicodeDecodeError
  └─ line 219-228: except Exception → logs → **re-raises**   ← kills the loop
       └─ task group dies → Router.run() exits → process exits 1
```

**Key insight:** `_networking_recv` has a broad try/except, but it **re-raises** (line 228: `raise`), so the *entire receive loop* — and therefore the whole process — dies on ONE malformed message. This is exactly what the `event_id` schema mismatch did (stale bundle rejected valid new-format events with `extra_forbidden`).

### The fix design (2 guards, minimal surface)

**Guard 1 — per-message, in `_networking_recv` (the critical one):**
```python
case FromSwarm.Message(topic, data):
    if topic not in self.topic_routers:
        logger.warning(f"Received message on unknown or inactive topic {topic}")
        continue
    router = self.topic_routers[topic]
    try:
        await router.publish_bytes(data)
    except Exception as exception:
        # NOTE: A single malformed message must never kill the gossip
        # receive loop (this previously caused exit-code-1 crashes when a
        # peer sent an event the local schema rejected).
        logger.warning(
            f"Dropping malformed message on {topic}: {exception}"
        )
```
This keeps the loop alive — the `except` at the bottom of `_networking_recv` remains only for genuinely fatal loop failures.

**Guard 2 — in `publish_bytes` (defense in depth, for direct callers):**
```python
async def publish_bytes(self, data: bytes):
    try:
        await self.publish(self.topic.deserialize(data))
    except Exception as exception:
        logger.warning(
            f"Dropping malformed {self.topic.topic} event: {exception}"
        )
```

**Exceptions the guard must catch (from `deserialize`):**
- `pydantic.ValidationError` (schema mismatch — the `event_id` case)
- `UnicodeDecodeError` (garbage bytes, `b.decode("utf-8")` in topics.py:36)
- `json.JSONDecodeError` (bad JSON payload)

**What NOT to catch:** `ClosedResourceError`/`BrokenResourceError` during shutdown — those are already handled by `publish`'s sender cleanup (router.py:78-83) and should propagate.

### Verification plan
1. Unit test: feed `b"not json"` to `publish_bytes` → expect `logger.warning`, no exception
2. Unit test: feed a valid event with wrong schema (e.g. missing required field) → same
3. Integration: run node A (new schema) + node B (old schema, or monkeypatch deserialize to raise) → B logs warnings, **keeps running**, cluster stays up

### Why this is low-risk
- Pure additive try/except; no behavior change for valid messages
- Matches the pattern already used for unknown topics (warning + continue)
- The warning is logged to the exo log, which fix A2 can surface in the UI

---

## C. Suggested Implementation Order (revised with mechanism)

| # | Change | Layer | Mechanism | Risk |
|---|--------|-------|-----------|------|
| 1 | `publish_bytes` guard | router.py | try/except per message (Guard 1 + 2) | Low |
| 2 | Dashboard runner errors | +page.svelte | render `error_message`/diagnostics in FAILED card (copy downloads pattern) | Low |
| 3 | Stable API error codes | api/main.py + types | `ExoError(code)` helper → ErrorInfo.code | Medium (client compat) |
| 4 | App shell exit reason | ExoProcessController.swift | read log tail on termination, map signatures → human reason | Medium (Swift) |

Items 1-3 require **no Swift changes**. Item 4 is independent of the rest.

---

## D. Key File Reference (for implementation)

| File:Line | Purpose |
|-----------|---------|
| `src/exo/routing/router.py:86` | `publish_bytes` — add Guard 2 |
| `src/exo/routing/router.py:211` | `publish_bytes` call site — add Guard 1 |
| `src/exo/routing/router.py:228` | re-raise that kills the loop |
| `src/exo/routing/topics.py:36` | `deserialize` → `model_validate_json` (throw point) |
| `dashboard/src/routes/+page.svelte:1960-2003` | `deriveInstanceStatus` — add errorMessage to return |
| `dashboard/src/routes/+page.svelte:5713` | downloads error render (pattern to copy) |
| `app/EXO/EXO/ExoProcessController.swift:188-204` | stderr discard + bare exit code |
| `src/exo/api/main.py:353-366` | http_exception_handler (code pass-through) |
| `src/exo/api/types/api.py:21-30` | ErrorInfo model (add `code: str`) |
