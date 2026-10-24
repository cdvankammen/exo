# Error UX Research — Making Error Codes Understandable

**Date:** 2026-08-06
**Branch:** fix-memory-error
**Scope:** Research only (no code changes) — how errors flow through exo and where information is lost, so we can make failures explain themselves instead of "failing with no information."

---

## 1. The Full Error Pipeline (end to end)

```
Python exception
   │
   ├─ API layer (src/exo/api/main.py)
   │    └─ HTTPException → ErrorResponse{ error: ErrorInfo{ message, type, code } }
   │
   ├─ Runner layer (src/exo/worker/)
   │    └─ RunnerTerminationError (bootstrap.py:22) → RunnerFailed (supervisor.py:438)
   │         └─ error_message + diagnostics[] (RunnerMetalGpuTimeout, RunnerRingTransportError, ...)
   │
   ├─ Routing layer (src/exo/routing/router.py)
   │    └─ publish_bytes (line 86) — NO try/except → unhandled → process crash (exit 1)
   │
   └─ App shell (app/EXO Swift)
        └─ ExoProcessController terminationHandler → "Exited with code 1" (no detail)
```

---

## 2. What Exists Already (good news — a solid foundation)

### 2a. API: structured error model ✅
`src/exo/api/types/api.py:21-30`
```python
class ErrorInfo(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: int

class ErrorResponse(BaseModel):
    error: ErrorInfo
```
- `src/exo/api/main.py:353-366` — global `http_exception_handler` wraps every `HTTPException` into this shape.
- `type` = HTTP status phrase (e.g. "Not Found", "Bad Request"), `code` = status code.

### 2b. Runner: structured diagnostics ✅
`src/exo/worker/runner/diagnostics.py` — has a real diagnostics classifier:
- `RunnerMetalGpuTimeout` (regex-matches `[METAL] GPU Timeout`)
- `RunnerRingTransportError` / `RunnerRingSocketReceivingError` (ring socket errno)
- `RunnerUnknown` fallback with 4-line stderr evidence

`src/exo/worker/runner/bootstrap.py:22-40` — `RunnerTerminationError` carries:
```python
exception_type, exception_message, exception_repr, traceback  # full chain!
```

`src/exo/worker/runner/supervisor.py:376-450` — `_check_runner` builds a **rich cause**:
```
cause = "signal=11 (Segmentation fault)"   # for negative rc
      or "exitcode=1"                       # for positive rc
```
→ `RunnerFailed(error_message=f"Terminated ({cause})", diagnostics=[...])`

### 2c. Chat streaming: ErrorChunk ✅
`src/exo/api/main.py:1197` — when a runner dies mid-generation, streams an SSE `ErrorChunk` with `error_message`, converted to a proper `ErrorResponse` (500) in the stream.

---

## 3. Where Information Is Lost (the gaps)

### 🔴 GAP 1: `publish_bytes` — unhandled → process crash (exit 1)
`src/exo/routing/router.py:86-87`
```python
async def publish_bytes(self, data: bytes):
    await self.publish(self.topic.deserialize(data))   # ← no try/except
```
**Problem:** Any malformed/desync event (like the `event_id` schema mismatch that caused the earlier exit-1 crash) raises here → propagates up → **kills the whole process** with no user-facing message beyond "Exited with code 1".
**Fix idea (deferred):** wrap in try/except → `logger.warning("dropping malformed event: ...")`.

### 🔴 GAP 2: App shell shows only "Exited with code N"
`app/EXO/EXO/ExoProcessController.swift:202-204`
```swift
case .failed: break
default:
    self.status = .failed(message: "Exited with code \(proc.terminationStatus)")
    self.lastError = "Process exited with code \(proc.terminationStatus)"
```
**Problem:** The menu bar / ContentView (`ContentView.swift:385`) shows `lastError` = "Process exited with code 1". No hint of *why* (port conflict vs. schema mismatch vs. missing module). The user sees a bare exit code.
**Note:** `standardError = FileHandle.nullDevice` — stderr is discarded! The diagnostic trail is thrown away unless it's in the exo log file.

### 🟡 GAP 3: Dashboard FAILED status shows no error detail
`dashboard/src/routes/+page.svelte:2003` — `RunnerFailed: "Failed"` just maps to status text "FAILED". The instance card shows the status but **not** the `error_message` / diagnostics (unlike downloads, which DO render `downloadInfo.errorMessage` at line 5713-5717).

### 🟡 GAP 4: Placement errors are bare
`src/exo/api/main.py:609,630` — placement preview errors are `str(exc)` / generic strings ("Expected exactly one new instance from placement"). No error *code* — just a message. The dashboard shows `onboardingError = "Failed to get placement: ${errorText}"` (`+page.svelte:726`) — decent, but unclassified.

### 🟡 GAP 5: HTTP status phrase as `type`
`main.py:361` — `type=HTTPStatus(exc.status_code).phrase` gives "Not Found", "Bad Request". Fine for clients, but not a stable machine-readable error code (e.g. `INSUFFICIENT_MEMORY`). The memory gate message (`main.py:493-495`) is informative (required vs available GB) but has no code.

### 🟢 GAP 6: Peer connect errors
`main.py:1982` — `error=f"Peer not reachable on API port {payload.api_port}: {exc}"` — actually decent context already.

---

## 4. Concrete Findings — What Could Be Improved

| # | Layer | Current | Would be better |
|---|-------|---------|-----------------|
| 1 | router.py `publish_bytes` | Process crash, exit 1 | try/except → log warning, keep running |
| 2 | ExoProcessController | "Exited with code 1" | Capture exo log tail + stderr; show "exited: <reason>" |
| 3 | Dashboard instance card | "FAILED" only | Show `error_message` + diagnostics (like downloads do) |
| 4 | API error `type` | HTTP phrase | Add stable error codes (`INSUFFICIENT_MEMORY`, `MODEL_NOT_FOUND`, `PEER_UNREACHABLE`, `PLACEMENT_FAILED`) |
| 5 | RunnerFailed | good (structured) | Already good — surface it in UI |

---

## 5. Suggested Implementation Order (for later)

1. **`publish_bytes` guard** (smallest, prevents crashes — already user-deferred)
2. **API stable error codes** — add a `code` enum to `ErrorInfo`, map HTTPExceptions
3. **Dashboard: render runner `error_message` + diagnostics** on FAILED cards
4. **App shell: capture stderr + show log tail** on exit (replace "Exited with code N" with a reason)
5. **Placement error classification** — wrap `ValueError` with codes

---

## 6. Key File Map

| File | What it does |
|------|--------------|
| `src/exo/api/types/api.py:21` | `ErrorInfo`/`ErrorResponse` models |
| `src/exo/api/main.py:353` | `http_exception_handler` |
| `src/exo/api/main.py:490-496` | memory gate (informative message, no code) |
| `src/exo/api/main.py:1197` | streaming `ErrorChunk` → SSE error |
| `src/exo/worker/runner/bootstrap.py:22` | `RunnerTerminationError` (type/message/repr/traceback) |
| `src/exo/worker/runner/supervisor.py:376-450` | `_check_runner` → rich cause + `RunnerFailed` |
| `src/exo/worker/runner/diagnostics.py` | stderr classifier (GPU timeout, ring errors) |
| `src/exo/routing/router.py:86` | `publish_bytes` — **unguarded crash point** |
| `app/EXO/EXO/ExoProcessController.swift:202` | "Exited with code N" |
| `dashboard/src/routes/+page.svelte:2003` | `RunnerFailed → "Failed"` (no detail shown) |
| `dashboard/src/routes/+page.svelte:5713` | downloads DO show `errorMessage` (good pattern to copy) |

---

## 7. Conclusion

The backend already has **strong structured error data** (ErrorResponse, RunnerTerminationError with full traceback, diagnostics classifier). The information is **mostly lost at the edges**:
- the **routing layer** (unguarded → crash),
- the **app shell** (discards stderr, shows bare exit code),
- and the **dashboard** (maps FAILED to a label but doesn't render the rich `error_message`/diagnostics).

Improving error UX = **surfacing data that already exists**, plus a small stable-code enum on the API. No architectural change needed.
