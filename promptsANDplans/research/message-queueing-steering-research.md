# Message Queueing + Steering — Research

**Date:** 2026-08-06
**Branch:** fix-memory-error
**Scope:** Research only (no code changes yet). Investigates: (1) existing queueing work in exo/upstream, (2) how message queueing could work for offline-model users, (3) whether "steering" (queue management / generation steering) is plausible given offline models + tokenization.

---

## PART 1: What already exists (verified in code, 2026-08-06)

### 1.1 Upstream: NO queueing feature exists
- Searched `exo-explore/exo` GitHub (code + issues + PRs) for "queue messages", "queue chat", "batching queue", "concurrent requests".
- **No PR or issue proposes a chat message queue.** The `pendingAutoMessage` code below is the ONLY queue-like mechanism, and it's a single-slot auto-launch handoff, not a message queue.

### 1.2 The backend ALREADY supports multiple concurrent generation requests

**Verified: `EXO_MAX_CONCURRENT_REQUESTS = 8`** (`src/exo/shared/constants.py:117`):
```python
EXO_MAX_CONCURRENT_REQUESTS = int(os.getenv("EXO_MAX_CONCURRENT_REQUESTS", "8"))
```

**The runner has a real work queue** (`src/exo/worker/runner/runner.py`):
- `self._work_queue: queue.Queue[WorkItem]` (line 125) — task reader thread feeds it (line 173)
- `handle_generation_tasks()` (line ~340) **pulls new work while active tasks run**:
  ```python
  while self.active_tasks:
      results = self.generator.step()
      ...
      try:
          item = self._work_queue.get_nowait()   # ← grabs next task mid-generation
      except queue.Empty:
          continue
      ...
      self.submit_generation(item)               # ← submits to batch generator
  ```
- So multiple `TextGeneration` tasks can be in-flight simultaneously.

**The BatchGenerator batches them** (`src/exo/worker/runner/llm_inference/batch_generator.py`):
- `_queue: deque[TextGeneration]` (line ~356)
- `step()` (line ~408) submits up to `EXO_MAX_CONCURRENT_REQUESTS` (8) tasks:
  ```python
  while self._queue and len(self._active_tasks) < EXO_MAX_CONCURRENT_REQUESTS:
      task = self._queue.popleft()
      ...
      self._active_tasks[uid] = (task, queue, output_generator)
  ```
- Continuous batching is ON by default (`--no-batch` disables → `SequentialGenerator`).

**Conclusion:** The backend can already accept N concurrent `/v1/chat/completions` and batch them (up to 8). **The queue gap is 100% frontend.**

### 1.3 The frontend BLOCKS queueing (the actual gap)

Three places silently drop or block messages while one is generating:

1. **`ChatForm.svelte:206`** — submit gated on loading:
   ```svelte
   function handleSubmit() {
     if ((!message.trim() && uploadedFiles.length === 0) || loading) return;
   ```
2. **`ChatForm.svelte:442`** — textarea `disabled={loading}` while generating
3. **`app.svelte.ts:2410`** — the store also gates:
   ```ts
   async sendMessage(...): Promise<void> {
     if ((!content.trim() && (!files || files.length === 0)) || this.isLoading)
       return;
   ```

**Result:** while a response streams, the input is disabled — the user **cannot type/send** the next message. No queue, no "pending (n)" indicator.

### 1.4 The existing single-slot mechanism: `pendingAutoMessage`

`+page.svelte:3112`:
```ts
let pendingAutoMessage = $state<{
  content: string;
  files?: {...}[];
} | null>(null);
```
Used by `handleChatSend` (line 3256) + `handleAutoSend` (line 2966): when the selected model isn't running, the message is parked here, the model is launched, and `sendMessage` fires once ready. **This proves the pattern works** — it's just single-slot and tied to auto-launch, not a general queue.

---

## PART 2: How message queueing WOULD work for offline-model users

### 2.1 The design (frontend-only, backend untouched)

**Concept:** Replace the "disabled while loading" behavior with a **client-side FIFO queue** + "N queued" indicator + per-item cancel. The backend already batches — we simply stop blocking.

**Frontend store changes (`app.svelte.ts`):**
```ts
interface QueuedMessage {
  id: string;
  content: string;
  files?: ...[];
  createdAt: number;
}

pendingQueue = $state<QueuedMessage[]>([]);   // the queue

// In sendMessage(): instead of returning when isLoading,
// enqueue and return:
if (this.isLoading) {
  this.pendingQueue.push({ id: crypto.randomUUID(), content, files, ... });
  return;
}
```

**The drain loop (in sendMessage after generation completes):**
```ts
// at end of sendMessage, after isLoading = false:
const next = this.pendingQueue.shift();
if (next) await this.sendMessage(next.content, next.files);
```

**UI changes (`ChatForm.svelte` / `+page.svelte`):**
- Remove `disabled={loading}` on the textarea — keep it typeable
- Replace the block in `handleSubmit()` with `onAutoSend(...)` always (store enqueues if busy)
- Show a small "**2 queued** ⏎ Send next automatically" chip near the input
- Per-item ✕ to remove from queue; maybe a "Send all" behavior

**Why offline models make this MORE valuable:** offline inference is slow (a 30B model on a single Mac could take minutes per response). The user wants to **fire several prompts and walk away**, then come back to N responses. Exactly what a queue enables.

### 2.2 Two queue modes to consider

| Mode | Behavior | Use case |
|---|---|---|
| **Sequential (default)** | One generation at a time; queue drains FIFO as each finishes | Offline/single-node; predictable memory; works with `--no-batch` too |
| **Parallel (advanced)** | Fire all N requests at once; backend batches up to 8 | Multi-node / small models; needs per-message stream UI (complex) |

Sequential is the right default for the user's offline use (memory-safe, matches SequentialGenerator, no interleaved-stream UI needed). Parallel is a later enhancement that the backend already supports.

### 2.3 What needs NO backend change
- FIFO queue, cancel, "N queued" indicator — all frontend
- Backend batching, work queue, `EXO_MAX_CONCURRENT_REQUESTS` — already there

### 2.4 Edge cases
- **Model switch mid-queue:** pin the queue to the model that was selected when messages were queued (like `targetConversationId` already does per-request)
- **Eject while queued:** queued messages for an ejected model → either drop with a toast or auto-queue on re-launch
- **Conversation switch:** keep queue per-conversation or global? Recommend global (simpler), with a note in UI
- **Tokenization:** each queued message tokenizes when it's processed (not at enqueue time) — no issue; the model context is built per-request from the conversation history as today

---

## PART 3: Steering — feasibility analysis

"Steering" can mean two things; both evaluated:

### 3.1 Queue steering (prioritize / reorder / cancel) — EASY, frontend-only
- Each queued item gets `priority` (or just drag-to-reorder)
- Cancel = remove from queue
- "Jump the line" = insert at head
- **Fully plausible** — it's just array manipulation + a drain loop that picks highest-priority. No model/tokenization involvement.

### 3.2 Generation steering (guiding the model's output) — PARTIAL, backend needed
What exo already exposes per request (`ChatCompletionRequest` in `src/exo/api/types/api.py:243-260`):
- `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`, `max_tokens`
- `logit_bias: dict[str, int] | None` — **exists in the API type**
- `response_format` (dict, e.g. JSON) — exists
- `tools` / function calling — exists (deepseek_v4_encoding has a response-format template)
- `reasoning_effort`, `enable_thinking`

What's MISSING for strong steering:
- **Guided/constrained decoding** (grammar/schema-constrained sampling, e.g. Outlines/guidance-style): not implemented; would require a sampling hook in the generator
- **logit_bias end-to-end:** the field exists in the API type — need to verify it's plumbed to the actual sampler (mlx_lm `sample_utils.pyi` has `logit_bias` in the stub, so mlx_lm supports it; exo may or may not forward it)

**Feasibility verdict:**
- **Basic steering (temperature/seed/top_p/stop/logit_bias via request params):** already largely present — the dashboard just doesn't expose them. Building a "sampling controls" panel in the UI is straightforward and works with offline models (these are per-request sampler params, nothing special about offline).
- **Constrained/guided decoding:** plausible but needs a generator-level hook (sampling overrides). MLX backend could support it (it's just a modified sampling step), but it's a backend feature, not a UI one. Offline/quantized models work fine with it in principle — tokenization is unaffected.
- **Steering vectors / activation steering:** NOT plausible through the current API — requires accessing hidden activations mid-generation, which exo's API/runner doesn't expose. Would need deep engine changes.

**Tokenization answer to the user's question:** "is steering plausible without offline models work and tokenising works?" — **Yes.** Steering via request params (temperature/seed/stop/logit_bias) operates at the sampling step, which happens *after* tokenization. It works identically for offline and hosted models. Only *activation-level* steering (vectors) needs engine internals — and that's a niche capability, not needed for queue-steering or basic generation-steering.

---

## PART 4: Recommended implementation order (for later, when user confirms)

| # | Feature | Scope | Effort |
|---|---|---|---|
| 1 | FIFO message queue (frontend) | store + ChatForm + chip UI | Small |
| 2 | Queue cancel + "N queued" indicator | frontend | Small |
| 3 | Queue reorder/priority (drag or ↑↓) | frontend | Small |
| 4 | Parallel drain (fire N at once, use backend batching) | frontend (backend ready) | Medium |
| 5 | Sampling controls panel (temperature/seed/top_p/stop) | frontend + verify logit_bias plumbing | Small-Med |
| 6 | Guided/constrained decoding | backend generator hook | Large |

**Recommendation:** Items 1–3 deliver the user's core ask (queue messages for slow offline models) with zero backend risk. Item 5 is the natural "steering" follow-up and needs only a UI + a check that logit_bias forwards.

---

## PART 5: Key file map

| File:Line | What it is |
|---|---|
| `src/exo/shared/constants.py:117` | `EXO_MAX_CONCURRENT_REQUESTS = 8` |
| `src/exo/worker/runner/runner.py:125` | runner `_work_queue` |
| `src/exo/worker/runner/runner.py:~340` | `handle_generation_tasks` — pulls next task mid-generation |
| `src/exo/worker/runner/llm_inference/batch_generator.py:356` | `_queue` deque |
| `src/exo/worker/runner/llm_inference/batch_generator.py:~408` | `step()` submits up to 8 |
| `src/exo/worker/engines/mlx/builder.py:86` | `EXO_NO_BATCH` → SequentialGenerator |
| `src/exo/api/types/api.py:243-260` | ChatCompletionRequest params (logit_bias, response_format, seed...) |
| `dashboard/src/lib/stores/app.svelte.ts:2410` | `sendMessage` — `isLoading` gate (the block) |
| `dashboard/src/lib/components/ChatForm.svelte:206` | `handleSubmit` — `loading` gate |
| `dashboard/src/lib/components/ChatForm.svelte:442` | textarea `disabled={loading}` |
| `dashboard/src/routes/+page.svelte:3112` | `pendingAutoMessage` (single-slot, proves the pattern) |
