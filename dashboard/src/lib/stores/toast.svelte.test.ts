// Toast store unit tests — EXO dashboard notification system.
//
// Covers the module-level toast store (Svelte 5 runes, no DOM deps):
//   - addToast lifecycle: append + id return, duration resolution
//     (persistent -> 0, no timer; defaults success/info 4000ms,
//     warning/error 6000ms; explicit duration override)
//   - auto-dismiss timing with fake timers
//   - dismissToast: pending-timer cleanup, unknown id safe no-op
//   - dismissByMessage: exact-match only, removes ALL matches
//   - stacking: capped at MAX_VISIBLE (5) with oldest-first eviction;
//     evicted toasts get their pending timer cleared via dismissToast
//   - truncateErrorMessage: FastAPI detail string, pydantic array .msg,
//     error_message, traceback final line, "For further information visit"
//     crop, whitespace collapse, 200-char clamp + ellipsis, empty/whitespace
//     -> "Unknown error"
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  addToast,
  dismissToast,
  dismissByMessage,
  toasts,
  truncateErrorMessage,
  type Toast,
} from "./toast.svelte";

describe("addToast lifecycle", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    // Fresh module state: dismiss everything added by prior tests.
    for (const t of toasts()) dismissToast(t.id);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("appends a toast and returns its id", () => {
    const id = addToast({ type: "success", message: "Model launched" });
    expect(id).toBeTruthy();
    expect(id).toMatch(/^toast-\d+-[a-z0-9]{6}$/);
    const list = toasts();
    expect(list).toHaveLength(1);
    expect(list[0].id).toBe(id);
    expect(list[0].type).toBe("success");
    expect(list[0].message).toBe("Model launched");
    expect(list[0].duration).toBe(4000);
    expect(list[0].createdAt).toBeGreaterThan(0);
  });

  it("persistent toasts get duration 0 and arm NO timer", () => {
    const spy = vi.spyOn(globalThis, "setTimeout");
    const id = addToast({ type: "error", message: "Connection to server lost", persistent: true });
    expect(spy).not.toHaveBeenCalled();
    const t = toasts().find((x) => x.id === id)!;
    expect(t.duration).toBe(0);
  });

  it("uses type defaults: 4000ms for success/info, 6000ms for warning/error", () => {
    for (const type of ["success", "info", "warning", "error"] as const) {
      addToast({ type, message: type });
    }
    const durations = new Map(toasts().map((t) => [t.type, t.duration]));
    expect(durations.get("success")).toBe(4000);
    expect(durations.get("info")).toBe(4000);
    expect(durations.get("warning")).toBe(6000);
    expect(durations.get("error")).toBe(6000);
  });

  it("honors an explicit duration override", () => {
    const id = addToast({ type: "info", message: "custom", duration: 1500 });
    expect(toasts().find((t) => t.id === id)!.duration).toBe(1500);
  });
});

describe("auto-dismiss", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    for (const t of toasts()) dismissToast(t.id);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("auto-dismisses a success toast after 4000ms", () => {
    const id = addToast({ type: "success", message: "Model ready" });
    expect(toasts()).toHaveLength(1);
    vi.advanceTimersByTime(3999);
    expect(toasts()).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(toasts()).toHaveLength(0);
    expect(toasts().find((t) => t.id === id)).toBeUndefined();
  });

  it("auto-dismisses an error toast after 6000ms", () => {
    addToast({ type: "error", message: "Model failed" });
    vi.advanceTimersByTime(5999);
    expect(toasts()).toHaveLength(1);
    vi.advanceTimersByTime(1);
    expect(toasts()).toHaveLength(0);
  });

  it("never auto-dismisses a persistent toast", () => {
    addToast({ type: "warning", message: "Connection lost", persistent: true });
    vi.advanceTimersByTime(60_000);
    expect(toasts()).toHaveLength(1);
  });
});

describe("dismissToast", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    for (const t of toasts()) dismissToast(t.id);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("clears the pending timer and removes the toast", () => {
    const id = addToast({ type: "info", message: "disappear" });
    dismissToast(id);
    // Timer was clearTimeout'd: advancing past the duration must not
    // attempt anything (idempotent), and the list stays empty.
    vi.advanceTimersByTime(10_000);
    expect(toasts()).toHaveLength(0);
    // No dangling timer: a second advance is still a no-op.
    vi.advanceTimersByTime(10_000);
    expect(toasts()).toHaveLength(0);
  });

  it("unknown id is a safe no-op", () => {
    addToast({ type: "error", message: "real" });
    expect(() => dismissToast("toast-does-not-exist")).not.toThrow();
    expect(toasts()).toHaveLength(1);
  });

  it("removes only the named toast, leaving siblings", () => {
    const a = addToast({ type: "success", message: "A" });
    addToast({ type: "success", message: "B" });
    dismissToast(a);
    const list = toasts();
    expect(list).toHaveLength(1);
    expect(list[0].message).toBe("B");
  });
});

describe("dismissByMessage", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    for (const t of toasts()) dismissToast(t.id);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("removes ALL exact-match toasts", () => {
    addToast({ type: "warning", message: "Connection to server lost", persistent: true });
    addToast({ type: "warning", message: "Connection to server lost", persistent: true });
    addToast({ type: "success", message: "Connection restored" });
    dismissByMessage("Connection to server lost");
    const list = toasts();
    expect(list).toHaveLength(1);
    expect(list[0].message).toBe("Connection restored");
  });

  it("does NOT substring-match", () => {
    addToast({ type: "info", message: "Model ready" });
    addToast({ type: "error", message: "Model failed: OOM" });
    dismissByMessage("Model");
    expect(toasts()).toHaveLength(2);
    dismissByMessage("Model failed");
    expect(toasts()).toHaveLength(2);
  });
});

describe("stacking", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    for (const t of toasts()) dismissToast(t.id);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("caps the stack at 5, evicting the oldest under a burst", () => {
    const N = 25;
    for (let i = 0; i < N; i++) addToast({ type: "info", message: `toast ${i}` });
    expect(toasts()).toHaveLength(5);
    expect(toasts().map((t: Toast) => t.message)).toEqual([
      "toast 20",
      "toast 21",
      "toast 22",
      "toast 23",
      "toast 24",
    ]);
  });

  it("keeps insertion order (newest last)", () => {
    addToast({ type: "info", message: "first" });
    addToast({ type: "info", message: "second" });
    addToast({ type: "info", message: "third" });
    const list = toasts().map((t: Toast) => t.message);
    expect(list).toEqual(["first", "second", "third"]);
  });
});

describe("MAX_VISIBLE cap (oldest-first eviction)", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    for (const t of toasts()) dismissToast(t.id);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("evicts the OLDEST toast when a 6th is pushed, keeping newest 5", () => {
    for (let i = 0; i < 5; i++) addToast({ type: "info", message: `toast ${i}` });
    expect(toasts()).toHaveLength(5);

    addToast({ type: "info", message: "toast 5" });
    const msgs = toasts().map((t: Toast) => t.message);
    expect(msgs).toEqual(["toast 1", "toast 2", "toast 3", "toast 4", "toast 5"]);
    expect(msgs).not.toContain("toast 0");
  });

  it("keeps at most 5 even under a rapid burst", () => {
    for (let i = 0; i < 12; i++) addToast({ type: "success", message: `Model ready ${i}` });
    expect(toasts()).toHaveLength(5);
  });

  it("clears the evicted toast's pending timer (no leak / no late no-op)", () => {
    const clearSpy = vi.spyOn(globalThis, "clearTimeout");
    try {
      for (let i = 0; i < 6; i++) addToast({ type: "info", message: `toast ${i}` });
      // 6 auto-dismiss timers armed, oldest evicted → its timer cleared exactly once.
      expect(clearSpy).toHaveBeenCalledTimes(1);

      // Late firing is impossible for the evicted id: the 5 survivors still
      // auto-dismiss cleanly and nothing throws or reappears.
      vi.advanceTimersByTime(10_000);
      expect(toasts()).toHaveLength(0);
    } finally {
      clearSpy.mockRestore();
    }
  });

  it("also evicts a persistent toast when it is the oldest", () => {
    addToast({ type: "error", message: "persistent oldest", persistent: true });
    for (let i = 0; i < 5; i++) addToast({ type: "info", message: `toast ${i}` });
    const msgs = toasts().map((t: Toast) => t.message);
    expect(msgs).toEqual(["toast 0", "toast 1", "toast 2", "toast 3", "toast 4"]);
    expect(msgs).not.toContain("persistent oldest");
    // Survivors are unaffected, including their own auto-dismiss timers.
    vi.advanceTimersByTime(10_000);
    expect(toasts()).toHaveLength(0);
  });
});

describe("truncateErrorMessage", () => {
  it("returns Unknown error for empty / whitespace-only input", () => {
    expect(truncateErrorMessage("")).toBe("Unknown error");
    expect(truncateErrorMessage("   ")).toBe("Unknown error");
    expect(truncateErrorMessage("\n\t ")).toBe("Unknown error");
  });

  it("extracts a JSON detail string (FastAPI style)", () => {
    expect(truncateErrorMessage('{"detail": "Model not found"}')).toBe("Model not found");
  });

  it("extracts first pydantic array .msg", () => {
    const raw =
      '{"detail": [{"loc": ["body", "model"], "msg": "field required", "type": "value_error"}]}';
    expect(truncateErrorMessage(raw)).toBe("field required");
  });

  it("falls back to error_message when detail is absent", () => {
    expect(truncateErrorMessage('{"error_message": "boom"}')).toBe("boom");
  });

  it("extracts message from exo unified ErrorResponse envelope (F2)", () => {
    const raw = JSON.stringify({
      error: {
        message: "Failed to load model card: Model does-not-exist requires authentication.",
        type: "Bad Request",
        param: null,
        code: 400,
        error_code: "MODEL_NOT_FOUND",
      },
    });
    expect(truncateErrorMessage(raw)).toBe(
      "Failed to load model card: Model does-not-exist requires authentication."
    );
  });

  it("extracts message from 422 ErrorResponse envelope with errors preserved (F2)", () => {
    const raw = JSON.stringify({
      error: {
        message: "model_id: Field required",
        type: "Unprocessable Entity",
        param: null,
        code: 422,
        error_code: "INVALID_REQUEST",
      },
      errors: [{ loc: ["query", "model_id"], msg: "Field required", type: "missing" }],
    });
    expect(truncateErrorMessage(raw)).toBe("model_id: Field required");
  });

 it("leaves other keys alone when envelope has no message (detail still wins)", () => {
    const raw = JSON.stringify({
      error: { code: 400, type: "Bad Request" },
      detail: "the actual reason",
    });
    expect(truncateErrorMessage(raw)).toBe("the actual reason");
  });

  it("keeps the FINAL line of a Python traceback", () => {
    const raw = [
      "Traceback (most recent call last):",
      '  File "/exo/runner.py", line 42, in run',
      "    raise RuntimeError(\"not enough memory\")",
      "RuntimeError: not enough memory",
    ].join("\n");
    expect(truncateErrorMessage(raw)).toBe("RuntimeError: not enough memory");
  });

  it("crops pydantic 'For further information visit' noise", () => {
    const raw = "1 validation error\nFor further information visit https://errors.pydantic.dev/";
    expect(truncateErrorMessage(raw)).toBe("1 validation error");
  });

  it("collapses whitespace / newlines into a single line", () => {
    expect(truncateErrorMessage("multi\n  line\t spaced   text")).toBe("multi line spaced text");
  });

  it("clamps to maxLength with a trailing ellipsis", () => {
    const long = "x".repeat(250);
    const out = truncateErrorMessage(long, 200);
    expect(out).toHaveLength(200);
    expect(out.endsWith("…")).toBe(true);
    expect(out.slice(0, 199)).toBe("x".repeat(199));
  });

  it("honors a custom maxLength", () => {
    const out = truncateErrorMessage("abcdefghij", 5);
    expect(out).toBe("abcd…");
  });

  it("returns Unknown error if extraction yields an empty result", () => {
    expect(truncateErrorMessage("   \n\t ")).toBe("Unknown error");
  });
});