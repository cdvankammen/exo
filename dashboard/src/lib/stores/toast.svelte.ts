/**
 * Toast notification store - Global notification system for the EXO dashboard.
 *
 * Usage:
 *   import { addToast, dismissToast, toasts } from "$lib/stores/toast.svelte";
 *   addToast({ type: "success", message: "Model launched" });
 *   addToast({ type: "error", message: "Connection lost", persistent: true });
 */

type ToastType = "success" | "error" | "warning" | "info";

export interface Toast {
  id: string;
  type: ToastType;
  message: string;
  /** Auto-dismiss after this many ms. 0 = persistent (must be dismissed manually). */
  duration: number;
  createdAt: number;
}

interface ToastInput {
  type: ToastType;
  message: string;
  /** If true, toast stays until manually dismissed. Default: false. */
  persistent?: boolean;
  /** Auto-dismiss duration in ms. Default: 4000 for success/info, 6000 for error/warning. */
  duration?: number;
}

const DEFAULT_DURATIONS: Record<ToastType, number> = {
  success: 4000,
  info: 4000,
  warning: 6000,
  error: 6000,
};

let toastList = $state<Toast[]>([]);
const timers = new Map<string, ReturnType<typeof setTimeout>>();

function generateId(): string {
  return `toast-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function addToast(input: ToastInput): string {
  const id = generateId();
  const duration = input.persistent
    ? 0
    : (input.duration ?? DEFAULT_DURATIONS[input.type]);

  const toast: Toast = {
    id,
    type: input.type,
    message: input.message,
    duration,
    createdAt: Date.now(),
  };

  toastList = [...toastList, toast];

  if (duration > 0) {
    const timer = setTimeout(() => dismissToast(id), duration);
    timers.set(id, timer);
  }

  return id;
}

export function dismissToast(id: string): void {
  const timer = timers.get(id);
  if (timer) {
    clearTimeout(timer);
    timers.delete(id);
  }
  toastList = toastList.filter((t) => t.id !== id);
}

/** Dismiss all toasts matching a message (useful for dedup). */
export function dismissByMessage(message: string): void {
  const matching = toastList.filter((t) => t.message === message);
  for (const t of matching) {
    dismissToast(t.id);
  }
}

export function toasts(): Toast[] {
  return toastList;
}

/**
 * Extract a short, human-readable message from a raw error body and clamp it
 * to `maxLength` chars. Raw server error bodies can be huge (pydantic
 * validation dumps, full Python tracebacks, JSON detail arrays) — toasts
 * should never show that verbatim.
 *
 * Handles:
 * - JSON bodies: {"detail": "..."} (FastAPI) / {"error_message": "..."}
 * - pydantic validation arrays: {"detail": [{"loc":..., "msg": "..."}, ...]}
 * - Python tracebacks: keeps the FINAL line (the actual exception message)
 * - plain text: collapses whitespace + truncates
 */
export function truncateErrorMessage(raw: string, maxLength = 200): string {
  if (!raw) return "Unknown error";
  let message = raw.trim();
  if (!message) return "Unknown error";

  // JSON error bodies (FastAPI style)
  if (message.startsWith("{") || message.startsWith("[")) {
    try {
      const parsed = JSON.parse(message) as {
        detail?: unknown;
        error_message?: unknown;
      };
      const detail = parsed.detail;
      if (typeof detail === "string") {
        message = detail;
      } else if (Array.isArray(detail) && detail.length > 0) {
        // pydantic-style validation errors: use the first error's msg
        const first = detail[0] as { msg?: unknown } | null;
        message =
          first && typeof first.msg === "string" ? first.msg : JSON.stringify(first);
      } else if (typeof parsed.error_message === "string") {
        message = parsed.error_message;
      }
    } catch {
      // Not valid JSON — fall through to text handling.
    }
  }

  // Python traceback: the final line is the actual exception message.
  if (message.includes("Traceback")) {
    const lines = message
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
    const last = lines[lines.length - 1];
    if (last) message = last;
  }

  // Cut pydantic noise that sometimes trails the real message.
  const markerIdx = message.indexOf("For further information visit");
  if (markerIdx > 0) message = message.slice(0, markerIdx);

  // Collapse whitespace/newlines into a single line.
  message = message.replace(/\s+/g, " ").trim();

  // Clamp.
  if (message.length > maxLength) {
    message = message.slice(0, maxLength - 1).trimEnd() + "…";
  }
  return message || "Unknown error";
}
