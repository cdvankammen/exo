<script lang="ts">
  import { toasts, dismissToast, type Toast } from "$lib/stores/toast.svelte";
  import { fly, fade } from "svelte/transition";
  import { flip } from "svelte/animate";

  const items = $derived(toasts());

  /** Toasts whose message is expanded past the clamp (per-toast toggle). */
  let expandedIds = $state<Set<string>>(new Set());

  /** Which toast is currently shown in the full overlay modal. */
  let overlayToast = $state<Toast | null>(null);

  /** View mode for the overlay: 'details' (structured), 'clean' (readable), 'raw' (full text), 'json' (highlighted). */
  type ViewMode = "details" | "clean" | "raw" | "json";
  let overlayViewMode = $state<ViewMode>("details");

  /** Messages longer than this get the 4-line clamp + "Show full" affordance. */
  const CLAMP_THRESHOLD = 160;

  function isLongMessage(message: string): boolean {
    return message.length > CLAMP_THRESHOLD;
  }

  function toggleExpanded(toastId: string): void {
    const next = new Set(expandedIds);
    if (next.has(toastId)) {
      next.delete(toastId);
    } else {
      next.add(toastId);
    }
    expandedIds = next;
  }

  function openOverlay(toast: Toast) {
    overlayToast = toast;
    overlayViewMode = "details";
  }

  /** Navigate to the full Logs page so the user can see raw context. */
  function viewInLogs() {
    closeOverlay();
    window.location.hash = "#/logs";
  }

  /** Format a message for display: unwrap common FastAPI/JSON wrappers. */
  function unwrapMessage(message: string): string {
    if (!message) return "";
    const trimmed = message.trim();
    if (trimmed.startsWith("{") || trimmed.startsWith("[")) {
      try {
        const parsed = JSON.parse(trimmed);
        if (parsed && typeof parsed === "object" && "detail" in parsed) {
          const detail = (parsed as { detail: unknown }).detail;
          if (typeof detail === "string") return detail;
          if (Array.isArray(detail) && detail.length > 0) {
            const first = detail[0] as { msg?: unknown } | null;
            if (first && typeof first.msg === "string") return first.msg;
          }
        }
      } catch { /* fall through */ }
    }
    return message;
  }

  function closeOverlay() {
    overlayToast = null;
  }

  /** Try to parse the message as JSON for structured viewing. */
  function tryParseJson(message: string): unknown | null {
    try {
      return JSON.parse(message);
    } catch {
      return null;
    }
  }

  /** Format a timestamp for display. */
  function formatTimestamp(id: string): string {
    try {
      const ts = parseInt(id.split("-")[0]);
      if (!isNaN(ts)) return new Date(ts).toLocaleString();
    } catch { /* ignore */ }
    return "";
  }

  const typeStyles: Record<
    Toast["type"],
    { border: string; icon: string; iconColor: string }
  > = {
    success: {
      border: "border-l-green-500",
      icon: "M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z",
      iconColor: "text-green-400",
    },
    error: {
      border: "border-l-red-500",
      icon: "M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z",
      iconColor: "text-red-400",
    },
    warning: {
      border: "border-l-yellow-500",
      icon: "M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126z",
      iconColor: "text-yellow-400",
    },
    info: {
      border: "border-l-blue-500",
      icon: "M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z",
      iconColor: "text-blue-400",
    },
  };
</script>

{#if items.length > 0}
  <div
    class="fixed bottom-6 right-6 z-[9999] flex flex-col gap-2 pointer-events-none"
    role="log"
    aria-live="polite"
    aria-label="Notifications"
  >
    {#each items as toast (toast.id)}
      {@const style = typeStyles[toast.type]}
      {@const long = isLongMessage(toast.message)}
      {@const expanded = expandedIds.has(toast.id)}
      <div
        class="pointer-events-auto w-80 max-w-sm bg-exo-dark-gray/95 backdrop-blur-sm border border-exo-medium-gray/60 border-l-[3px] {style.border} rounded shadow-lg shadow-black/40 cursor-pointer hover:border-exo-medium-gray/80 transition-colors"
        in:fly={{ x: 80, duration: 250 }}
        out:fade={{ duration: 150 }}
        animate:flip={{ duration: 200 }}
        role="alert"
        onclick={() => openOverlay(toast)}
        onkeydown={(e) => { if (e.key === "Enter" || e.key === " ") openOverlay(toast); }}
      >
        <div class="flex items-start gap-3 px-4 py-3">
          <!-- Icon -->
          <svg
            class="w-5 h-5 flex-shrink-0 mt-0.5 {style.iconColor}"
            fill="none"
            viewBox="0 0 24 24"
            stroke-width="1.5"
            stroke="currentColor"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              d={style.icon}
            />
          </svg>

          <!-- Message: clamped to 4 lines unless expanded; long messages
               always stay inside the toast box. -->
          <div class="flex-1 min-w-0">
            <p
              class="text-sm text-white/90 font-mono leading-snug break-words {long && !expanded
                ? 'clamped-message'
                : ''}"
              title={long && !expanded ? "Click 'Show full' to expand" : undefined}
            >
              {toast.message}
            </p>
            {#if long}
              <button
                type="button"
                onclick={() => toggleExpanded(toast.id)}
                class="mt-1 text-[11px] font-mono uppercase tracking-wide text-exo-yellow/80 hover:text-exo-yellow transition-colors cursor-pointer"
              >
                {expanded ? "Show less" : "Show full"}
              </button>
            {/if}
          </div>

          <!-- Dismiss button -->
          <button
            onclick={() => dismissToast(toast.id)}
            class="flex-shrink-0 p-0.5 text-white/40 hover:text-white/80 transition-colors cursor-pointer"
            aria-label="Dismiss notification"
          >
            <svg
              class="w-4 h-4"
              fill="none"
              viewBox="0 0 24 24"
              stroke-width="2"
              stroke="currentColor"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        <!-- Auto-dismiss progress bar -->
        {#if toast.duration > 0}
          <div class="h-0.5 bg-white/5 rounded-b overflow-hidden">
            <div
              class="h-full {style.border.replace('border-l-', 'bg-')}/60"
              style="animation: shrink {toast.duration}ms linear forwards"
            ></div>
          </div>
        {/if}
      </div>
    {/each}
  </div>
{/if}

<!-- Full Error Overlay Modal -->
{#if overlayToast}
  {@const jsonParsed = tryParseJson(overlayToast.message)}
  <div
    class="fixed inset-0 z-[10000] flex items-center justify-center p-4"
    role="dialog"
    aria-modal="true"
    onclick={closeOverlay}
    onkeydown={(e) => { if (e.key === "Escape") closeOverlay(); }}
  >
    <!-- Backdrop -->
    <div class="absolute inset-0 bg-black/70 backdrop-blur-sm" in:fade={{ duration: 150 }}></div>
    <!-- Modal -->
    <div
      class="relative w-full max-w-2xl max-h-[80vh] bg-exo-dark-gray border border-exo-medium-gray/60 rounded-lg shadow-2xl flex flex-col overflow-hidden"
      onclick={(e) => e.stopPropagation()}
      in:fly={{ y: 20, duration: 200 }}
    >
      <!-- Header -->
      <div class="flex items-center justify-between px-4 py-3 border-b border-exo-medium-gray/30">
        <div class="flex items-center gap-2">
          <span class="text-xs font-mono uppercase tracking-wider {typeStyles[overlayToast.type].iconColor}">
            {overlayToast.type}
          </span>
          <span class="text-xs text-white/30 font-mono">
            {overlayToast.createdAt ? new Date(overlayToast.createdAt).toLocaleString() : ""}
          </span>
        </div>
        <button
          type="button"
          onclick={closeOverlay}
          class="p-1 text-white/40 hover:text-white/80 transition-colors cursor-pointer"
          aria-label="Close"
        >
          <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      <!-- View mode tabs -->
      <div class="flex gap-1 px-4 py-2 border-b border-exo-medium-gray/20">
        {#each ["details", "clean", "raw", "json"] as mode}
          <button
            type="button"
            onclick={() => overlayViewMode = mode as ViewMode}
            class="px-2 py-0.5 text-[10px] font-mono uppercase tracking-wider rounded transition-colors cursor-pointer {overlayViewMode === mode
              ? 'bg-exo-yellow/20 text-exo-yellow border border-exo-yellow/30'
              : 'text-white/40 hover:text-white/60 border border-transparent'}"
            disabled={mode === "json" && !jsonParsed}
          >
            {mode}
          </button>
        {/each}
      </div>
      <!-- Content -->
      <div class="flex-1 overflow-y-auto p-4">
        {#if overlayViewMode === "details"}
          <div class="space-y-3">
            <div class="grid grid-cols-2 gap-2">
              <div class="bg-white/[0.03] border border-white/10 rounded p-2">
                <div class="text-[9px] uppercase tracking-wider text-white/40">Type</div>
                <div class="text-xs text-white/90 mt-0.5">{overlayToast.type}</div>
              </div>
              <div class="bg-white/[0.03] border border-white/10 rounded p-2">
                <div class="text-[9px] uppercase tracking-wider text-white/40">ID</div>
                <div class="text-xs text-white/70 mt-0.5 break-all">{overlayToast.id}</div>
              </div>
              <div class="bg-white/[0.03] border border-white/10 rounded p-2">
                <div class="text-[9px] uppercase tracking-wider text-white/40">Created</div>
                <div class="text-xs text-white/70 mt-0.5">{overlayToast.createdAt ? new Date(overlayToast.createdAt).toLocaleString() : "—"}</div>
              </div>
              <div class="bg-white/[0.03] border border-white/10 rounded p-2">
                <div class="text-[9px] uppercase tracking-wider text-white/40">Duration</div>
                <div class="text-xs text-white/70 mt-0.5">{overlayToast.duration > 0 ? `${(overlayToast.duration / 1000).toFixed(0)}s` : "persistent"}</div>
              </div>
            </div>
            <div class="bg-white/[0.03] border border-white/10 rounded p-2">
              <div class="text-[9px] uppercase tracking-wider text-white/40">Message</div>
              <pre class="text-xs font-mono text-white/85 whitespace-pre-wrap break-words leading-relaxed mt-1">{unwrapMessage(overlayToast.message)}</pre>
            </div>
          </div>
        {:else if overlayViewMode === "clean"}
          <pre class="text-xs font-mono text-white/80 whitespace-pre-wrap break-words leading-relaxed">{overlayToast.message}</pre>
        {:else if overlayViewMode === "raw"}
          <pre class="text-[10px] font-mono text-white/50 whitespace-pre-wrap break-words">{JSON.stringify(overlayToast, null, 2)}</pre>
        {:else if overlayViewMode === "json" && jsonParsed}
          <pre class="text-[11px] font-mono text-green-400/80 whitespace-pre-wrap break-words">{JSON.stringify(jsonParsed, null, 2)}</pre>
        {/if}
      </div>
      <!-- Footer -->
      <div class="flex items-center justify-between px-4 py-2 border-t border-exo-medium-gray/20">
        <button
          type="button"
          onclick={viewInLogs}
          class="text-[10px] font-mono text-exo-yellow/70 hover:text-exo-yellow transition-colors cursor-pointer"
        >
          View in Logs →
        </button>
        <div class="flex items-center gap-4">
          <button
            type="button"
            onclick={() => { navigator.clipboard.writeText(overlayToast?.message ?? ""); }}
            class="text-[10px] font-mono text-white/40 hover:text-exo-yellow transition-colors cursor-pointer"
          >
            Copy to clipboard
          </button>
          <button
            type="button"
            onclick={() => { if (overlayToast) dismissToast(overlayToast.id); closeOverlay(); }}
            class="text-[10px] font-mono text-white/40 hover:text-red-400 transition-colors cursor-pointer"
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  </div>
{/if}

<style>
  /* Hard clamp: even a huge error message stays inside the toast box. */
  .clamped-message {
    display: -webkit-box;
    -webkit-line-clamp: 4;
    line-clamp: 4;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  @keyframes shrink {
    from {
      width: 100%;
    }
    to {
      width: 0%;
    }
  }
</style>
