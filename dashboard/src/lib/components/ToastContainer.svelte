<script lang="ts">
  import { toasts, dismissToast, type Toast } from "$lib/stores/toast.svelte";
  import { fly, fade } from "svelte/transition";
  import { flip } from "svelte/animate";

  const items = $derived(toasts());

  /** Toasts whose message is expanded past the clamp (per-toast toggle). */
  let expandedIds = $state<Set<string>>(new Set());

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
        class="pointer-events-auto w-80 max-w-sm bg-exo-dark-gray/95 backdrop-blur-sm border border-exo-medium-gray/60 border-l-[3px] {style.border} rounded shadow-lg shadow-black/40"
        in:fly={{ x: 80, duration: 250 }}
        out:fade={{ duration: 150 }}
        animate:flip={{ duration: 200 }}
        role="alert"
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
