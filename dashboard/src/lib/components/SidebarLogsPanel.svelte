<script lang="ts">
  // Sidebar Logs/Errors panel.
  // - Header bar toggles collapse/expand. Click header to collapse/expand.
  // - Expanded = auto-height panel that fits the number of errors, bounded by
  //   a viewport-relative max (max-h-[60vh]) so it works on any screen size.
  //   No manual drag-resize: the panel grows/shrinks with its content.
  // - Collapsed = slim footer-style row with toggle + error-count badge.
  import { onMount, onDestroy } from "svelte";
  import { listLogErrors, getLogTail, type LogErrorEntry } from "$lib/stores/app.svelte";

  const LEVEL_STYLES: Record<string, string> = {
    CRITICAL: "text-red-300 bg-red-500/20 border-red-500/40",
    ERROR: "text-red-400 bg-red-500/10 border-red-500/30",
    WARNING: "text-yellow-300 bg-yellow-500/10 border-yellow-500/30",
  };

  const LEVEL_ORDER = ["CRITICAL", "ERROR", "WARNING"] as const;
  const EXPANDED_KEY = "exo-sidebar-logs-expanded";
  const ERRORS_SECTION_KEY = "exo-sidebar-logs-errors-section";
  const TAIL_SECTION_KEY = "exo-sidebar-logs-tail-section";
  const ERRORS_HEIGHT_KEY = "exo-sidebar-logs-errors-height";

  let errors = $state<LogErrorEntry[]>([]);
  let loading = $state(true);
  let errorMsg = $state<string | null>(null);
  let tailContent = $state<string>("");
  let tailLoading = $state(false);
  let refreshTimer: ReturnType<typeof setInterval> | undefined;
  let expanded = $state(false);
  let errorsExpanded = $state(true);
  let tailExpanded = $state(true);
  let errorsSectionHeight = $state(200);
  let dragging = $state(false);
  let dismissedErrors = $state<Set<string>>(new Set());

  // Errors we can show, ordered by severity (critical first), capped so a
  // burst of errors can't freeze the UI. Dismissed errors are filtered out.
  const visibleErrors = $derived(
    [...errors]
      .filter((e) => !dismissedErrors.has(errorKey(e)))
      .sort((a, b) => levelRank(a.level) - levelRank(b.level))
      .slice(0, 50),
  );

  const activeErrorCount = $derived(visibleErrors.length);

  // Expandable error cards: click to show the full message inline.
  let expandedErrors = $state<Set<string>>(new Set());

  function errorKey(e: LogErrorEntry): string {
    return e.id ?? `${e.level}-${e.message}-${e.timestamp}`;
  }

  function toggleError(e: LogErrorEntry) {
    const key = errorKey(e);
    const next = new Set(expandedErrors);
    if (next.has(key)) {
      next.delete(key);
    } else {
      next.add(key);
    }
    expandedErrors = next;
  }

  function dismissError(e: LogErrorEntry) {
    const key = errorKey(e);
    dismissedErrors = new Set([...dismissedErrors, key]);
  }

  function clearAllErrors() {
    dismissedErrors = new Set(errors.map((e) => errorKey(e)));
  }

  function toggleErrorsSection() {
    errorsExpanded = !errorsExpanded;
    try {
      localStorage.setItem(ERRORS_SECTION_KEY, errorsExpanded ? "1" : "0");
    } catch { /* ignore */ }
  }

  function toggleTailSection() {
    tailExpanded = !tailExpanded;
    try {
      localStorage.setItem(TAIL_SECTION_KEY, tailExpanded ? "1" : "0");
    } catch { /* ignore */ }
  }

  const criticalCount = $derived(
    errors.filter((e) => e.level === "CRITICAL" || e.level === "ERROR").length,
  );

  function levelRank(level: string): number {
    const idx = LEVEL_ORDER.indexOf(level as (typeof LEVEL_ORDER)[number]);
    return idx === -1 ? LEVEL_ORDER.length : idx;
  }

  function toggleExpanded() {
    expanded = !expanded;
    try {
      localStorage.setItem(EXPANDED_KEY, expanded ? "1" : "0");
    } catch {
      // private mode — preference just won't persist
    }
    if (expanded) refreshAll();
  }

  // --- Flicker-free refresh: only reassign if content actually changed ---
  async function refreshErrors() {
    loading = true;
    errorMsg = null;
    try {
      const response = await listLogErrors();
      const next = response.errors;
      // Diff: only replace if the serialized content differs
      if (JSON.stringify(next) !== JSON.stringify(errors)) {
        errors = next;
      }
    } catch (e) {
      errorMsg = e instanceof Error ? e.message : "Failed to load errors";
    } finally {
      loading = false;
    }
  }

  async function refreshTail() {
    tailLoading = true;
    try {
      const result = (await getLogTail("main", 30)).content;
      // Only update if content actually changed to prevent flicker
      if (result !== tailContent) {
        tailContent = result;
      }
    } catch {
      if (tailContent !== "(unavailable)") {
        tailContent = "(unavailable)";
      }
    } finally {
      tailLoading = false;
    }
  }

  // --- Resizable divider between errors and log tail ---
  function onDividerPointerDown(e: PointerEvent) {
    e.preventDefault();
    dragging = true;
    const pointerId = e.pointerId;
    (e.target as HTMLElement).setPointerCapture(pointerId);

    const startY = e.clientY;
    const startHeight = errorsSectionHeight;

    function onMove(ev: PointerEvent) {
      const delta = ev.clientY - startY;
      const newHeight = Math.max(60, Math.min(startHeight + delta, 500));
      errorsSectionHeight = newHeight;
    }

    function onUp() {
      dragging = false;
      (e.target as HTMLElement).releasePointerCapture(pointerId);
      document.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerup", onUp);
      try {
        localStorage.setItem(ERRORS_HEIGHT_KEY, String(errorsSectionHeight));
      } catch {
        // ignore
      }
    }

    document.addEventListener("pointermove", onMove);
    document.addEventListener("pointerup", onUp);
  }

  function refreshAll() {
    refreshErrors();
    refreshTail();
  }

  onMount(() => {
    try {
      expanded = localStorage.getItem(EXPANDED_KEY) === "1";
      errorsExpanded = localStorage.getItem(ERRORS_SECTION_KEY) !== "0";
      tailExpanded = localStorage.getItem(TAIL_SECTION_KEY) !== "0";
      const storedHeight = localStorage.getItem(ERRORS_HEIGHT_KEY);
      if (storedHeight) errorsSectionHeight = Number(storedHeight) || 200;
    } catch {
      // ignore
    }
    refreshAll();
    refreshTimer = setInterval(() => {
      if (expanded) refreshAll();
    }, 5000);
  });

  onDestroy(() => {
    if (refreshTimer) clearInterval(refreshTimer);
  });
</script>

<!-- Collapsed: slim footer-style row -->
{#if !expanded}
  <button
    type="button"
    onclick={toggleExpanded}
    class="w-full flex items-center justify-between px-3 py-2 text-[10px] font-mono tracking-widest uppercase text-exo-light-gray/70 hover:text-exo-yellow hover:bg-white/[0.03] transition-colors cursor-pointer"
    title="Expand logs & errors panel"
  >
    <span class="flex items-center gap-1.5">
      <svg fill="currentColor" viewBox="0 0 24 24" class="w-3.5 h-3.5">
        <path d="M20 4h-6l-1.5 2H4a1 1 0 0 0-1 1v12a1 1 0 0 0 1 1h16a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1Zm-1 12H5V8h12.17l.83-1.11V16ZM7 13h10v-2H7v2Z"></path>
      </svg>
      MAIN LOGS
    </span>
    {#if criticalCount > 0}
      <span class="px-1.5 py-0.5 rounded bg-red-500/20 border border-red-500/40 text-red-300 text-[9px]">
        {criticalCount}
      </span>
    {/if}
  </button>
{:else}
  <div class="flex flex-col border-t border-exo-yellow/10 bg-exo-black/30 min-h-0 overflow-hidden">
    <!-- Header bar: click to collapse entire panel + clear-all button -->
    <div class="px-3 py-2 flex items-center justify-between flex-shrink-0 select-none hover:bg-exo-yellow/10 transition-colors">
      <span
        class="text-[10px] font-mono tracking-widest uppercase text-exo-light-gray/70 cursor-pointer"
        onclick={toggleExpanded}
        role="button"
        title="Click to collapse"
      >
        Main Logs
      </span>
      <div class="flex items-center gap-2">
        {#if activeErrorCount > 0}
          <button
            type="button"
            onclick={(ev) => { ev.stopPropagation(); clearAllErrors(); }}
            class="text-[9px] font-mono text-red-400/60 hover:text-red-400 transition-colors cursor-pointer uppercase"
            title="Dismiss all warnings"
          >
            Clear All
          </button>
        {/if}
        {#if activeErrorCount > 0}
          <span class="text-[10px] font-mono text-red-400">{activeErrorCount}</span>
        {/if}
        <span
          class="text-[10px] font-mono text-exo-light-gray/40 cursor-pointer"
          onclick={toggleExpanded}
          role="button"
          title="Click to collapse"
        >⠿</span>
      </div>
    </div>

    <!-- Errors section (collapsible) -->
    <div class="flex-shrink-0">
      <button
        type="button"
        onclick={toggleErrorsSection}
        class="w-full px-2 py-1 flex items-center justify-between text-[9px] font-mono uppercase tracking-widest text-exo-light-gray/50 hover:text-exo-light-gray/70 hover:bg-white/[0.02] transition-colors cursor-pointer"
      >
        <span class="flex items-center gap-1">
          <span class="text-[8px]">{errorsExpanded ? "▾" : "▸"}</span>
          Warnings & Errors
        </span>
        {#if visibleErrors.length > 0}
          <span class="text-[9px] text-red-400/70">{visibleErrors.length}</span>
        {/if}
      </button>

      {#if errorsExpanded}
        <div
          class="overflow-y-auto px-2 pb-1 space-y-1"
          style="max-height: {errorsSectionHeight}px;"
        >
          {#if loading}
            <div class="text-[10px] text-exo-light-gray/50 font-mono px-1">Loading…</div>
          {:else if errorMsg}
            <div class="text-[10px] text-red-400 font-mono px-1">{errorMsg}</div>
          {:else if visibleErrors.length === 0}
            <div class="text-[10px] text-exo-light-gray/40 font-mono px-1">
              {dismissedErrors.size > 0 ? "All warnings dismissed." : "No errors in cluster."}
            </div>
          {:else}
            {#each visibleErrors as e (errorKey(e))}
              {@const isExpanded = expandedErrors.has(errorKey(e))}
              <div
                class="w-full text-left px-1.5 py-1 rounded border text-[10px] font-mono leading-snug transition-colors {LEVEL_STYLES[e.level] ?? 'text-exo-light-gray/70'} hover:border-exo-yellow/40"
              >
                <div class="flex items-center gap-1.5">
                  <button
                    type="button"
                    onclick={() => toggleError(e)}
                    class="flex-1 flex items-center gap-1.5 text-left cursor-pointer"
                    title={isExpanded ? "Click to collapse" : "Click to expand full error"}
                  >
                    <span class="uppercase font-bold">{e.level}</span>
                    <span class="text-exo-light-gray/50 truncate">{e.source ?? "node"}</span>
                    <span class="ml-auto flex-shrink-0 text-exo-light-gray/40" aria-hidden="true">
                      {isExpanded ? "▾" : "▸"}
                    </span>
                  </button>
                  <button
                    type="button"
                    onclick={() => dismissError(e)}
                    class="flex-shrink-0 p-0.5 text-exo-light-gray/30 hover:text-red-400 transition-colors cursor-pointer"
                    title="Dismiss this warning"
                  >
                    <svg class="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                <button
                  type="button"
                  onclick={() => toggleError(e)}
                  class="w-full text-left cursor-pointer"
                  title={isExpanded ? "Click to collapse" : "Click to expand full error"}
                >
                  <div class={isExpanded ? "whitespace-pre-wrap break-words" : "truncate"}>
                    {e.message}
                  </div>
                  {#if isExpanded}
                    {#if e.source}
                      <div class="mt-1 pt-1 border-t border-white/10 text-exo-light-gray/40">
                        {e.source}
                      </div>
                    {/if}
                    {#if e.timestamp}
                      <div class="mt-0.5 text-exo-light-gray/30">
                        {new Date(e.timestamp).toLocaleString()}
                      </div>
                    {/if}
                  {/if}
                </button>
              </div>
            {/each}
          {/if}
        </div>
      {/if}
    </div>

    <!-- Draggable divider between errors and log tail -->
    {#if errorsExpanded && tailExpanded}
      <div
        class="h-1 flex-shrink-0 cursor-row-resize hover:bg-exo-yellow/20 active:bg-exo-yellow/30 transition-colors flex items-center justify-center select-none"
        onpointerdown={onDividerPointerDown}
        role="separator"
        aria-label="Resize errors and logs sections"
        title="Drag to resize"
      >
        <div class="w-8 h-0.5 rounded-full bg-exo-light-gray/20 {dragging ? 'bg-exo-yellow/40' : ''}"></div>
      </div>
    {/if}

    <!-- Log tail section (collapsible) -->
    <div class="flex-shrink-0 min-h-0">
      <button
        type="button"
        onclick={toggleTailSection}
        class="w-full px-2 py-1 flex items-center justify-between text-[9px] font-mono uppercase tracking-widest text-exo-light-gray/50 hover:text-exo-light-gray/70 hover:bg-white/[0.02] transition-colors cursor-pointer"
      >
        <span class="flex items-center gap-1">
          <span class="text-[8px]">{tailExpanded ? "▾" : "▸"}</span>
          Log Tail
        </span>
      </button>

      {#if tailExpanded}
        <div class="px-2 pb-1 overflow-y-auto" style="max-height: {280 - errorsSectionHeight}px; min-height: 40px;">
          <pre
            class="text-[9px] font-mono leading-tight text-exo-light-gray/60 whitespace-pre-wrap break-words m-0"
          >{tailLoading ? "Loading…" : tailContent}</pre>
        </div>
      {/if}
    </div>
  </div>
{/if}
