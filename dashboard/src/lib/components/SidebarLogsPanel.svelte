<script lang="ts">
  // Sidebar Logs/Errors panel.
  // Phase 1: static inline error list + log tail.
  // Phase 2: collapsible — collapsed = slim footer-style row with toggle +
  //          error-count badge; expanded = the full panel. Preference
  //          persisted to localStorage (like devMode).
  // Phase 3: resizable height.
  import { onMount, onDestroy } from "svelte";
  import { listLogErrors, getLogTail, type LogErrorEntry } from "$lib/stores/app.svelte";

  const LEVEL_STYLES: Record<string, string> = {
    CRITICAL: "text-red-300 bg-red-500/20 border-red-500/40",
    ERROR: "text-red-400 bg-red-500/10 border-red-500/30",
    WARNING: "text-yellow-300 bg-yellow-500/10 border-yellow-500/30",
  };

  const LEVEL_ORDER = ["CRITICAL", "ERROR", "WARNING"] as const;
  const EXPANDED_KEY = "exo-sidebar-logs-expanded";
  const HEIGHT_KEY = "exo-sidebar-logs-height";

  let errors = $state<LogErrorEntry[]>([]);
  let loading = $state(true);
  let errorMsg = $state<string | null>(null);
  let tailContent = $state<string>("");
  let tailLoading = $state(false);
  let refreshTimer: ReturnType<typeof setInterval> | undefined;
  let expanded = $state(false);
  // Height in px when expanded. Persisted so it survives navigation.
  let panelHeight = $state(256);
  let dragging = $state(false);
  let dragStartY = $state(0);
  let dragStartHeight = $state(0);
  let panelEl = $state<HTMLElement | null>(null);

  const visibleErrors = $derived(
    errors
      .filter((e) => LEVEL_ORDER.includes(e.level as (typeof LEVEL_ORDER)[number]))
      .slice(0, 30),
  );

  const criticalCount = $derived(
    errors.filter((e) => e.level === "CRITICAL" || e.level === "ERROR").length,
  );

  function toggleExpanded() {
    expanded = !expanded;
    try {
      localStorage.setItem(EXPANDED_KEY, expanded ? "1" : "0");
    } catch {
      // private mode — preference just won't persist
    }
    if (expanded) refreshAll();
  }

  // --- Resize (Phase 3) ------------------------------------------------
  // Drag the handle strip to resize. Clamped so the panel never gets tiny
  // (min 96px) and never overlaps the search bar: we measure the panel's
  // top edge and stop the bottom edge from exceeding the sidebar's search
  // header (found via the nearest sibling above). Because the panel sits
  // above the footer, dragging up grows it; dragging down shrinks it.
  function onDragStart(e: PointerEvent) {
    dragging = true;
    dragStartY = e.clientY;
    dragStartHeight = panelHeight;
    e.preventDefault();
  }

  function onDragMove(e: PointerEvent) {
    if (!dragging || !panelEl) return;
    const delta = dragStartY - e.clientY; // drag up → grow
    // Upper bound: don't let the panel's top rise above the search header.
    // Find the conversation list header (search bar) top edge.
    const sidebar = panelEl.closest("aside");
    const searchHeader = sidebar?.querySelector(".p-4") as HTMLElement | null;
    const minTop = searchHeader
      ? searchHeader.getBoundingClientRect().bottom + 8
      : 0;
    const panelTop = panelEl.getBoundingClientRect().top;
    const maxHeight = Math.max(96, panelTop - minTop);
    panelHeight = Math.min(Math.max(96, dragStartHeight + delta), maxHeight);
  }

  function onDragEnd() {
    dragging = false;
    try {
      localStorage.setItem(HEIGHT_KEY, String(Math.round(panelHeight)));
    } catch {
      // ignore
    }
  }

  async function refreshErrors() {
    loading = true;
    errorMsg = null;
    try {
      errors = await listLogErrors();
    } catch (e) {
      errorMsg = e instanceof Error ? e.message : "Failed to load errors";
    } finally {
      loading = false;
    }
  }

  async function refreshTail() {
    tailLoading = true;
    try {
      tailContent = (await getLogTail("main", 30)).content;
    } catch {
      tailContent = "(unavailable)";
    } finally {
      tailLoading = false;
    }
  }

  function refreshAll() {
    refreshErrors();
    refreshTail();
  }

  onMount(() => {
    try {
      expanded = localStorage.getItem(EXPANDED_KEY) === "1";
      const savedHeight = Number(localStorage.getItem(HEIGHT_KEY));
      if (savedHeight >= 96 && savedHeight <= 800) panelHeight = savedHeight;
    } catch {
      // ignore
    }
    refreshAll();
    refreshTimer = setInterval(() => {
      // Only keep polling while expanded — collapsed state doesn't need live data.
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
      LOGS &amp; ERRORS
    </span>
    {#if criticalCount > 0}
      <span class="px-1.5 py-0.5 rounded bg-red-500/20 border border-red-500/40 text-red-300 text-[9px]">
        {criticalCount}
      </span>
    {/if}
  </button>
{:else}
  <div
    bind:this={panelEl}
    class="flex flex-col min-h-0 border-t border-exo-yellow/10 bg-exo-black/30"
    style="height: {panelHeight}px"
  >
    <!-- Resize handle (drag up/down to change height) -->
    <div
      class="h-1.5 flex-shrink-0 cursor-row-resize hover:bg-exo-yellow/30 transition-colors"
      onpointerdown={onDragStart}
      onpointermove={onDragMove}
      onpointerup={onDragEnd}
      onpointerleave={onDragEnd}
      title="Drag to resize"
    ></div>
    <!-- Header -->
    <div class="px-3 py-1.5 flex items-center justify-between">
      <span class="text-[10px] font-mono tracking-widest uppercase text-exo-light-gray/70">
        Cluster Errors
      </span>
      <div class="flex items-center gap-2">
        {#if errors.length > 0}
          <span class="text-[10px] font-mono text-red-400">{errors.length}</span>
        {/if}
        <button
          type="button"
          onclick={toggleExpanded}
          class="text-[10px] font-mono text-exo-light-gray/60 hover:text-exo-yellow transition-colors cursor-pointer"
          title="Collapse logs & errors panel"
        >
          [—]
        </button>
      </div>
    </div>

    <!-- Error list -->
    <div class="flex-1 overflow-y-auto min-h-0 px-2 pb-1 space-y-1">
      {#if loading}
        <div class="text-[10px] text-exo-light-gray/50 font-mono px-1">Loading…</div>
      {:else if errorMsg}
        <div class="text-[10px] text-red-400 font-mono px-1">{errorMsg}</div>
      {:else if visibleErrors.length === 0}
        <div class="text-[10px] text-exo-light-gray/40 font-mono px-1">
          No errors in cluster.
        </div>
      {:else}
        {#each visibleErrors as e (e.id ?? `${e.level}-${e.message}-${e.timestamp}`)}
          <div
            class="px-1.5 py-1 rounded border text-[10px] font-mono leading-snug {LEVEL_STYLES[e.level] ?? 'text-exo-light-gray/70'}"
            title={e.message}
          >
            <div class="flex items-center gap-1.5">
              <span class="uppercase font-bold">{e.level}</span>
              <span class="text-exo-light-gray/50 truncate">{e.source ?? "node"}</span>
            </div>
            <div class="truncate">{e.message}</div>
          </div>
        {/each}
      {/if}
    </div>

    <!-- Log tail -->
    <div class="px-3 py-1 border-t border-exo-yellow/10">
      <div class="text-[10px] font-mono tracking-widest uppercase text-exo-light-gray/70 mb-0.5">
        Log Tail
      </div>
      <pre
        class="text-[9px] font-mono leading-tight text-exo-light-gray/60 whitespace-pre-wrap break-words max-h-16 overflow-y-auto m-0"
      >{tailLoading ? "Loading…" : tailContent}</pre>
    </div>
  </div>
{/if}
