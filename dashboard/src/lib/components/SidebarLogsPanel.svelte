<script lang="ts">
  // Sidebar Logs/Errors panel — Phase 1: static inline error list + log tail.
  // Phases 2/3 will add collapsible + resizable. Shows cluster-aggregated
  // errors (the API already merges across nodes) + the local log tail.
  import { onMount, onDestroy } from "svelte";
  import { listLogErrors, getLogTail, type LogErrorEntry } from "$lib/stores/app.svelte";

  const LEVEL_STYLES: Record<string, string> = {
    CRITICAL: "text-red-300 bg-red-500/20 border-red-500/40",
    ERROR: "text-red-400 bg-red-500/10 border-red-500/30",
    WARNING: "text-yellow-300 bg-yellow-500/10 border-yellow-500/30",
  };

  const LEVEL_ORDER = ["CRITICAL", "ERROR", "WARNING"] as const;

  let errors = $state<LogErrorEntry[]>([]);
  let loading = $state(true);
  let errorMsg = $state<string | null>(null);
  let tailContent = $state<string>("");
  let tailLoading = $state(false);
  let refreshTimer: ReturnType<typeof setInterval> | undefined;

  const visibleErrors = $derived(
    errors
      .filter((e) => LEVEL_ORDER.includes(e.level as (typeof LEVEL_ORDER)[number]))
      .slice(0, 30),
  );

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
    refreshAll();
    refreshTimer = setInterval(refreshAll, 5000);
  });

  onDestroy(() => {
    if (refreshTimer) clearInterval(refreshTimer);
  });
</script>

<div class="flex flex-col h-full min-h-0 border-t border-exo-yellow/10 bg-exo-black/30">
  <!-- Header -->
  <div class="px-3 py-1.5 flex items-center justify-between">
    <span class="text-[10px] font-mono tracking-widest uppercase text-exo-light-gray/70">
      Cluster Errors
    </span>
    {#if errors.length > 0}
      <span class="text-[10px] font-mono text-red-400">{errors.length}</span>
    {/if}
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

  <!-- Log tail (collapsed strip, Phase 2 will fold this into a tab) -->
  <div class="px-3 py-1 border-t border-exo-yellow/10">
    <div class="text-[10px] font-mono tracking-widest uppercase text-exo-light-gray/70 mb-0.5">
      Log Tail
    </div>
    <pre
      class="text-[9px] font-mono leading-tight text-exo-light-gray/60 whitespace-pre-wrap break-words max-h-16 overflow-y-auto m-0"
    >{tailLoading ? "Loading…" : tailContent}</pre>
  </div>
</div>
