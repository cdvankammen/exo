<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import {
    listLogs,
    getLogTail,
    getLogAll,
    getLogRawUrl,
    listLogErrors,
    nodeIdentities,
    type LogErrorEntry,
    type AllNodesLogResponse,
    type LogFileListItem,
  } from "$lib/stores/app.svelte";
  import HeaderNav from "$lib/components/HeaderNav.svelte";

  const ALL_NODES_KEY = "__all_nodes__";

  // Metadata for the aggregated "Main (All Nodes)" view.
  let allNodesMeta = $state<AllNodesLogResponse["nodes"] | null>(null);

  const LOG_LABELS: Record<string, string> = {
    main: "Main Log",
    runner_stdout: "Runner Stdout",
    runner_stderr: "Runner Stderr",
  };

  const ERROR_LEVELS = ["CRITICAL", "ERROR", "WARNING"] as const;
  const LEVEL_STYLES: Record<string, string> = {
    CRITICAL: "text-red-300 bg-red-500/20 border-red-500/40",
    ERROR: "text-red-400 bg-red-500/10 border-red-500/30",
    WARNING: "text-yellow-300 bg-yellow-500/10 border-yellow-500/30",
  };

  let logs = $state<LogFileListItem[]>([]);
  let selectedName = $state<string | null>(null);
  let content = $state<string>("");
  let truncated = $state(false);
  let loadingList = $state(true);
  let loadingContent = $state(false);
  let error = $state<string | null>(null);
  let autoRefresh = $state(true);

  // Errors view
  let viewMode = $state<"tail" | "errors">("errors");
  let errors = $state<LogErrorEntry[]>([]);
  let errorsTruncated = $state(false);
  let loadingErrors = $state(false);
  let errorLevelFilter = $state<Set<string>>(new Set(["ERROR", "CRITICAL"]));

  // Node filter for errors view: Set of node_id prefixes (first 8 chars).
  // Empty = show all nodes.
  let errorNodeFilter = $state<Set<string>>(new Set());

  // Derive unique node prefixes from error source_log fields.
  // Local entries have source_log like "main" or "runner_stdout".
  // Remote entries are prefixed: "<node_id>::<source_log>".
  const NODE_PREFIX_RE = /^([0-9a-f-]{8})::(.+)$/;
  const uniqueNodePrefixes = $derived(
    [...new Set(
      errors.map((e) => {
        const m = e.sourceLog.match(NODE_PREFIX_RE);
        return m ? m[1] : null;
      }).filter((p): p is string => p !== null),
    )].sort(),
  );

  // Resolve a node_id prefix to a friendly name or fall back to the prefix.
  function resolveNodeName(prefix: string): string {
    const identities = nodeIdentities();
    for (const [nodeId, identity] of Object.entries(identities)) {
      if (nodeId.startsWith(prefix)) {
        return (identity as { friendlyName?: string }).friendlyName || prefix;
      }
    }
    return prefix;
  }

  const filteredErrors = $derived(
    errors.filter((e) => errorLevelFilter.has(e.level)),
  );

  // Node-filtered errors: when filter is empty, show all.
  const nodeFilteredErrors = $derived(
    errorNodeFilter.size === 0
      ? filteredErrors
      : filteredErrors.filter((e) => {
          const m = e.sourceLog.match(NODE_PREFIX_RE);
          const nodePrefix = m ? m[1] : "local";
          return errorNodeFilter.has(nodePrefix);
        }),
  );

  // --- State persistence across page navigation ---------------------------
  // The user asked: when leaving the Logs page (Home, Settings, ...) and
  // coming back, the page should restore the section (Errors/Tail), the
  // selected log file, the level filter, and the scroll position. We use
  // sessionStorage so it survives SPA navigation but resets on a fresh
  // browser session.
  const LOGS_STATE_KEY = "exo-logs-page-state-v1";

  function saveLogsState() {
    try {
      sessionStorage.setItem(
        LOGS_STATE_KEY,
        JSON.stringify({
          viewMode,
          selectedName,
          errorLevelFilter: [...errorLevelFilter],
          stickToBottom,
          scrollTop: logViewerEl?.scrollTop ?? 0,
        }),
      );
    } catch {
      // sessionStorage unavailable (private mode etc.) — state just won't persist.
    }
  }

  function loadLogsState() {
    try {
      const raw = sessionStorage.getItem(LOGS_STATE_KEY);
      if (!raw) return;
      const saved = JSON.parse(raw) as {
        viewMode?: "tail" | "errors";
        selectedName?: string | null;
        errorLevelFilter?: string[];
        stickToBottom?: boolean;
        scrollTop?: number;
      };
      if (saved.viewMode === "tail" || saved.viewMode === "errors") {
        viewMode = saved.viewMode;
      }
      if (typeof saved.selectedName === "string") {
        selectedName = saved.selectedName;
      }
      if (Array.isArray(saved.errorLevelFilter) && saved.errorLevelFilter.length > 0) {
        errorLevelFilter = new Set(saved.errorLevelFilter);
      }
      if (typeof saved.stickToBottom === "boolean") {
        stickToBottom = saved.stickToBottom;
      }
      if (typeof saved.scrollTop === "number" && saved.scrollTop > 0) {
        // Restored after the log content renders (see onMount).
        pendingScrollTop = saved.scrollTop;
      }
    } catch {
      // Corrupt or unavailable storage — start fresh.
    }
  }

  let pendingScrollTop = $state(0);

  // Persist on every state change — hash navigation (/#/logs) does NOT
  // unmount the page, so onDestroy never fires. Saving in an $effect that
  // tracks the view state guarantees the last position is always stored.
  // The first run is skipped so the effect never overwrites the state
  // restored by loadLogsState() in onMount with the pre-restore defaults.
  let persistenceReady = $state(false);
  $effect(() => {
    // Read all tracked state so the effect re-runs on any change.
    const snapshot = {
      viewMode,
      selectedName,
      errorLevelFilter: [...errorLevelFilter],
      stickToBottom,
    };
    if (persistenceReady) {
      saveLogsState();
    }
    void snapshot;
  });

  let refreshTimer: ReturnType<typeof setInterval> | undefined;
  // Auto-scroll: start at the bottom, stay at the bottom while the user is
  // at the bottom, and re-engage when they scroll back down. Manual scroll
  // up pauses it.
  let logViewerEl = $state<HTMLPreElement | null>(null);
  let stickToBottom = $state(true);

  function labelFor(name: string): string {
    return LOG_LABELS[name] ?? name;
  }

  function formatBytes(bytes: number): string {
    if (!bytes || bytes <= 0) return "0B";
    const units = ["B", "KB", "MB", "GB"];
    const i = Math.min(
      Math.floor(Math.log(bytes) / Math.log(1024)),
      units.length - 1,
    );
    const val = bytes / Math.pow(1024, i);
    return `${val.toFixed(val >= 10 ? 0 : 1)}${units[i]}`;
  }

  function formatDate(isoString: string): string {
    return new Date(isoString).toLocaleString();
  }

  async function refreshList() {
    loadingList = true;
    error = null;
    try {
      const response = await listLogs();
      logs = response.logs;
      if (!selectedName && logs.length > 0) {
        selectLog(logs[0].name);
      }
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load logs";
    } finally {
      loadingList = false;
    }
  }

  async function refreshContent() {
    if (!selectedName) return;
    loadingContent = true;
    error = null;
    allNodesMeta = null;
    try {
      if (selectedName === ALL_NODES_KEY) {
        const response = await getLogAll();
        content = response.content;
        truncated = response.truncated;
        allNodesMeta = response.nodes;
      } else {
        const response = await getLogTail(selectedName);
        content = response.content;
        truncated = response.truncated;
      }
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load log content";
    } finally {
      loadingContent = false;
      // Restore the scroll position saved from the previous visit to this
      // page (only meaningful once the content has rendered).
      if (pendingScrollTop > 0 && logViewerEl) {
        logViewerEl.scrollTop = pendingScrollTop;
        pendingScrollTop = 0;
        // Pinning to the bottom would override the restored position.
        stickToBottom = false;
      } else if (stickToBottom && logViewerEl) {
        // Keep the view pinned to the newest lines if the user is at (or
        // near) the bottom (or hasn't scrolled up yet).
        logViewerEl.scrollTop = logViewerEl.scrollHeight;
      }
    }
  }

  function onLogScroll() {
    if (!logViewerEl) return;
    // "At the bottom" = within 40px of the bottom. Scrolling up (reading
    // history) turns stickToBottom off; scrolling back down re-enables it.
    const atBottom =
      logViewerEl.scrollHeight - logViewerEl.scrollTop - logViewerEl.clientHeight <
      40;
    stickToBottom = atBottom;
  }

  function selectLog(name: string) {
    selectedName = name;
    refreshContent();
  }

  async function downloadLog(name: string) {
    const response = await fetch(getLogRawUrl(name));
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${name}.log`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function toggleAutoRefresh() {
    autoRefresh = !autoRefresh;
  }

  async function refreshErrors() {
    loadingErrors = true;
    error = null;
    try {
      const response = await listLogErrors();
      errors = response.errors;
      errorsTruncated = response.truncated;
    } catch (e) {
      error = e instanceof Error ? e.message : "Failed to load log errors";
    } finally {
      loadingErrors = false;
    }
  }

  function toggleErrorLevel(level: string) {
    const next = new Set(errorLevelFilter);
    if (next.has(level)) {
      next.delete(level);
    } else {
      next.add(level);
    }
    errorLevelFilter = next;
  }

  function toggleErrorNodeFilter(prefix: string) {
    const next = new Set(errorNodeFilter);
    if (next.has(prefix)) {
      next.delete(prefix);
    } else {
      next.add(prefix);
    }
    errorNodeFilter = next;
  }

  function setViewMode(mode: "tail" | "errors") {
    viewMode = mode;
    if (mode === "errors" && errors.length === 0) {
      refreshErrors();
    }
  }

  $effect(() => {
    if (refreshTimer) clearInterval(refreshTimer);
    if (autoRefresh) {
      refreshTimer = setInterval(() => {
        if (viewMode === "errors") {
          refreshErrors();
        } else {
          refreshContent();
        }
      }, 3000);
    }
  });

  onMount(() => {
    // Restore the last section/log/filter/scroll position from the previous
    // visit to this page (survives SPA navigation; resets on fresh session).
    loadLogsState();
    // Now that restored state is applied, the persistence effect may save.
    persistenceReady = true;
    refreshList();
    refreshErrors();
  });

  onDestroy(() => {
    if (refreshTimer) clearInterval(refreshTimer);
    // Persist section/log/filter/scroll so returning to Logs restores them.
    saveLogsState();
  });
</script>

<div class="min-h-screen bg-exo-dark-gray text-white">
  <HeaderNav showHome={true} />
  <div class="max-w-7xl mx-auto px-4 lg:px-8 py-6 space-y-6">
    <div class="flex items-center justify-between gap-4 flex-wrap">
      <div>
        <h1
          class="text-2xl font-mono tracking-[0.2em] uppercase text-exo-yellow"
        >
          Logs
        </h1>
        <div class="text-xs text-exo-light-gray/70 font-mono mt-1">
          Logs and errors from all cluster nodes.
        </div>
      </div>
      <div class="flex items-center gap-3">
        <div class="flex rounded border border-exo-medium-gray/40 overflow-hidden">
          <button
            type="button"
            class="text-xs font-mono uppercase px-3 py-1.5 transition-colors {viewMode ===
            'errors'
              ? 'text-exo-yellow bg-exo-yellow/10'
              : 'text-exo-light-gray hover:text-exo-yellow'}"
            onclick={() => setViewMode("errors")}
          >
            Errors
          </button>
          <button
            type="button"
            class="text-xs font-mono uppercase px-3 py-1.5 transition-colors {viewMode ===
            'tail'
              ? 'text-exo-yellow bg-exo-yellow/10'
              : 'text-exo-light-gray hover:text-exo-yellow'}"
            onclick={() => setViewMode("tail")}
          >
            Tail
          </button>
        </div>
        <button
          type="button"
          class="text-xs font-mono uppercase transition-colors border px-2 py-1 rounded {autoRefresh
            ? 'text-exo-yellow border-exo-yellow/40'
            : 'text-exo-light-gray hover:text-exo-yellow border-exo-medium-gray/40'}"
          onclick={toggleAutoRefresh}
        >
          Auto-refresh {autoRefresh ? "on" : "off"}
        </button>
        <button
          type="button"
          class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-2 py-1 rounded"
          onclick={viewMode === "errors" ? refreshErrors : refreshContent}
          disabled={viewMode === "errors"
            ? loadingErrors
            : loadingContent || !selectedName}
        >
          Refresh
        </button>
      </div>
    </div>

    {#if error}
      <div
        class="rounded border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-400"
      >
        {error}
      </div>
    {/if}

    {#if viewMode === "errors"}
      <!-- Level filter chips -->
      <div class="flex items-center gap-2 flex-wrap">
        <span class="text-xs font-mono uppercase text-exo-light-gray/70"
          >Levels:</span
        >
        {#each ERROR_LEVELS as level}
          <button
            type="button"
            class="text-xs font-mono uppercase border px-3 py-1 rounded transition-colors {errorLevelFilter.has(
              level,
            )
              ? (LEVEL_STYLES[level] ?? 'text-exo-yellow border-exo-yellow/40')
              : 'text-exo-light-gray/70 border-exo-medium-gray/30 hover:text-exo-light-gray'}"
            onclick={() => toggleErrorLevel(level)}
          >
            {level}
            <span class="ml-1 normal-case"
              >({errors.filter((e) => e.level === level).length})</span
            >
          </button>
        {/each}
      </div>

      <!-- Node filter chips -->
      {#if uniqueNodePrefixes.length > 0}
        <div class="flex items-center gap-2 flex-wrap">
          <span class="text-xs font-mono uppercase text-exo-light-gray/70"
            >Nodes:</span
          >
          <button
            type="button"
            class="text-xs font-mono uppercase border px-3 py-1 rounded transition-colors {errorNodeFilter.size ===
            0
              ? 'text-exo-yellow border-exo-yellow/40'
              : 'text-exo-light-gray/70 border-exo-medium-gray/30 hover:text-exo-light-gray'}"
            onclick={() => (errorNodeFilter = new Set())}
          >
            All
          </button>
          {#each uniqueNodePrefixes as prefix}
            <button
              type="button"
              class="text-xs font-mono border px-3 py-1 rounded transition-colors {errorNodeFilter.has(
                prefix,
              )
                ? 'text-exo-yellow border-exo-yellow/40'
                : 'text-exo-light-gray/70 border-exo-medium-gray/30 hover:text-exo-light-gray'}"
              onclick={() => toggleErrorNodeFilter(prefix)}
            >
              {resolveNodeName(prefix)}
              <span class="ml-1 normal-case text-exo-light-gray/70"
                >({errors.filter((e) => {
                  const m = e.sourceLog.match(NODE_PREFIX_RE);
                  return m ? m[1] === prefix : false;
                }).length})</span
              >
            </button>
          {/each}
        </div>
      {/if}

      {#if errorsTruncated}
        <div class="text-xs text-exo-light-gray/70 font-mono">
          Scanning the tail of each log file — very old entries may be missed.
        </div>
      {/if}

      {#if loadingErrors}
        <div
          class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-6 text-center text-exo-light-gray"
        >
          <div class="text-sm">Loading errors...</div>
        </div>
      {:else if nodeFilteredErrors.length === 0}
        <div
          class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-6 text-center text-exo-light-gray"
        >
          <div class="text-sm">No matching WARNING/ERROR/CRITICAL entries.</div>
        </div>
      {:else}
        <div class="overflow-x-auto rounded border border-exo-medium-gray/30">
          <table class="w-full text-left text-xs font-mono">
            <thead class="bg-exo-black/50 text-exo-light-gray/70 uppercase text-[10px]">
              <tr>
                <th class="px-3 py-2 font-normal">Time</th>
                <th class="px-3 py-2 font-normal">Level</th>
                <th class="px-3 py-2 font-normal">Source</th>
                <th class="px-3 py-2 font-normal">Message</th>
                <th class="px-3 py-2 font-normal">Log</th>
              </tr>
            </thead>
            <tbody>
              {#each nodeFilteredErrors as entry (entry.timestamp + entry.source + entry.message)}
                <tr
                  class="border-t border-exo-medium-gray/20 hover:bg-exo-yellow/5"
                >
                  <td
                    class="px-3 py-1.5 whitespace-nowrap text-exo-light-gray/70"
                    >{entry.timestamp}</td
                  >
                  <td class="px-3 py-1.5 whitespace-nowrap">
                    <span
                      class="border rounded px-1.5 py-0.5 text-[10px] uppercase {(LEVEL_STYLES[
                        entry.level
                      ] ?? 'text-exo-light-gray border-exo-medium-gray/40')}"
                      >{entry.level}</span
                    >
                  </td>
                  <td class="px-3 py-1.5 whitespace-nowrap text-exo-light-gray/80 max-w-[200px] truncate"
                    title={entry.source}
                    >{entry.source}</td
                  >
                  <td class="px-3 py-1.5 break-words text-white/90 max-w-[520px]"
                    >{entry.message}</td
                  >
                  <td class="px-3 py-1.5 whitespace-nowrap text-exo-light-gray/60"
                    >{labelFor(entry.sourceLog)}</td
                  >
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    {:else if loadingList}
      <div
        class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-6 text-center text-exo-light-gray"
      >
        <div class="text-sm">Loading logs...</div>
      </div>
    {:else if logs.length === 0}
      <div
        class="rounded border border-exo-medium-gray/30 bg-exo-black/30 p-6 text-center text-exo-light-gray"
      >
        <div class="text-sm">No log files found on this node.</div>
      </div>
    {:else}
      <div class="flex gap-2 flex-wrap">
        <button
          type="button"
          class="text-xs font-mono uppercase transition-colors border px-3 py-1.5 rounded {selectedName ===
          ALL_NODES_KEY
            ? 'text-exo-yellow border-exo-yellow/40 bg-exo-yellow/10'
            : 'text-exo-light-gray hover:text-exo-yellow border-exo-medium-gray/40'}"
          onclick={() => selectLog(ALL_NODES_KEY)}
        >
          Main (All Nodes)
        </button>
        {#each logs as log}
          <button
            type="button"
            class="text-xs font-mono uppercase transition-colors border px-3 py-1.5 rounded {selectedName ===
            log.name
              ? 'text-exo-yellow border-exo-yellow/40 bg-exo-yellow/10'
              : 'text-exo-light-gray hover:text-exo-yellow border-exo-medium-gray/40'}"
            onclick={() => selectLog(log.name)}
          >
            {labelFor(log.name)}
            <span class="text-exo-light-gray/60 normal-case"
              >&nbsp;&bull; {formatBytes(log.fileSize)} &bull; {formatDate(
                log.modifiedAt,
              )}</span
            >
          </button>
        {/each}
        {#if selectedName && selectedName !== ALL_NODES_KEY}
          <button
            type="button"
            class="text-xs font-mono text-exo-light-gray hover:text-exo-yellow transition-colors uppercase border border-exo-medium-gray/40 px-3 py-1.5 rounded"
            onclick={() => downloadLog(selectedName!)}
          >
            Download full log
          </button>
        {/if}
      </div>

      {#if allNodesMeta && allNodesMeta.length > 0}
        <div class="text-xs text-exo-light-gray/60 font-mono flex gap-3 flex-wrap">
          {#each allNodesMeta as node}
            <span>{node.nodeId} &bull; {node.lineCount} lines</span>
          {/each}
        </div>
      {/if}

      {#if truncated}
        <div class="text-xs text-exo-light-gray/70 font-mono">
          Showing the tail of this log. Use "Download full log" for the complete
          file.
        </div>
      {/if}

      <pre
        bind:this={logViewerEl}
        onscroll={onLogScroll}
        class="rounded border border-exo-medium-gray/30 bg-exo-black/50 p-4 text-xs font-mono text-exo-light-gray whitespace-pre-wrap break-words overflow-y-auto max-h-[70vh]">{content ||
          (loadingContent ? "Loading..." : "No content.")}</pre>
    {/if}
  </div>
</div>
