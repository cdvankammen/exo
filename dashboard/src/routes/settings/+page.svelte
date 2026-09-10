<script lang="ts">
  import { browser } from "$app/environment";
  import HeaderNav from "$lib/components/HeaderNav.svelte";

  // ── Types ────────────────────────────────────────────────────────────────
  interface SettingEntry {
    var: string;
    description: string;
    type: "bool" | "int" | "float" | "str";
    requires_restart: boolean;
    has_override: boolean;
    source: "override" | "env" | "default" | null;
    value: boolean | number | string | null;
    raw_value: string | null;
    invalid?: boolean;
  }

  type SectionId = "memory-kv" | "cluster" | "models" | "debug";

  interface Section {
    id: SectionId;
    label: string;
    description: string;
    vars: string[];
  }

  // ── State ────────────────────────────────────────────────────────────────
  let settings = $state<SettingEntry[]>([]);
  let loading = $state(true);
  let error = $state<string | null>(null);
  let activeSection = $state<SectionId>("memory-kv");
  let savingVar = $state<string | null>(null);
  let saveError = $state<string | null>(null);
  let undoFlash = $state<string | null>(null);

  // ── Section definitions ──────────────────────────────────────────────────
  const sections: Section[] = [
    {
      id: "memory-kv",
      label: "Memory / KV-Cache",
      description: "KV cache quantisation, disk persistence, and memory thresholds.",
      vars: [
        "EXO_KV_CACHE_BITS",
        "EXO_KV_CACHE_GROUP_SIZE",
        "EXO_KV_DISK_PERSISTENCE",
        "EXO_KV_DISK_PATH",
        "EXO_KV_DISK_MAX_SIZE_GB",
        "EXO_KV_DISK_TTL_HOURS",
        "EXO_MEMORY_THRESHOLD",
        "EXO_PREFILL_MEMORY_THRESHOLD",
        "EXO_PREFILL_STEP_SIZE",
        "EXO_MAX_CHUNK_SIZE",
      ],
    },
    {
      id: "cluster",
      label: "Cluster / Placement",
      description: "Concurrency, retries, peer discovery, and node identity.",
      vars: [
        "EXO_MAX_CONCURRENT_REQUESTS",
        "EXO_MAX_INSTANCE_RETRIES",
        "EXO_BOOTSTRAP_PEERS",
        "EXO_NODE_ZID",
        "EXO_ZENOH_NAMESPACE",
      ],
    },
    {
      id: "models",
      label: "Models",
      description: "Model directories, image models, and offline mode.",
      vars: [
        "EXO_MODELS_DIRS",
        "EXO_MODELS_READ_ONLY_DIRS",
        "EXO_ENABLE_IMAGE_MODELS",
        "EXO_OFFLINE",
      ],
    },
    {
      id: "debug",
      label: "Debug / Perf",
      description: "Performance toggles, tracing, and tool execution settings.",
      vars: [
        "EXO_DSV4_FUSED_MOE",
        "EXO_NO_BATCH",
        "EXO_FAST_SYNCH",
        "EXO_TRACING_ENABLED",
        "EXO_ENABLE_SERVERSIDE_TOOLCALLS",
      ],
    },
  ];

  // ── Derived ──────────────────────────────────────────────────────────────
  let settingsMap = $derived.by(() => {
    const m: Record<string, SettingEntry> = {};
    for (const s of settings) m[s.var] = s;
    return m;
  });

  let activeSectionData = $derived(
    sections.find((s) => s.id === activeSection)!,
  );

  let currentEntries = $derived(
    activeSectionData.vars
      .map((v) => settingsMap[v])
      .filter(Boolean) as SettingEntry[],
  );

  let anyRequiresRestart = $derived(
    currentEntries.some((e) => e.requires_restart && e.has_override),
  );

  // ── Load settings from API ───────────────────────────────────────────────
  async function loadSettings() {
    if (!browser) return;
    loading = true;
    error = null;
    try {
      const resp = await fetch("/v1/settings");
      if (!resp.ok) throw new Error(`GET /v1/settings ${resp.status}`);
      settings = await resp.json();
    } catch (e) {
      error = String(e);
    } finally {
      loading = false;
    }
  }

  // ── Save a setting ──────────────────────────────────────────────────────
  async function saveSetting(varName: string, value: string | null) {
    savingVar = varName;
    saveError = null;
    try {
      const resp = await fetch("/v1/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ var: varName, value }),
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}));
        throw new Error(body?.error?.message ?? `PUT ${resp.status}`);
      }
      // Reload to reflect new state
      await loadSettings();
      // Flash confirmation
      undoFlash = varName;
      setTimeout(() => {
        if (undoFlash === varName) undoFlash = null;
      }, 1500);
    } catch (e) {
      saveError = String(e);
    } finally {
      savingVar = null;
    }
  }

  // ── Handlers ─────────────────────────────────────────────────────────────
  function handleBoolToggle(entry: SettingEntry) {
    const currentBool =
      entry.value === true ||
      entry.value === "1" ||
      entry.value === "true";
    // If currently on (via any source), turn off (clear override)
    // If currently off or unset, turn on (set "1")
    if (currentBool) {
      saveSetting(entry.var, null); // Clear override
    } else {
      saveSetting(entry.var, "1");
    }
  }

  function handleReset(entry: SettingEntry) {
    saveSetting(entry.var, null);
  }

  let inputBuffer: Record<string, string> = {};
  function getBuffer(entry: SettingEntry): string {
    if (inputBuffer[entry.var] !== undefined) return inputBuffer[entry.var];
    return entry.raw_value ?? "";
  }
  function setBuffer(varName: string, val: string) {
    inputBuffer[varName] = val;
  }
  function submitBuffer(entry: SettingEntry) {
    const val = inputBuffer[entry.var];
    if (val === undefined) return;
    // If empty, clear the override
    if (val.trim() === "") {
      saveSetting(entry.var, null);
    } else {
      saveSetting(entry.var, val.trim());
    }
    // Clear buffer after save
    delete inputBuffer[entry.var];
  }

  // ── Helpers ──────────────────────────────────────────────────────────────
  function sourceLabel(source: string | null): string {
    switch (source) {
      case "override":
        return "Custom";
      case "env":
        return "Env var";
      case "default":
        return "Default";
      default:
        return "Not set";
    }
  }

  function sourceBadgeClass(source: string | null): string {
    switch (source) {
      case "override":
        return "bg-exo-yellow/15 text-exo-yellow border border-exo-yellow/30";
      case "env":
        return "bg-blue-500/15 text-blue-400 border border-blue-500/30";
      case "default":
        return "bg-white/10 text-white/50 border border-white/15";
      default:
        return "bg-white/5 text-white/30 border border-white/10";
    }
  }

  function formatValue(entry: SettingEntry): string {
    if (entry.invalid) return `⚠ ${entry.raw_value} (invalid)`;
    if (entry.value === null || entry.value === undefined) return "—";
    if (entry.type === "bool")
      return entry.value === true || entry.value === "1" ? "On" : "Off";
    if (entry.type === "int") return String(Math.round(Number(entry.value)));
    return String(entry.value);
  }

  function specTypeLabel(type: string): string {
    switch (type) {
      case "bool":
        return "toggle";
      case "int":
        return "integer";
      case "float":
        return "decimal";
      default:
        return "text";
    }
  }

  function needsRestart(entry: SettingEntry): boolean {
    return entry.requires_restart && entry.has_override;
  }

  // ── Init ─────────────────────────────────────────────────────────────────
  $effect(() => {
    if (browser) loadSettings();
  });
</script>

<svelte:head>
  <title>EXO · Settings</title>
</svelte:head>

<div class="min-h-screen bg-exo-dark-gray flex flex-col">
  <HeaderNav showHome={true} />

  <main class="flex-1 max-w-5xl mx-auto w-full px-4 md:px-6 py-6 md:py-10">
    <!-- ── Page header ──────────────────────────────────────────────── -->
    <div class="mb-6">
      <h1
        class="text-white text-xl md:text-2xl font-semibold tracking-wide mb-1"
      >
        Settings
      </h1>
      <p class="text-exo-light-gray/60 text-sm">
        Cluster-level configuration overrides. Changes persist to disk and take
        effect on restart unless marked otherwise.
      </p>
    </div>

    <!-- ── Error banner ─────────────────────────────────────────────── -->
    {#if error}
      <div
        class="border border-red-500/30 text-red-400 p-3 rounded text-sm mb-4"
      >
        Failed to load settings: {error}
        <button
          onclick={loadSettings}
          class="ml-2 underline hover:text-red-300 cursor-pointer"
          >Retry</button
        >
      </div>
    {/if}

    {#if loading}
      <div class="text-exo-light-gray/40 text-sm py-12 text-center">
        Loading settings…
      </div>
    {:else}
      <!-- ── Section tabs ──────────────────────────────────────────── -->
      <div
        class="flex flex-wrap gap-2 mb-6 border-b border-exo-light-gray/10 pb-3"
      >
        {#each sections as section (section.id)}
          <button
            onclick={() => (activeSection = section.id)}
            class="px-3 py-1.5 text-xs rounded-md transition-all cursor-pointer
              {activeSection === section.id
              ? 'bg-exo-yellow/15 text-exo-yellow border border-exo-yellow/30'
              : 'text-exo-light-gray/60 hover:text-white/80 border border-transparent hover:border-exo-light-gray/20'}"
          >
            {section.label}
          </button>
        {/each}
      </div>

      <!-- ── Section description ───────────────────────────────────── -->
      <p class="text-exo-light-gray/50 text-xs mb-4">
        {activeSectionData.description}
      </p>

      <!-- ── Save error ─────────────────────────────────────────────── -->
      {#if saveError}
        <div
          class="border border-red-500/30 text-red-400 p-2 rounded text-xs mb-4"
        >
          Save failed: {saveError}
        </div>
      {/if}

      <!-- ── Settings list ──────────────────────────────────────────── -->
      <div class="space-y-2">
        {#each currentEntries as entry (entry.var)}
          <div
            class="border border-white/10 rounded-lg p-4 bg-white/[0.02] hover:bg-white/[0.04] transition-colors"
          >
            <!-- Row 1: var name + badges + reset -->
            <div class="flex items-start justify-between gap-3 mb-2">
              <div class="flex-1 min-w-0">
                <div class="flex items-center gap-2 flex-wrap">
                  <span class="font-mono text-sm text-exo-light-gray truncate">
                    {entry.var}
                  </span>
                  <span
                    class="text-[10px] px-1.5 py-0.5 rounded {sourceBadgeClass(
                      entry.source,
                    )}"
                  >
                    {sourceLabel(entry.source)}
                  </span>
                  {#if needsRestart(entry)}
                    <span
                      class="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-400 border border-amber-500/30"
                    >
                      Restart needed
                    </span>
                  {/if}
                  {#if entry.invalid}
                    <span
                      class="text-[10px] px-1.5 py-0.5 rounded bg-red-500/15 text-red-400 border border-red-500/30"
                    >
                      Invalid value
                    </span>
                  {/if}
                </div>
                <p class="text-white/50 text-xs mt-0.5">
                  {entry.description}
                  <span class="text-white/30 ml-1"
                    >({specTypeLabel(entry.type)})</span
                  >
                </p>
              </div>

              {#if entry.has_override}
                <button
                  onclick={() => handleReset(entry)}
                  disabled={savingVar === entry.var}
                  class="text-[10px] px-2 py-1 rounded border border-white/15 text-white/40 hover:text-white/70 hover:border-white/30 transition-colors cursor-pointer disabled:opacity-50 flex-shrink-0"
                  title="Reset to env/default"
                >
                  Reset
                </button>
              {/if}
            </div>

            <!-- Row 2: value display + input -->
            {#if entry.type === "bool"}
              <!-- Toggle -->
              <div class="flex items-center gap-3">
                <button
                  onclick={() => handleBoolToggle(entry)}
                  disabled={savingVar === entry.var}
                  class="relative w-10 h-5 rounded-full transition-colors cursor-pointer disabled:opacity-50
                    {entry.value === true ||
                    entry.value === '1' ||
                    entry.value === 'true'
                    ? 'bg-exo-yellow/60'
                    : 'bg-white/15'}"
                  role="switch"
                  aria-checked={entry.value === true ||
                    entry.value === "1" ||
                    entry.value === "true"}
                  aria-label={entry.var}
                >
                  <span
                    class="absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform
                      {entry.value === true ||
                      entry.value === '1' ||
                      entry.value === 'true'
                      ? 'translate-x-5'
                      : ''}"
                  ></span>
                </button>
                <span class="text-xs text-white/50">
                  {entry.value === true ||
                  entry.value === "1" ||
                  entry.value === "true"
                    ? "On"
                    : "Off"}
                </span>
                {#if undoFlash === entry.var}
                  <span class="text-[10px] text-exo-yellow animate-pulse"
                    >Saved</span
                  >
                {/if}
              </div>
            {:else}
              <!-- Text/number input -->
              <div class="flex items-center gap-2">
                <span class="text-xs text-white/40 min-w-[3rem]">
                  Current:
                </span>
                <span
                  class="font-mono text-sm {entry.invalid
                    ? 'text-red-400'
                    : 'text-white/80'}"
                >
                  {formatValue(entry)}
                </span>
              </div>
              <div class="mt-2 flex items-center gap-2">
                <input
                  type={entry.type === "int" || entry.type === "float"
                    ? "text"
                    : "text"}
                  placeholder="New value…"
                  value={getBuffer(entry)}
                  oninput={(e) =>
                    setBuffer(entry.var, (e.target as HTMLInputElement).value)}
                  onkeydown={(e) => {
                    if (e.key === "Enter") submitBuffer(entry);
                    if (e.key === "Escape") {
                      delete inputBuffer[entry.var];
                      (e.target as HTMLInputElement).blur();
                    }
                  }}
                  disabled={savingVar === entry.var}
                  class="flex-1 px-3 py-1.5 bg-white/5 border border-white/15 rounded text-sm font-mono
                    text-white/80 placeholder-white/25 focus:border-exo-yellow/50 focus:outline-none
                    transition-colors disabled:opacity-50"
                />
                <button
                  onclick={() => submitBuffer(entry)}
                  disabled={savingVar === entry.var ||
                    inputBuffer[entry.var] === undefined}
                  class="px-3 py-1.5 text-xs rounded bg-exo-yellow/15 text-exo-yellow border border-exo-yellow/30
                    hover:bg-exo-yellow/25 transition-colors cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  {savingVar === entry.var ? "…" : "Apply"}
                </button>
                {#if undoFlash === entry.var}
                  <span class="text-[10px] text-exo-yellow animate-pulse"
                    >Saved</span
                  >
                {/if}
              </div>
              <p class="text-[10px] text-white/30 mt-1">
                Press Enter to apply, Esc to cancel. Leave blank and Apply to
                reset.
              </p>
            {/if}
          </div>
        {/each}

        {#if currentEntries.length === 0 && !loading}
          <p class="text-white/30 text-sm text-center py-8">
            No settings in this section.
          </p>
        {/if}
      </div>

      <!-- ── Restart reminder ────────────────────────────────────────── -->
      {#if settings.some((s) => needsRestart(s))}
        <div
          class="mt-6 border border-amber-500/30 rounded-lg p-4 bg-amber-500/5"
        >
          <div class="flex items-center gap-2 mb-1">
            <svg
              class="w-4 h-4 text-amber-400 flex-shrink-0"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <span class="text-amber-400 text-sm font-medium">
              Restart required
            </span>
          </div>
          <p class="text-amber-400/70 text-xs">
            Some overrides require the EXO process to restart before they take
            effect. Restart the cluster to apply all pending changes.
          </p>
        </div>
      {/if}

      <!-- ── Legend ──────────────────────────────────────────────────── -->
      <div class="mt-8 border-t border-white/10 pt-4">
        <h3 class="text-white/50 text-xs font-mono uppercase tracking-wider mb-2">
          Source Legend
        </h3>
        <div class="flex flex-wrap gap-3">
          <div class="flex items-center gap-1.5">
            <span
              class="text-[10px] px-1.5 py-0.5 rounded bg-exo-yellow/15 text-exo-yellow border border-exo-yellow/30"
              >Custom</span
            >
            <span class="text-white/40 text-[11px]"
              >User override (persisted)</span
            >
          </div>
          <div class="flex items-center gap-1.5">
            <span
              class="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/15 text-blue-400 border border-blue-500/30"
              >Env var</span
            >
            <span class="text-white/40 text-[11px]"
              >Set via process environment</span
            >
          </div>
          <div class="flex items-center gap-1.5">
            <span
              class="text-[10px] px-1.5 py-0.5 rounded bg-white/10 text-white/50 border border-white/15"
              >Default</span
            >
            <span class="text-white/40 text-[11px]"
              >Built-in fallback value</span
            >
          </div>
          <div class="flex items-center gap-1.5">
            <span
              class="text-[10px] px-1.5 py-0.5 rounded bg-white/5 text-white/30 border border-white/10"
              >Not set</span
            >
            <span class="text-white/40 text-[11px]">No value configured</span>
          </div>
        </div>
      </div>
    {/if}
  </main>
</div>
