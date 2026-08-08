<script lang="ts">
  import { onMount } from "svelte";
  import HeaderNav from "$lib/components/HeaderNav.svelte";

  // ── Live cluster state (detected values — read-only) ──────────────────────
  let state = $state<{
    nodeBackends: Record<string, string[]>;
    nodeMemory: Record<string, { ramTotal: { inBytes: number }; ramAvailable: { inBytes: number } }>;
    topologyNodes: string[];
  }>({ nodeBackends: {}, nodeMemory: {}, topologyNodes: [] });
  let loadError = $state<string | null>(null);

  // ── Editable EXO_* overrides (GET/PUT /v1/settings) ───────────────────────
  type SettingEntry = {
    var: string;
    description: string;
    type: "bool" | "int" | "float" | "str";
    requires_restart: boolean;
    has_override: boolean;
    source: string | null;
    value?: string | number | boolean | null;
    raw_value?: string | null;
    invalid?: boolean;
  };

  let settings: SettingEntry[] = $state([]);
  let settingsError: string | null = $state(null);
  let savingVar: string | null = $state(null);
  let savedMsg: { var: string; ok: boolean; text: string } | null = $state(null);
  let drafts: Record<string, string> = $state({});

  async function loadSettings() {
    try {
      const resp = await fetch("/v1/settings");
      if (!resp.ok) throw new Error(`settings ${resp.status}`);
      settings = await resp.json();
      settingsError = null;
    } catch (e) {
      settingsError = String(e);
    }
  }

  async function saveSetting(entry: SettingEntry, raw: string) {
    savingVar = entry.var;
    savedMsg = null;
    try {
      const resp = await fetch("/v1/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ var: entry.var, value: raw }),
      });
      if (!resp.ok) {
        const body = await resp.json().catch(() => null);
        throw new Error(body?.error?.message ?? `PUT ${resp.status}`);
      }
      delete drafts[entry.var];
      savedMsg = { var: entry.var, ok: true, text: "Saved" };
      await loadSettings();
    } catch (e) {
      savedMsg = { var: entry.var, ok: false, text: String(e) };
    } finally {
      savingVar = null;
    }
  }

  async function clearSetting(entry: SettingEntry) {
    savingVar = entry.var;
    savedMsg = null;
    try {
      const resp = await fetch("/v1/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ var: entry.var, value: null }),
      });
      if (!resp.ok) throw new Error(`PUT ${resp.status}`);
      delete drafts[entry.var];
      savedMsg = { var: entry.var, ok: true, text: "Cleared (env/default now apply)" };
      await loadSettings();
    } catch (e) {
      savedMsg = { var: entry.var, ok: false, text: String(e) };
    } finally {
      savingVar = null;
    }
  }

  function sourceBadgeClass(source: string | null): string {
    switch (source) {
      case "override":
        return "border-exo-yellow/40 text-exo-yellow";
      case "env":
        return "border-blue-400/40 text-blue-400";
      case "default":
        return "border-white/20 text-white/40";
      default:
        return "border-white/20 text-white/40";
    }
  }

  function draftFor(entry: SettingEntry): string {
    return drafts[entry.var] ?? entry.raw_value ?? "";
  }

  function boolChecked(entry: SettingEntry): boolean {
    const raw = draftFor(entry);
    return raw === "1" || raw.toLowerCase() === "true";
  }

  function toggleBool(entry: SettingEntry) {
    drafts[entry.var] = boolChecked(entry) ? "0" : "1";
  }

  function boolValueLabel(raw: string): string {
    return raw === "1" || raw.toLowerCase() === "true" ? "on" : "off";
  }

  async function loadState() {
    try {
      const resp = await fetch("/state");
      if (!resp.ok) throw new Error(`state ${resp.status}`);
      const d = await resp.json();
      state = {
        nodeBackends: d.nodeBackends ?? {},
        nodeMemory: d.nodeMemory ?? {},
        topologyNodes: (d.topology?.nodes ?? []).map((n: { nodeId?: string } | string) =>
          typeof n === "string" ? n : n.nodeId ?? "",
        ),
      };
      loadError = null;
    } catch (e) {
      loadError = String(e);
    }
  }

  onMount(() => {
    loadState();
    loadSettings();
    const t = setInterval(loadState, 5000);
    return () => clearInterval(t);
  });

  // ── Derived helpers ────────────────────────────────────────────────────────
  function fmtBytes(b: number): string {
    if (!b) return "—";
    const gb = b / 1e9;
    return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(b / 1e6).toFixed(0)} MB`;
  }

  function isVramNode(backends: string[] | undefined): boolean {
    return backends?.includes("MlxCuda") ?? false;
  }

  function backendBadgeClass(b: string): string {
    switch (b) {
      case "MlxMetal":
        return "bg-blue-500/15 text-blue-400 border-blue-500/30";
      case "MlxCuda":
        return "bg-green-500/15 text-green-400 border-green-500/30";
      case "Vllm":
        return "bg-purple-500/15 text-purple-400 border-purple-500/30";
      default:
        return "bg-white/10 text-white/60 border-white/15";
    }
  }

  // ── Launch-env reference (read-only; env vars are process-level) ──────────
  const ENV_REFERENCE: { var: string; meaning: string; suggested: string }[] = [
    { var: "EXO_KV_CACHE_BITS", meaning: "KV cache quantization bits (4/8)", suggested: "4 (default)" },
    { var: "EXO_KV_DISK_PERSISTENCE", meaning: "SSD-as-RAM: persist KV cache to disk", suggested: "1 = on" },
    { var: "EXO_MEMORY_THRESHOLD", meaning: "RAM headroom reserved before prefill", suggested: "e.g. 8GB" },
    { var: "EXO_PREFILL_STEP_SIZE", meaning: "Prefill chunk size in tokens", suggested: "512 (default)" },
    { var: "EXO_MAX_CONCURRENT_REQUESTS", meaning: "API concurrency limit", suggested: "unset = unlimited" },
    { var: "OVERRIDE_MEMORY_MB", meaning: "Force-reported node memory (MB) — e.g. GPU VRAM", suggested: "e.g. 16384" },
    { var: "CUDA_HOME / CUDA_PATH", meaning: "CUDA toolkit location (Linux JIT)", suggested: "/usr/local/cuda" },
    { var: "EXO_MODELS_DIRS", meaning: "Writable model directories", suggested: "~/.cache/exo/models" },
    { var: "EXO_MODELS_READ_ONLY_DIRS", meaning: "Read-only dirs (incl. HF cache)", suggested: "~/.cache/huggingface/hub" },
  ];
</script>

<svelte:head>
  <title>EXO · Settings</title>
</svelte:head>

<div class="min-h-screen bg-exo-dark-gray text-white">
  <HeaderNav showHome={true} />
  <div class="p-6 max-w-5xl mx-auto">
  <h1 class="text-2xl font-mono tracking-wider text-exo-light-gray mb-1">SETTINGS</h1>
  <p class="text-white/50 text-sm mb-6">
    Live cluster detection, editable <code class="font-mono">EXO_*</code> overrides and launch
    environment reference. Overrides persist to
    <code class="font-mono">~/.exo/settings.json</code> and take precedence over launch env
    vars (which still override built-in defaults).
  </p>

  {#if loadError}
    <div class="border border-red-500/30 text-red-400 p-3 rounded text-sm mb-4">
      Failed to load cluster state: {loadError}
    </div>
  {/if}

  <!-- ═══ Per-node detection ═══ -->
  <section class="mb-8">
    <h2 class="text-exo-yellow font-mono text-sm tracking-wider mb-3">NODE DETECTION</h2>
    <div class="grid gap-3 md:grid-cols-3">
      {#each state.topologyNodes as nodeId (nodeId)}
        {@const backends = state.nodeBackends[nodeId] ?? []}
        {@const mem = state.nodeMemory[nodeId]}
        <div class="border border-white/10 rounded-lg p-4 bg-white/[0.02]">
          <div class="flex items-center justify-between mb-2">
            <span class="font-mono text-sm text-exo-light-gray">{nodeId.slice(0, 8)}…</span>
            {#if isVramNode(backends)}
              <span class="text-[10px] px-2 py-0.5 rounded border border-green-500/40 text-green-400 uppercase tracking-wider">GPU VRAM</span>
            {:else}
              <span class="text-[10px] px-2 py-0.5 rounded border border-white/20 text-white/50 uppercase tracking-wider">System RAM</span>
            {/if}
          </div>
          <div class="flex flex-wrap gap-1 mb-3">
            {#each backends as b (b)}
              <span class="text-[10px] px-2 py-0.5 rounded border font-mono {backendBadgeClass(b)}">{b}</span>
            {/each}
            {#if backends.length === 0}<span class="text-white/30 text-xs">…</span>{/if}
          </div>
          <div class="text-xs text-white/60 space-y-0.5 font-mono">
            <div class="flex justify-between"><span>total</span><span>{fmtBytes(mem?.ramTotal?.inBytes)}</span></div>
            <div class="flex justify-between"><span>available</span><span>{fmtBytes(mem?.ramAvailable?.inBytes)}</span></div>
          </div>
          <p class="text-[10px] text-white/40 mt-2 leading-relaxed">
            {#if isVramNode(backends)}
              Advertises CUDA GPU memory — placement treats this node as an accelerator.
            {:else}
              Advertises system memory — CPU/Metal node.
            {/if}
          </p>
        </div>
      {/each}
      {#if state.topologyNodes.length === 0 && !loadError}
        <p class="text-white/40 text-sm col-span-full">Waiting for cluster state…</p>
      {/if}
    </div>
  </section>

  <!-- ═══ Editable EXO_* overrides ═══ -->
  <section class="mb-8">
    <h2 class="text-exo-yellow font-mono text-sm tracking-wider mb-3">ENVIRONMENT OVERRIDES</h2>
    {#if settingsError}
      <div class="border border-red-500/30 text-red-400 p-3 rounded text-sm mb-4">
        Failed to load settings: {settingsError}
      </div>
    {:else if settings.length === 0}
      <p class="text-white/40 text-sm">Loading settings…</p>
    {:else}
      <div class="border border-white/10 rounded-lg overflow-hidden">
        <table class="w-full text-sm">
          <thead>
            <tr class="text-left text-white/50 text-xs uppercase tracking-wider border-b border-white/10">
              <th class="px-4 py-2 font-mono">Variable</th>
              <th class="px-4 py-2">Meaning</th>
              <th class="px-4 py-2">Source</th>
              <th class="px-4 py-2">Value</th>
              <th class="px-4 py-2">Action</th>
            </tr>
          </thead>
          <tbody>
            {#each settings as entry (entry.var)}
              <tr class="border-b border-white/5 last:border-0 hover:bg-white/[0.02]">
                <td class="px-4 py-2 font-mono text-exo-light-gray whitespace-nowrap">
                  {entry.var}
                  {#if entry.requires_restart}
                    <span
                      class="ml-1 text-[9px] px-1.5 py-0.5 rounded border border-white/15 text-white/40 uppercase"
                      title="Takes effect on the next node/runner start">restart</span>
                  {/if}
                </td>
                <td class="px-4 py-2 text-white/70">{entry.description}</td>
                <td class="px-4 py-2">
                  <span class="text-[10px] px-2 py-0.5 rounded border uppercase tracking-wider {sourceBadgeClass(entry.source)}">
                    {entry.source ?? "unset"}
                  </span>
                </td>
                <td class="px-4 py-2">
                  {#if entry.type === "bool"}
                    <button
                      class="px-3 py-1 rounded border text-xs font-mono {boolChecked(entry)
                        ? 'border-green-500/40 text-green-400 bg-green-500/10'
                        : 'border-white/20 text-white/40'}"
                      onclick={() => toggleBool(entry)}>{boolValueLabel(draftFor(entry))}</button>
                  {:else}
                    <input
                      type="text"
                      value={draftFor(entry)}
                      placeholder={entry.type === "int" ? "integer" : "text"}
                      class="bg-white/[0.04] border border-white/15 rounded px-2 py-1 text-xs font-mono text-exo-light-gray w-48 focus:outline-none focus:border-exo-yellow/50"
                      oninput={(e) => (drafts[entry.var] = (e.currentTarget as HTMLInputElement).value)}
                    />
                  {/if}
                  {#if entry.invalid}
                    <span class="text-[10px] text-red-400 ml-1">invalid current value</span>
                  {/if}
                </td>
                <td class="px-4 py-2 whitespace-nowrap">
                  <button
                    class="px-2.5 py-1 rounded border border-exo-yellow/40 text-exo-yellow text-xs hover:bg-exo-yellow/10 disabled:opacity-40"
                    disabled={savingVar === entry.var}
                    onclick={() => saveSetting(entry, draftFor(entry))}>Set</button>
                  {#if entry.has_override}
                    <button
                      class="ml-1 px-2.5 py-1 rounded border border-white/20 text-white/50 text-xs hover:bg-white/5 disabled:opacity-40"
                      disabled={savingVar === entry.var}
                      onclick={() => clearSetting(entry)}>Clear</button>
                  {/if}
                </td>
              </tr>
              {#if savedMsg && savedMsg.var === entry.var}
                <tr class="border-b border-white/5">
                  <td colspan="5" class="px-4 py-1 text-xs {savedMsg.ok ? 'text-green-400' : 'text-red-400'}">
                    {savedMsg.text}
                  </td>
                </tr>
              {/if}
            {/each}
          </tbody>
        </table>
      </div>
      <p class="text-[11px] text-white/35 mt-2">
        Precedence: override (this page) &gt; launch env var &gt; built-in default. Most knobs apply to
        new runners — restart the node for a full effect.
      </p>
    {/if}
  </section>

  <!-- ═══ Launch env reference ═══ -->
  <section>
    <h2 class="text-exo-yellow font-mono text-sm tracking-wider mb-3">LAUNCH ENVIRONMENT</h2>
    <div class="border border-white/10 rounded-lg overflow-hidden">
      <table class="w-full text-sm">
        <thead>
          <tr class="text-left text-white/50 text-xs uppercase tracking-wider border-b border-white/10">
            <th class="px-4 py-2 font-mono">Variable</th>
            <th class="px-4 py-2">Meaning</th>
            <th class="px-4 py-2">Typical value</th>
          </tr>
        </thead>
        <tbody>
          {#each ENV_REFERENCE as row (row.var)}
            <tr class="border-b border-white/5 last:border-0 hover:bg-white/[0.02]">
              <td class="px-4 py-2 font-mono text-exo-light-gray whitespace-nowrap">{row.var}</td>
              <td class="px-4 py-2 text-white/70">{row.meaning}</td>
              <td class="px-4 py-2 text-white/50 font-mono">{row.suggested}</td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
    <p class="text-[11px] text-white/35 mt-2">
      Full inventory (incl. advanced knobs) tracked in
      <code class="font-mono">promptsANDplans/research/settings-manager-design.md</code>.
      Linux GPU troubleshooting: <code class="font-mono">scripts/setup_linux_gpu.sh</code>.
    </p>
  </section>
  </div>
</div>
