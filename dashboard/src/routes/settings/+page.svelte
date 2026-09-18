<script lang="ts">
  import { onMount } from "svelte";
  import HeaderNav from "$lib/components/HeaderNav.svelte";
  import { appStore } from "$lib/stores/app.svelte";

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

  // ── Placement guardrails (src/exo/master/placement.py) ────────────────────
  // Every guardrail can be bypassed by enabling FORCE in the Memory Override
  // section above (sets force_override=True on the PlaceInstance command).
  type Guardrail = {
    id: number;
    name: string;
    category: "Ring" | "Tensor" | "Pipeline" | "Memory" | "Backend" | "Topology";
    description: string;
    errorMessage: string;
    bypassable: boolean;
  };

  const GUARDRAILS: Guardrail[] = [
    {
      id: 1, name: "Ring transport", category: "Ring",
      description: "Ring attention requires the MlxRing transport engine. Cannot be used with MlxJaccl.",
      errorMessage: "Ring attention requires the MlxRing transport",
      bypassable: true,
    },
    {
      id: 2, name: "Ring minimum nodes", category: "Ring",
      description: "Ring attention requires at least 2 nodes to form a ring buffer.",
      errorMessage: "Ring attention requires at least two nodes",
      bypassable: true,
    },
    {
      id: 3, name: "Ring model support", category: "Ring",
      description: "Model card must declare Ring attention support (supports_ring=true). Only verified for dense LlamaForCausalLM and Qwen3ForCausalLM architectures.",
      errorMessage: "Model does not declare Ring attention support",
      bypassable: true,
    },
    {
      id: 4, name: "Manual layer allocation", category: "Pipeline",
      description: "Per-node layer counts (node_layers) are only valid for Pipeline sharding. Tensor/Ring handle layer distribution automatically.",
      errorMessage: "Manual layer allocation requires Pipeline sharding",
      bypassable: false,
    },
    {
      id: 5, name: "Exact node match", category: "Topology",
      description: "When node_ids are selected in the UI, only cycles whose node set exactly matches are considered. No supersets.",
      errorMessage: "(silently filters cycles — no matching cycle means placement fails)",
      bypassable: false,
    },
    {
      id: 6, name: "Ring memory admission", category: "Memory",
      description: "Every ring rank replicates the full model weights plus KV cache working set. Each node must hold the full model size.",
      errorMessage: "No cycles found with sufficient memory",
      bypassable: true,
    },
    {
      id: 7, name: "Pipeline/Tensor memory", category: "Memory",
      description: "Total cycle memory must exceed the model storage size. For Pipeline: layers split across nodes. For Tensor: weights split per node.",
      errorMessage: "No cycles found with sufficient memory",
      bypassable: true,
    },
    {
      id: 8, name: "Tensor model support", category: "Tensor",
      description: "Model card must declare supports_tensor=true. Architecture-gated in ConfigData.supports_tensor (e.g. LlamaForCausalLM, Qwen3NextForCausalLM, DeepseekV4ForCausalLM).",
      errorMessage: "Requested Tensor sharding but this model does not support tensor parallelism",
      bypassable: true,
    },
    {
      id: 9, name: "Tensor hidden_size divisibility", category: "Tensor",
      description: "hidden_size must be divisible by the number of nodes in the cycle. e.g. hidden_size=2048 works on 2/4/8 nodes but not 3.",
      errorMessage: "No tensor sharding found for model with hidden_size=…",
      bypassable: true,
    },
    {
      id: 10, name: "Tensor kv_heads divisibility", category: "Tensor",
      description: "num_key_value_heads must be divisible by node count (MQA models like DeepSeek V4 are exempt — they shard MoE experts instead of KV heads).",
      errorMessage: "No tensor sharding found … num_key_value_heads=…",
      bypassable: true,
    },
    {
      id: 11, name: "DeepSeek V3.1 Pipeline block", category: "Pipeline",
      description: "DeepSeek-V3.1-8bit specifically blocks Pipeline parallelism due to quantization constraints. Use Tensor.",
      errorMessage: "Pipeline parallelism is not supported for DeepSeek V3.1 (8-bit)",
      bypassable: true,
    },
    {
      id: 12, name: "Gemma 4 Pipeline block", category: "Pipeline",
      description: "Gemma 4 models restrict Pipeline to single-node cycles. Multi-node Gemma 4 requires Tensor sharding.",
      errorMessage: "Pipeline parallelism is not supported for Gemma 4; use tensor parallelism instead",
      bypassable: true,
    },
    {
      id: 13, name: "Backend compatibility", category: "Backend",
      description: "The model's declared backends must overlap with the instance engine's required backends (e.g. MlxRing needs MlxMetal/MlxCuda/MlxCpu).",
      errorMessage: "Model backends cannot satisfy engine",
      bypassable: true,
    },
    {
      id: 14, name: "Backend cycle filter", category: "Backend",
      description: "Every node in the selected cycle must support at least one of the required backends. A single node without a matching backend disqualifies the cycle.",
      errorMessage: "No cycle where every node supports a backend in …",
      bypassable: true,
    },
    {
      id: 15, name: "RDMA for MlxJaccl", category: "Backend",
      description: "MlxJaccl (RDMA transport) requires every node in the cycle to be RDMA-connected AND have rdma_ctl enabled with a verbs device enumerated.",
      errorMessage: "Requested RDMA (MlxJaccl) but no RDMA-connected cycles available",
      bypassable: true,
    },
    {
      id: 16, name: "Single-node multi-sharding", category: "Topology",
      description: "Tensor, Ring, and MlxJaccl all require at least 2 nodes. If only a single-node cycle is viable, placement forces Pipeline/Ring rewrite (or fails if force_override is off).",
      errorMessage: "… requires at least 2 nodes, but only a single-node cycle is available",
      bypassable: true,
    },
  ];

  const guardrailCategories = ["Ring", "Tensor", "Pipeline", "Memory", "Backend", "Topology"] as const;

  function categoryColor(cat: Guardrail["category"]): string {
    switch (cat) {
      case "Ring": return "border-purple-500/40 text-purple-400 bg-purple-500/10";
      case "Tensor": return "border-blue-500/40 text-blue-400 bg-blue-500/10";
      case "Pipeline": return "border-green-500/40 text-green-400 bg-green-500/10";
      case "Memory": return "border-exo-yellow/40 text-exo-yellow bg-exo-yellow/10";
      case "Backend": return "border-orange-500/40 text-orange-400 bg-orange-500/10";
      case "Topology": return "border-cyan-500/40 text-cyan-400 bg-cyan-500/10";
    }
  }
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

  <!-- ═══ Memory override (LM Studio-style tiering) ═══ -->
  <section class="mb-8">
    <h2 class="text-exo-yellow font-mono text-sm tracking-wider mb-3">MEMORY OVERRIDE</h2>
    <div class="border border-white/10 rounded-lg p-4 bg-white/[0.02] space-y-4">
      <p class="text-[11px] text-white/50 leading-relaxed">
        Controls how strictly placement checks available memory before loading a
        model. Higher levels let you push past the reported limits — the model
        may load slowly or fail if the hardware truly can't hold it.
      </p>
      <div class="grid gap-2 md:grid-cols-3">
        <button
          class="text-left border rounded-lg p-3 transition-colors {appStore.getMemoryOverrideLevel() === 0
            ? 'border-exo-yellow/60 bg-exo-yellow/10'
            : 'border-white/10 hover:border-white/30'}"
          onclick={() => appStore.setMemoryOverrideLevel(0)}
        >
          <div class="text-xs font-mono text-exo-light-gray mb-1">AUTO</div>
          <div class="text-[11px] text-white/50">Strict check — model must fit available memory.</div>
        </button>
        <button
          class="text-left border rounded-lg p-3 transition-colors {appStore.getMemoryOverrideLevel() === 1
            ? 'border-exo-yellow/60 bg-exo-yellow/10'
            : 'border-white/10 hover:border-white/30'}"
          onclick={() => appStore.setMemoryOverrideLevel(1)}
        >
          <div class="text-xs font-mono text-exo-light-gray mb-1">RELAXED</div>
          <div class="text-[11px] text-white/50">Admit cycles holding at least the tolerance fraction of the model.</div>
        </button>
        <button
          class="text-left border rounded-lg p-3 transition-colors {appStore.getMemoryOverrideLevel() === 2
            ? 'border-exo-yellow/60 bg-exo-yellow/10'
            : 'border-white/10 hover:border-white/30'}"
          onclick={() => appStore.setMemoryOverrideLevel(2)}
        >
          <div class="text-xs font-mono text-exo-light-gray mb-1">FORCE</div>
          <div class="text-[11px] text-white/50">Bypass memory checks entirely — load the model anyway.</div>
        </button>
      </div>
      {#if appStore.getMemoryOverrideLevel() === 1}
        <div class="flex items-center gap-3">
          <label class="text-xs text-white/60 font-mono whitespace-nowrap">
            Tolerance: {Math.round(appStore.getMemoryTolerance() * 100)}%
          </label>
          <input
            type="range"
            min="0.1"
            max="1"
            step="0.05"
            value={appStore.getMemoryTolerance()}
            oninput={(e) => appStore.setMemoryTolerance(parseFloat(e.currentTarget.value))}
            class="flex-1 accent-exo-yellow"
          />
          <span class="text-[10px] text-white/40 font-mono">% of model size a cycle must hold</span>
        </div>
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
                      title="Read when the node starts — save then restart the node for this to take effect">restart</span>
                  {:else}
                    <span
                      class="ml-1 text-[9px] px-1.5 py-0.5 rounded border border-white/15 text-white/40 uppercase"
                      title="Read dynamically at call time — applies to new runners without a node restart">new runners</span>
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
        Precedence: override (this page) &gt; launch env var &gt; built-in default. Knobs marked
        <span class="uppercase text-white/45">restart</span> are read when the node starts — save,
        then restart the node for them to take effect. Knobs marked
        <span class="uppercase text-white/45">new runners</span> are read at call time and apply to
        newly spawned runners without a node restart.
      </p>
    {/if}
  </section>

  <!-- ═══ Placement guardrails ═══ -->
  <section class="mb-8">
    <h2 class="text-exo-yellow font-mono text-sm tracking-wider mb-3">PLACEMENT GUARDRAILS</h2>
    <div class="border border-white/10 rounded-lg p-4 bg-white/[0.02] space-y-3 mb-4">
      <p class="text-[11px] text-white/60 leading-relaxed">
        Every model launch runs through <code class="font-mono text-exo-light-gray">{GUARDRAILS.length}</code> guardrails in
        <code class="font-mono text-exo-light-gray">src/exo/master/placement.py</code>. Each one is a specific check that
        can reject placement with an actionable error message.
      </p>
      <p class="text-[11px] text-white/50 leading-relaxed">
        <span class="text-exo-yellow">Tip:</span> Every guardrail except the hard routing ones (manual layers, exact node match)
        can be bypassed by enabling <span class="font-mono text-exo-light-gray">FORCE</span> in the Memory Override section above.
        Force override sets <code class="font-mono">force_override=true</code> on the PlaceInstance command, which skips memory,
        backend, tensor divisibility, and model-support checks — useful for pushing past reported limits on known-good hardware.
      </p>
    </div>

    <div class="space-y-2">
      {#each guardrailCategories as cat (cat)}
        {@const rails = GUARDRAILS.filter((g) => g.category === cat)}
        {#if rails.length > 0}
          <div class="border border-white/10 rounded-lg overflow-hidden">
            <div class="px-4 py-2 border-b border-white/10 bg-white/[0.02] flex items-center gap-2">
              <span class="text-[10px] px-2 py-0.5 rounded border font-mono uppercase tracking-wider {categoryColor(cat)}">
                {cat}
              </span>
              <span class="text-xs text-white/50">{rails.length} guardrail{rails.length > 1 ? "s" : ""}</span>
            </div>
            <div class="divide-y divide-white/5">
              {#each rails as rail (rail.id)}
                <div class="px-4 py-3 hover:bg-white/[0.02]">
                  <div class="flex items-start justify-between gap-3 mb-1">
                    <div class="flex items-center gap-2">
                      <span class="text-[10px] font-mono text-white/30">#{rail.id}</span>
                      <span class="text-sm font-mono text-exo-light-gray">{rail.name}</span>
                    </div>
                    {#if rail.bypassable}
                      <span class="text-[9px] px-1.5 py-0.5 rounded border border-exo-yellow/30 text-exo-yellow/80 uppercase tracking-wider whitespace-nowrap" title="Can be bypassed with force_override=true">
                        force-overridable
                      </span>
                    {:else}
                      <span class="text-[9px] px-1.5 py-0.5 rounded border border-white/20 text-white/50 uppercase tracking-wider whitespace-nowrap" title="Always enforced — structural requirement">
                        always enforced
                      </span>
                    {/if}
                  </div>
                  <p class="text-xs text-white/60 leading-relaxed mb-1.5">{rail.description}</p>
                  <div class="flex items-start gap-1.5">
                    <span class="text-[9px] font-mono text-red-400/70 uppercase tracking-wider mt-0.5 whitespace-nowrap">error:</span>
                    <code class="text-[10px] font-mono text-red-400/80 break-words">{rail.errorMessage}</code>
                  </div>
                </div>
              {/each}
            </div>
          </div>
        {/if}
      {/each}
    </div>

    <p class="text-[11px] text-white/35 mt-3">
      Source: <code class="font-mono">src/exo/master/placement.py</code> · placement_utils helpers ·
      per-model gates. Full trace available in the logs page when a placement fails.
    </p>
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
