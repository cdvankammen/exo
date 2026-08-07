<script lang="ts">
  import { onMount } from "svelte";

  // ── Live cluster state (detected values — read-only) ──────────────────────
  let state = $state<{
    nodeBackends: Record<string, string[]>;
    nodeMemory: Record<string, { ramTotal: { inBytes: number }; ramAvailable: { inBytes: number } }>;
    topologyNodes: string[];
  }>({ nodeBackends: {}, nodeMemory: {}, topologyNodes: [] });
  let loadError = $state<string | null>(null);

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

<div class="p-6 max-w-5xl mx-auto">
  <h1 class="text-2xl font-mono tracking-wider text-exo-light-gray mb-1">SETTINGS</h1>
  <p class="text-white/50 text-sm mb-6">
    Live cluster detection + launch environment reference. Env vars are set at process
    launch (edit the launch script / app config) — this page shows what each node
    actually advertises and how to change the knobs.
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
