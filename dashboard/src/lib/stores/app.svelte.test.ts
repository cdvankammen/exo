// Store tests for the EXO dashboard AppStore (Svelte 5 runes).
// Covers fetch/state success + error paths, connection-loss threshold,
// conversation lifecycle, and persisted UI preferences.
import { describe, it, expect, beforeEach, vi } from "vitest";
import {
  appStore,
  conversations,
  activeConversationId,
  createConversation,
  loadConversation,
  deleteConversation,
  renameConversation,
  createInstanceLink,
  updateInstanceLink,
  deleteInstanceLink,
  listLogs,
  getLogTail,
  listLogErrors,
  fetchTraceStats,
  listTraces,
  checkTraceExists,
  deleteTraces,
  getTraceRawUrl,
  pendingQueue,
  removeFromQueue,
  updateQueuedMessage,
  moveQueuedMessage,
  queueParallelism,
  setQueueParallelism,
  clearQueue,
} from "./app.svelte";

// jsdom provides localStorage; SvelteKit's `$app/environment` browser flag is
// resolved by the vitest alias (see vite.config.ts test section) to true.

describe("AppStore conversation lifecycle", () => {
  beforeEach(() => {
    // Fresh store state + clean storage between tests.
    appStore.deleteAllConversations();
    appStore.clearChat();
    localStorage.clear();
  });

  it("createConversation creates a conversation and activates it", () => {
    const id = createConversation("Test Chat");
    expect(id).toBeTruthy();
    const list = conversations();
    expect(list).toHaveLength(1);
    expect(list[0].name).toBe("Test Chat");
    expect(activeConversationId()).toBe(id);
    expect(appStore.hasStartedChat).toBe(true);
  });

  it("loadConversation returns false for unknown ids and true for known ones", () => {
    expect(loadConversation("nope")).toBe(false);
    const id = createConversation("Chat A");
    expect(loadConversation(id)).toBe(true);
    expect(activeConversationId()).toBe(id);
  });

  it("deleteConversation removes a conversation and resets active state", () => {
    const id = createConversation("Chat A");
    deleteConversation(id);
    expect(conversations()).toHaveLength(0);
    expect(activeConversationId()).toBe(null);
    expect(appStore.hasStartedChat).toBe(false);
  });

  it("renameConversation updates the name", () => {
    const id = createConversation("Old Name");
    renameConversation(id, "New Name");
    expect(conversations()[0].name).toBe("New Name");
  });

  it("persists conversations to localStorage and restores on reload", () => {
    const id = createConversation("Persisted");
    const saved = localStorage.getItem("exo-conversations");
    expect(saved).toBeTruthy();
    appStore.deleteAllConversations();
    localStorage.removeItem("exo-conversations");
    // simulate reload by re-setting storage then re-loading via constructor path
    localStorage.setItem("exo-conversations", saved!);
    const reloaded = new (appStore.constructor as new () => typeof appStore)();
    expect(reloaded.conversations).toHaveLength(1);
    expect(reloaded.conversations[0].id).toBe(id);
    expect(reloaded.conversations[0].name).toBe("Persisted");
    reloaded.stopPolling?.();
  });
});

describe("AppStore feature-flag resilience", () => {
  beforeEach(() => {
    appStore.deleteAllConversations();
    appStore.clearChat();
    localStorage.clear();
    vi.restoreAllMocks();
    appStore.stopPolling();
    appStore.isConnected = true;
    // Reset feature-flag state so tests start from a clean slate.
    appStore.featureFlags = {};
    (appStore as unknown as { featureFlagsLoaded: boolean }).featureFlagsLoaded = false;
    (appStore as unknown as { featureFlagsAttempt: number }).featureFlagsAttempt = 0;
    // Shorten retry backoff so tests don't wait real seconds.
    (appStore as unknown as { featureFlagRetryBaseMs: number }).featureFlagRetryBaseMs = 5;
  });

  it("fetchFeatureFlags stores flags and marks the loaded signal", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: true, json: async () => ({ disaggregation: true }) }),
    );
    await appStore.fetchFeatureFlags();
    expect(appStore.featureFlags).toEqual({ disaggregation: true });
    expect((appStore as unknown as { featureFlagsLoaded: boolean }).featureFlagsLoaded).toBe(true);
  });

  it("fetchFeatureFlags retries with exponential backoff on failure, then succeeds", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, status: 500 })
      .mockResolvedValueOnce({ ok: false, status: 503 })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ disaggregation: false }) });
    vi.stubGlobal("fetch", fetchMock);

    await appStore.fetchFeatureFlags();
    // First call failed: attempt counter incremented, loaded still false.
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect((appStore as unknown as { featureFlagsLoaded: boolean }).featureFlagsLoaded).toBe(false);

    // Let the backoff retries fire (5ms base → 10ms → capped).
    await new Promise((resolve) => setTimeout(resolve, 60));
    expect(fetchMock.mock.calls.length).toBeGreaterThanOrEqual(2);
    expect((appStore as unknown as { featureFlagsLoaded: boolean }).featureFlagsLoaded).toBe(true);
    expect(appStore.featureFlags).toEqual({ disaggregation: false });
  });

  it("fetchFeatureFlags keeps retrying on persistent failure and never marks loaded", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 500 });
    vi.stubGlobal("fetch", fetchMock);

    await appStore.fetchFeatureFlags();
    await new Promise((resolve) => setTimeout(resolve, 40));
    expect(fetchMock.mock.calls.length).toBeGreaterThan(1);
    expect((appStore as unknown as { featureFlagsLoaded: boolean }).featureFlagsLoaded).toBe(false);
  });

  it("stopPolling clears a pending feature-flags retry timer", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new Error("network down"));
    vi.stubGlobal("fetch", fetchMock);

    await appStore.fetchFeatureFlags();
    expect((appStore as unknown as { featureFlagsLoaded: boolean }).featureFlagsLoaded).toBe(false);
    appStore.stopPolling();

    const callsAfterStop = fetchMock.mock.calls.length;
    await new Promise((resolve) => setTimeout(resolve, 30));
    expect(fetchMock.mock.calls.length).toBe(callsAfterStop);
  });

  it("fetchState re-fetches feature flags on reconnect after a prior failure", async () => {
    const flagFetch = vi
      .fn()
      .mockResolvedValueOnce({ ok: false, status: 500 })
      .mockResolvedValue({ ok: true, json: async () => ({ disaggregation: true }) });
    const stateFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    });
    // First call is the feature-flags fetch, subsequent are /state.
    const fetchMock = vi
      .fn()
      .mockImplementationOnce(flagFetch)
      .mockImplementation(stateFetch);
    vi.stubGlobal("fetch", fetchMock);

    // Initial flag fetch fails → backoff scheduled.
    await appStore.fetchFeatureFlags();
    expect((appStore as unknown as { featureFlagsLoaded: boolean }).featureFlagsLoaded).toBe(false);

    // Simulate reconnect after connection loss.
    appStore.isConnected = false;
    await appStore.fetchState();
    expect(appStore.isConnected).toBe(true);

    // Reconnect triggers another feature-flags fetch (the flag cache is not loaded).
    await new Promise((resolve) => setTimeout(resolve, 30));
    const flagCalls = fetchMock.mock.calls.filter(([url]) => url === "/v1/feature-flags");
    expect(flagCalls.length).toBeGreaterThanOrEqual(2);
  });
});

describe("AppStore fetchState success path", () => {
  beforeEach(() => {
    appStore.deleteAllConversations();
    appStore.clearChat();
    localStorage.clear();
    vi.restoreAllMocks();
    appStore.stopPolling();
    appStore.isConnected = true;
  });

  it("parses /state into topology nodes and edges", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          topology: {
            nodes: ["node-a", "node-b"],
            connections: {
              "node-a": {
                "node-b": [
                  {
                    sinkMultiaddr: { ip_address: "10.0.0.2", address: "/ip4/10.0.0.2/tcp/4567" },
                    latencyMs: 1.5,
                    bandwidthMbps: 800,
                  },
                ],
              },
            },
          },
          nodeIdentities: {
            "node-a": { modelId: "mlx-community/Llama-3.2-1B", friendlyName: "Alpha", osVersion: "26.6" },
          },
          nodeMemory: {
            "node-a": { ramTotal: { inBytes: 32_000_000_000 }, ramAvailable: { inBytes: 16_000_000_000 } },
          },
          nodeSystem: { "node-a": { gpuUsage: 42.5, temp: 61.2, sysPower: 18.3 } },
          nodeNetwork: { "node-a": { interfaces: [{ name: "en0", ipAddress: "10.0.0.1" }] } },
          nodeDisk: {
            "node-a": { total: { inBytes: 1_000_000_000_000 }, available: { inBytes: 900_000_000_000 } },
          },
          instanceLinks: { link1: { linkId: "link1", prefillInstances: ["a"], decodeInstances: ["b"] } },
          thunderboltBridgeCycles: [["node-a", "node-b"]],
        }),
      }),
    );

    await appStore.fetchState();

    expect(appStore.isConnected).toBe(true);
    expect(appStore.topologyData).not.toBeNull();
    expect(Object.keys(appStore.topologyData!.nodes)).toEqual(["node-a", "node-b"]);
    expect(appStore.topologyData!.edges).toHaveLength(1);
    const edge = appStore.topologyData!.edges[0];
    expect(edge.source).toBe("node-a");
    expect(edge.target).toBe("node-b");
    expect(edge.sendBackIp).toBe("10.0.0.2");
    expect(edge.latency_ms).toBe(1.5);
    expect(edge.bandwidth_mbps).toBe(800);

    const nodeA = appStore.topologyData!.nodes["node-a"];
    expect(nodeA.system_info?.model_id).toBe("mlx-community/Llama-3.2-1B");
    expect(nodeA.system_info?.chip).toBeUndefined();
    expect(nodeA.friendly_name).toBe("Alpha");
    expect(nodeA.macmon_info?.memory?.ram_usage).toBe(16_000_000_000);
    expect(nodeA.macmon_info?.gpu_usage?.[1]).toBe(42.5);
    expect(nodeA.macmon_info?.temp?.gpu_temp_avg).toBe(61.2);

    expect(appStore.instanceLinks["link1"].prefillInstances).toEqual(["a"]);
    expect(appStore.nodeDisk["node-a"].available.inBytes).toBe(900_000_000_000);
    expect(appStore.thunderboltBridgeCycles).toEqual([["node-a", "node-b"]]);
    expect(appStore.lastUpdate).not.toBeNull();
  });

  it("sets instances and derives conversation model info", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          instances: {
            runner1: {
              MlxRingInstance: {
                shardAssignments: {
                  modelId: "mlx-community/Qwen2.5-0.5B",
                  runnerToShard: { r1: { PipelineShardMetadata: {} } },
                },
              },
            },
          },
        }),
      }),
    );

    await appStore.fetchState();
    expect(Object.keys(appStore.instances)).toEqual(["runner1"]);
    // conversation model derivation is exercised via createConversation below
    const id = createConversation();
    const conv = appStore.getActiveConversation();
    expect(conv?.modelId).toBe("mlx-community/Qwen2.5-0.5B");
    expect(conv?.instanceType).toBe("MLX Ring");
    expect(conv?.sharding).toBe("Pipeline");
    appStore.deleteConversation(id);
  });
});

describe("AppStore fetchState error path", () => {
  beforeEach(() => {
    appStore.deleteAllConversations();
    appStore.clearChat();
    localStorage.clear();
    vi.restoreAllMocks();
    appStore.stopPolling();
    appStore.isConnected = true;
  });

  it("keeps isConnected=true for transient failures below the threshold", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network down")));
    await appStore.fetchState();
    await appStore.fetchState();
    expect(appStore.isConnected).toBe(true);
  });

  it("flips isConnected=false after 3 consecutive failures", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network down")));
    await appStore.fetchState();
    await appStore.fetchState();
    await appStore.fetchState();
    expect(appStore.isConnected).toBe(false);
  });

  it("recovers isConnected=true on the next successful fetch", async () => {
    const fail = vi.fn().mockRejectedValue(new Error("down"));
    const ok = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fail);
    await appStore.fetchState();
    await appStore.fetchState();
    expect(appStore.isConnected).toBe(false);
    vi.stubGlobal("fetch", ok);
    await appStore.fetchState();
    expect(appStore.isConnected).toBe(true);
  });

  it("handles non-ok /state responses as failures", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 503 }),
    );
    await appStore.fetchState();
    expect(appStore.isConnected).toBe(true); // one failure only
    await appStore.fetchState();
    await appStore.fetchState();
    expect(appStore.isConnected).toBe(false);
  });
});

describe("AppStore persisted UI preferences", () => {
  beforeEach(() => {
    appStore.deleteAllConversations();
    appStore.clearChat();
    localStorage.clear();
    vi.restoreAllMocks();
    appStore.stopPolling();
  });

  it("setLogsDefaultTab persists and getLogsDefaultTab restores it", () => {
    // invalid values are ignored on load: a fresh store keeps the default
    appStore.setLogsDefaultTab("tail");
    expect(appStore.getLogsDefaultTab()).toBe("tail");
    expect(localStorage.getItem("exo-logs-default-tab")).toBe("tail");
    localStorage.setItem("exo-logs-default-tab", "bogus");
    const reloaded = new (appStore.constructor as new () => typeof appStore)();
    expect(reloaded.getLogsDefaultTab()).toBe("errors"); // default preserved
    reloaded.stopPolling?.();
  });

  it("setImageGenerationParams merges and resets to defaults", () => {
    appStore.setImageGenerationParams({ quality: "high", numImages: 2 });
    const params = appStore.getImageGenerationParams();
    expect(params.quality).toBe("high");
    expect(params.numImages).toBe(2);
    expect(params.size).toBe("auto");
    appStore.resetImageGenerationParams();
    expect(appStore.getImageGenerationParams().quality).toBe("medium");
    expect(appStore.getImageGenerationParams().numImages).toBe(1);
  });

  it("setMemoryOverrideLevel clamps and produces override params", () => {
    appStore.setMemoryOverrideLevel(0);
    expect(appStore.getMemoryOverrideParams()).toEqual({});
    appStore.setMemoryOverrideLevel(1);
    appStore.setMemoryTolerance(0.75);
    expect(appStore.getMemoryOverrideParams()).toEqual({ memory_tolerance: 0.75 });
    appStore.setMemoryOverrideLevel(2);
    expect(appStore.getMemoryOverrideParams()).toEqual({ force_override: true });
    appStore.setMemoryOverrideLevel(99); // clamp
    expect(appStore.getMemoryOverrideLevel()).toBe(2);
    appStore.setMemoryOverrideLevel(-5);
    expect(appStore.getMemoryOverrideLevel()).toBe(0);
  });

  it("sampling params: non-finite and float ints are coerced to ints on load", () => {
    // Simulate a previously persisted object with float topK and string maxTokens
    localStorage.setItem(
      "exo-sampling-params-v1",
      JSON.stringify({ topK: 3.5, maxTokens: "512", seed: 42.9 }),
    );
    const reloaded = new (appStore.constructor as new () => typeof appStore)();
    expect(reloaded.samplingParams.topK).toBe(4); // Math.round
    expect(reloaded.samplingParams.maxTokens).toBe(512);
    expect(reloaded.samplingParams.seed).toBe(43);
    reloaded.stopPolling?.();
  });
});

describe("AppStore API helpers", () => {
  beforeEach(() => {
    appStore.deleteAllConversations();
    appStore.clearChat();
    localStorage.clear();
    vi.restoreAllMocks();
    appStore.stopPolling();
  });

  it("createInstanceLink POSTs the correct body", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);
    await createInstanceLink(["pre"], ["dec"]);
    expect(fetchMock).toHaveBeenCalledWith("/v1/instance-links", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prefill_instances: ["pre"], decode_instances: ["dec"] }),
    });
  });

  it("updateInstanceLink PUTs to the link id", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);
    await updateInstanceLink("link-1", ["pre"], ["dec"]);
    expect(fetchMock).toHaveBeenCalledWith("/v1/instance-links/link-1", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prefill_instances: ["pre"], decode_instances: ["dec"] }),
    });
  });

  it("deleteInstanceLink DELETEs the link id", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);
    await deleteInstanceLink("link-1");
    expect(fetchMock).toHaveBeenCalledWith("/v1/instance-links/link-1", {
      method: "DELETE",
    });
  });

  it("listLogs returns parsed log file list", async () => {
    const payload = { logs: [{ name: "main.log", fileSize: 10, modifiedAt: "2026-09-17" }] };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => payload }));
    const result = await listLogs();
    expect(result.logs).toHaveLength(1);
    expect(result.logs[0].name).toBe("main.log");
  });

  it("getLogTail requests the correct lines query", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ name: "main", content: "hello", truncated: false }),
    });
    vi.stubGlobal("fetch", fetchMock);
    await getLogTail("main", 30);
    expect(fetchMock).toHaveBeenCalledWith("/v1/logs/main?lines=30");
  });

  it("listLogErrors includes level param when provided", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ errors: [], truncated: false }),
    });
    vi.stubGlobal("fetch", fetchMock);
    await listLogErrors("ERROR", 100);
    expect(fetchMock).toHaveBeenCalledWith("/v1/logs/errors?lines=100&level=ERROR");
  });

  it("fetchTraceStats returns stats or throws on non-ok", async () => {
    const payload = { taskId: "t1", totalWallTimeUs: 123, byCategory: {}, byRank: {} };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => payload }));
    const stats = await fetchTraceStats("t1");
    expect(stats.taskId).toBe("t1");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 404 }));
    await expect(fetchTraceStats("t1")).rejects.toThrow("404");
  });
});

describe("AppStore trace API helpers", () => {
  beforeEach(() => {
    appStore.deleteAllConversations();
    appStore.clearChat();
    localStorage.clear();
    vi.restoreAllMocks();
    appStore.stopPolling();
  });

  it("listTraces fetches /v1/traces and returns the parsed list", async () => {
    const payload = {
      traces: [
        { taskId: "t1", createdAt: "2026-09-17T00:00:00+00:00", fileSize: 100 },
        { taskId: "t2", createdAt: "2026-09-17T00:00:01+00:00", fileSize: 200 },
      ],
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => payload });
    vi.stubGlobal("fetch", fetchMock);

    const result = await listTraces();

    expect(fetchMock).toHaveBeenCalledWith("/v1/traces");
    expect(result.traces).toHaveLength(2);
    expect(result.traces[0].taskId).toBe("t1");
    expect(result.traces[1].fileSize).toBe(200);
  });

  it("listTraces throws on non-ok responses", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 500 }));
    await expect(listTraces()).rejects.toThrow("500");
  });

  it("listTraces propagates network errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network down")));
    await expect(listTraces()).rejects.toThrow("network down");
  });

  it("checkTraceExists returns true on ok and false on 404", async () => {
    const okMock = vi.fn().mockResolvedValue({ ok: true });
    vi.stubGlobal("fetch", okMock);
    expect(await checkTraceExists("t1")).toBe(true);
    expect(okMock).toHaveBeenCalledWith("/v1/traces/t1");

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 404 }));
    expect(await checkTraceExists("t1")).toBe(false);
  });

  it("checkTraceExists returns false on network errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("down")));
    expect(await checkTraceExists("t1")).toBe(false);
  });

  it("checkTraceExists URL-encodes the task id", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true });
    vi.stubGlobal("fetch", fetchMock);
    await checkTraceExists("my task/id?");
    expect(fetchMock).toHaveBeenCalledWith("/v1/traces/my%20task%2Fid%3F");
  });

  it("deleteTraces POSTs task ids and returns the split", async () => {
    const payload = { deleted: ["t1"], notFound: ["t2"] };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => payload });
    vi.stubGlobal("fetch", fetchMock);

    const result = await deleteTraces(["t1", "t2"]);

    expect(fetchMock).toHaveBeenCalledWith("/v1/traces/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ taskIds: ["t1", "t2"] }),
    });
    expect(result.deleted).toEqual(["t1"]);
    expect(result.notFound).toEqual(["t2"]);
  });

  it("deleteTraces throws on non-ok responses", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 503 }));
    await expect(deleteTraces(["t1"])).rejects.toThrow("503");
  });

  it("getTraceRawUrl returns the raw endpoint with an encoded task id", () => {
    expect(getTraceRawUrl("t1")).toBe("/v1/traces/t1/raw");
    expect(getTraceRawUrl("my task/id?")).toBe("/v1/traces/my%20task%2Fid%3F/raw");
  });
});// ── Message queue (T24/T24b/T25/T26) ────────────────────────────────────────
// The frontend FIFO queue parks messages sent while a generation is in flight
// and drains them when a slot frees up. Tests drive the pure queue mutations
// with a seeded store state so no network is involved.
describe("AppStore message queue (FIFO)", () => {
  beforeEach(() => {
    appStore.deleteAllConversations();
    appStore.clearChat();
    appStore.pendingQueue = [];
    appStore.activeGenerations = 0;
    appStore.queueParallelism = 1; // sequential default for queue tests
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it("queues messages in FIFO order while the gate is saturated", () => {
    // Simulate one in-flight generation with parallelism 1 → gate saturated.
    appStore.activeGenerations = 1;
    createConversation("Q");

    const result = appStore.sendMessage("first");
    // sendMessage is async; the gate branch is synchronous so the queue is
    // populated before the first await. Await a microtask to settle any
    // internal promise churn.
    void result;

    expect(pendingQueue().length).toBe(1);
    expect(pendingQueue()[0].content).toBe("first");

    // Second send while still saturated appends at the tail.
    appStore.activeGenerations = 1;
    void appStore.sendMessage("second");
    expect(pendingQueue().map((m) => m.content)).toEqual(["first", "second"]);
  });

  it("removeFromQueue cancels a specific queued message", () => {
    appStore.pendingQueue = [
      { id: "a", content: "one" },
      { id: "b", content: "two" },
      { id: "c", content: "three" },
    ] as never;

    removeFromQueue("b");

    expect(pendingQueue().map((m) => m.content)).toEqual(["one", "three"]);
  });

  it("clearQueue empties the queue", () => {
    appStore.pendingQueue = [
      { id: "a", content: "one" },
      { id: "b", content: "two" },
    ] as never;

    clearQueue();

    expect(pendingQueue().length).toBe(0);
  });

  it("updateQueuedMessage edits content in place before it is sent", () => {
    appStore.pendingQueue = [
      { id: "a", content: "old" },
      { id: "b", content: "two" },
    ] as never;

    updateQueuedMessage("a", "new");

    expect(pendingQueue().map((m) => m.content)).toEqual(["new", "two"]);
  });

  it("moveQueuedMessage reorders FIFO positions with clamped bounds", () => {
    appStore.pendingQueue = [
      { id: "a", content: "one" },
      { id: "b", content: "two" },
      { id: "c", content: "three" },
    ] as never;

    // Move "two" up one (send sooner).
    moveQueuedMessage(1, -1);
    expect(pendingQueue().map((m) => m.content)).toEqual(["two", "one", "three"]);

    // Move "one" down (send later).
    moveQueuedMessage(1, 1);
    expect(pendingQueue().map((m) => m.content)).toEqual(["two", "three", "one"]);

    // Out-of-bounds moves are no-ops.
    const before = pendingQueue().map((m) => m.content);
    moveQueuedMessage(0, -1);
    moveQueuedMessage(2, 1);
    expect(pendingQueue().map((m) => m.content)).toEqual(before);
  });

  it("queueParallelism clamps to 1..8 and persists", () => {
    setQueueParallelism(4);
    expect(queueParallelism()).toBe(4);
    expect(localStorage.getItem("exo-queue-parallelism-v1")).toBe("4");

    setQueueParallelism(99);
    expect(queueParallelism()).toBe(8);

    setQueueParallelism(0);
    expect(queueParallelism()).toBe(1);

    setQueueParallelism(1);
    expect(queueParallelism()).toBe(1);
  });
});
