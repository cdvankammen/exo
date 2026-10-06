<script lang="ts" module>
  // Shared trace export/Perfetto handlers, used by both trace pages.
  //
  // Extracted verbatim from routes/traces/+page.svelte and
  // routes/traces/[taskId]/+page.svelte (previously duplicated ~50 lines).
  // These functions THROW on failure so the host page can surface the error
  // in its own UI state (detail page: inline error panel; list page: top
  // error banner). Behavior contract (identical to the original
  // implementations):
  //   - downloadTrace: fetch raw trace -> object URL -> programmatic anchor
  //     click (filename `trace_<taskId>.json`) -> revoke URL. Throws on
  //     non-2xx.
  //   - openInPerfetto: fetch raw trace as ArrayBuffer -> open ui.perfetto.dev
  //     -> PING every 50ms until PONG, then postMessage the trace buffer;
  //     give up after 10s; alert when the popup is blocked. Throws on
  //     non-2xx.

  import { getTraceRawUrl } from "$lib/stores/app.svelte";

  export async function downloadTrace(taskId: string): Promise<void> {
    const response = await fetch(getTraceRawUrl(taskId));
    if (!response.ok) {
      throw new Error(`Failed to download trace: ${response.status}`);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trace_${taskId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  export async function openInPerfetto(taskId: string): Promise<void> {
    const response = await fetch(getTraceRawUrl(taskId));
    if (!response.ok) {
      throw new Error(`Failed to fetch trace for Perfetto: ${response.status}`);
    }
    const traceData = await response.arrayBuffer();

    // Open Perfetto UI
    const perfettoWindow = window.open("https://ui.perfetto.dev");
    if (!perfettoWindow) {
      alert("Failed to open Perfetto. Please allow popups.");
      return;
    }

    // Wait for Perfetto to be ready, then send trace via postMessage
    const onMessage = (e: MessageEvent) => {
      if (e.data === "PONG") {
        window.removeEventListener("message", onMessage);
        perfettoWindow.postMessage(
          {
            perfetto: {
              buffer: traceData,
              title: `Trace ${taskId}`,
            },
          },
          "https://ui.perfetto.dev",
        );
      }
    };
    window.addEventListener("message", onMessage);

    // Ping Perfetto until it responds
    const pingInterval = setInterval(() => {
      perfettoWindow.postMessage("PING", "https://ui.perfetto.dev");
    }, 50);

    // Clean up after 10 seconds
    setTimeout(() => {
      clearInterval(pingInterval);
      window.removeEventListener("message", onMessage);
    }, 10000);
  }
</script>