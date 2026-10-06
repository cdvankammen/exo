// TraceActions component tests: shared downloadTrace / openInPerfetto
// handlers used by both trace pages.
//
// The functions live in `<script module>` so they are imported directly by
// the pages. They are pure DOM/network mechanics and THROW on failure so the
// host page can surface errors in its own UI state.
import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import {
  downloadTrace,
  openInPerfetto,
} from "./TraceActions.svelte";

const RAW_URL_PATTERN = (taskId: string) =>
  `/v1/traces/${encodeURIComponent(taskId)}/raw`;

beforeEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("TraceActions.downloadTrace", () => {
  it("fetches the raw URL and triggers a same-named anchor download", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      blob: async () => new Blob(["{}"], { type: "application/json" }),
    });
    vi.stubGlobal("fetch", fetchMock);

    const click = vi.fn();
    const createObjectURL = vi.fn(() => "blob:mock");
    const revokeObjectURL = vi.fn();
    // Patch only the URL static methods; keep the rest of URL intact.
    vi.stubGlobal(
      "URL",
      Object.assign(Object.create(URL), {
        createObjectURL,
        revokeObjectURL,
      }),
    );
    // Patch document.createElement to return a tracking anchor; keep the rest
    // of document intact (jsdom provides a real createElement otherwise).
    const originalCreateElement = document.createElement.bind(document);
    const anchor = { href: "", download: "", click };
    const createElement = vi.fn((tag: string) => {
      if (tag === "a") return anchor;
      return originalCreateElement(tag);
    }) as unknown as typeof document.createElement;
    vi.spyOn(document, "createElement").mockImplementation(createElement);

    await downloadTrace("abc-123");

    expect(fetchMock).toHaveBeenCalledWith(RAW_URL_PATTERN("abc-123"));
    expect(createObjectURL).toHaveBeenCalledWith(expect.any(Blob));
    expect(click).toHaveBeenCalledTimes(1);
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:mock");
  });

  it("throws with status message on non-2xx", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 404 });
    vi.stubGlobal("fetch", fetchMock);

    await expect(downloadTrace("missing")).rejects.toThrow(
      "Failed to download trace: 404",
    );
  });
});

describe("TraceActions.openInPerfetto", () => {
  it("fetches, opens ui.perfetto.dev, PINGs until PONG, then posts the buffer", async () => {
    vi.useFakeTimers();
    try {
      const buffer = new ArrayBuffer(8);
      const fetchMock = vi.fn().mockResolvedValue({
        ok: true,
        arrayBuffer: async () => buffer,
      });
      vi.stubGlobal("fetch", fetchMock);

      const postMessage = vi.fn();
      const perfettoWindow = { postMessage } as unknown as Window;
      const openMock = vi.fn().mockReturnValue(perfettoWindow);
      // window.open is an own property on jsdom's window; direct assignment
      // works (spyOn fails because the property isn't on the prototype).
      window.open = openMock;

      const addSpy = vi.spyOn(window, "addEventListener");
      const removeSpy = vi.spyOn(window, "removeEventListener");

      const promise = openInPerfetto("abc-123");
      await Promise.resolve();
      await Promise.resolve();

      expect(fetchMock).toHaveBeenCalledWith(RAW_URL_PATTERN("abc-123"));

      // Advance one tick: the fetch's arrayBuffer() resolves on the microtask
      // queue and openInPerfetto then calls window.open; the 50ms PING
      // interval also fires here.
      await vi.advanceTimersByTimeAsync(50);
      expect(openMock).toHaveBeenCalledWith("https://ui.perfetto.dev");
      expect(postMessage).toHaveBeenCalledWith(
        "PING",
        "https://ui.perfetto.dev",
      );
      const pong = new MessageEvent("message", { data: "PONG" });
      const listener = addSpy.mock.calls
        .find(([type]) => type === "message")?.[1] as EventListener;
      expect(listener).toBeTypeOf("function");
      listener(pong);

      expect(
        postMessage.mock.calls.some(
          ([msg, targetOrigin]) =>
            msg.perfetto?.buffer === buffer &&
            targetOrigin === "https://ui.perfetto.dev",
        ),
      ).toBe(true);
      expect(removeSpy).toHaveBeenCalledWith(
        "message",
        expect.any(Function),
      );

      await promise;
    } finally {
      vi.useRealTimers();
    }
  });

  it("throws with status message on non-2xx", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: false, status: 500 });
    vi.stubGlobal("fetch", fetchMock);

    await expect(openInPerfetto("missing")).rejects.toThrow(
      "Failed to fetch trace for Perfetto: 500",
    );
  });
});