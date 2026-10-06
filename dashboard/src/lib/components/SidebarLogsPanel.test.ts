// SidebarLogsPanel component tests: collapsed footer, expand,
// error list ordering, dismiss/clear-all, node attribution chips.
//
// The component imports `listLogErrors` / `getLogTail` from the AppStore
// module and calls them on mount. It also imports the store singleton
// (which starts polling at construction, fetching `/state`). In jsdom/node
// relative fetch URLs throw, so we:
//  1. vi.mock the store module to return controlled fixtures, and
//  2. stub global fetch with an absolute-URL passthrough so the AppStore
//     singleton's constructor polling doesn't crash the test file.
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, fireEvent, waitFor } from "@testing-library/svelte";

const ERROR_PAYLOAD = {
  errors: [
    {
      id: "e1",
      timestamp: "2026-09-17T19:00:00Z",
      level: "ERROR",
      source: "api",
      message: "Failed to reach node",
      sourceLog: "ab12cd34::main.log",
      context: ["line1", "line2"],
    },
    {
      id: "e2",
      timestamp: "2026-09-17T19:01:00Z",
      level: "CRITICAL",
      source: "runner",
      message: "OOM on device",
      sourceLog: "runner.log",
    },
    {
      id: "e3",
      timestamp: "2026-09-17T19:02:00Z",
      level: "WARNING",
      source: "placement",
      message: "Slow ring detected",
      sourceLog: "main.log",
    },
  ],
  truncated: false,
};

const TAIL_PAYLOAD = {
  name: "main",
  content: "2026-09-17 19:00:00 INFO started",
  truncated: false,
};

// Mock the store module the component imports. This also prevents the real
// AppStore singleton (module side effect) from being constructed — the
// component only needs the two functions.
vi.mock("$lib/stores/app.svelte", () => ({
  listLogErrors: vi.fn(),
  getLogTail: vi.fn(),
  LogErrorEntry: {},
}));

import SidebarLogsPanel from "./SidebarLogsPanel.svelte";
import { listLogErrors, getLogTail } from "$lib/stores/app.svelte";

let fetchStub: ReturnType<typeof vi.fn>;

beforeEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  localStorage.clear();

  // Default stub: everything resolves to sensible empty data; individual
  // tests override listLogErrors/getLogTail as needed.
  fetchStub = vi.fn(async (url: string) => {
    // Make relative URLs absolute so undici doesn't throw.
    const abs = url.startsWith("http") ? url : `http://localhost:3000${url}`;
    if (abs.includes("/state")) {
      return { ok: true, json: async () => ({}) };
    }
    return { ok: true, json: async () => ({}) };
  });
  vi.stubGlobal("fetch", fetchStub);

  (listLogErrors as ReturnType<typeof vi.fn>).mockResolvedValue(ERROR_PAYLOAD);
  (getLogTail as ReturnType<typeof vi.fn>).mockResolvedValue(TAIL_PAYLOAD);
});

describe("SidebarLogsPanel", () => {
  it("renders collapsed footer row with error count badge", async () => {
    const { getByText, queryByText } = render(SidebarLogsPanel);
    // Collapsed footer shows "MAIN LOGS" and the count of CRITICAL+ERROR
    // (2: e1 ERROR + e2 CRITICAL). WARNING is not counted.
    expect(getByText("MAIN LOGS")).toBeTruthy();
    await waitFor(() => expect(getByText("2")).toBeTruthy());
    // not expanded yet
    expect(queryByText("Warnings & Errors")).toBeNull();
    expect(queryByText("Log Tail")).toBeNull();
  });

  it("expands on header click and shows errors sorted critical-first", async () => {
    const { getByText, getAllByText } = render(SidebarLogsPanel);
    await waitFor(() => expect(getByText("2")).toBeTruthy());
    await fireEvent.click(getByText("MAIN LOGS"));
    await waitFor(() => expect(getByText("Warnings & Errors")).toBeTruthy());
    // CRITICAL appears before ERROR (severity ordering)
    const critical = getAllByText("CRITICAL")[0];
    const error = getAllByText("ERROR")[0];
    expect(critical.compareDocumentPosition(error)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );
  });

  it("shows node attribution chip for remote-node source logs", async () => {
    const { getByText, getAllByText } = render(SidebarLogsPanel);
    await waitFor(() => expect(getByText("2")).toBeTruthy());
    await fireEvent.click(getByText("MAIN LOGS"));
    await waitFor(() => expect(getByText("Warnings & Errors")).toBeTruthy());
    // The chip only renders when the error card is expanded. Click the
    // message to toggle expansion, then the node chip appears.
    await fireEvent.click(getByText("Failed to reach node"));
    await waitFor(() => expect(getByText("node ab12cd34")).toBeTruthy());
    expect(getAllByText("ab12cd34::main.log").length).toBeGreaterThan(0);
  });

  it("clears all errors via Clear All button (becomes 'All warnings dismissed.')", async () => {
    const { getByText } = render(SidebarLogsPanel);
    await waitFor(() => expect(getByText("2")).toBeTruthy());
    await fireEvent.click(getByText("MAIN LOGS"));
    await waitFor(() => expect(getByText("Warnings & Errors")).toBeTruthy());
    await fireEvent.click(getByText("Clear All"));
    await waitFor(() =>
      expect(getByText("All warnings dismissed.")).toBeTruthy(),
    );
  });

  it("collapsed panel shows no errors section when expanded=false maintained from storage", async () => {
    // Force collapsed via the expanded key.
    localStorage.setItem("exo-sidebar-logs-expanded", "0");
    const { getByText, queryByText } = render(SidebarLogsPanel);
    await waitFor(() => expect(getByText("MAIN LOGS")).toBeTruthy());
    // still collapsed: no errors section
    expect(queryByText("Warnings & Errors")).toBeNull();
  });

  it("shows error message when the API call fails", async () => {
    (listLogErrors as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error("network down"),
    );
    const { getByText } = render(SidebarLogsPanel);
    await waitFor(() => expect(getByText("MAIN LOGS")).toBeTruthy());
    await fireEvent.click(getByText("MAIN LOGS"));
    await waitFor(() => expect(getByText("network down")).toBeTruthy());
  });

  it("loads log tail when expanded and shows tail content", async () => {
    const { getByText } = render(SidebarLogsPanel);
    await waitFor(() => expect(getByText("MAIN LOGS")).toBeTruthy());
    await fireEvent.click(getByText("MAIN LOGS"));
    await waitFor(() =>
      expect(getByText(/2026-09-17 19:00:00 INFO started/)).toBeTruthy(),
    );
  });
});