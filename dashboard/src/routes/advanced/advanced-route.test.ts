// AdvancedRoute tests: gate semantics of routes/advanced/+page.svelte
// (named advanced-route.test.ts because Vitest rejects `+`-prefixed files:
// SvelteKit reserves the `+` namespace).
//
// The page is a feature-flag access gate: while the one-shot /state refresh
// runs it renders "Loading…"; once loaded it either bounces home (when the
// disaggregation flag is off AND the flag store has settled) or renders the
// Advanced shell. The crucial guard is `flagsSettled` (t_1eca3af5 fix): if
// the initial /v1/feature-flags fetch never succeeded, the page must NOT
// bounce — doing so would strand users outside an enabled-but-unreachable
// tab.
//
// We mock the store module ($lib/stores/app.svelte) with controllable
// featureFlags / featureFlagsLoaded / refreshState so each scenario can be
// exercised in isolation. PrefillDecodeDisaggregation is left real: with the
// store mocked it renders its static explanation details and empty "No
// routes" state, and its 3s polling interval is cleared on unmount.
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, waitFor } from "@testing-library/svelte";

import AdvancedPage from "./+page.svelte";

// ---- module mocks ----------------------------------------------------------

// Shared mutable state driving the mocked store module. Tests set this in
// beforeEach; the mock store functions read it.
const mockStoreState = {
  featureFlags: {} as Record<string, boolean>,
  featureFlagsLoaded: false,
  refreshStateImpl: null as null | (() => Promise<unknown>),
};

vi.mock("$lib/stores/app.svelte", () => ({
  featureFlags: () => mockStoreState.featureFlags,
  featureFlagsLoaded: () => mockStoreState.featureFlagsLoaded,
  refreshState: () => {
    if (mockStoreState.refreshStateImpl) {
      return mockStoreState.refreshStateImpl();
    }
    return Promise.resolve();
  },
  // HeaderNav also imports devMode from the store module.
  devMode: () => false,
  // PrefillDecodeDisaggregation (real child) reads these; empty values
  // render its "No instances / No routes yet" empty states.
  instances: () => ({}),
  instanceLinks: () => ({}),
  nodeIdentities: () => ({}),
}));

// ---- helpers ---------------------------------------------------------------

function mockFlags(flags: Record<string, boolean>, settled: boolean) {
  mockStoreState.featureFlags = flags;
  mockStoreState.featureFlagsLoaded = settled;
}

describe("routes/advanced/+page.svelte gate semantics", () => {
  beforeEach(() => {
    mockStoreState.featureFlags = {};
    mockStoreState.featureFlagsLoaded = false;
    mockStoreState.refreshStateImpl = null;
    // Fresh jsdom location between tests. jsdom normalizes hash to "#/".
    window.location.hash = "";
  });

  it("shows Loading while the state refresh is in flight", async () => {
    // refreshState never resolves until we let it: keep the promise pending
    // so flagsLoaded stays false.
    let resolveRefresh: (() => void) | undefined;
    mockStoreState.refreshStateImpl = () =>
      new Promise<void>((resolve) => {
        resolveRefresh = resolve;
      });

    const { getByText, queryByText } = render(AdvancedPage);
    // onMount fired; still not latched → loading branch.
    expect(getByText(/Loading/)).toBeTruthy();

    resolveRefresh!();
    // After the refresh settles the latch flips → no longer loading.
    await waitFor(() => expect(queryByText(/Loading/)).toBeNull());
  });

  it("renders the no-advanced message and bounces home when disabled and flags settled", async () => {
    mockFlags({ disaggregation: false }, true);

    const { getByText } = render(AdvancedPage);
    // refreshState resolves immediately; loading latch flips on a microtask.
    await waitFor(() =>
      expect(getByText(/No advanced features enabled/)).toBeTruthy(),
    );
    // All three conditions hold (flagsLoaded + flagsSettled + !enabled):
    // the effect must have bounced to the hash root (#/ in jsdom).
    await waitFor(() => expect(window.location.hash).toBe("#/"));
  });

  it("renders the Advanced shell (header, tab bar, editor) when enabled", async () => {
    mockFlags({ disaggregation: true }, true);

    const { getByText } = render(AdvancedPage);
    // Header h1 is unambiguous; the nav also has an "Advanced" link.
    await waitFor(() => expect(getByText("Advanced", { selector: "h1" })).toBeTruthy());
    expect(getByText(/Cluster-level configuration/)).toBeTruthy();
    // Tab bar renders the single tab.
    expect(getByText("Prefill / Decode")).toBeTruthy();
    // The real child renders its prefill-vs-decode explanation.
    expect(getByText("Prefill vs Decode")).toBeTruthy();
  });

  it("does NOT bounce home while flags are unresolved (t_1eca3af5 regression guard)", async () => {
    // Store never got a successful /v1/feature-flags response: flags are
    // empty AND flagsSettled is false — but refreshState completed.
    mockFlags({}, false);
    mockStoreState.refreshStateImpl = () => Promise.resolve();

    const { getByText, queryByText } = render(AdvancedPage);
    // Wait for the refresh latch to flip (loading → non-loading): the
    // disabled message appears because the flags default to off.
    await waitFor(() =>
      expect(getByText(/No advanced features enabled/)).toBeTruthy(),
    );
    // But the page must NOT bounce (pre-fix code without the flagsSettled
    // gate would have bounced here, stranding the user).
    await new Promise<void>((resolve) => setTimeout(resolve, 20));
    expect(window.location.hash).not.toBe("#/");
    // And no enabled shell is rendered.
    expect(queryByText("Advanced", { selector: "h1" })).toBeNull();
  });

  it("bounces ONLY when all three conditions hold (loaded + settled + disabled)", async () => {
    // Condition A: refresh done, flags settled, but enabled → no bounce.
    mockFlags({ disaggregation: true }, true);
    const enabledView = render(AdvancedPage);
    await waitFor(() =>
      expect(enabledView.getByText("Advanced", { selector: "h1" })).toBeTruthy(),
    );
    expect(window.location.hash).not.toBe("#/");
    enabledView.unmount();

    // Condition B: refresh done, flags settled, disabled → bounce.
    window.location.hash = "";
    mockFlags({ disaggregation: false }, true);
    const disabledView = render(AdvancedPage);
    await waitFor(() =>
      expect(disabledView.getByText(/No advanced features enabled/)).toBeTruthy(),
    );
    await waitFor(() => expect(window.location.hash).toBe("#/"));
    disabledView.unmount();

    // Condition C: refresh done, flags NOT settled, disabled → no bounce.
    window.location.hash = "";
    mockFlags({}, false);
    const pendingView = render(AdvancedPage);
    await waitFor(() =>
      expect(pendingView.getByText(/No advanced features enabled/)).toBeTruthy(),
    );
    await new Promise<void>((resolve) => setTimeout(resolve, 20));
    expect(window.location.hash).not.toBe("#/");
    pendingView.unmount();
  });
});