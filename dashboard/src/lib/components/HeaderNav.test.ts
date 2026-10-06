// HeaderNav component tests: the Advanced link visibility must share the
// feature-flag "settled" signal (featureFlagsLoaded) and latch once the
// disaggregation capability has EVER resolved true, so a transient blank
// (failed re-fetch, partial payload) never hides a surface the backend has
// already enabled — and it must appear immediately after flags resolve,
// without requiring a reconnect/refetch cycle.
//
// We use the REAL AppStore singleton (same pattern as app.svelte.test.ts)
// and mutate its public $state fields directly, keeping $derived / $effect
// reactivity real. The singleton starts polling in its constructor, so we
// stop polling in beforeEach and stub global fetch for any stray request.
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { render, waitFor } from "@testing-library/svelte";
import { flushSync } from "svelte";
import HeaderNav from "./HeaderNav.svelte";
import { appStore } from "$lib/stores/app.svelte";

const ADVANCED_TITLE = "Advanced cluster settings";

beforeEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  localStorage.clear();

  // Stop the singleton's polling so the store doesn't refetch/rewrite state
  // mid-test (same discipline as app.svelte.test.ts).
  appStore.stopPolling();
  appStore.isConnected = true;

  // Clean feature-flag slate: unresolved (loaded=false, blank flags).
  flushSync(() => {
    appStore.featureFlags = {};
    (appStore as unknown as { featureFlagsLoaded: boolean }).featureFlagsLoaded =
      false;
  });

  // Stub fetch for anything that still fires (e.g. retry timers scheduled
  // before stopPolling); relative URLs must be made absolute for undici.
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      const abs = url.startsWith("http") ? url : `http://localhost:3000${url}`;
      if (abs.includes("/v1/feature-flags")) {
        return { ok: true, json: async () => ({}) };
      }
      return { ok: true, json: async () => ({}) };
    }),
  );
});

afterEach(() => {
  appStore.stopPolling();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("HeaderNav Advanced link", () => {
  it("is hidden while feature flags are unresolved", () => {
    const { queryByTitle } = render(HeaderNav);
    expect(queryByTitle(ADVANCED_TITLE)).toBeNull();
  });

  it("appears as soon as flags resolve with disaggregation enabled", async () => {
    const { queryByTitle } = render(HeaderNav);
    // Irrelevant while unresolved → still hidden.
    expect(queryByTitle(ADVANCED_TITLE)).toBeNull();

    flushSync(() => {
      appStore.featureFlags = { disaggregation: true };
      (appStore as unknown as { featureFlagsLoaded: boolean }).featureFlagsLoaded =
        true;
    });

    await waitFor(() => expect(queryByTitle(ADVANCED_TITLE)).toBeTruthy());
  });

  it("stays hidden when flags resolve without disaggregation", async () => {
    const { queryByTitle } = render(HeaderNav);

    flushSync(() => {
      appStore.featureFlags = { ring_attention: true };
      (appStore as unknown as { featureFlagsLoaded: boolean }).featureFlagsLoaded =
        true;
    });

    await waitFor(() => expect(queryByTitle(ADVANCED_TITLE)).toBeNull());
  });

  it("latches visible once ever enabled — transient blank is not treated as disabled", async () => {
    const { queryByTitle } = render(HeaderNav);

    flushSync(() => {
      appStore.featureFlags = { disaggregation: true };
      (appStore as unknown as { featureFlagsLoaded: boolean }).featureFlagsLoaded =
        true;
    });
    await waitFor(() => expect(queryByTitle(ADVANCED_TITLE)).toBeTruthy());

    // Simulate a transient blank: re-fetch returns a payload WITHOUT the key
    // while featureFlagsLoaded stays true (store retry semantics keep the
    // loaded signal latched). The nav link must remain visible.
    flushSync(() => {
      appStore.featureFlags = {};
    });
    await waitFor(() => expect(queryByTitle(ADVANCED_TITLE)).toBeTruthy());
  });
});