// Vitest setup file (runs after jsdom environment is initialized).
//
// Context: Node >= 22.4 ships an experimental `localStorage`/`sessionStorage`
// global (disabled unless --localstorage-file is passed). In Node 26 the
// experimental webstorage integration *strips* localStorage/sessionStorage
// from jsdom's created window, so vitest's jsdom environment ends up with
// neither a global nor a window implementation of Storage — stores that use
// bare `localStorage` see `undefined`.
//
// Fix: provide a spec-shaped in-memory Storage on both window and globalThis.

function createMemoryStorage(): Storage {
  const store = new Map<string, string>();
  return {
    get length() {
      return store.size;
    },
    clear() {
      store.clear();
    },
    getItem(key: string): string | null {
      return store.has(String(key)) ? store.get(String(key))! : null;
    },
    key(index: number): string | null {
      return Array.from(store.keys())[index] ?? null;
    },
    removeItem(key: string) {
      store.delete(String(key));
    },
    setItem(key: string, value: string) {
      store.set(String(key), String(value));
    },
  };
}

if (typeof window !== "undefined") {
  if (typeof window.localStorage === "undefined") {
    Object.defineProperty(window, "localStorage", {
      value: createMemoryStorage(),
      configurable: true,
      writable: true,
    });
  }
  if (typeof window.sessionStorage === "undefined") {
    Object.defineProperty(window, "sessionStorage", {
      value: createMemoryStorage(),
      configurable: true,
      writable: true,
    });
  }
}

// Mirror onto globalThis for code that uses the bare `localStorage` global.
for (const name of ["localStorage", "sessionStorage"]) {
  if (typeof globalThis[name] === "undefined") {
    Object.defineProperty(globalThis, name, {
      value: window[name],
      configurable: true,
      writable: true,
    });
  }
}

// Browser APIs jsdom does not implement but the dashboard components use.
// Without these, mounting a component that touches one throws on render
// instead of failing the assertion that was actually being tested.
if (typeof globalThis.matchMedia !== "function") {
  globalThis.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  })) as typeof globalThis.matchMedia;
}

if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof globalThis.ResizeObserver;
}

if (typeof globalThis.IntersectionObserver === "undefined") {
  globalThis.IntersectionObserver = class IntersectionObserver {
    readonly root = null;
    readonly rootMargin = "";
    readonly thresholds: readonly number[] = [];
    observe() {}
    unobserve() {}
    disconnect() {}
    takeRecords() {
      return [];
    }
  } as unknown as typeof globalThis.IntersectionObserver;
}

// @testing-library/svelte auto-registers its afterEach(cleanup) hook only when a
// GLOBAL `afterEach` already exists at the moment its module is first
// evaluated. Vitest does not expose beforeEach/afterEach as globals unless
// `globals: true` is set, so that auto-cleanup never ran: every render() leaked
// its DOM into the next test, which is how IntegrationCard ended up matching two
// "Copy" buttons and cascaded "Found multiple elements" failures through the
// component suite.
//
// Register the cleanup explicitly instead, so it does not depend on globals.
import { afterEach } from "vitest";
import { act, cleanup } from "@testing-library/svelte";

afterEach(async () => {
  await act();
  cleanup();
});