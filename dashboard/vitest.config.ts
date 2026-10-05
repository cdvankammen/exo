// Vitest config for the exo dashboard.
//
// Why this file exists: the dashboard test suite was written against a jsdom
// environment (src/lib/testing/setup.ts patches window/globalThis storage, and
// every component/store test calls @testing-library/svelte). There was no
// vitest config at all, so vitest fell back to its default `node`
// environment and every DOM-dependent test died with "window is not defined".
//
// Two non-obvious requirements, both found the hard way — see §1 and §2.
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { defineConfig } from "vitest/config";

const root = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [svelte()],

  // §1 — MUST be the object form, not an array of {find, replacement}.
  // With the array form the `$lib/*` aliases silently do not register: every
  // spec still loads, but `$lib/types/api` fails with
  // "Failed to resolve import ... from src/lib/utils/api-errors.ts".
  // Object form is longest-prefix-wins, which is also what makes the
  // `$app/environment` alias win over the broader `$app` one.
  resolve: {
    alias: {
      "$app/environment": path.resolve(root, "src/lib/testing/app-env.ts"),
      $app: path.resolve(root, "src/lib/testing"),
      $lib: path.resolve(root, "src/lib"),
      $components: path.resolve(root, "src/lib/components"),
    },

    // §2 — MUST be present. Vitest transforms test modules through Vite's SSR
    // pipeline, which resolves Svelte's "node"/server condition. That makes
    // `import { mount } from "svelte"` land in svelte/src/index-server.js,
    // whose mount() throws `lifecycle_function_unavailable` — so every
    // @testing-library/svelte render() blew up even with a correct jsdom
    // environment. Forcing the browser condition points Svelte back at
    // index-client.js. Gated on VITEST so `vite build` keeps its own
    // resolution untouched.
    conditions: process.env.VITEST ? ["browser"] : [],
  },

  test: {
    environment: "jsdom",
    setupFiles: ["src/lib/testing/setup.ts"],
    include: ["src/**/*.test.ts"],
    server: {
      deps: {
        inline: [/@testing-library/],
      },
    },
    clearMocks: true,
    restoreMocks: true,
  },
});