// Test-only stand-in for SvelteKit's `$app/environment`.
// Vitest/jsdom simulates a browser context, so `browser` is always true here.
export const browser = true;
export const building = false;
export const dev = true;
export const version = "test";