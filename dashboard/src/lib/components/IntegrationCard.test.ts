// IntegrationCard component tests: render, copy-to-clipboard flow, states.
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, fireEvent, waitFor } from "@testing-library/svelte";
import IntegrationCard from "./IntegrationCard.svelte";

beforeEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

// copyText (utils/clipboard) first checks `window.isSecureContext &&
// navigator.clipboard?.writeText`; only then falls back to execCommand.
// jsdom's default navigator has no clipboard API, so tests must add it
// explicitly — but must NOT replace the whole navigator, or isSecureContext
// (a window property) is lost.
function stubSecureClipboard(writeTextImpl: () => Promise<unknown>) {
  Object.defineProperty(window, "isSecureContext", {
    value: true,
    configurable: true,
  });
  Object.defineProperty(window.navigator, "clipboard", {
    value: { writeText: vi.fn(writeTextImpl) },
    configurable: true,
  });
}

function stubInsecureNoClipboard() {
  Object.defineProperty(window, "isSecureContext", {
    value: false,
    configurable: true,
  });
  Object.defineProperty(window.navigator, "clipboard", {
    value: undefined,
    configurable: true,
  });
}

describe("IntegrationCard", () => {
  it("renders title, subtitle, description and config block", () => {
    const { getByText } = render(IntegrationCard, {
      props: {
        title: "My Integration",
        subtitle: "v1.2.3",
        description: "A helpful description",
        config: '{"key": "value"}',
      },
    });
    expect(getByText("My Integration")).toBeTruthy();
    expect(getByText("v1.2.3")).toBeTruthy();
    expect(getByText("A helpful description")).toBeTruthy();
    expect(getByText('{"key": "value"}')).toBeTruthy();
  });

  it("omits description paragraph when not provided", () => {
    const { queryByText } = render(IntegrationCard, {
      props: { title: "No Desc", subtitle: "x", config: "{}" },
    });
    expect(queryByText("A helpful description")).toBeNull();
  });

  it("copies config to clipboard on button click — success path", async () => {
    const copyMock = vi.fn().mockResolvedValue(true);
    stubSecureClipboard(copyMock);
    const { getByText, getByRole } = render(IntegrationCard, {
      props: { title: "T", subtitle: "S", config: "payload" },
    });
    await fireEvent.click(getByRole("button", { name: "Copy" }));
    await waitFor(() => expect(getByText("Copied!")).toBeTruthy());
    expect(copyMock).toHaveBeenCalledWith("payload");
  });

  it("copies config to clipboard — failure path", async () => {
    const copyMock = vi
      .fn()
      .mockRejectedValue(new Error("clipboard permission denied"));
    stubSecureClipboard(copyMock);
    const { getByText, getByRole } = render(IntegrationCard, {
      props: { title: "T", subtitle: "S", config: "payload" },
    });
    await fireEvent.click(getByRole("button", { name: "Copy" }));
    await waitFor(() => expect(getByText("Copy failed")).toBeTruthy());
  });

  it("execCommand fallback is used when clipboard API is unavailable", async () => {
    stubInsecureNoClipboard();
    const execCommandMock = vi.fn().mockReturnValue(true);
    // jsdom's document has no execCommand; define it before spying.
    Object.defineProperty(document, "execCommand", {
      value: execCommandMock,
      configurable: true,
      writable: true,
    });
    const { getByRole, getByText } = render(IntegrationCard, {
      props: { title: "T", subtitle: "S", config: "fallback" },
    });
    await fireEvent.click(getByRole("button", { name: "Copy" }));
    await waitFor(() => expect(getByText("Copied!")).toBeTruthy());
    expect(execCommandMock).toHaveBeenCalledWith("copy");
  });

  it("button shows correct class states for copied/failed/default", async () => {
    const copyMock = vi.fn().mockResolvedValue(true);
    stubSecureClipboard(copyMock);
    const { getByRole } = render(IntegrationCard, {
      props: { title: "T", subtitle: "S", config: "x" },
    });
    const btn = getByRole("button", { name: "Copy" });
    // default
    expect(btn.className).toContain("border-exo-light-gray/30");
    await fireEvent.click(btn);
    await waitFor(() => expect(btn.className).toContain("border-green-500/50"));
    // failure state: a writeText that rejects flips the button red
    const failing = vi.fn().mockRejectedValue(new Error("denied"));
    stubSecureClipboard(failing);
    const { getByRole: getByRole2 } = render(IntegrationCard, {
      props: { title: "T", subtitle: "S", config: "x" },
    });
    const btn2 = getByRole2("button", { name: "Copy" });
    await fireEvent.click(btn2);
    await waitFor(() => expect(btn2.className).toContain("border-red-500/50"));
  });
});