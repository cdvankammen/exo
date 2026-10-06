// MarkdownContent component tests: markdown rendering, code blocks,
// LaTeX display/inline math, lists, and empty content.
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, fireEvent, waitFor } from "@testing-library/svelte";
import MarkdownContent from "./MarkdownContent.svelte";

beforeEach(() => {
  vi.restoreAllMocks();
  // highlight.js and katex import CSS; jsdom handles it, but silence any
  // window.matchMedia / CSSOM noise is unnecessary — just clear state.
});

describe("MarkdownContent", () => {
  it("renders basic markdown (heading, bold, paragraph)", async () => {
    const { container } = render(MarkdownContent, {
      props: { content: "# Hello\n\nThis is **bold** text." },
    });
    await waitFor(() => {
      const h1 = container.querySelector("h1");
      expect(h1).toBeTruthy();
      expect(h1!.textContent).toContain("Hello");
    });
    expect(container.querySelector("strong")?.textContent).toBe("bold");
    expect(container.querySelector("p")?.textContent).toContain("This is bold text.");
  });

  it("renders a fenced code block with language label and copy button", async () => {
    const { container } = render(MarkdownContent, {
      props: { content: "```python\nprint('hi')\n```" },
    });
    await waitFor(() => {
      expect(container.querySelector("pre code")).toBeTruthy();
    });
    const lang = container.querySelector(".code-language");
    expect(lang?.textContent).toBe("python");
    expect(container.querySelector(".copy-code-btn")).toBeTruthy();
    // highlighted content survives (hljs should have applied classes)
    const code = container.querySelector("pre code");
    expect(code!.textContent).toContain("print");
  });

  it("renders inline code", async () => {
    const { container } = render(MarkdownContent, {
      props: { content: "Use `katex` here." },
    });
    await waitFor(() => {
      expect(container.querySelector(".inline-code")?.textContent).toBe("katex");
    });
  });

  it("renders lists ordered and unordered", async () => {
    const { container } = render(MarkdownContent, {
      props: { content: "- a\n- b\n\n1. one\n2. two" },
    });
    await waitFor(() => {
      expect(container.querySelectorAll("li")).toHaveLength(4);
    });
    expect(container.querySelector("ul")).toBeTruthy();
    expect(container.querySelector("ol")).toBeTruthy();
  });

  it("renders display math via KaTeX when content contains $$...$$", async () => {
    const { container } = render(MarkdownContent, {
      props: { content: "Echo: $$E = mc^2$$" },
    });
    await waitFor(() => {
      expect(container.querySelector(".math-display-wrapper")).toBeTruthy();
    });
    expect(container.querySelector(".katex")).toBeTruthy();
    expect(container.querySelector(".math-label")?.textContent).toBe("LaTeX");
  });

  it("renders inline math via KaTeX for $...$", async () => {
    const { container } = render(MarkdownContent, {
      props: { content: "Inline $x^2$ math" },
    });
    await waitFor(() => {
      expect(container.querySelector(".math-inline .katex")).toBeTruthy();
    });
  });

  it("renders empty content without crash", async () => {
    const { container } = render(MarkdownContent, {
      props: { content: "" },
    });
    expect(container.querySelector(".markdown-content")).toBeTruthy();
  });

  it("renders links", async () => {
    const { container } = render(MarkdownContent, {
      props: { content: "[exo](https://github.com/exo-explore/exo)" },
    });
    await waitFor(() => {
      const a = container.querySelector("a");
      expect(a).toBeTruthy();
      expect(a!.getAttribute("href")).toBe("https://github.com/exo-explore/exo");
      expect(a!.textContent).toBe("exo");
    });
  });

  it("copy code button writes raw code to clipboard", async () => {
    const writeTextMock = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { clipboard: { writeText: writeTextMock } });
    const { container } = render(MarkdownContent, {
      props: { content: "```js\nconst a = 1;\n```" },
    });
    await waitFor(() => {
      expect(container.querySelector(".copy-code-btn")).toBeTruthy();
    });
    const btn = container.querySelector<HTMLButtonElement>(".copy-code-btn")!;
    await fireEvent.click(btn);
    await waitFor(() => expect(writeTextMock).toHaveBeenCalled());
    const encoded = btn.getAttribute("data-code")!;
    const raw = decodeURIComponent(encoded);
    expect(raw).toContain("const a = 1;");
  });
});