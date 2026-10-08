// WidgetFrame component tests (ticket W1, card t_0d092272):
// renders title/subtitle/toolbar/footer/children; each optional prop present
// and absent; class merge onto the shell container.
import { describe, it, expect } from "vitest";
import { render } from "@testing-library/svelte";
import { createRawSnippet } from "svelte";
import WidgetFrame from "./WidgetFrame.svelte";

// Snippet props can't be written inline in a .ts test — build them with
// createRawSnippet (public Svelte 5 API), render returns raw HTML.
const toolbarSnippet = createRawSnippet(() => ({
  render: () => '<button type="button">Toolbar action</button>',
}));
const footerSnippet = createRawSnippet(() => ({
  render: () => '<pre>footer config</pre>',
}));
const bodySnippet = createRawSnippet(() => ({
  render: () => "<p>body content</p>",
}));

const required = {
  title: "Panel title",
  children: bodySnippet,
};

describe("WidgetFrame", () => {
  it("renders title and children body (minimal props)", () => {
    const { getByText, container } = render(WidgetFrame, { props: { ...required } });
    expect(getByText("Panel title")).toBeTruthy();
    expect(getByText("body content")).toBeTruthy();
    // shell anatomy from IntegrationCard:36
    const shell = container.firstElementChild!;
    expect(shell.className).toContain("border-exo-light-gray/20");
    expect(shell.className).toContain("rounded-lg");
    expect(shell.className).toContain("bg-exo-medium-gray/20");
  });

  it("renders subtitle when present", () => {
    const { getByText } = render(WidgetFrame, {
      props: { ...required, subtitle: "v2.0" },
    });
    expect(getByText("v2.0")).toBeTruthy();
  });

  it("omits subtitle when absent", () => {
    const { container, getByText, queryByText } = render(WidgetFrame, {
      props: { ...required },
    });
    expect(getByText("Panel title")).toBeTruthy();
    expect(queryByText("v2.0")).toBeNull();
    // no <p> inside the header row (body content lives outside it)
    const headerRow = container.firstElementChild!.firstElementChild!;
    expect(headerRow.querySelector("p")).toBeNull();
  });

  it("renders toolbar snippet in the header row when present", () => {
    const { getByRole } = render(WidgetFrame, {
      props: { ...required, toolbar: toolbarSnippet },
    });
    expect(getByRole("button", { name: "Toolbar action" })).toBeTruthy();
  });

  it("omits toolbar when absent", () => {
    const { queryByRole } = render(WidgetFrame, { props: { ...required } });
    expect(queryByRole("button")).toBeNull();
  });

  it("renders footer block with top divider when present", () => {
    const { getByText, container } = render(WidgetFrame, {
      props: { ...required, footer: footerSnippet },
    });
    expect(getByText("footer config")).toBeTruthy();
    const footerRow = getByText("footer config").parentElement!;
    expect(footerRow.className).toContain("border-t");
    expect(footerRow.className).toContain("border-exo-light-gray/10");
  });

  it("omits footer divider when footer absent", () => {
    const { container } = render(WidgetFrame, { props: { ...required } });
    expect(container.querySelector(".border-t")).toBeNull();
  });

  it("merges a custom class onto the shell container", () => {
    const { container } = render(WidgetFrame, {
      props: { ...required, class: "my-extra-class" },
    });
    const shell = container.firstElementChild!;
    expect(shell.className).toContain("my-extra-class");
    expect(shell.className).toContain("border-exo-light-gray/20"); // defaults kept
  });

  it("renders all optional slots together", () => {
    const { getByText, getByRole } = render(WidgetFrame, {
      props: {
        title: "Full",
        subtitle: "all parts",
        toolbar: toolbarSnippet,
        footer: footerSnippet,
        children: bodySnippet,
        class: "extra",
      },
    });
    expect(getByText("Full")).toBeTruthy();
    expect(getByText("all parts")).toBeTruthy();
    expect(getByRole("button", { name: "Toolbar action" })).toBeTruthy();
    expect(getByText("footer config")).toBeTruthy();
    expect(getByText("body content")).toBeTruthy();
  });
});
