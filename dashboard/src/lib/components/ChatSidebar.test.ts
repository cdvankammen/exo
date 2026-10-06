// ChatSidebar component tests: render, conversation list/selection,
// rename + delete interactions, delete-all, mobile drawer behavior.
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, fireEvent, waitFor } from "@testing-library/svelte";
import ChatSidebar from "./ChatSidebar.svelte";
import { appStore, createConversation } from "../stores/app.svelte";

// Stub fetch so the embedded SidebarLogsPanel's onMount refreshes resolve
// cleanly instead of hitting the network.
function stubOkFetch() {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      if (url.includes("/errors")) {
        return { ok: true, json: async () => ({ errors: [], truncated: false }) };
      }
      if (url.includes("/v1/logs/")) {
        return { ok: true, json: async () => ({ name: "main", content: "", truncated: false }) };
      }
      return { ok: true, json: async () => ({}) };
    }),
  );
}

beforeEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  localStorage.clear();
  appStore.stopPolling();
  (appStore as unknown as { consecutiveFailures: number }).consecutiveFailures = 0;
  appStore.isConnected = true;
  appStore.deleteAllConversations();
  appStore.clearChat();
  stubOkFetch();
});

describe("ChatSidebar", () => {
  it("renders the new-chat button and empty state", () => {
    const { getByText } = render(ChatSidebar);
    expect(getByText("NEW CHAT")).toBeTruthy();
    expect(getByText("NO CONVERSATIONS")).toBeTruthy();
  });

  it("fires onNewChat when NEW CHAT is clicked", async () => {
    const onNewChat = vi.fn();
    const { getByText } = render(ChatSidebar, { props: { onNewChat } });
    await fireEvent.click(getByText("NEW CHAT"));
    expect(onNewChat).toHaveBeenCalledTimes(1);
  });

  it("lists conversations and selects one on click", async () => {
    const onSelectConversation = vi.fn();
    const id = createConversation("Things to try");
    const { getByText } = render(ChatSidebar, { props: { onSelectConversation } });
    await waitFor(() => expect(getByText("Things to try")).toBeTruthy());
    await fireEvent.click(getByText("Things to try"));
    expect(onSelectConversation).toHaveBeenCalledTimes(1);
    expect(appStore.activeConversationId).toBe(id);
  });

  it("renames a conversation via the inline edit flow", async () => {
    const id = createConversation("Old Name");
    const { getByTitle, getByText, container } = render(ChatSidebar);
    await waitFor(() => expect(getByText("Old Name")).toBeTruthy());
    await fireEvent.click(getByTitle("Rename"));
    // The rename input is the second text input (first is search); bound to "Old Name".
    const inputs = [...container.querySelectorAll<HTMLInputElement>("input[type='text']")];
    const renameInput = inputs.find((i) => i.value === "Old Name");
    expect(renameInput).toBeTruthy();
    await fireEvent.input(renameInput!, { target: { value: "Renamed Chat" } });
    await fireEvent.click(getByText("SAVE"));
    await waitFor(() =>
      expect(appStore.getActiveConversation()?.name).toBe("Renamed Chat"),
    );
    expect(appStore.getActiveConversation()?.id).toBe(id);
  });

  it("deletes a conversation through the inline confirm flow", async () => {
    createConversation("Doomed");
    const { getByTitle, getByText } = render(ChatSidebar);
    await waitFor(() => expect(getByText("Doomed")).toBeTruthy());
    await fireEvent.click(getByTitle("Delete"));
    await waitFor(() =>
      expect(getByText(/Delete "Doomed"\?/)).toBeTruthy(),
    );
    await fireEvent.click(getByText("DELETE"));
    await waitFor(() => expect(appStore.conversations.length).toBe(0));
    expect(appStore.hasStartedChat).toBe(false);
  });

  it("deletes all conversations from the footer confirm flow", async () => {
    createConversation("A");
    createConversation("B");
    const { getByText } = render(ChatSidebar);
    await waitFor(() => expect(getByText("DELETE ALL CHATS")).toBeTruthy());
    await fireEvent.click(getByText("DELETE ALL CHATS"));
    await waitFor(() => expect(getByText("DELETE ALL")).toBeTruthy());
    await fireEvent.click(getByText("DELETE ALL"));
    await waitFor(() => expect(appStore.conversations.length).toBe(0));
  });

  it("mobile drawer: renders overlay and fires onClose via the backdrop", async () => {
    const onClose = vi.fn();
    const { getByLabelText } = render(ChatSidebar, {
      props: { isMobileDrawer: true, isOpen: true, onClose },
    });
    await fireEvent.click(getByLabelText("Close sidebar"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});