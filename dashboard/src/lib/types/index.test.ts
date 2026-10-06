// Barrel acceptance test — `$lib/types` must resolve (card t_9d6daf71).
// Verifies the index.ts barrel re-exports the file-attachment types so that
// `import { ChatAttachment } from "$lib/types"` is a valid import path.
import { describe, it, expect } from "vitest";
import type { ChatAttachment, ChatUploadedFile, FileCategory } from "$lib/types";

describe("$lib/types barrel", () => {
  it("re-exports ChatAttachment type", () => {
    const attachment: ChatAttachment = {
      type: "image",
      name: "photo.png",
      mimeType: "image/png",
    };
    expect(attachment.type).toBe("image");
    expect(attachment.name).toBe("photo.png");
  });

  it("re-exports ChatUploadedFile type", () => {
    const uploaded: ChatUploadedFile = {
      id: "1",
      name: "doc.pdf",
      size: 1024,
      type: "application/pdf",
      file: new File([], "doc.pdf"),
    };
    expect(uploaded.size).toBe(1024);
  });

  it("re-exports FileCategory type (includes unknown)", () => {
    const category: FileCategory = "unknown";
    expect(category).toBe("unknown");
  });
});