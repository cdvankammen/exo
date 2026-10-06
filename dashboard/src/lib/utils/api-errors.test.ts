// Unit tests for api-errors helper: stable code→hint mapping + error-envelope
// parsing used by app.svelte.ts and +page.svelte.
import { describe, it, expect } from "vitest";
import {
  parseErrorResponse,
  errorCodeHint,
  describeApiError,
  ERROR_CODE_HINTS,
} from "./api-errors";

describe("errorCodeHint", () => {
  it("maps each stable code to a non-empty hint", () => {
    for (const [code, hint] of Object.entries(ERROR_CODE_HINTS)) {
      expect(hint.length, `${code} hint`).toBeGreaterThan(10);
    }
  });

  it("returns a generic message for unknown/null/undefined codes", () => {
    expect(errorCodeHint(null)).toMatch(/failed/i);
    expect(errorCodeHint(undefined)).toMatch(/failed/i);
    // @ts-expect-error - unknown string is not a valid ErrorCode
    expect(errorCodeHint("BOGUS_CODE")).toMatch(/failed/i);
  });

  it("gives the INSUFFICIENT_MEMORY hint mentioning memory", () => {
    expect(errorCodeHint("INSUFFICIENT_MEMORY")).toMatch(/memory/i);
  });

  it("gives the UNAUTHORIZED hint mentioning key", () => {
    expect(errorCodeHint("UNAUTHORIZED")).toMatch(/key/i);
  });
});

describe("parseErrorResponse", () => {
  it("parses a valid OpenAI-style error envelope", () => {
    const info = parseErrorResponse({
      error: {
        message: "Insufficient memory",
        type: "Bad Request",
        code: 400,
        error_code: "INSUFFICIENT_MEMORY",
      },
    });
    expect(info).not.toBeNull();
    expect(info!.error_code).toBe("INSUFFICIENT_MEMORY");
    expect(info!.code).toBe(400);
    expect(info!.message).toBe("Insufficient memory");
  });

  it("normalizes an unknown error_code to INTERNAL_ERROR", () => {
    const info = parseErrorResponse({
      error: {
        message: "x",
        type: "Bad Request",
        code: 500,
        error_code: "SOME_NEW_CODE",
      },
    });
    expect(info!.error_code).toBe("INTERNAL_ERROR");
  });

  it("returns null for non-envelope bodies", () => {
    expect(parseErrorResponse(null)).toBeNull();
    expect(parseErrorResponse({ foo: "bar" })).toBeNull();
    expect(parseErrorResponse("plain text")).toBeNull();
    expect(parseErrorResponse({ error: "not an object" })).toBeNull();
  });

  it("returns null when message/code are missing", () => {
    expect(parseErrorResponse({ error: { type: "x" } })).toBeNull();
  });
});

describe("describeApiError", () => {
  const jsonResponse = (body: unknown) => ({
    status: 400,
    text: async () => JSON.stringify(body),
  });

  it("prefers error_code hint + message for JSON envelopes", async () => {
    const msg = await describeApiError(
      jsonResponse({
        error: {
          message: "Required 24GB, available 8GB",
          type: "Bad Request",
          code: 400,
          error_code: "INSUFFICIENT_MEMORY",
        },
      }),
    );
    expect(msg).toContain("INSUFFICIENT_MEMORY");
    expect(msg).toContain("Required 24GB");
    expect(msg).toMatch(/memory/i);
  });

  it("falls back to HTTP status + raw text for non-JSON bodies", async () => {
    const msg = await describeApiError({
      status: 503,
      text: async () => "Service Unavailable plain",
    });
    expect(msg).toContain("503");
    expect(msg).toContain("Service Unavailable plain");
  });

  it("falls back to HTTP status only for empty bodies", async () => {
    const msg = await describeApiError({ status: 404, text: async () => "" });
    expect(msg).toContain("404");
  });
});