/**
 * Stable API error-code → human hint text mapping.
 *
 * Mirrors src/exo/api/types/api.py ErrorCode and gives the dashboard a
 * single place to translate machine codes into user-facing guidance.
 * Unknown codes fall back to a generic message that still surfaces the
 * raw status + server detail.
 */
import type { ErrorCode, ErrorInfo } from "$lib/types/api";
import { ERROR_CODES } from "$lib/types/api";

export const ERROR_CODE_HINTS: Record<ErrorCode, string> = {
  INVALID_REQUEST: "The request was malformed or contained invalid parameters. Check your input and try again.",
  UNAUTHORIZED: "This request is missing a valid API key. Set the Authorization: Bearer header on your request.",
  INSUFFICIENT_MEMORY: "Not enough memory is available on the cluster for this model. Free memory by unloading other models, or use a smaller model.",
  PLACEMENT_FAILED: "The model could not be placed on the cluster. Check per-node memory/backends and retry.",
  MODEL_NOT_FOUND: "The requested model is not available. Check the model ID or download it first.",
  INSTANCE_NOT_FOUND: "The requested instance no longer exists. It may have been stopped or the cluster state changed.",
  IMAGE_NOT_FOUND: "The requested image is not available or has expired.",
  PEER_UNREACHABLE: "The peer node could not be reached. Check its address/network and retry.",
  NODE_NOT_FOUND: "The requested node was not found in the cluster topology.",
  INSTANCES_RUNNING: "Operation blocked because instances are still running. Stop the instances first.",
  NOT_FOUND: "The requested resource was not found.",
  INPUT_TOO_LONG: "The input is too long for this model's context window. Reduce the input or use a model with a larger context.",
  RATE_LIMITED: "You are sending requests too quickly. Slow down and retry shortly.",
  QUOTA_EXCEEDED: "Your daily token quota has been exceeded. Check your API key quota or wait for reset.",
  INTERNAL_ERROR: "An internal error occurred. Check the server logs for details and retry.",
};

const GENERIC_HINT =
  "The request failed. Check the server status and logs for details.";

/**
 * Parse an API error response body (OpenAI-style `{"error": {...}}`) into a
 * stable ErrorInfo, or null if the body is not a recognized error envelope.
 */
export function parseErrorResponse(body: unknown): ErrorInfo | null {
  if (body && typeof body === "object" && "error" in body) {
    const err = (body as { error?: unknown }).error;
    if (err && typeof err === "object") {
      const e = err as Record<string, unknown>;
      if (typeof e.message === "string" && typeof e.code === "number") {
        return {
          message: e.message,
          type: typeof e.type === "string" ? e.type : "",
          param: typeof e.param === "string" ? e.param : null,
          code: e.code,
          error_code: ERROR_CODES.includes(e.error_code as ErrorCode)
            ? (e.error_code as ErrorCode)
            : "INTERNAL_ERROR",
        };
      }
    }
  }
  return null;
}

/** Human-readable hint for an error code (falls back to a generic message). */
export function errorCodeHint(code: ErrorCode | undefined | null): string {
  if (!code) return GENERIC_HINT;
  return ERROR_CODE_HINTS[code] ?? GENERIC_HINT;
}

/**
 * Build a user-facing error string from a failed fetch Response, preferring
 * the stable error_code hint when the body carries one.
 */
export async function describeApiError(
  response: { status: number; text(): Promise<string> },
): Promise<string> {
  const raw = await response.text();
  let parsed: ErrorInfo | null = null;
  try {
    parsed = parseErrorResponse(JSON.parse(raw));
  } catch {
    // Not JSON — fall through to the raw text form.
  }
  if (parsed) {
    return `${parsed.error_code}: ${errorCodeHint(parsed.error_code)}${
      parsed.message ? ` — ${parsed.message}` : ""
    }`;
  }
  return `HTTP ${response.status}${raw ? ` — ${raw.slice(0, 500)}` : ""}`;
}