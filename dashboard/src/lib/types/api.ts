/**
 * Canonical API-mirror types for the dashboard (hand-maintained mirror of
 * src/exo/api/types/api.py — the /v1 HTTP surface; camelCase wire shape via
 * Pydantic to_camel alias).
 *
 * Trace stats + list types were previously duplicated between
 * stores/app.svelte.ts and utils/trace-stats.ts; the SSE streaming chunk
 * shapes were re-declared inline at 4 call sites in app.svelte.ts. Both now
 * live here as single-point-of-truth types so backend shape changes are
 * single-point edits.
 *
 * NOTE: this module is imported only for `import type` — clean up consumers to
 * import from `$lib/types/api` as they change. The new `$lib/types/index.ts`
 * barrel (t_9d6daf71) re-exports ./files.ts; this api.ts is intentionally
 * standalone to avoid build-order coupling between the sibling cards.
 */

// ---- Trace API response types (api.py: TraceCategoryStats / TraceRankStats /
// TraceStatsResponse / TraceListItem / TraceListResponse) ----

export interface TraceCategoryStats {
  totalUs: number;
  count: number;
  minUs: number;
  maxUs: number;
  avgUs: number;
}

export interface TraceRankStats {
  byCategory: Record<string, TraceCategoryStats>;
}

export interface TraceStatsResponse {
  taskId: string;
  totalWallTimeUs: number;
  byCategory: Record<string, TraceCategoryStats>;
  byRank: Record<number, TraceRankStats>;
}

export interface TraceListItem {
  taskId: string;
  createdAt: string;
  fileSize: number;
}

export interface TraceListResponse {
  traces: TraceListItem[];
}

// ---- SSE streaming chunk wire shapes (chunks.py / api.py streaming models).
// The SSE parser in app.svelte.ts matches on these exact fields; keep
// byte-identical with the inline declarations they replace. ----

export interface ChatCompletionChunk {
  choices?: Array<{
    delta?: { content?: string; reasoning_content?: string };
    logprobs?: {
      content?: Array<{
        token: string;
        logprob: number;
        top_logprobs?: Array<{
          token: string;
          logprob: number;
          bytes: number[] | null;
        }>;
      }>;
    };
  }>;
}

export interface ImageGenerationChunk {
  data?: { b64_json?: string };
  format?: string;
  type?: "partial" | "final";
  image_index?: number;
  partial_index?: number;
  total_partials?: number;
}

export interface ImageEditChunk {
  data?: { b64_json?: string };
  format?: string;
  type?: "partial" | "final";
  partial_index?: number;
  total_partials?: number;
}

// ---- Stable API error codes (api.py: ErrorCode Literal + ErrorInfo/ErrorResponse).

export const ERROR_CODES = [
  "INVALID_REQUEST",
  "UNAUTHORIZED",
  "INSUFFICIENT_MEMORY",
  "PLACEMENT_FAILED",
  "MODEL_NOT_FOUND",
  "INSTANCE_NOT_FOUND",
  "IMAGE_NOT_FOUND",
  "PEER_UNREACHABLE",
  "NODE_NOT_FOUND",
  "INSTANCES_RUNNING",
  "NOT_FOUND",
  "INPUT_TOO_LONG",
  "RATE_LIMITED",
  "QUOTA_EXCEEDED",
  "INTERNAL_ERROR",
] as const;

export type ErrorCode = (typeof ERROR_CODES)[number];

export interface ErrorInfo {
  message: string;
  type: string;
  param?: string | null;
  code: number;
  error_code: ErrorCode;
}

export interface ErrorResponse {
  error: ErrorInfo;
}