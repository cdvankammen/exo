/**
 * Type barrel for the dashboard's shared type layer.
 *
 * NOTE: `lib/types.ts` (a flat file) does not exist in this repo — this
 * directory is the canonical home for shared API-mirror types. The bulk of
 * the dashboard's API-mirroring interfaces currently live in
 * `lib/stores/app.svelte.ts` (imported directly by 30+ consumers — do not
 * move them here); this barrel re-exports the file-attachment types so that
 * `import { ChatAttachment } from "$lib/types"` resolves.
 *
 * New shared API-mirror types should be added to this directory (e.g.
 * `api.ts`) and re-exported here.
 */
export type { ChatAttachment, ChatUploadedFile, FileCategory } from "./files.js";