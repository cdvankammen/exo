# DEPRECATIONS

This registry tracks every feature that is formally deprecated in this fork of exo.
It is the single source of truth for removal planning: anything listed here is
scheduled for removal, and anything removed must be recorded first.

Policy (per cross-cutting deprecation analysis `t_99c1804c`):

1. Mark the feature `@deprecated` in code with a comment naming this file and the
   tracking issue.
2. Add a registry row here with the tracking issue, the deprecation date, the
   replacement, and the removal window.
3. Keep the deprecated path working for **at least one minor-version window**
   (back-compat).
4. Remove only after the window closes; then migrate remaining call sites and
   delete the registry row (git history keeps the record).

## Registry

| Feature | Deprecated | Replacement | Tracking issue | Remove after |
|---|---|---|---|---|
| `allowMemoryOverride` dashboard boolean (`exo-allow-memory-override` localStorage key) | 2026-09-18 | tiered `memoryOverrideLevel` (`exo-memory-override-level`) | exo-explore/exo#2315 | >= v0.4.x |
| `/ollama/api/api/chat` + `/ollama/api/api/tags` typo-alias routes | 2026-09-18 | canonical `/ollama/api/chat` + `/ollama/api/tags` | exo-explore/exo#2314 | >= v0.4.x |

## Notes

- **`allowMemoryOverride`** — legacy single boolean gate for memory-guard
  bypass in the dashboard (model picker "Memory guard"/"Override on" toggle).
  Added in fork commit `6b8d3a78`. Still wired to `force_override` on placement
  previews and instance launches. Back-compat shim: stays in sync while the
  replacement storage format lands; old call sites keep working.
- **`/ollama/api/api/*` typo aliases** — doubled `/api/api` paths registered
  silently since commit `addf73a14` (2026-02-20) for clients that copied the
  wrong Ollama path. Every aliased response carries an `X-EXO-Deprecation:
  true` header, an RFC 8594 `Deprecation` header, and a server-side warning
  log. Remove only after the canonical routes have been stable for one
  minor-version window.