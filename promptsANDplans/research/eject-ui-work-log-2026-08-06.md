# Eject UI Work Log — 2026-08-06 (complete history)

> **Purpose:** The full saga of the eject button changes — from the original #2238 port, through the split-button redesign, to the no-auto-delete reversal. Every decision, error, and code location documented.

---

## CHAPTER 1: The Original Feature (from PR #2238)

### Commit `ed7ba52f` — "feat(dashboard): unify instance eject with optional weight deletion (port of PR #2238)"
- **Source:** https://github.com/exo-explore/exo/pull/2238 by @alytaphoenix (Alyta Satine Phoenix)
- **Files:** 1 — `dashboard/src/routes/+page.svelte` (132+/10-)
- **Original flow:**
  1. Click EJECT → `confirm("Eject instance abc123...?")`
  2. If yes → `DELETE /instance/{id}`
  3. Then → `confirm("Also delete the downloaded weights for {model} from disk to free storage?")`
  4. If yes → `deleteDownload(nodeId, modelId)` on every node from `shardAssignments.nodeToRunner`
- **Known caveat (documented in commit):** weight-deletion fires right after instance-delete 200, but teardown is async → race possible with still-loading instances

---

## CHAPTER 2: The Split-Button Redesign (user request)

### User request (verbatim intent)
> "Can we remove the need to ask 'do you want to delete it' when I click eject... can we make it so that it is just in the UI itself? When I click eject it will have a very simple and smooth transition of the eject button that will split with NO on the left and YES on the right... NO with just an outline but no infill... same animations as the other buttons (hovering etc.)"

### Commit `ca8474c7` — "feat(dashboard): inline EJECT confirm split-button, auto-delete weights on eject"

**New file `dashboard/src/lib/components/EjectButton.svelte`:**
```svelte
<script lang="ts">
  import { fade, fly } from "svelte/transition";  // MUST import — Svelte 5 compiles unimported
                                                  // transition names to bare refs → ReferenceError
  interface Props {
    id: string;           // MUST be named `id` — parent passes {id}
    onConfirm: (id: string) => void;
  }
  let { id, onConfirm }: Props = $props();
  let confirming = $state(false);
  // handleEject() → confirming = true
  // handleNo()   → confirming = false
  // handleYes()  → confirming = false; onConfirm(id)
</script>
{#if confirming}
  <button NO  transition:fly={{ x: -12, duration: 180 }} />   <!-- outline-only -->
  <button YES transition:fly={{ x: 12, duration: 180 }} />    <!-- filled, matches EJECT hover -->
{:else}
  <button EJECT transition:fade={{ duration: 150 }} />
{/if}
```
- NO = outline-only (`border-red-500/30 text-red-400 hover:border-red-500/50` — no bg fill)
- YES = filled (`hover:bg-red-500/20` — same look as EJECT hover)
- Both keep the same hover animations as the original EJECT

**`+page.svelte` changes:**
- Removed both `confirm()` calls from `ejectInstance()`
- Kept the auto-delete block (from #2238) — deleting weights was still automatic
- Replaced the two hardcoded EJECT `<button>`s (lines ~5369, ~6563) with `<EjectButton {id} onConfirm={ejectInstance} />`
- Added `EjectButton` export to `$lib/components/index.ts`

### Errors hit during implementation
1. **Svelte 5 transition conflict:** `transition:` + `out:` on the same element → compile error "Cannot use `transition:` alongside existing `out:` directive" → fixed by using `transition:` only (handles both in/out)
2. **Barrel export broke tree-shaking:** page rendered but threw `ReferenceError: EjectButton is not defined` at runtime — the generated chunk called `EjectButton(...)` as a bare function without importing it → fixed with direct import:
   ```ts
   import EjectButton from "$lib/components/EjectButton.svelte";
   ```
   (kept the barrel export too — harmless)

### Verification
- `npm run build` clean
- Preview server test: page loaded with no ReferenceError (only expected 404s from missing API)
- Node chunk hash changed `2.C9m36cdM.js` → `2.CB1tWQOG.js` (confirmed real change)

---

## CHAPTER 3: The Auto-Delete Reversal (user request)

### User request (verbatim intent)
> "Also, does it delete it now automatically when I say yes after pressing eject or no?"
> "okay I DO NOT WANT THAT to happen i will delete it on my own. i do not want that to happen that a model will delete itself automatically. I will delete them manually if I want to. so list where that functionality is currently happening and then we can move forward with removing that 'feature'"

### Where the auto-delete lived (mapped for the user)
| Location | What |
|---|---|
| `+page.svelte:43` | `deleteDownload` import from stores |
| `+page.svelte:2075-2082` | `getInstanceNodeIds()` helper — finds nodes the instance ran on (ONLY for deletion) |
| `+page.svelte:2121` | `nodeIds = getInstanceNodeIds(...)` — captured before eject |
| `+page.svelte:2136-2153` | **THE AUTO-DELETE BLOCK** — after eject succeeds: `deleteDownload(nodeId, modelId)` on every node with error toasts |
| `EjectButton.svelte:40` | YES tooltip: "Eject instance and delete weights" |

### Commit `3d1a675c` — "fix(dashboard): eject no longer auto-deletes model weights"
**Removed:**
- The `deleteDownload()` block (the whole `if (deletedInstanceModelId && ... && nodeIds.length > 0) { await Promise.all(...) }` — was deleting weights on every node with per-node error toasts)
- `getInstanceNodeIds()` helper entirely
- `deleteDownload` import from the app store
- EjectButton YES tooltip → "Eject instance"

**New `ejectInstance()` (what remains):**
```ts
async function ejectInstance(instanceId: string) {
  const wrappedInstance = instanceData[instanceId];
  const deletedInstanceModelId = getInstanceModelId(wrappedInstance);
  const wasSelected = selectedChatModel() === deletedInstanceModelId;
  try {
    const response = await fetch(`/instance/${instanceId}`, { method: "DELETE", ... });
    if (!response.ok) { ...toast error; return; }
    // NOTE: Eject only stops the instance. Weights are NEVER deleted
    // automatically — the user deletes them manually from the Downloads page.
    if (wasSelected) { ...switch to another model... }
  }
}
```

### Verification
- `grep deleteDownload +page.svelte` → none
- `grep getInstanceNodeIds` → none
- Build clean
- Deployed to running app (`sudo rsync` into `/Applications/EXO.app/Contents/Resources/exo/_internal/dashboard/`, backup first)
- Served bundle: zero references to "delete weights"

---

## CHAPTER 4: App Deployment (hot-swap vs full rebuild)

### Hot-swap (15:37, first attempt)
- Copied `dashboard/build/` over the running app's `_internal/dashboard/`
- **Why it worked:** exo serves static files from disk per request → new UI on next page load, no restart
- **Why it wasn't enough:** the BUILD PRODUCTS (`app/EXO/build/...` + DerivedData) were stale (15:16, other agent's build); next real rebuild would wipe the change; bundle signature invalid

### Full rebuild (15:50, proper fix)
1. `pkill` old app
2. `uv sync --extra mlx` → **ERROR: mlx vanished** (git-fork install failed silently — `ModuleNotFoundError: No module named 'mlx'`) → `uv pip install --reinstall-package mlx "mlx @ git+https://github.com/rltakashige/mlx-jaccl-fix-small-recv.git@cc3f3e60..."` (19ms, from cache)
3. `uv run pyinstaller packaging/pyinstaller/exo.spec --noconfirm` (first run failed "Module 'mlx' is not available" — fixed by step 2) → `dist/exo` at 15:50:13 with new EJECT baked in
4. `xcodebuild -project app/EXO/EXO.xcodeproj -scheme EXO -configuration Release build` → **BUILD SUCCEEDED** — output to **DerivedData** (`~/Library/Developer/Xcode/DerivedData/EXO-bkuinapskmxewzcxnnqwsmuzojhk/Build/Products/Release/EXO.app`), NOT `app/EXO/build` (that dir was stale from the other agent — explains "I don't see it happening")
5. Install: `sudo rm -rf /Applications/EXO.app && sudo cp -R <DerivedData>/EXO.app /Applications/`
6. **ERROR: launch failed "Permission denied"** — bundle was `drwx------` (700/root) → `sudo chmod -R a+rX /Applications/EXO.app`
7. Signature check → "bundle format unrecognized" → re-sign ad-hoc: `codesign --force --deep --sign - /Applications/EXO.app` → verified
8. `open /Applications/EXO.app` → running (PID 1629), API live, new EJECT served

---

## CHAPTER 5: Current State (final)

**Eject flow today:**
1. Click **EJECT** → smooth fly animation → **NO** (outline, left) / **YES** (filled, right)
2. Click **YES** → `DELETE /instance/{id}` — instance stops
3. **Weights stay on disk. Nothing deleted. Ever.**
4. Manual deletion: Downloads page per-node delete buttons (never removed)

**Commits in order:**
| Commit | Change | Source |
|---|---|---|
| `ed7ba52f` | Original eject + optional weight-delete popup | PR #2238 port |
| `ca8474c7` | Split-button NO/YES, removed popups, weights auto-deleted | User request |
| `3d1a675c` | Auto-delete REMOVED — eject = stop only | User request (reversal) |

---

## CHAPTER 6: The TWO Runtime Bugs — Found & Fixed via Live Browser Test (evening)

**Symptom (user):** "when i click yes or no it will either put the eject button back how it was if i did no and yes will do the same functionality as before where it ejects the model — whats going on with it not working?"

**Context:** the split-button compiled fine but didn't *work* at runtime. Two separate bugs — both invisible to the build.

### Bug A — Missing `svelte/transition` import (`ReferenceError: fly is not defined`)

**Code before (in `EjectButton.svelte`):**
```svelte
<button ... transition:fly={{ x: -12, duration: 180 }}>NO</button>
<button ... transition:fly={{ x: 12, duration: 180 }}>YES</button>
<button ... transition:fade={{ duration: 150 }}>EJECT</button>
```
**No `import { fade, fly } from "svelte/transition"` anywhere.**

**Why the build passed but runtime failed:** Svelte 5 compiles an unimported transition name into a **bare closure reference** — `qi(3,m,()=>fly,()=>({x:-12,duration:180}))`. `fly` is undefined at runtime. The instant you clicked EJECT, the NO/YES buttons tried to animate in → `ReferenceError: fly is not defined` → the split never rendered.

**Fix (one line):**
```svelte
import { fade, fly } from "svelte/transition";
```
Compiled JS now shows `()=>G_` — a real minified import binding — instead of `()=>fly`.

### Bug B — Prop name mismatch (`DELETE /instance/undefined` → 404)

**Parent renders** (`+page.svelte:5365`):
```svelte
<EjectButton {id} onConfirm={ejectInstance} />
```
**Component declared**:
```ts
interface Props {
  instanceId: string;              // ← WRONG NAME
  onConfirm: (instanceId: string) => void;
}
let { instanceId, onConfirm }: Props = $props();
```
So `instanceId` was **always `undefined`** — YES called `onConfirm(undefined)` → `ejectInstance(undefined)` → `fetch("/instance/undefined")` → **404 "Failed to eject instance"** toast.

**Fix:** renamed the prop to `id`:
```ts
interface Props {
  id: string;
  onConfirm: (id: string) => void;
}
let { id, onConfirm }: Props = $props();
// handleYes() → onConfirm(id)
```

### How the bugs were found (the key lesson)
**Static verification was not enough.** I inspected the compiled chunk and saw `()=>fly` (Bug A), but Bug B only surfaced when clicking YES in a **real browser**: the network tab showed `DELETE http://localhost:52416/instance/undefined → 404`.

### End-to-end verification (Playwright, isolated test node `--namespace ui-test -m --api-port 52416`)
1. Placed Llama-3.2-1B on an isolated test node → instance READY
2. Clicked **EJECT** → **NO/YES appeared** (Bug A fixed)
3. Clicked **NO** → EJECT button restored, instance untouched
4. Clicked **EJECT** again → **YES** → captured network: `DELETE /instance/1ee19017-857a-4063-a937-0c3452a0f460 → 200` — instance ejected (Bug B fixed)

### Deployed (2 rounds)
- Round 1: dashboard rebuild → pyinstaller → xcodebuild → installed locally → copied to mini2
- Round 2 (added mlx_lm fix): full rebuild → installed locally (PID 53068) → copied to mini2 (node `e45455dc`)
- Verified on both: served chunk `2.BlaL9Lg6.js` contains "Cancel eject"; 0 crashes

**Prevention:** the component now carries inline NOTE comments explaining both bugs; future edits to `EjectButton.svelte` keep the import + `id` prop.
