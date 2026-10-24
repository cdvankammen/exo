# Build Architecture — Using the Backend WITHOUT Rebuilding the Swift App

**Date:** 2026-08-06
**Branch:** fix-memory-error
**Scope:** Research only. Explains how EXO.app is assembled and exactly which rebuild steps each type of change requires.

---

## 1. What EXO.app actually is (2 layers glued together)

```
EXO.app/
└── Contents/
    ├── MacOS/EXO                    ← SwiftUI shell (Xcode build)
    │                                  • menu bar, Settings UI, process management
    └── Resources/
        └── exo/                     ← PyInstaller backend bundle (embedded folder)
            ├── exo                  ← the Python backend executable (~76MB)
            └── _internal/
                ├── (all Python, MLX, torch, etc.)
                ├── mlx/lib/mlx.metallib      ← Metal shaders
                ├── dashboard/                ← the BUILT Svelte dashboard (HTML/JS)
                └── resources/                ← model cards etc.
```

**How they're glued:** `app/EXO/EXO.xcodeproj/project.pbxproj:35`
```
path = ../../dist/exo; sourceTree = "<group>"
```
Xcode's "Copy Bundle Resources" phase embeds the **PyInstaller output folder** (`dist/exo`) into the app. So the app is: *Xcode shell wrapping a PyInstaller bundle*.

**How the app launches the backend:** `ExoProcessController.swift` `resolveRuntimeDirectory()` (line 398) resolves the runtime in this priority order:
1. **`EXO_RUNTIME_DIR`** env var (dev override — checked FIRST)
2. `Bundle.main.resourceURL/exo` (the embedded bundle)
3. `currentDirectoryPath/dist/exo` (repo fallback — if you run the app from the repo root!)

Then `makeEnvironment()` (line 411) sets `EXO_RUNTIME_DIR` + PATH (runtime dir + `_internal`) for the child process.

---

## 2. How the backend finds the dashboard (runtime resolution)

`src/exo/utils/dashboard_path.py` `find_dashboard()`:
1. **Repo:** walk up from `src/exo/` → `dashboard/build/` (if it has `index.html`)
2. **Bundle:** `sys._MEIPASS/dashboard` (PyInstaller extraction dir)

Overrides in `src/exo/shared/constants.py:74-77`:
```python
_DASHBOARD_DIR_ENV = os.environ.get("EXO_DASHBOARD_DIR", None)
DASHBOARD_DIR = find_dashboard() if _DASHBOARD_DIR_ENV is None else Path.home() / _DASHBOARD_DIR_ENV
```
`EXO_DASHBOARD_DIR` (absolute path works via pathlib absolute-join semantics) overrides where the dashboard is served from.

---

## 3. The 4 ways to run the backend (fastest → slowest)

### Level 0 — Source backend, no app at all (FASTEST for Python)
```bash
uv run exo
```
- Uses `src/exo/` directly + repo `dashboard/build/` (via `find_dashboard` repo branch)
- **Python changes: instant** — restart and it's live, zero rebuild
- Dashboard: `cd dashboard && npm run dev` (hot-reload Svelte) or `npm run build` + restart

### Level 1 — The built bundle directly (skip the shell)
```bash
/Applications/EXO.app/Contents/Resources/exo/exo
```
- Same binary the app runs, no Swift UI
- **Python changes: need `uv run pyinstaller`** (the bundle is a frozen snapshot)
- Dashboard override: `EXO_DASHBOARD_DIR=/abs/path/dashboard/build` (after rebuild) — no repackaging

### Level 2 — Point the INSTALLED app at a custom runtime (KEY TRICK)
```bash
# build a fresh backend bundle only
uv run pyinstaller packaging/pyinstaller/exo.spec
# then launch the app against it — NO xcodebuild needed
EXO_RUNTIME_DIR=/path/to/dist/exo /Applications/EXO.app/Contents/MacOS/EXO
```
- `resolveRuntimeDirectory()` checks `EXO_RUNTIME_DIR` **first** — the installed Swift shell runs your freshly-built `dist/exo`
- **Python changes: only `uv run pyinstaller`** — skip xcodebuild entirely
- ⚠️ `open` does NOT pass env vars — launch the binary directly or use `launchctl setenv EXO_RUNTIME_DIR <path>` before `open`

### Level 3 — Full app rebuild (ONLY needed for Swift shell changes)
```bash
uv sync --extra mlx                                   # mlx deps (if changed)
uv sync --reinstall-package exo_rs                    # Rust bindings (if changed)
cd dashboard && npm run build                         # dashboard (if changed)
uv run pyinstaller packaging/pyinstaller/exo.spec     # backend bundle
cd app/EXO && xcodebuild clean build ...              # Swift shell
sudo rm -rf /Applications/EXO.app && sudo cp -R ...   # install
open /Applications/EXO.app
```

---

## 4. The COMPLETE matrix — what to rebuild for each change

| You changed… | Rebuild needed | Rebuild steps skipped |
|---|---|---|
| **Dashboard UI** (`.svelte`, `app.svelte.ts`) | `npm run build` only | pyinstaller, xcode, install |
| **Python backend** (`src/exo/**/*.py`) | `uv run pyinstaller` (or run Level 0 source) | xcode, install |
| **Rust bindings** (`rust/`, `exo_rs`) | `uv sync --reinstall-package exo_rs` + pyinstaller | xcode, install |
| **Swift shell** (menu bar, Settings, process mgmt) | `xcodebuild` only (uses existing `dist/exo`) | pyinstaller (keep dist/exo) |
| **New Python dependency** | `uv sync` + pyinstaller | xcode, install |
| **Model cards / resources** (`resources/`) | pyinstaller (or `EXO_RESOURCES_DIR` override) | xcode, install |
| **pyinstaller spec** (`exo.spec`) | pyinstaller | xcode, install |

**Key rule of thumb:**
- Swift shell changes → the ONLY case needing `xcodebuild` + `sudo install`
- Everything else → `npm run build` and/or `uv run pyinstaller`, served via the dev override (Levels 0–2)
- `EXO_RUNTIME_DIR` (Level 2) is the "hot-swap" that makes the installed app run any freshly-built bundle

---

## 5. File map

| File | Role |
|---|---|
| `app/EXO/EXO.xcodeproj/project.pbxproj:35` | embeds `../../dist/exo` as a folder resource |
| `app/EXO/EXO/ExoProcessController.swift:398` | `resolveRuntimeDirectory()` — EXO_RUNTIME_DIR → bundle → repo |
| `app/EXO/EXO/ExoProcessController.swift:411` | `makeEnvironment()` — EXO_RUNTIME_DIR, PATH |
| `src/exo/utils/dashboard_path.py` | `find_dashboard()` — repo `dashboard/build` or bundle `_internal/dashboard` |
| `src/exo/shared/constants.py:74` | `EXO_DASHBOARD_DIR` / `EXO_RESOURCES_DIR` overrides |
| `packaging/pyinstaller/exo.spec` | the bundle build (Python + MLX + dashboard + resources) |
| `README.md:598` | the full build doc ("⚡ The Easy Steps") |

---

## 6. Recommended dev loop (no Swift rebuilds)

```
Python change  →  uv run exo                      (source, instant)
Dashboard UI   →  npm run dev (hot reload)        (or npm run build + restart)
Backend change in app →  pyinstaller + EXO_RUNTIME_DIR (hot-swap into installed app)
Swift shell    →  xcodebuild + install            (rare, only shell changes)
```
