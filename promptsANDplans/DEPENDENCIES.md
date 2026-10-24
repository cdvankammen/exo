# EXO — Dependencies, Libraries & Frameworks

> Companion to `PROJECT-OVERVIEW.md`. Complete inventory of the technology stack.

---

## 1. Python Dependencies (`pyproject.toml`)

**Python version:** `==3.13.*` (pinned)
**Build backend:** `uv_build`
**Package version:** `0.3.70`

### Core runtime dependencies

| Package | Purpose |
|---|---|
| `exo-rs` | Rust PyO3 bindings (networking, pidfile) — built from `rust/exo_rs` |
| `anyio==4.11.0` | Async runtime abstraction (task groups, channels, streams) |
| `pydantic>=2.11.7` | All data models (`FrozenModel`, `TaggedModel`) |
| `fastapi` | REST API framework |
| `hypercorn` | ASGI server (serves FastAPI) |
| `rustworkx` | Graph library for topology (`PyDiGraph`) |
| `huggingface-hub` | Model downloads, snapshot_download |
| `loguru` | Logging |
| `msgspec` | Fast serialization (disaggregated protocol) |
| `zstandard` | Compression (disaggregated KV transfer) |
| `openai-harmony` | OpenAI-compatible tool-call parsing |
| `transformers` | Chat templates, tokenizers |
| `python-daemon` | Daemonization (double-fork) |
| `tiktoken` | Token counting |

### MLX extras (mutually exclusive)
| Extra | Purpose |
|---|---|
| `mlx` | Default — **forked** mlx + mlx-lm from `rltakashige/mlx-jaccl-fix-small-recv` and `leo/deepseek-v4` branch |
| `mlx-cpu` | CPU-only MLX |
| `mlx-cuda12` | CUDA 12 MLX |
| `mlx-cuda13` | CUDA 13 MLX |
| `mlx-none` | No MLX |

### uv workspace members
`rust/exo_rs`, `bench`, `tools`

### Type checking
Strict `basedpyright` config (`reportAny=error`, etc.)

---

## 2. Rust Dependencies (`Cargo.toml`)

**Workspace:** `resolver = "3"`, members `["rust/exo_rs", "rust/networking"]`, edition 2024

### Shared workspace dependencies

| Crate | Version | Purpose |
|---|---|---|
| `pyo3` | 0.28.3 | Python bindings |
| `pyo3-async-runtimes` | 0.28.0 | Tokio runtime integration with PyO3 |
| `pyo3-log` | 0.13.3 | Rust log → Python logging |
| `pyo3-stub-gen` | 0.22.3 | `.pyi` stub generation |
| `tokio` | 1.46 | Async runtime |
| `zenoh` | =1.9.0 (pinned) | Pub/sub networking (forked) |
| `zenoh-plugin-storage-manager` | 1.9.0 | Cluster KV storage |
| `zenoh-plugin-trait` | 1.9.0 | Plugin framework |
| `netwatcher` | 0.6.0 | Network interface monitoring |
| `blake3` | 1.8.5 | Namespace hashing |
| `smol` | 2.0.2 | Async runtime (dev) |
| `socket2` | 0.6.4 | Raw sockets (discovery) |
| `pidfile-rs` | git | PID file locking (libbsd flopen) |
| `parking_lot` | — | Mutexes |
| `rand` | — | Randomness |
| `futures-lite` | — | Async utilities |
| `async-stream` | — | Streams |
| `extend` | — | Extension traits |
| `delegate` | — | Method delegation |
| `nix` | 0.31 | Unix syscalls |

### `rust/networking` additional deps
`zenoh` (features `internal`, `plugins`, `unstable`), `bytemuck` (derive — wire types), `log`, dev: `env_logger`, `smol`, `tracing`

### `rust/exo_rs` additional deps
`serde_json`, `pin-project`, `zenoh`, `rand`, `parking_lot`, `tokio` (full), `futures-lite`

### ⚠️ Critical: zenoh is forked
All ~25 `zenoh*` crates are patched via `[patch.crates-io]` to `github.com/evanev7/zenoh` branch `exo` — a fork by exo maintainer Evan Quiney. The custom networking depends on this fork.

---

## 3. Frontend (Dashboard)

**Framework:** Svelte 5 + TypeScript + Vite 6 + Tailwind CSS 4

### Dependencies (`dashboard/package.json`)
| Package | Purpose |
|---|---|
| `svelte ^5.0.0` | UI framework |
| `@sveltejs/kit ^2.48.4` | SvelteKit |
| `@sveltejs/adapter-static ^3.0.10` | Static build |
| `@sveltejs/vite-plugin-svelte ^5.0.0` | Vite integration |
| `@tailwindcss/vite ^4.0.0` | Tailwind 4 |
| `tailwindcss ^4.0.0` | CSS framework |
| `tw-animate-css ^1.3.5` | Animations |
| `d3 ^7.9.0` | Topology graph visualization |
| `marked ^17.0.1` | Markdown rendering |
| `highlight.js ^11.11.1` | Code syntax highlighting |
| `katex ^0.16.27` | Math rendering |
| `pdfjs-dist ^5.6.205` | PDF attachments |
| `devalue ^5.6.2` | Serialization |
| `mode-watcher ^1.1.0` | Dark mode |
| `prettier` + `prettier-plugin-svelte` | Formatting |
| `svelte-check` | Type checking |

---

## 4. Native macOS App (`app/EXO/`)

SwiftUI app with:
- **Sparkle** — auto-update framework
- **ServiceManagement** — launch at login
- **CoreImage** — image processing
- **UserNotifications** — notifications
- **os.log** — logging

Services: `ExoProcessController` (manages the exo subprocess), `ClusterStateService`, `NetworkStatusService`, `LocalNetworkChecker`, `ThunderboltBridgeService`/`ThunderboltBridgeDetector`, `NetworkSetupHelper`, `BugReportService`.

---

## 5. Build & Packaging Tooling

| Tool | Purpose |
|---|---|
| **uv** | Python package/dependency manager (uv workspace) |
| **maturin** | Rust → Python wheel build |
| **Nix** (flake) | Reproducible builds; inputs: nixpkgs (nixos-25.11), flake-parts, crane, fenix, treefmt-nix, dream2nix, pyproject-nix, uv2nix, pyproject-build-systems, nixglhost |
| **crane** | Rust builds in Nix |
| **fenix** | Rust toolchains in Nix |
| **treefmt-nix** | Formatting (nix fmt) |
| **PyInstaller** | Standalone binary packaging (`packaging/pyinstaller/exo.spec`) |
| **create-dmg** | macOS DMG creation (`packaging/dmg/create-dmg.sh`) |
| **xcodebuild** | macOS app build |
| **just** | Command runner (`justfile`) |
| **ruff** | Python linting |
| **basedpyright** | Python type checking |
| **pytest** | Python testing |
| **cargo-nextest** | Rust testing |
| **macmon** | GPU monitoring (pinned fork `vladkens/macmon`) |

### Nix specifics
- **`nix/apple-sdk-overlay.nix`** — patches nixpkgs apple-sdk to SDK 26.2, builds `apple-sdk_26`
- **`nix/metal-toolchain.nix`** — repackages Apple's Metal toolchain NAR (`metal-toolchain-17C48.nar`) with `bin/metal` + `bin/metallib`; `passthru.metalVersion = "400"`; unfree, aarch64-darwin only
- **`nix/darwin-build-fixes.patch`** — MLX build patches: removes `xcrun` invocations, substitutes SDK/Metal versions as CMake variables — hermetic Nix builds without Xcode
- **Cachix:** `exo.cachix.org` + `cache.nixos-cuda.org`

---

## 6. Test Dependencies

- `pytest` + `pytest-asyncio` (`asyncio_mode = "auto"`)
- `exo_tools` (workspace package: client, cluster, harness)
- Playwright (for dashboard screenshots, per AGENTS.md)

---

## 7. External Services

| Service | Purpose |
|---|---|
| **HuggingFace Hub** | Model downloads, model search |
| **zenoh fork** (`github.com/evanev7/zenoh`) | Networking substrate |
| **MLX / mlx-lm** (forked) | Inference engine |
| **mflux** (forked) | Image generation (FLUX) |
| **mlx-vlm** | Vision-language models |
| **macmon** (pinned fork) | GPU monitoring |

---

## 8. Version Pins Summary

| Component | Version |
|---|---|
| Python | `==3.13.*` |
| exo package | `0.3.70` |
| Rust edition | 2024 |
| pyo3 | 0.28.3 |
| zenoh | `=1.9.0` (fork-patched) |
| anyio | `==4.11.7` |
| pydantic | `>=2.11.7` |
| Svelte | `^5.0.0` |
| Tailwind | `^4.0.0` |
| nixpkgs | nixos-25.11 |