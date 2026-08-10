"""PyInstaller runtime hook: verify MLX's Metal metallib is reachable.

MLX (mlx.core) locates its default Metal library (mlx.metallib) via
``load_colocated_library`` (mlx/backend/metal/device.cpp): it dladdr()s the
*loaded* libmlx.dylib and looks for ``mlx.metallib`` next to it. In a
PyInstaller onedir bundle PyInstaller hoists libmlx.dylib to the top of
``_internal`` (it is a dependency of mlx's core extension), so the metallib
must ship at ``_internal/mlx.metallib`` — the spec adds it there (see
exo.spec). cwd is irrelevant to this lookup.

Previously this hook chdir()d into ``_internal`` assuming a relative
resolution; that only masked the problem on machines where the compiled-in
METAL_PATH (a build-machine uv cache path) still existed.

This hook now verifies the invariant and prints a clear diagnostic instead
of letting mlx fail later with an opaque "Failed to load the default
metallib" deep inside a runner.
"""

import os
import sys
from typing import cast


def _find_internal_dir() -> str | None:
    # PyInstaller onedir: sys._MEIPASS points at the _internal directory.
    meipass = cast(str | None, getattr(sys, "_MEIPASS", None))
    if meipass and os.path.isdir(meipass):
        return meipass
    # Fallback: derive from the executable location.
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    candidate = os.path.join(exe_dir, "_internal")
    if os.path.isdir(candidate):
        return candidate
    candidate2 = os.path.join(exe_dir, "Contents", "Resources", "exo", "_internal")
    if os.path.isdir(candidate2):
        return candidate2
    return None

_internal = _find_internal_dir()
if _internal:
    colocated = os.path.join(_internal, "mlx.metallib")
    if not os.path.isfile(colocated):
        print(
            f"[hook-mlx-metallib] mlx.metallib missing at {colocated}; "
            "MLX Metal will fail to load ('Failed to load the default metallib'). "
            "Rebuild with packaging/pyinstaller/exo.spec which ships the "
            "metallib colocated with the loaded libmlx.dylib.",
            file=sys.stderr,
        )
