"""PyInstaller runtime hook: make MLX find its Metal metallib.

MLX (mlx.core) locates its default Metal library (mlx.metallib) relative to
the process working directory / compiled-in paths. In a PyInstaller bundle
the metallib ships at ``_internal/mlx/lib/mlx.metallib`` but MLX has no
``__file__`` (compiled extension) and no env override, so it fails with:

    RuntimeError: Failed to load the default metallib. library not found

Running from ``_internal`` makes MLX resolve ``mlx/lib/mlx.metallib``
(verified live 2026-08-07: ``mx.metal.is_available() == True``). This hook
chdir()s into the PyInstaller ``_internal`` directory before any MLX import.

Discovered while diagnosing the T29 RDMA retest: every runner failed with
"Failed to load the default metallib" even though the file was bundled.
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
if _internal and os.path.isfile(os.path.join(_internal, "mlx", "lib", "mlx.metallib")):
    os.chdir(_internal)
