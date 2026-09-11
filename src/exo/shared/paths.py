"""XDG Base Directory path computation.

Leaf module — imported by both ``exo.shared.constants`` (re-exported for
backward compatibility) and ``exo.utils.settings`` (for the settings file
location). Keeping this logic in its own module breaks the circular import
that would otherwise arise when ``constants.py`` wants to resolve feature
knobs through SettingsManager.

Nothing in this module imports from ``exo.shared.constants`` or
``exo.utils.settings``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_EXO_HOME_ENV = os.environ.get("EXO_HOME", None)


def _get_xdg_dir(env_var: str, fallback: str) -> Path:
    """Get XDG directory, prioritising EXO_HOME environment variable if its set. On non-Linux platforms, default to ~/.exo."""

    if _EXO_HOME_ENV is not None:
        return Path.home() / _EXO_HOME_ENV

    if sys.platform != "linux":
        return Path.home() / ".exo"

    xdg_value = os.environ.get(env_var, None)
    if xdg_value is not None:
        return Path(xdg_value) / "exo"
    return Path.home() / fallback / "exo"


EXO_CONFIG_HOME = _get_xdg_dir("XDG_CONFIG_HOME", ".config")
EXO_DATA_HOME = _get_xdg_dir("XDG_DATA_HOME", ".local/share")
EXO_CACHE_HOME = _get_xdg_dir("XDG_CACHE_HOME", ".cache")
