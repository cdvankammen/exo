"""SettingsManager — persisted EXO_* environment-variable overrides.

Precedence: **persisted UI override > process environment > default**.

exo has ~50 ``EXO_*`` environment variables controlling memory/KV-cache,
cluster, model and debug behavior. Nothing exposes them in a UI today — they
only work as process-launch env vars. This module persists user overrides to
``EXO_CONFIG_HOME/settings.json`` (``~/.exo/settings.json`` on macOS) so the
dashboard / macOS app can edit them, while env vars remain the headless/CLI
escape hatch and defaults fill the rest.

The settings file is a flat JSON object mapping ``EXO_*`` var name → string
value. Values are stored as strings (matching env semantics); typed coercion
happens on read via the catalog spec.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Literal, cast

from loguru import logger

from exo.shared.paths import EXO_CONFIG_HOME

SettingType = Literal["bool", "int", "float", "str"]


class SettingSpec:
    """Catalog entry describing one known EXO_* variable."""

    __slots__ = ("var", "type", "description", "default", "requires_restart")

    def __init__(
        self,
        var: str,
        value_type: SettingType,
        description: str,
        default: str | None = None,
        requires_restart: bool = True,
    ) -> None:
        self.var = var
        self.type = value_type
        self.description = description
        self.default = default
        self.requires_restart = requires_restart

    def coerce(self, raw: str) -> bool | int | float | str:
        """Coerce a raw string to the spec's type, raising ValueError on bad input."""
        match self.type:
            case "bool":
                lowered = raw.strip().lower()
                if lowered in ("1", "true", "yes", "on"):
                    return True
                if lowered in ("0", "false", "no", "off", ""):
                    return False
                raise ValueError(f"{self.var}: expected a boolean, got {raw!r}")
            case "int":
                return int(raw)
            case "float":
                return float(raw)
            case _:
                return raw


# The canonical catalog of UI-editable knobs. Kept deliberately smaller than
# the ~50 EXO_* vars: paths/format internals stay env-only.
CATALOG: dict[str, SettingSpec] = {
    spec.var: spec
    for spec in [
        # Memory / KV-cache
        SettingSpec("EXO_KV_CACHE_BITS", "str", "KV cache quantization bits (4/8)", "4"),
        SettingSpec("EXO_KV_CACHE_GROUP_SIZE", "int", "KV cache quantization group size", "64"),
        SettingSpec("EXO_KV_DISK_PERSISTENCE", "bool", "SSD-as-RAM: persist KV cache to disk", "0"),
        SettingSpec("EXO_KV_DISK_PATH", "str", "KV cache disk directory", None),
        SettingSpec("EXO_KV_DISK_MAX_SIZE_GB", "int", "KV cache disk budget (GB)", None),
        SettingSpec("EXO_KV_DISK_TTL_HOURS", "int", "KV cache expiry (hours)", None),
        SettingSpec("EXO_MEMORY_THRESHOLD", "str", "RAM headroom reserved before prefill", None),
        SettingSpec("EXO_PREFILL_MEMORY_THRESHOLD", "str", "Prefill-specific RAM headroom", None),
        SettingSpec("EXO_PREFILL_STEP_SIZE", "int", "Prefill chunk size in tokens", "512"),
        SettingSpec("EXO_MAX_CHUNK_SIZE", "int", "Token chunk size", None),
        # Cluster / placement
        SettingSpec("EXO_MAX_CONCURRENT_REQUESTS", "int", "API concurrency limit", None, requires_restart=False),
        SettingSpec("EXO_MAX_INSTANCE_RETRIES", "int", "Runner retry budget", "5"),
        SettingSpec("EXO_BOOTSTRAP_PEERS", "str", "Manual peer list (comma-separated)", None),
        SettingSpec("EXO_NODE_ZID", "str", "Fixed node identity (keypair seed)", None),
        SettingSpec("EXO_ZENOH_NAMESPACE", "str", "Cluster namespace isolation", None),
        # Models
        SettingSpec("EXO_MODELS_DIRS", "str", "Writable model directories (colon-separated)", None),
        SettingSpec("EXO_MODELS_READ_ONLY_DIRS", "str", "Read-only model dirs (colon-separated)", None),
        SettingSpec("EXO_ENABLE_IMAGE_MODELS", "bool", "Show image model cards", "0"),
        SettingSpec("EXO_OFFLINE", "bool", "No network (Hub disabled)", "0"),
        # Debug / perf
        SettingSpec("EXO_DSV4_FUSED_MOE", "bool", "DeepSeek V4 fused gate+up MoE", "1"),
        SettingSpec("EXO_NO_BATCH", "bool", "Disable request batching", "0"),
        SettingSpec("EXO_FAST_SYNCH", "bool", "Metal fast synchronisation", "1"),
        SettingSpec("EXO_TRACING_ENABLED", "bool", "Per-request tracing", "0"),
        SettingSpec(
            "EXO_ENABLE_SERVERSIDE_TOOLCALLS",
            "bool",
            "Server-side tool execution (strip tools when 0)",
            "1",
            requires_restart=False,
        ),
    ]
}


class SettingsManager:
    """Load/save EXO_* overrides and resolve values with env-var fallback."""

    def __init__(self, settings_file: Path | None = None) -> None:
        self._settings_file = settings_file or EXO_CONFIG_HOME / "settings.json"
        self._lock = threading.Lock()
        self._overrides: dict[str, str] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────
    def _load(self) -> None:
        try:
            if self._settings_file.is_file():
                raw: object = json.loads(  # pyright: ignore[reportAny]
                    self._settings_file.read_text(encoding="utf-8")
                )
                if isinstance(raw, dict):
                    raw_dict = cast(dict[str, object], raw)
                    self._overrides = {
                        key: str(val)
                        for key, val in raw_dict.items()
                        if val is not None
                    }
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Failed to load settings file {self._settings_file}: {exc}")

    def _save(self) -> None:
        try:
            self._settings_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._settings_file.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(self._overrides, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            tmp.replace(self._settings_file)
        except OSError as exc:
            logger.error(f"Failed to persist settings file {self._settings_file}: {exc}")

    # ── resolution ───────────────────────────────────────────────────────────
    def resolve(self, var: str) -> tuple[str, str] | None:
        """Return (value, source) for a catalogued var, or None if unknown.

        Sources: ``override`` (persisted UI setting) > ``env`` > ``default``.
        """
        spec = CATALOG.get(var)
        if spec is None:
            return None
        with self._lock:
            override = self._overrides.get(var)
        if override is not None:
            return override, "override"
        env_value = os.environ.get(var)
        if env_value is not None:
            return env_value, "env"
        if spec.default is not None:
            return spec.default, "default"
        return None

    def get_value(self, var: str, default: str | None = None) -> str | None:
        """Effective raw value for wiring call sites: override > env > ``default``.

        Unlike :meth:`resolve` — whose final tier is the *catalog* default and
        which powers the settings UI snapshot — this falls back to the
        caller-supplied ``default``. Routing a knob through the manager this
        way never changes a call site's existing in-code default when neither
        a persisted UI override nor a process env var is present. Returns
        ``None`` only when all three tiers are unset (or ``var`` is unknown).
        """
        if CATALOG.get(var) is None:
            return default
        with self._lock:
            override = self._overrides.get(var)
        if override is not None:
            return override
        env_value = os.environ.get(var)
        if env_value is not None:
            return env_value
        return default

    def snapshot(self) -> list[dict[str, object]]:
        """Full catalog snapshot for the API: var, description, type, resolved
        value/source, requires_restart, and whether an override is persisted."""
        entries: list[dict[str, object]] = []
        for var in sorted(CATALOG):
            spec = CATALOG[var]
            resolved = self.resolve(var)
            with self._lock:
                has_override = var in self._overrides
            entry: dict[str, object] = {
                "var": var,
                "description": spec.description,
                "type": spec.type,
                "requires_restart": spec.requires_restart,
                "has_override": has_override,
                "source": resolved[1] if resolved else None,
            }
            if resolved is not None:
                try:
                    entry["value"] = spec.coerce(resolved[0])
                    entry["raw_value"] = resolved[0]
                except ValueError:
                    entry["value"] = None
                    entry["raw_value"] = resolved[0]
                    entry["invalid"] = True
            entries.append(entry)
        return entries

    def apply_override(self, var: str, value: str | None) -> None:
        """Persist (or clear, when value is None) an override for a catalogued var."""
        spec = CATALOG.get(var)
        if spec is None:
            raise ValueError(f"Unknown setting: {var}")
        with self._lock:
            if value is None:
                self._overrides.pop(var, None)
            else:
                # Validate + normalise via the spec's coercion.
                spec.coerce(value)
                self._overrides[var] = value
            self._save()


# Process-wide singleton; the API constructs it lazily so the settings file
# path follows EXO_CONFIG_HOME at first use.
_manager: SettingsManager | None = None
_manager_lock = threading.Lock()


def get_settings_manager() -> SettingsManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = SettingsManager()
        return _manager
