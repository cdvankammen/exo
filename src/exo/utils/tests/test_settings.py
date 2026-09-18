import json
import os
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from exo.utils.settings import CATALOG, SettingsManager, SettingSpec


@pytest.fixture
def settings_manager(tmp_path: Path) -> SettingsManager:
    return SettingsManager(settings_file=tmp_path / "settings.json")


def test_catalog_has_expected_knobs() -> None:
    for var in (
        "EXO_KV_CACHE_BITS",
        "EXO_MAX_KV_SIZE",
        "EXO_KEEP_KV_SIZE",
        "EXO_KV_DISK_PERSISTENCE",
        "EXO_MAX_CONCURRENT_REQUESTS",
        "EXO_BOOTSTRAP_PEERS",
        "EXO_NODE_ZID",
        "EXO_OFFLINE",
        "EXO_TRACING_ENABLED",
    ):
        assert var in CATALOG, f"{var} should be catalogued"


def test_resolve_precedence_override_beats_env(
    settings_manager: SettingsManager, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setenv("EXO_KV_CACHE_BITS", "8")
    settings_manager.apply_override("EXO_KV_CACHE_BITS", "4")

    resolved = settings_manager.resolve("EXO_KV_CACHE_BITS")
    assert resolved is not None
    value, source = resolved
    assert value == "4"
    assert source == "override"


def test_resolve_precedence_env_beats_default(
    settings_manager: SettingsManager, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setenv("EXO_KV_CACHE_BITS", "8")

    resolved = settings_manager.resolve("EXO_KV_CACHE_BITS")
    assert resolved is not None
    value, source = resolved
    assert value == "8"
    assert source == "env"


def test_resolve_default_when_unset(
    settings_manager: SettingsManager, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.delenv("EXO_KV_CACHE_BITS", raising=False)

    resolved = settings_manager.resolve("EXO_KV_CACHE_BITS")
    # Opt-in feature: unset means OFF (None), not a default of 4.
    assert resolved is None


def test_resolve_unknown_var_returns_none(settings_manager: SettingsManager) -> None:
    assert settings_manager.resolve("EXO_NOT_A_REAL_VAR") is None


def test_resolve_max_kv_size_default(settings_manager: SettingsManager) -> None:
    """EXO_MAX_KV_SIZE defaults to 16384 and requires restart (import-time read)."""
    assert settings_manager.resolve("EXO_MAX_KV_SIZE") == ("16384", "default")
    assert CATALOG["EXO_MAX_KV_SIZE"].requires_restart is True
    assert settings_manager.resolve("EXO_KEEP_KV_SIZE") == ("8000", "default")


def test_apply_override_unknown_var_raises(
    settings_manager: SettingsManager,
) -> None:
    with pytest.raises(ValueError, match="Unknown setting"):
        settings_manager.apply_override("EXO_NOT_A_REAL_VAR", "1")


def test_apply_override_validates_type(settings_manager: SettingsManager) -> None:
    with pytest.raises(ValueError, match="expected a boolean"):
        settings_manager.apply_override("EXO_KV_DISK_PERSISTENCE", "not-a-bool")


def test_apply_override_clears_with_none(settings_manager: SettingsManager) -> None:
    settings_manager.apply_override("EXO_KV_CACHE_BITS", "4")
    resolved = settings_manager.resolve("EXO_KV_CACHE_BITS")
    assert resolved is not None and resolved[1] == "override"

    settings_manager.apply_override("EXO_KV_CACHE_BITS", None)

    resolved = settings_manager.resolve("EXO_KV_CACHE_BITS")
    # Override cleared and the opt-in knob has no default: fully unset.
    assert resolved is None


def test_overrides_persist_to_disk(tmp_path: Path) -> None:
    settings_file = tmp_path / "settings.json"
    manager = SettingsManager(settings_file=settings_file)
    manager.apply_override("EXO_KV_CACHE_BITS", "4")

    assert settings_file.is_file()
    assert json.loads(settings_file.read_text(encoding="utf-8")) == {
        "EXO_KV_CACHE_BITS": "4"
    }

    # A fresh manager instance sees the persisted override.
    reloaded = SettingsManager(settings_file=settings_file)
    resolved = reloaded.resolve("EXO_KV_CACHE_BITS")
    assert resolved is not None
    value, source = resolved
    assert value == "4"
    assert source == "override"


def test_snapshot_includes_catalog_metadata(
    settings_manager: SettingsManager,
) -> None:
    entries = settings_manager.snapshot()

    assert len(entries) == len(CATALOG)
    entry = next(e for e in entries if e["var"] == "EXO_KV_CACHE_BITS")
    assert entry["description"]
    assert entry["type"] == "str"
    assert entry["requires_restart"] is True
    # Opt-in knob with no default: unset resolves to None source.
    assert entry["source"] in ("default", "env", "override", None)


def test_catalog_restart_semantics_match_read_sites() -> None:
    """UI truthfulness: dynamic-read vars must not carry the 'restart' badge.

    Verified read sites (2026-09-18 audit, t_4db7e78a):
    - EXO_KV_DISK_PATH / MAX_SIZE_GB / TTL_HOURS: os.environ.get inside methods
      (mlx/cache.py, mlx/kv_offload.py) -> dynamic.
    - EXO_PREFILL_STEP_SIZE: os.getenv inside prefill (mlx/generator/generate.py:477).
    - EXO_NO_BATCH: builder call (mlx/builder.py:76).
    - EXO_FAST_SYNCH: runner bootstrap (worker/runner/bootstrap.py:142).
    """
    dynamic = {
        "EXO_KV_DISK_PATH",
        "EXO_KV_DISK_MAX_SIZE_GB",
        "EXO_KV_DISK_TTL_HOURS",
        "EXO_PREFILL_STEP_SIZE",
        "EXO_NO_BATCH",
        "EXO_FAST_SYNCH",
        "EXO_MAX_CONCURRENT_REQUESTS",
    }
    for var in dynamic:
        spec = CATALOG[var]
        assert spec.requires_restart is False, (
            f"{var} is read dynamically and must not show 'restart'"
        )
    # Import-time knobs must keep the restart badge.
    for var in ("EXO_KV_CACHE_BITS", "EXO_MAX_KV_SIZE", "EXO_KEEP_KV_SIZE",
                "EXO_KV_TIERED", "EXO_MEMORY_THRESHOLD",
                "EXO_MAX_CHUNK_SIZE", "EXO_MAX_INSTANCE_RETRIES", "EXO_OFFLINE",
                "EXO_DSV4_FUSED_MOE", "EXO_TRACING_ENABLED"):
        assert CATALOG[var].requires_restart is True, (
            f"{var} is read at import time and must show 'restart'"
        )


def test_apply_overrides_to_environ(
    settings_manager: SettingsManager, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.delenv("EXO_KV_CACHE_BITS", raising=False)
    monkeypatch.delenv("EXO_KV_DISK_PERSISTENCE", raising=False)

    settings_manager.apply_override("EXO_KV_CACHE_BITS", "4")
    settings_manager.apply_override("EXO_KV_DISK_PERSISTENCE", "1")

    settings_manager.apply_overrides_to_environ()

    assert os.environ.get("EXO_KV_CACHE_BITS") == "4"
    assert os.environ.get("EXO_KV_DISK_PERSISTENCE") == "1"


def test_apply_overrides_to_environ_ignores_invalid(
    settings_manager: SettingsManager, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.delenv("EXO_KV_DISK_PERSISTENCE", raising=False)
    # A raw file written by hand can hold invalid values; the bridge must not
    # crash the node start, just skip the offending override.
    settings_manager._overrides["EXO_KV_DISK_PERSISTENCE"] = "not-a-bool"
    settings_manager.apply_overrides_to_environ()
    assert "EXO_KV_DISK_PERSISTENCE" not in os.environ


def test_spec_coerce_bool() -> None:
    spec = SettingSpec("X", "bool", "desc")
    assert spec.coerce("1") is True
    assert spec.coerce("true") is True
    assert spec.coerce("on") is True
    assert spec.coerce("0") is False
    assert spec.coerce("false") is False
    with pytest.raises(ValueError):
        spec.coerce("maybe")
