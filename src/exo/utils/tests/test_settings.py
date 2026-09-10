import json
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
    assert resolved is not None
    value, source = resolved
    assert value == "4"
    assert source == "default"


def test_resolve_unknown_var_returns_none(settings_manager: SettingsManager) -> None:
    assert settings_manager.resolve("EXO_NOT_A_REAL_VAR") is None


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
    assert resolved is not None and resolved[1] != "override"


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
    assert entry["source"] in ("default", "env", "override")


def test_get_value_override_beats_env_and_default(
    settings_manager: SettingsManager, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setenv("EXO_KV_CACHE_BITS", "8")
    settings_manager.apply_override("EXO_KV_CACHE_BITS", "4")
    assert settings_manager.get_value("EXO_KV_CACHE_BITS", "fallback") == "4"


def test_get_value_env_beats_caller_default(
    settings_manager: SettingsManager, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setenv("EXO_KV_CACHE_BITS", "8")
    assert settings_manager.get_value("EXO_KV_CACHE_BITS", "fallback") == "8"


def test_get_value_falls_back_to_caller_default_not_catalog(
    settings_manager: SettingsManager, monkeypatch: MonkeyPatch
) -> None:
    """With no override and no env, get_value must return the CALLER default,
    not the catalog default — this is what keeps wiring behavior-preserving
    (e.g. EXO_KV_CACHE_BITS catalog default is '4' but code default is off)."""
    monkeypatch.delenv("EXO_KV_CACHE_BITS", raising=False)
    assert settings_manager.get_value("EXO_KV_CACHE_BITS", "fallback") == "fallback"


def test_get_value_unknown_var_returns_caller_default(
    settings_manager: SettingsManager,
) -> None:
    assert settings_manager.get_value("EXO_NOT_A_REAL_VAR", "d") == "d"
    assert settings_manager.get_value("EXO_NOT_A_REAL_VAR") is None


def test_spec_coerce_bool() -> None:
    spec = SettingSpec("X", "bool", "desc")
    assert spec.coerce("1") is True
    assert spec.coerce("true") is True
    assert spec.coerce("on") is True
    assert spec.coerce("0") is False
    assert spec.coerce("false") is False
    with pytest.raises(ValueError):
        spec.coerce("maybe")
