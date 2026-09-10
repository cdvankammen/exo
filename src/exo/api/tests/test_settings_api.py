# type: ignore
"""API-level tests for GET/PUT /v1/settings."""

from typing import Any
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api.main import API
from exo.shared.types.state import State
from exo.utils.settings import SettingsManager


def _make_api() -> tuple[Any, TestClient]:
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api.state = State()
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    app.get("/v1/settings")(api.get_settings)
    app.put("/v1/settings")(api.update_settings)
    return api, TestClient(app)


def test_get_settings_returns_catalog(tmp_path) -> None:
    mgr = SettingsManager(settings_file=tmp_path / "settings.json")
    _, client = _make_api()
    with patch("exo.api.main.get_settings_manager", return_value=mgr):
        resp = client.get("/v1/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    vars_present = {entry["var"] for entry in data}
    assert "EXO_KV_CACHE_BITS" in vars_present
    assert "EXO_PREFILL_STEP_SIZE" in vars_present


def test_put_settings_persists_override(tmp_path) -> None:
    mgr = SettingsManager(settings_file=tmp_path / "settings.json")
    _, client = _make_api()
    with patch("exo.api.main.get_settings_manager", return_value=mgr):
        resp = client.put(
            "/v1/settings",
            json={"var": "EXO_KV_CACHE_BITS", "value": "8"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["var"] == "EXO_KV_CACHE_BITS"
    assert body["value"] == "8"
    # Verify it persisted through the manager.
    resolved = mgr.resolve("EXO_KV_CACHE_BITS")
    assert resolved is not None
    assert resolved == ("8", "override")


def test_put_settings_clears_override_with_null(tmp_path) -> None:
    mgr = SettingsManager(settings_file=tmp_path / "settings.json")
    mgr.apply_override("EXO_KV_CACHE_BITS", "8")
    _, client = _make_api()
    with patch("exo.api.main.get_settings_manager", return_value=mgr):
        resp = client.put(
            "/v1/settings",
            json={"var": "EXO_KV_CACHE_BITS", "value": None},
        )
    assert resp.status_code == 200
    resolved = mgr.resolve("EXO_KV_CACHE_BITS")
    assert resolved is not None
    assert resolved[1] != "override"


def test_put_settings_rejects_unknown_var(tmp_path) -> None:
    mgr = SettingsManager(settings_file=tmp_path / "settings.json")
    _, client = _make_api()
    with patch("exo.api.main.get_settings_manager", return_value=mgr):
        resp = client.put(
            "/v1/settings",
            json={"var": "EXO_NOT_A_REAL_VAR", "value": "1"},
        )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["error_code"] == "INVALID_REQUEST"


def test_put_settings_rejects_bad_type(tmp_path) -> None:
    mgr = SettingsManager(settings_file=tmp_path / "settings.json")
    _, client = _make_api()
    with patch("exo.api.main.get_settings_manager", return_value=mgr):
        resp = client.put(
            "/v1/settings",
            json={"var": "EXO_KV_DISK_PERSISTENCE", "value": "not-a-bool"},
        )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["error_code"] == "INVALID_REQUEST"
