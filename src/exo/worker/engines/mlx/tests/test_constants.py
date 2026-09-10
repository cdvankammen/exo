"""Tests for mlx engine constants — T9 RotatingKVCache default + env override."""

import importlib

import pytest

from exo.worker.engines.mlx import constants


def test_default_max_kv_size_is_safe_bounded() -> None:
    """Default MAX_KV_SIZE is 16384 (4x the old 3200 cap, still bounded)."""
    assert constants.MAX_KV_SIZE == 16384
    assert constants.MAX_KV_SIZE < 131072  # not the risky PR default


def test_default_keep_kv_size() -> None:
    """Default KEEP_KV_SIZE is set (bounded rotating cache keep)."""
    assert constants.KEEP_KV_SIZE is not None
    assert constants.KEEP_KV_SIZE > 0


def test_env_override_max_kv_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """EXO_MAX_KV_SIZE env var overrides the default (power-user escape)."""
    monkeypatch.setenv("EXO_MAX_KV_SIZE", "131072")
    reloaded = importlib.reload(constants)
    assert reloaded.MAX_KV_SIZE == 131072  # type: ignore[reportAny]
