"""Tests for per-module log filters (TODO #27)."""

from pytest import MonkeyPatch

from exo.shared.logging import (
    LOG_LEVEL_NO,
    make_module_filter,
    parse_module_levels,
    resolve_base_level,
)


class _Level:
    def __init__(self, no: int) -> None:
        self.no = no


def _rec(name: str, level: str) -> dict[str, object]:
    return {"name": name, "level": _Level(LOG_LEVEL_NO[level])}


def test_parse_module_levels_empty() -> None:
    assert parse_module_levels(None) == {}
    assert parse_module_levels("") == {}


def test_parse_module_levels_basic() -> None:
    parsed = parse_module_levels("exo.master=DEBUG,exo.worker=WARNING")
    assert parsed == {"exo.master": "DEBUG", "exo.worker": "WARNING"}


def test_parse_module_levels_skips_invalid() -> None:
    parsed = parse_module_levels("exo.master=DEBUG,=INFO,garbage,exo.api=BANANA")
    assert parsed == {"exo.master": "DEBUG"}


def test_parse_module_levels_tolerates_whitespace() -> None:
    parsed = parse_module_levels(" exo.master = DEBUG , exo.worker=info ")
    assert parsed == {"exo.master": "DEBUG", "exo.worker": "INFO"}


def test_filter_base_level_applies_without_override() -> None:
    filt = make_module_filter({}, "INFO")
    assert filt(_rec("exo.master", "INFO"))
    assert filt(_rec("exo.master", "DEBUG")) is False


def test_filter_module_override_raises_level() -> None:
    filt = make_module_filter({"exo.master": "DEBUG"}, "INFO")
    assert filt(_rec("exo.master", "DEBUG"))
    assert filt(_rec("exo.master.placement", "DEBUG"))
    # other modules still use the base level
    assert filt(_rec("exo.worker", "DEBUG")) is False


def test_filter_module_override_lowers_level() -> None:
    filt = make_module_filter({"exo.worker": "WARNING"}, "DEBUG")
    assert filt(_rec("exo.worker", "WARNING"))
    assert filt(_rec("exo.worker", "INFO")) is False
    assert filt(_rec("exo.master", "DEBUG"))  # base still DEBUG


def test_filter_most_specific_prefix_wins() -> None:
    filt = make_module_filter(
        {"exo": "WARNING", "exo.master": "DEBUG"}, "INFO"
    )
    assert filt(_rec("exo.master", "DEBUG"))  # specific override wins
    assert filt(_rec("exo.worker", "DEBUG")) is False  # broad override applies


def test_filter_unknown_module_name_uses_base() -> None:
    filt = make_module_filter({"exo.master": "DEBUG"}, "INFO")
    assert filt(_rec("", "INFO"))
    assert filt(_rec("some_other", "INFO"))
    assert filt(_rec("some_other", "DEBUG")) is False


def test_resolve_base_level_env_wins(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("EXO_LOG_LEVEL", "WARNING")
    assert resolve_base_level(0) == "WARNING"
    assert resolve_base_level(2) == "WARNING"


def test_resolve_base_level_verbosity(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("EXO_LOG_LEVEL", raising=False)
    assert resolve_base_level(0) == "INFO"
    assert resolve_base_level(1) == "DEBUG"


def test_resolve_base_level_invalid_env_ignored(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("EXO_LOG_LEVEL", "BANANA")
    assert resolve_base_level(0) == "INFO"
