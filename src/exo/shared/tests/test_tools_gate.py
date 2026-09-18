"""Tests for the T13 server-side tool-calling gate helper.

Covers `tools_enabled()` in exo.shared.constants: env parsing semantics
(default ON / backward compatible; 0, false, no, off disable it).
"""

import pytest

from exo.shared.constants import tools_enabled


def test_tools_enabled_default_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXO_ENABLE_SERVERSIDE_TOOLCALLS", raising=False)
    assert tools_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", "OFF"])
def test_tools_enabled_disabled_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("EXO_ENABLE_SERVERSIDE_TOOLCALLS", value)
    assert tools_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "anything-else"])
def test_tools_enabled_enabled_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("EXO_ENABLE_SERVERSIDE_TOOLCALLS", value)
    assert tools_enabled() is True
