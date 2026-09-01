import sys

import exo.main as exo_main
from pytest import MonkeyPatch


class _SettingsManagerStub:
    def resolve(self, variable: str) -> tuple[str, str] | None:
        assert variable == "EXO_BOOTSTRAP_PEERS"
        return ("peer-a:52414, peer-b:6000,  ", "override")


def test_bootstrap_peers_setting_becomes_startup_default(
    monkeypatch: MonkeyPatch,
) -> None:
    """A peer list saved in Settings is used without requiring a CLI flag."""
    monkeypatch.setattr(exo_main, "get_settings_manager", _SettingsManagerStub)
    monkeypatch.setattr(sys, "argv", ["exo"])

    args = exo_main.Args.parse()

    assert args.bootstrap_peers == ["peer-a:52414", "peer-b:6000"]


def test_bootstrap_peer_cli_argument_strips_whitespace(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["exo", "--bootstrap-peers", "peer-a, peer-b:52414,  "])

    args = exo_main.Args.parse()

    assert args.bootstrap_peers == [
        "peer-a",
        "peer-b:52414",
    ]