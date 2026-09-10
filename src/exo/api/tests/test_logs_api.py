# pyright: reportUnusedFunction=false, reportAny=false
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api import main as api_main
from exo.api.main import (
    API,
    _tail_file,  # pyright: ignore[reportPrivateUsage]
)
from exo.shared.types.state import State


def _make_api() -> Any:
    """Create a minimal API instance with the log routes registered."""
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    app.get("/v1/logs")(api.list_logs)
    # Match main.py registration order: local {name}/raw before proxy routes
    # so Starlette's trie matches /v1/logs/main/raw correctly.
    app.get("/v1/logs/{name}/raw")(api.get_log_raw)
    app.get("/v1/logs/{node_id}/{name}")(api.get_node_log_tail)
    app.get("/v1/logs/{node_id}/{name}/raw")(api.get_node_log_raw)
    app.get("/v1/logs/{name}")(api.get_log_tail)
    api.state = State()
    return api


@pytest.fixture(autouse=True)
def _isolated_log_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the whitelist at a temp dir so tests never touch a real ~/.exo log."""
    log_path = tmp_path / "exo.log"
    monkeypatch.setattr(api_main, "_LOG_FILES", {"main": log_path})
    return log_path


def test_tail_file_returns_last_n_lines(tmp_path: Path) -> None:
    path = tmp_path / "a.log"
    path.write_text("\n".join(f"line{i}" for i in range(10)))

    content, truncated = _tail_file(path, max_lines=3)

    assert content == "line7\nline8\nline9"
    assert truncated is True


def test_tail_file_returns_everything_when_under_limit(tmp_path: Path) -> None:
    path = tmp_path / "a.log"
    path.write_text("line0\nline1")

    content, truncated = _tail_file(path, max_lines=1000)

    assert content == "line0\nline1"
    assert truncated is False


def test_tail_file_zero_lines_returns_empty(tmp_path: Path) -> None:
    """Regression test: max_lines<=0 must not fall back to returning everything
    (Python's `lines[-0:]` is the full list, not an empty slice)."""
    path = tmp_path / "a.log"
    path.write_text("line0\nline1\nline2")

    content, truncated = _tail_file(path, max_lines=0)

    assert content == ""
    assert truncated is True


def test_tail_file_negative_lines_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "a.log"
    path.write_text("line0\nline1")

    content, truncated = _tail_file(path, max_lines=-5)

    assert content == ""
    assert truncated is True


def test_list_logs_only_includes_existing_files(_isolated_log_files: Path) -> None:
    api = _make_api()
    client = TestClient(api.app)

    response = client.get("/v1/logs")
    assert response.status_code == 200
    assert response.json() == {"logs": []}

    _isolated_log_files.write_text("hello\n")
    response = client.get("/v1/logs")
    data = response.json()
    assert len(data["logs"]) == 1
    assert data["logs"][0]["name"] == "main"
    assert data["logs"][0]["fileSize"] == len("hello\n")


def test_get_log_tail_unknown_name_returns_404() -> None:
    api = _make_api()
    client = TestClient(api.app)

    response = client.get("/v1/logs/does_not_exist")
    assert response.status_code == 404


def test_get_log_tail_missing_file_returns_404(_isolated_log_files: Path) -> None:
    api = _make_api()
    client = TestClient(api.app)

    response = client.get("/v1/logs/main")
    assert response.status_code == 404


def test_get_log_tail_returns_content(_isolated_log_files: Path) -> None:
    _isolated_log_files.write_text("line0\nline1\nline2")
    api = _make_api()
    client = TestClient(api.app)

    response = client.get("/v1/logs/main?lines=2")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "main"
    assert data["content"] == "line1\nline2"
    assert data["truncated"] is True


def test_get_log_raw_unknown_name_returns_404() -> None:
    api = _make_api()
    client = TestClient(api.app)

    response = client.get("/v1/logs/does_not_exist/raw")
    assert response.status_code == 404


def test_get_log_raw_returns_full_file(_isolated_log_files: Path) -> None:
    _isolated_log_files.write_text("full contents\n")
    api = _make_api()
    client = TestClient(api.app)

    response = client.get("/v1/logs/main/raw")
    assert response.status_code == 200
    assert response.text == "full contents\n"


# ── Per-node log proxy tests ────────────────────────────────────────────────


def _register_node(api: Any, node_id: str, host: str, port: int) -> None:
    """Inject a NodeIdentity into the API state so the proxy can resolve it."""
    from exo.shared.types.profiling import NodeIdentity

    identities = dict(api.state.node_identities)
    identities[node_id] = NodeIdentity(
        friendly_name=f"test-{node_id[:8]}",
        api_host=host,
        api_port=port,
    )
    # State is frozen — rebuild with updated identities.
    api.state = api.state.model_copy(update={"node_identities": identities})


def test_node_log_tail_unknown_node_returns_404() -> None:
    api = _make_api()
    client = TestClient(api.app)

    response = client.get("/v1/logs/nonexistent-node/main")
    assert response.status_code == 404
    assert "Unknown or unreachable" in response.json()["error"]["message"]


def test_node_log_tail_unknown_log_name_returns_404() -> None:
    api = _make_api()
    _register_node(api, "abc-123", "127.0.0.1", 9999)
    client = TestClient(api.app)

    response = client.get("/v1/logs/abc-123/bogus_log")
    assert response.status_code == 404
    assert "Unknown log" in response.json()["error"]["message"]


def test_node_log_tail_proxies_to_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the proxy forwards to the advertised node endpoint and returns
    the remote's LogTailResponse shape."""
    import httpx

    api = _make_api()
    _register_node(api, "remote-node-1", "127.0.0.1", 9999)

    async def _fake_get(self: Any, url: str, *args: Any, **kwargs: Any):
        # The proxy must hit the *advertised* endpoint, not the dashboard
        # path, and forward the requested log + lines.
        assert "127.0.0.1:9999" in url
        assert "/v1/logs/main" in url
        assert "lines=500" in url

        class _FakeResp:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, object]:
                return {
                    "name": "main",
                    "content": "remote-line-1\nremote-line-2",
                    "truncated": False,
                }

        return _FakeResp()

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "_FakeClient":
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        get = _fake_get

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    client = TestClient(api.app)

    response = client.get("/v1/logs/remote-node-1/main?lines=500")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "main"
    assert "remote-line-1" in data["content"]
    assert data["truncated"] is False


def test_node_log_tail_remote_404_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 404 from the remote node (e.g. missing log file there) must surface
    as a local 404 with the remote's detail message."""
    import httpx

    api = _make_api()
    _register_node(api, "remote-node-1", "127.0.0.1", 9999)

    class _FakeResp:
        status_code = 404

        def raise_for_status(self) -> None:
            raise httpx.HTTPStatusError(
                "Not Found",
                request=httpx.Request("GET", "http://127.0.0.1:9999/v1/logs/main"),
                response=self,  # pyright: ignore[reportCallIssue]
            )

        def json(self) -> dict[str, str]:
            return {"detail": "Log file not found: main"}

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "_FakeClient":
            return self

        async def __aexit__(self, *args: Any) -> bool:
            return False

        async def get(self, url: str, *args: Any, **kwargs: Any) -> _FakeResp:
            return _FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    client = TestClient(api.app)

    response = client.get("/v1/logs/remote-node-1/main")
    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Log file not found: main"


def test_node_log_raw_unknown_node_returns_404() -> None:
    api = _make_api()
    client = TestClient(api.app)

    response = client.get("/v1/logs/nonexistent-node/main/raw")
    assert response.status_code == 404


def test_node_log_tail_unreachable_node_returns_502() -> None:
    """A registered node whose port isn't listening should yield 502."""
    api = _make_api()
    # Port 1 is almost certainly not running an exo API
    _register_node(api, "dead-node", "127.0.0.1", 1)
    client = TestClient(api.app)

    response = client.get("/v1/logs/dead-node/main")
    assert response.status_code == 502
    assert "Could not reach node" in response.json()["error"]["message"]


# ── _parse_log_errors bare-segfault/OOM detection ─────────────────────────────


def test_parse_log_errors_catches_bare_segfault() -> None:
    """Bare 'Segmentation fault: 11' lines (no loguru prefix) must surface
    as synthetic ERROR entries with source='(signal)'."""
    from exo.api.main import _parse_log_errors  # pyright: ignore[reportPrivateUsage]

    content = (
        "[ 2026-09-09 12:00:00.000 | INFO  | exo.worker ] starting\n"
        "Segmentation fault: 11\n"
        "[ 2026-09-09 12:00:01.000 | INFO  | exo.worker ] next\n"
    )
    entries = _parse_log_errors(content, "runner_stderr", context_lines=2)
    assert len(entries) == 1
    assert entries[0].level == "ERROR"
    assert entries[0].source == "(signal)"
    assert entries[0].message == "Segmentation fault: 11"
    assert entries[0].context is not None
    assert any("Segmentation fault" in c for c in entries[0].context)


def test_parse_log_errors_catches_oom_killed() -> None:
    """Linux OOM-killer lines ('Killed process ...') must be captured."""
    from exo.api.main import _parse_log_errors  # pyright: ignore[reportPrivateUsage]

    content = "Killed - process 12345 (exo) OOM killed\n"
    entries = _parse_log_errors(content, "runner_stderr", context_lines=0)
    assert len(entries) == 1
    assert entries[0].level == "ERROR"
    assert entries[0].source == "(signal)"
    assert "OOM killed" in entries[0].message


def test_parse_log_errors_segfault_does_not_contaminate_prior_warning() -> None:
    """A bare segfault following a WARNING must NOT be appended to the
    WARNING's collapsed message — it must become its own entry."""
    from exo.api.main import _parse_log_errors  # pyright: ignore[reportPrivateUsage]

    content = (
        "[ 2026-09-09 12:00:00.000 | WARNING | exo.master ] peer timeout\n"
        "Segmentation fault: 11\n"
    )
    entries = _parse_log_errors(content, "runner_stderr", context_lines=2)
    assert len(entries) == 2
    assert entries[0].level == "WARNING"
    assert entries[0].message == "peer timeout"
    assert "Segmentation" not in entries[0].message
    assert entries[1].level == "ERROR"
    assert entries[1].source == "(signal)"


def test_parse_log_errors_strips_ansi_codes() -> None:
    """ANSI color escapes from TTY-mode loguru must be stripped."""
    from exo.api.main import _parse_log_errors  # pyright: ignore[reportPrivateUsage]

    content = "[ 2026-09-09 12:00:00.000 | ERROR | exo.api ] \x1b[31mred\x1b[0m\n"
    entries = _parse_log_errors(content, "main", context_lines=0)
    assert len(entries) == 1
    assert "\x1b" not in entries[0].message
    assert entries[0].message == "red"
