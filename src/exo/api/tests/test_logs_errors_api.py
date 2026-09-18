# pyright: reportUnusedFunction=false
"""Tests for GET /v1/logs/errors — structured WARNING/ERROR/CRITICAL parsing.

Covers the loguru-line parser, bare crash-marker detection (Traceback /
Segmentation fault / OOM / GPU faults), node_id attribution, level filtering,
and cluster merge behavior.
"""
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from exo.api import main as api_main
from exo.api.main import (
    API,
    _LOG_LINE_RE,
    _looks_like_crash_line,
    _parse_log_errors,
    _tail_file,  # pyright: ignore[reportPrivateUsage]
)

SAMPLE_LOG = """\
[ 2026-08-08 06:03:28.551 | INFO  | exo.main:startup:86 ] hello
[ 2026-08-08 06:03:29.001 | WARNING | exo.worker.main:run:12 ] mem pressure high
[ 2026-08-08 06:03:30.222 | ERROR | exo.worker.plan:plan:40 ] failed to place model
[ 2026-08-08 06:03:31.333 | CRITICAL | exo.worker.runner:run:9 ] runner crashed
[ 2026-08-08 06:03:32.444 | INFO  | exo.main:startup:86 ] done
"""


def _make_api() -> Any:
    """Create a minimal API instance with the log error route registered."""
    app = FastAPI()
    api = object.__new__(API)
    api.app = app
    api._setup_exception_handlers()  # pyright: ignore[reportPrivateUsage]
    api.node_id = "node-abc-123"  # type: ignore[attr-defined]
    api.state = type("S", (), {"node_identities": {}})()  # type: ignore[attr-defined]
    app.get("/v1/logs/errors")(api.get_log_errors)
    return api


@pytest.fixture(autouse=True)
def _isolated_log_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point the whitelist at a temp dir so tests never touch a real ~/.exo log."""
    log_path = tmp_path / "exo.log"
    monkeypatch.setattr(api_main, "_LOG_FILES", {"main": log_path})
    return log_path


# ── Parser ───────────────────────────────────────────────────────────────


def test_parse_log_errors_extracts_structured_lines() -> None:
    entries = _parse_log_errors(SAMPLE_LOG, source_log="main")

    assert len(entries) == 3
    levels = [e.level for e in entries]
    assert levels == ["WARNING", "ERROR", "CRITICAL"]
    assert entries[0].timestamp == "2026-08-08 06:03:29.001"
    assert entries[1].source == "exo.worker.plan:plan:40"
    assert entries[1].message == "failed to place model"
    assert entries[1].source_log == "main"


def test_parse_log_errors_attaches_node_id() -> None:
    entries = _parse_log_errors(SAMPLE_LOG, source_log="main", node_id="node-xyz")

    assert all(e.node_id == "node-xyz" for e in entries)


def test_parse_log_errors_context_around_error() -> None:
    entries = _parse_log_errors(SAMPLE_LOG, source_log="main", context_lines=1)

    # The ERROR entry should carry 1 line before + 1 line after.
    error = next(e for e in entries if e.level == "ERROR")
    assert error.context is not None
    assert len(error.context) == 3
    assert "mem pressure high" in error.context[0]
    assert "failed to place model" in error.context[1]
    assert "runner crashed" in error.context[2]


def test_parse_log_errors_no_context_when_disabled() -> None:
    entries = _parse_log_errors(SAMPLE_LOG, source_log="main", context_lines=0)

    assert all(e.context is None for e in entries)


def test_parse_log_errors_skips_info_lines() -> None:
    entries = _parse_log_errors(SAMPLE_LOG, source_log="main")

    assert all(e.level in ("WARNING", "ERROR", "CRITICAL") for e in entries)


# ── Bare crash markers ───────────────────────────────────────────────────


def test_looks_like_crash_line_detects_patterns() -> None:
    assert _looks_like_crash_line("Traceback (most recent call last):") == "ERROR"
    assert _looks_like_crash_line("Segmentation fault: 11") == "CRITICAL"
    assert _looks_like_crash_line("SegmentationFault") == "CRITICAL"
    assert _looks_like_crash_line("Killed") == "CRITICAL"
    assert _looks_like_crash_line("Out of memory while loading model") == "CRITICAL"
    assert _looks_like_crash_line("Metal API error: invalid resource") == "ERROR"
    assert _looks_like_crash_line("CUDA error: out of memory") == "ERROR"
    assert _looks_like_crash_line("some INFO line") is None
    assert _looks_like_crash_line("  File \"/a/b.py\", line 10") is None
    assert _looks_like_crash_line("    raise ValueError()") is None


def test_parse_log_errors_detects_bare_traceback() -> None:
    content = """\
plain line before
Traceback (most recent call last):
  File "/app/x.py", line 23, in main
    run()
  File "/app/x.py", line 45, in run
    raise ValueError("boom")
ValueError: boom
plain line after
"""
    entries = _parse_log_errors(content, source_log="runner_stderr")

    assert len(entries) == 1
    assert entries[0].level == "ERROR"
    assert entries[0].message == "Traceback (most recent call last):"
    assert entries[0].source_log == "runner_stderr"
    assert entries[0].source == ""
    ctx = entries[0].context or []
    assert any("ValueError: boom" in line for line in ctx)


def test_parse_log_errors_detects_bare_segfault_with_timestamp_from_prev_line() -> None:
    content = """\
[ 2026-08-08 06:10:00.000 | INFO | exo.runner:main:1 ] starting
Segmentation fault: 11
"""
    entries = _parse_log_errors(content, source_log="main")

    assert len(entries) == 1
    assert entries[0].level == "CRITICAL"
    assert entries[0].message == "Segmentation fault: 11"
    assert entries[0].timestamp == "2026-08-08 06:10:00.000"


def test_parse_log_errors_mixed_structured_and_crash() -> None:
    content = """\
[ 2026-08-08 06:03:29.001 | WARNING | exo.a:b:1 ] warn
[ 2026-08-08 06:03:30.222 | ERROR | exo.b:c:2 ] err
Killed
"""
    entries = _parse_log_errors(content, source_log="main")

    levels = sorted(e.level for e in entries)
    assert levels == ["CRITICAL", "ERROR", "WARNING"]


# ── Endpoint ─────────────────────────────────────────────────────────────


def test_get_log_errors_empty_when_no_file(_isolated_log_files: Path) -> None:
    api = _make_api()
    client = TestClient(api.app)

    response = client.get("/v1/logs/errors")
    assert response.status_code == 200
    assert response.json() == {"errors": [], "truncated": False}


def test_get_log_errors_parses_and_sorts_newest_first(
    _isolated_log_files: Path,
) -> None:
    _isolated_log_files.write_text(SAMPLE_LOG)
    api = _make_api()
    client = TestClient(api.app)

    response = client.get("/v1/logs/errors")
    assert response.status_code == 200
    data = response.json()
    assert len(data["errors"]) == 3
    assert data["errors"][0]["level"] == "CRITICAL"
    assert data["errors"][0]["nodeId"] == "node-abc-123"
    assert data["errors"][2]["level"] == "WARNING"


def test_get_log_errors_level_filter(_isolated_log_files: Path) -> None:
    _isolated_log_files.write_text(SAMPLE_LOG)
    api = _make_api()
    client = TestClient(api.app)

    response = client.get("/v1/logs/errors", params={"level": "ERROR"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["errors"]) == 1
    assert data["errors"][0]["level"] == "ERROR"

    # WARNING filter only
    response = client.get("/v1/logs/errors", params={"level": "WARNING"})
    data = response.json()
    assert len(data["errors"]) == 1
    assert data["errors"][0]["level"] == "WARNING"


def test_get_log_errors_invalid_level_returns_400(_isolated_log_files: Path) -> None:
    _isolated_log_files.write_text(SAMPLE_LOG)
    api = _make_api()
    client = TestClient(api.app)

    response = client.get("/v1/logs/errors", params={"level": "BOGUS"})
    assert response.status_code == 400


def test_get_log_errors_merges_remote_nodes(
    _isolated_log_files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cluster merge: remote identities with an API endpoint contribute entries."""
    _isolated_log_files.write_text(SAMPLE_LOG)

    class _Identity:
        api_host = "10.0.0.9"
        api_port = 8000
        friendly_name = "mini2"

    api = _make_api()
    api.state = type("S", (), {"node_identities": {"node-remote-9": _Identity()}})()  # type: ignore[attr-defined]

    remote_payload = {
        "errors": [
            {
                "timestamp": "2026-08-08 07:00:00.000",
                "level": "ERROR",
                "source": "exo.worker:main:1",
                "message": "remote failure",
                "source_log": "main",
            }
        ],
        "truncated": False,
    }

    async def _fake_get(url: str, timeout: float | None = None) -> Any:
        class _Resp:
            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict[str, Any]:
                return remote_payload

        return _Resp()

    import httpx

    class _FakeAsyncClient:
        def __init__(self, **kw: object) -> None:
            pass

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, *a: object) -> None:
            return None

        async def get(self, url: str, timeout: float | None = None) -> Any:
            return await _fake_get(url, timeout)

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    client = TestClient(api.app)
    response = client.get("/v1/logs/errors")
    assert response.status_code == 200
    data = response.json()
    # 3 local + 1 remote
    assert len(data["errors"]) == 4
    remote = next(e for e in data["errors"] if e["message"] == "remote failure")
    assert remote["nodeId"] == "node-remote-9"
    assert remote["sourceLog"].startswith("node-rem::")