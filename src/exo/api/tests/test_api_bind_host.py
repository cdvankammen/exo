# pyright: reportUnusedFunction=false, reportAny=false
"""Tests for the configurable API bind host (T12).

Covers the exact ``cfg.bind`` value produced by ``API.run_api``:
- the bind string must be ``<EXO_API_HOST>:<port>``
- a hardened bind (127.0.0.1) must produce a localhost-only bind
- the default (0.0.0.0) must preserve cluster behavior
"""

from typing import Any

import pytest

from exo.api.main import API


def _make_api() -> API:
    api = object.__new__(API)  # type: ignore[call-arg]
    from fastapi import FastAPI

    api.app = FastAPI()  # type: ignore[attr-defined]
    api.port = 52415  # type: ignore[attr-defined]
    return api


async def _capture_bind(api: API, monkeypatch: pytest.MonkeyPatch, host: str) -> str:
    """Run API.run_api with hypercorn.serve stubbed out; return cfg.bind[0].

    EXO_API_HOST is read at import time (module constant), so we patch the
    module-level name run_api dereferences — same effect as launching with the
    env var set.
    """
    import exo.api.main as api_main

    captured: dict[str, Any] = {}

    async def fake_serve(app: Any, cfg: Any, **_: Any) -> None:  # noqa: ANN001
        captured["bind"] = list(cfg.bind)

    monkeypatch.setattr(api_main, "serve", fake_serve)
    monkeypatch.setattr(api_main, "EXO_API_HOST", host)
    # Shutdown trigger event: run_api waits on ev.wait(); we drive it with a set event.
    import anyio

    ev = anyio.Event()
    ev.set()
    await api.run_api(ev)
    return captured["bind"][0]


@pytest.mark.anyio
async def test_bind_defaults_to_all_interfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXO_API_HOST", raising=False)
    bind = await _capture_bind(_make_api(), monkeypatch, "0.0.0.0")
    assert bind == "0.0.0.0:52415"


@pytest.mark.anyio
async def test_bind_honors_exo_api_host(monkeypatch: pytest.MonkeyPatch) -> None:
    bind = await _capture_bind(_make_api(), monkeypatch, "127.0.0.1")
    assert bind == "127.0.0.1:52415"