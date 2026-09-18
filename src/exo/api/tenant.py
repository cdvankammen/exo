"""Multi-tenant API key authentication and per-key quota enforcement.

Augments (never replaces) the single shared ``EXO_API_TOKEN`` dashboard token
flow. When multi-tenancy is enabled the API installs this middleware in
addition to (or instead of) the legacy bearer-token middleware:

* ``Authorization: Bearer <legacy EXO_API_TOKEN>`` — still accepted as a
  wildcard admin credential (dashboard/operator flow keeps working).
* ``Authorization: Bearer <tenant key>`` — authenticated against the tenant
  key store and subject to per-key quotas (rate + daily token caps).

Key sources (highest precedence first):
1. ``EXO_TENANT_CONFIG`` env — inline JSON mapping key -> quota spec.
2. ``~/.exo/tenants.json`` — same JSON shape, operator-managed fleets.
3. ``~/.exo/tenant.db`` — SQLite key store managed via admin endpoints
   (``POST /v1/tenants``, ``DELETE /v1/tenants/{key}``).

A key present in both a file source and the DB takes quotas from the DB (the
DB is the runtime source of truth once it exists); file keys act as
bootstrap/air-gapped config.

Quota semantics
---------------
* ``rate_limit``: max requests per 60s rolling window (per process).
* ``daily_tokens``: max *estimated* tokens per UTC day (input chars // 4).

Coarse character-based estimates are used because the API layer has no
tokenizer — the same heuristic ``EXO_CHARS_PER_TOKEN`` uses for input-length
guarding. Metering applies to chat-style endpoints only (``/v1/chat/completions``,
``/v1/messages``, ``/v1/responses``, ``/ollama/*``); admin/state routes are
not metered and don't consume quota.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from exo.shared.constants import EXO_CHARS_PER_TOKEN

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TENANT_DB_PATH = Path(
    os.getenv("EXO_TENANT_DB", str(Path.home() / ".exo" / "tenant.db"))
)
TENANT_JSON_PATH = Path(
    os.getenv("EXO_TENANT_JSON", str(Path.home() / ".exo" / "tenants.json"))
)
TENANT_CONFIG_ENV = "EXO_TENANT_CONFIG"

_RATE_WINDOW_SECONDS = 60
_DAILY_WINDOW_SECONDS = 24 * 60 * 60

# Keys with no explicit quota get these defaults when multi-tenancy is on.
_DEFAULT_RATE_LIMIT = int(os.getenv("EXO_TENANT_DEFAULT_RATE_LIMIT", "60"))
_DEFAULT_DAILY_TOKENS = int(os.getenv("EXO_TENANT_DEFAULT_DAILY_TOKENS", "1000000"))

# Endpoints that consume quota (token metering). Anything else is admin/state.
_METERED_SUFFIXES = ("/chat/completions",)
_METERED_EXACT = ("/v1/messages", "/v1/responses")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TenantKeySpec:
    """Quota spec for a tenant key."""

    key: str
    rate_limit: int
    daily_tokens: int
    # Optional model allowlist: model names (exact) or "*" for all. Empty = all.
    models: tuple[str, ...] = ("*",)
    note: str = ""


@dataclass
class TenantUsage:
    """In-memory rolling usage counters for one key (per process)."""

    key: str
    request_stamps: list[tuple[float, int]] = field(default_factory=list)
    daily_tokens: dict[str, int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def prune(self, now_ts: float, window: int = _RATE_WINDOW_SECONDS) -> None:
        cutoff = now_ts - window
        self.request_stamps = [s for s in self.request_stamps if s[0] >= cutoff]

    def rate_count(self, now_ts: float) -> int:
        return len(self.request_stamps)

    def day_total(self, day: str) -> int:
        return self.daily_tokens.get(day, 0)


# ---------------------------------------------------------------------------
# SQLite key store
# ---------------------------------------------------------------------------


class TenantDB:
    """SQLite-backed tenant key store (stdlib sqlite3, no new dependency)."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS tenants (
        key          TEXT PRIMARY KEY,
        display_name TEXT NOT NULL DEFAULT '',
        rate_limit   INTEGER NOT NULL,
        daily_tokens INTEGER NOT NULL,
        models       TEXT NOT NULL DEFAULT '*',
        note         TEXT NOT NULL DEFAULT '',
        created_at   TEXT NOT NULL,
        last_used_at TEXT
    );
    """

    def __init__(self, path: Path = TENANT_DB_PATH) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._ensure()

    def _ensure(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            conn = sqlite3.connect(str(self.path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.executescript(self._SCHEMA)
            conn.commit()
            self._conn = conn

    def _row_to_spec(self, row: sqlite3.Row) -> TenantKeySpec:
        models = tuple(
            m.strip() for m in (row["models"] or "*").split(",") if m.strip()
        )
        return TenantKeySpec(
            key=row["key"],
            rate_limit=row["rate_limit"],
            daily_tokens=row["daily_tokens"],
            models=models or ("*",),
            note=row["note"],
        )

    def get(self, key: str) -> TenantKeySpec | None:
        with self._lock:
            if self._conn is None:
                return None
            cur = self._conn.execute("SELECT * FROM tenants WHERE key = ?", (key,))
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_spec(row)

    def upsert(
        self,
        key: str,
        *,
        rate_limit: int,
        daily_tokens: int,
        models: tuple[str, ...] = ("*",),
        note: str = "",
        display_name: str = "",
    ) -> None:
        with self._lock:
            if self._conn is None:
                return
            self._conn.execute(
                """
                INSERT INTO tenants (key, display_name, rate_limit, daily_tokens,
                                     models, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    rate_limit = excluded.rate_limit,
                    daily_tokens = excluded.daily_tokens,
                    models = excluded.models,
                    note = excluded.note
                """,
                (
                    key,
                    display_name,
                    int(rate_limit),
                    int(daily_tokens),
                    ",".join(models),
                    note,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                ),
            )
            self._conn.commit()

    def delete(self, key: str) -> bool:
        with self._lock:
            if self._conn is None:
                return False
            cur = self._conn.execute("DELETE FROM tenants WHERE key = ?", (key,))
            self._conn.commit()
            return cur.rowcount > 0

    def list(self) -> list[TenantKeySpec]:
        with self._lock:
            if self._conn is None:
                return []
            cur = self._conn.execute("SELECT * FROM tenants ORDER BY created_at")
            return [self._row_to_spec(r) for r in cur.fetchall()]

    def touch(self, key: str) -> None:
        with self._lock:
            if self._conn is None:
                return
            self._conn.execute(
                "UPDATE tenants SET last_used_at = ? WHERE key = ?",
                (datetime.now(timezone.utc).isoformat(timespec="seconds"), key),
            )
            self._conn.commit()


# ---------------------------------------------------------------------------
# Static sources + merged registry
# ---------------------------------------------------------------------------


def _load_static_specs() -> dict[str, TenantKeySpec]:
    """Load key specs from env JSON and the JSON file (env wins)."""
    merged: dict[str, TenantKeySpec] = {}

    def _absorb(raw: Any) -> None:
        if not isinstance(raw, dict):
            return
        for key, spec in raw.items():
            if not isinstance(key, str) or not key:
                continue
            if isinstance(spec, dict):
                rate_limit = int(spec.get("rate_limit", _DEFAULT_RATE_LIMIT))
                daily_tokens = int(spec.get("daily_tokens", _DEFAULT_DAILY_TOKENS))
                models_raw = spec.get("models", "*")
                if isinstance(models_raw, str):
                    models = tuple(m.strip() for m in models_raw.split(",") if m.strip()) or ("*",)
                elif isinstance(models_raw, list):
                    models = tuple(str(m) for m in models_raw) or ("*",)
                else:
                    models = ("*",)
                note = str(spec.get("note", ""))
            else:
                rate_limit = _DEFAULT_RATE_LIMIT
                daily_tokens = _DEFAULT_DAILY_TOKENS
                models = ("*",)
                note = ""
            merged[key] = TenantKeySpec(
                key=key,
                rate_limit=rate_limit,
                daily_tokens=daily_tokens,
                models=models,
                note=note,
            )

    env_raw = os.getenv(TENANT_CONFIG_ENV)
    if env_raw:
        try:
            _absorb(json.loads(env_raw))
        except json.JSONDecodeError:
            from loguru import logger

            logger.warning("EXO_TENANT_CONFIG is not valid JSON; ignoring tenant env config")

    if TENANT_JSON_PATH.exists():
        try:
            _absorb(json.loads(TENANT_JSON_PATH.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            from loguru import logger

            logger.warning(f"Could not read tenant JSON file {TENANT_JSON_PATH}; ignoring it")

    return merged


class TenantRegistry:
    """Merged view of static key specs + DB-backed keys (DB wins)."""

    def __init__(
        self,
        static: dict[str, TenantKeySpec] | None = None,
        db: TenantDB | None = None,
    ) -> None:
        self.static = static if static is not None else _load_static_specs()
        self.db = db if db is not None else TenantDB()
        self._usage: dict[str, TenantUsage] = {}

    # -- key resolution ----------------------------------------------------

    def resolve(self, key: str) -> TenantKeySpec | None:
        """Resolve a key: DB first (runtime source of truth), then static."""
        spec = self.db.get(key)
        if spec is not None:
            return spec
        return self.static.get(key)

    def all_keys(self) -> dict[str, TenantKeySpec]:
        out: dict[str, TenantKeySpec] = {}
        for spec in self.db.list():
            out[spec.key] = spec
        out.update(self.static)  # static fills gaps only
        return out

    # -- usage tracking ----------------------------------------------------

    def usage(self, key: str) -> TenantUsage:
        if key not in self._usage:
            self._usage[key] = TenantUsage(key=key)
        return self._usage[key]

    def check_rate(self, key: str) -> tuple[bool, int | None]:
        """Check (and record) the 60s rate window for *key*.

        Returns ``(allowed, retry_after_seconds)``. Rejected requests are not
        recorded, so a burst of rejects cannot push the key further into
        rate-limit debt.
        """
        usage = self.usage(key)
        now_ts = time.monotonic()
        with usage.lock:
            usage.prune(now_ts)
            spec = self.resolve(key)
            if spec is None:
                return (False, None)
            if usage.rate_count(now_ts) >= spec.rate_limit:
                if usage.request_stamps:
                    retry_after = int(
                        _RATE_WINDOW_SECONDS - (now_ts - usage.request_stamps[0][0])
                    )
                    retry_after = max(1, retry_after)
                else:
                    retry_after = _RATE_WINDOW_SECONDS
                return (False, retry_after)
            usage.request_stamps.append((now_ts, 0))
            return (True, None)

    def record_tokens(self, key: str, est_tokens: int) -> bool:
        """Record *est_tokens* against the key's daily cap.

        Returns True if the tokens fit under the cap (and were recorded);
        False if the daily quota would be exceeded (nothing recorded).
        """
        if est_tokens <= 0:
            return True
        usage = self.usage(key)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with usage.lock:
            spec = self.resolve(key)
            if spec is None:
                return False
            if usage.day_total(day) + est_tokens > spec.daily_tokens:
                return False
            usage.daily_tokens[day] = usage.day_total(day) + est_tokens
            return True

    def is_allowed_model(self, key: str, model: str) -> bool:
        spec = self.resolve(key)
        if spec is None:
            return False
        if "*" in spec.models:
            return True
        return model in spec.models

    # -- admin ----------------------------------------------------------------

    def create_key(
        self,
        *,
        rate_limit: int,
        daily_tokens: int,
        models: tuple[str, ...] = ("*",),
        note: str = "",
        display_name: str = "",
        key: str | None = None,
    ) -> str:
        """Create (or update) a key; returns the key string."""
        new_key = key if key else secrets.token_urlsafe(24)
        self.db.upsert(
            new_key,
            rate_limit=rate_limit,
            daily_tokens=daily_tokens,
            models=models,
            note=note,
            display_name=display_name,
        )
        return new_key

    def delete_key(self, key: str) -> bool:
        return self.db.delete(key)

    def list_keys(self) -> list[TenantKeySpec]:
        return self.db.list()


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


def _unauthorized_body() -> dict[str, Any]:
    return {
        "error": {
            "message": "Unauthorized. Set the Authorization: Bearer <tenant key> header.",
            "type": "Unauthorized",
            "code": 401,
            "error_code": "UNAUTHORIZED",
        }
    }


def _rate_limited_body(error_code: str, message: str) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": "RateLimitError",
            "code": 429,
            "error_code": error_code,
        }
    }


def multi_tenant_auth_middleware(
    registry: TenantRegistry,
    legacy_token: str | None = None,
):
    """FastAPI http middleware enforcing per-key auth + rate quotas.

    * Public dashboard paths stay exempt (defense in depth; the API wiring
      also orders this after the legacy middleware).
    * ``legacy_token`` (EXO_API_TOKEN) is accepted as an admin/wildcard key
      that bypasses quotas.
    * Authenticated tenant keys are attached to ``request.state.tenant``
      for downstream metering.
    """
    from exo.api.auth import is_public_dashboard_path

    async def middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[StreamingResponse]],
    ) -> StreamingResponse | JSONResponse:
        if is_public_dashboard_path(request.url.path):
            return await call_next(request)

        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return JSONResponse(status_code=401, content=_unauthorized_body())
        key = auth[len("Bearer ") :].strip()
        if not key:
            return JSONResponse(status_code=401, content=_unauthorized_body())

        if legacy_token is not None and secrets.compare_digest(key, legacy_token):
            request.state.tenant = {"key": legacy_token, "admin": True}
            return await call_next(request)

        spec = registry.resolve(key)
        if spec is None:
            return JSONResponse(status_code=401, content=_unauthorized_body())

        allowed, retry_after = registry.check_rate(key)
        if not allowed:
            headers = {"Retry-After": str(retry_after)} if retry_after else {}
            return JSONResponse(
                status_code=429,
                content=_rate_limited_body(
                    "RATE_LIMITED", "Rate limit exceeded for this API key."
                ),
                headers=headers,
            )

        request.state.tenant = {
            "key": key,
            "admin": False,
            "models": spec.models,
        }
        response = await call_next(request)
        try:  # best-effort last-used bookkeeping
            registry.db.touch(key)
        except Exception:
            pass
        return response

    return middleware


def _estimate_tokens_from_body(body: Any) -> int:
    """Estimate input tokens from a parsed chat-style JSON body."""
    if not isinstance(body, dict):
        return 0
    total_chars = 0
    messages = body.get("messages") or body.get("input")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        total_chars += len(part["text"])
    if total_chars <= 0:
        return 0
    return max(1, total_chars // max(EXO_CHARS_PER_TOKEN, 1))


def meter_chat_endpoint_middleware(registry: TenantRegistry):
    """Middleware that meters token usage on chat-style endpoints.

    Placed AFTER the auth middleware (which sets ``request.state.tenant``).
    Reads the request body once (caching it into ``request.state._body`` so
    the route handlers can re-read it via ``await request.json()`` without
    error), records estimated tokens against the authenticated key's daily
    cap, and returns 429 ``QUOTA_EXCEEDED`` when the cap would be exceeded.
    """

    async def middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[StreamingResponse]],
    ) -> StreamingResponse | JSONResponse:
        tenant = getattr(request.state, "tenant", None)
        path = request.url.path
        metered = path.endswith(_METERED_SUFFIXES) or path in _METERED_EXACT
        # Only meter tenant (non-admin) keys on chat-style endpoints.
        if tenant is None or tenant.get("admin") or not metered:
            return await call_next(request)

        raw = await request.body()
        body: Any = None
        if raw:
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = None
        request.state._body = body

        est_tokens = _estimate_tokens_from_body(body)
        if not registry.record_tokens(tenant["key"], est_tokens):
            return JSONResponse(
                status_code=429,
                content=_rate_limited_body(
                    "QUOTA_EXCEEDED", "Daily token quota exceeded for this API key."
                ),
            )
        return await call_next(request)

    return middleware