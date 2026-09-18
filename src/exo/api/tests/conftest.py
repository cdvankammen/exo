# type: ignore
"""Pytest configuration for API tests.

Stubs the exo_rs Rust extension so API tests can run without a compiled
binary. The stub provides empty placeholder classes for symbols that are
imported at module level by exo.routing, but not exercised by these tests.
"""

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# Load the integration-test token env BEFORE any exo.* import. exo.shared.constants
# reads EXO_API_TOKEN once at import time (constants.py:171); pytest collects test
# modules alphabetically, so without this, test_add_peer.py etc. would freeze
# EXO_API_TOKEN=None before test_api_auth_integration.py's module-level env setup
# runs, silently disabling the auth middleware for the whole session.
_ENV_FILE = Path(__file__).with_name(".exo_api_token.env")
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())
del _ENV_FILE, _line, _k, _v

# Only install the stub if the real extension is not already available.
if "exo_rs" not in sys.modules:
    # If the REAL compiled extension exists, prefer it: the stub replaces the
    # whole module in sys.modules, which would otherwise shadow exo_rs for
    # later consumers in the same pytest session (e.g. utils pidfile tests).
    _real_rs = None
    try:
        import exo_rs as _real_rs  # type: ignore[import-not-found]
    except (ImportError, ModuleNotFoundError):
        _real_rs = None

    if _real_rs is not None:
        sys.modules["exo_rs"] = _real_rs

        # Still provide the API-test-specific placeholders for symbols the
        # real extension may not expose in every build. FromSwarm is only
        # stubbed when the real extension is absent (it is a pure-Python
        # compat shim in some builds; never replace a real one).
        for _name, _fallback in (
            ("AllQueuesFullError", type("AllQueuesFullError", (Exception,), {})),
            ("MessageTooLargeError", type("MessageTooLargeError", (Exception,), {})),
            (
                "NoPeersSubscribedToTopicError",
                type("NoPeersSubscribedToTopicError", (Exception,), {}),
            ),
            ("Keypair", MagicMock),
            ("NetworkingHandle", MagicMock),
            ("PidfileError", type("PidfileError", (Exception,), {})),
        ):
            if not hasattr(_real_rs, _name):
                setattr(_real_rs, _name, _fallback)
        del _name, _fallback
    else:
        _stub = types.ModuleType("exo_rs")

        # Symbols imported by exo.routing.connection_message
        class _FromSwarm:
            class Connection:
                peer_id: str = ""
                connected: bool = False

        _stub.FromSwarm = _FromSwarm

        # Symbols imported by exo.routing.router
        _stub.AllQueuesFullError = type("AllQueuesFullError", (Exception,), {})
        _stub.MessageTooLargeError = type("MessageTooLargeError", (Exception,), {})
        _stub.NoPeersSubscribedToTopicError = type(
            "NoPeersSubscribedToTopicError", (Exception,), {}
        )
        _stub.Keypair = MagicMock
        _stub.NetworkingHandle = MagicMock

        # Symbols imported by exo.main
        _stub.Pidfile = MagicMock
        _stub.PidfileError = type("PidfileError", (Exception,), {})

        sys.modules["exo_rs"] = _stub
