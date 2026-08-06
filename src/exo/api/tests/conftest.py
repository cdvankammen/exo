# type: ignore
"""Pytest configuration for API tests.

Stubs the exo_rs Rust extension so API tests can run without a compiled
binary. The stub provides empty placeholder classes for symbols that are
imported at module level by exo.routing, but not exercised by these tests.
"""

import sys
import types
from unittest.mock import MagicMock

# Only install the stub if the real extension is not already available.
if "exo_rs" not in sys.modules:
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
