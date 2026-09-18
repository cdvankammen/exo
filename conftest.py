"""Session-start environment bootstrap (root conftest).

Loaded by pytest BEFORE any package/test-module conftest and before any
test-module import. This is the ONLY place that can safely set environment
variables that exo.shared.constants reads at import time (e.g.
EXO_API_TOKEN) — package-level conftests load too late: pytest imports
test modules in collection order, and modules from earlier directories
(shared/master/...) pull in exo.shared.constants while the env is still
unset, freezing EXO_API_TOKEN=None for the whole session.
"""

import os
from pathlib import Path

_ENV_FILE = Path(__file__).parent / "src" / "exo" / "api" / "tests" / ".exo_api_token.env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())