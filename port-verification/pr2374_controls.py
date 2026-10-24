"""Negative controls for exo issue #2374 (prefill memory budget guard).

Why controls at all: `test_prefill_budget.py` proves the arithmetic is right
for injected numbers. It cannot prove the guard is *wired into production* --
a `_enforce_prefill_budget` that is defined but never called, or called in a
branch nothing reaches, passes every unit test while the runner still
SIGABRTs. These checks close that gap by asserting the call path on disk.

Run against the worktree:

    cd /Users/chris/.hermes/worktrees/exo-issue2374-wired-preflight
    PYTHONPATH=$PWD/src /Users/chris/exo/.venv/bin/python \
        /Users/chris/exo/port-verification/pr2374_controls.py

Exits non-zero on the first control that fails, naming it.
"""

from __future__ import annotations

import ast
import pathlib
import sys

WORKTREE = pathlib.Path("/Users/chris/.hermes/worktrees/exo-issue2374-wired-preflight")
GENERATE = WORKTREE / "src/exo/worker/engines/mlx/generator/generate.py"
BATCH_GEN = WORKTREE / "src/exo/worker/runner/llm_inference/batch_generator.py"
PREBILL = WORKTREE / "src/exo/worker/engines/mlx/generator/prefill_budget.py"

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def _calls_to(path: pathlib.Path, func_name: str) -> list[tuple[str, int]]:
    """Every call site of `func_name` in `path`, as (function, line)."""
    tree = ast.parse(path.read_text())
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call):
                target = inner.func
                name = getattr(target, "id", None) or getattr(target, "attr", None)
                if name == func_name:
                    found.append((node.name, inner.lineno))
    return found


print("=" * 72)
print("#2374 prefill budget guard -- production wiring controls")
print("=" * 72)
print(f"worktree: {WORKTREE}\n")

# ---------------------------------------------------------------------------
print("1. The guard is invoked from the real prefill entrypoint")
# ---------------------------------------------------------------------------
sites = _calls_to(GENERATE, "_enforce_prefill_budget")
check(
    "_enforce_prefill_budget is actually called",
    len(sites) >= 1,
    f"call sites: {sites}",
)
check(
    "the call is inside prefill() itself, not a dead helper",
    any(fn == "prefill" for fn, _ in sites),
    f"enclosing functions: {[fn for fn, _ in sites]}",
)
check(
    "called exactly once (a second call site would double-refuse)",
    len(sites) == 1,
    f"{len(sites)} call sites",
)

# ---------------------------------------------------------------------------
print("\n2. The refusal is caught where a task can fail cleanly")
# ---------------------------------------------------------------------------
handlers = _calls_to(BATCH_GEN, "_send_error")
check(
    "_send_error is reachable in batch_generator",
    len(handlers) >= 1,
    f"{len(handlers)} call sites",
)
budget_handlers: list[str] = []
tree = ast.parse(BATCH_GEN.read_text())
for node in ast.walk(tree):
    if isinstance(node, ast.ExceptHandler) and node.type is not None:
        rendered = ast.unparse(node.type)
        if "PrefillBudgetError" in rendered:
            budget_handlers.append(node.name or "<unnamed>")
check(
    "an `except PrefillBudgetError` arm exists",
    len(budget_handlers) == 1,
    f"arms: {budget_handlers}",
)
check(
    "the arm binds the exception (needed for _send_error)",
    budget_handlers and budget_handlers[0] != "<unnamed>",
    f"arm names: {budget_handlers}",
)

# Ordering: the specific arm must precede the catch-all `except Exception`,
# or the generic handler wins and the guard loses its distinct handling.
try:
    lines = BATCH_GEN.read_text().splitlines()
    budget_line = next(
        i for i, ln in enumerate(lines) if "except PrefillBudgetError" in ln
    )
    generic_line = next(i for i, ln in enumerate(lines) if "except Exception" in ln)
    check(
        "the specific arm precedes the catch-all `except Exception`",
        budget_line < generic_line,
        f"PrefillBudgetError@line {budget_line + 1} vs Exception@line {generic_line + 1}",
    )
except StopIteration:
    check("located both handler arms", False, "one of the arms is missing")

# ---------------------------------------------------------------------------
print("\n3. The module the handlers import is the one under test")
# ---------------------------------------------------------------------------
sys.path.insert(0, str(WORKTREE / "src"))
try:
    from exo.worker.engines.mlx.generator.prefill_budget import PrefillBudgetError

    imported = PrefillBudgetError
except Exception as exc:  # pragma: no cover
    imported = None  # type: ignore[assignment]
    check("importing the guard module succeeds", False, repr(exc))

if imported is not None:
    check(
        "PrefillBudgetError subclasses RuntimeError",
        issubclass(imported, RuntimeError),
        f"MRO: {[c.__name__ for c in imported.__mro__[:4]]}",
    )
    # `__module__` is only the dotted name, so it cannot prove provenance.
    # The .pth in the project venv points at the MAIN repo's src, which would
    # shadow this worktree -- so check the file the class actually came from.
    module_file = getattr(sys.modules[imported.__module__], "__file__", "") or ""
    check(
        "the imported class comes from THIS worktree",
        str(WORKTREE) in module_file,
        f"module file: {module_file}",
    )
    check(
        "it is not a bare Exception subclass (would change handler shape)",
        imported.__bases__ == (RuntimeError,),
        f"direct bases: {[b.__name__ for b in imported.__bases__]}",
    )

# ---------------------------------------------------------------------------
print("\n4. The message cannot be mistaken for a retryable OOM")
# ---------------------------------------------------------------------------
if imported is not None:
    try:
        from exo.worker.engines.mlx.generator.prefill_budget import (
            KvGeometry,
            check_prefill_budget,
        )
        from exo.shared.types.memory import Memory

        class _Cfg:
            num_hidden_layers = 48
            num_attention_heads = 64
            num_key_value_heads = 8
            hidden_size = 8192
            torch_dtype = "bfloat16"

        class _M:
            args = _Cfg()

        err = check_prefill_budget(
            model=_M(),
            prompt_tokens=19_000,
            kv_layers=24,
            weights=Memory.from_gb(8),
            weights_source="control",
            limit=Memory.from_gb(11),
        )
        check("a real refusal is produced for the reported case", err is not None)
        if err is not None:
            msg = str(err).lower()
            markers = (
                "failed to allocate",
                "insufficient memory",
                "out of memory",
                "oom",
                "bad_alloc",
                "allocator",
            )
            tripped = [m for m in markers if m in msg]
            check(
                "message trips NO runner._is_oom() marker",
                not tripped,
                f"tripped: {tripped}" if tripped else "clean",
            )
            check(
                "message names the sysctl remedy",
                "iogpu.wired_limit_mb" in str(err),
            )
            check(
                "message advertises a workable ceiling",
                err.max_prompt_tokens > 0,
                f"ceiling={err.max_prompt_tokens}",
            )
        # The geometry object is only needed to prove the import surface is
        # intact for the arithmetic tests.
        check("KvGeometry still importable", KvGeometry is not None)
    except Exception as exc:  # pragma: no cover
        check("refusal-path control ran", False, repr(exc))

# ---------------------------------------------------------------------------
print("\n5. guard() is defined with the signature the call site uses")
# ---------------------------------------------------------------------------
try:
    src = PREBILL.read_text()
    tree = ast.parse(src)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "check_prefill_budget"
    )
    # kwonlyargs are plain `ast.arg` (no .default); the matching defaults
    # sit in `kw_defaults`, positionally aligned (None == required).
    kwonly = [a.arg for a in fn.args.kwonlyargs]
    defaults = dict(zip(kwonly, fn.args.kw_defaults))
    check(
        "check_prefill_budget takes the kwargs generate.py passes",
        {"model", "prompt_tokens", "kv_layers", "kv_bits"} <= set(kwonly),
        f"kwonly args: {sorted(kwonly)}",
    )
    limit_default = defaults.get("limit")
    check(
        "limit defaults to None (= auto-detect, NOT 'no ceiling')",
        limit_default is not None and ast.unparse(limit_default) == "None",
        f"limit default: {ast.unparse(limit_default) if limit_default else 'REQUIRED'}",
    )
    returns = ast.unparse(fn.returns) if fn.returns else ""
    check(
        "returns Optional[PrefillBudgetError] (raises nothing itself)",
        "PrefillBudgetError | None" in returns,
        f"returns: {returns}",
    )
except Exception as exc:  # pragma: no cover
    check("parsed check_prefill_budget", False, repr(exc))

print("\n" + "=" * 72)
if FAILURES:
    print(f"CONTROLS FAILED ({len(FAILURES)}):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("All controls passed.")
sys.exit(0)
