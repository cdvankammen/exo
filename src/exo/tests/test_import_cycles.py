"""Regression guard for exo's import-cycle break-points.

exo's module graph contains one logical 21-module strongly connected
component (MLX engine + image engine + runner/bootstrap cluster).  The
component is *not* a runtime import loop only because every edge that
would close a cycle is a function-level (lazy) import.  If any of those
lazy imports is hoisted to module scope, exo dies with a circular
import at startup.

This module guards both directions:

1. *Pin the break-points* (AST-based, no runtime imports): each known
   lazy edge must stay inside a function body.  A hoist fails here with
   a message naming the implicated module.  Known inversions owned by a
   FIX card live in ``TRACKED_EAGER_OVERRIDES``: they must stay in their
   documented state until the FIX lands (removal or silent reversal
   fails loudly instead).
2. *Pin the invariant globally* (eager-import graph): walking all of
   ``src/exo`` and keeping only imports that actually execute at import
   time (module scope, excluding ``if TYPE_CHECKING`` and its deferred
   alias ``exec("if TYPE_CHECKING...")``), the graph must be acyclic.
   A *new* eager cycle anywhere in the tree fails here.

The scan is pure static analysis (``ast`` only), so it runs anywhere
pytest runs — no MLX, no torch, no network.

Break-point line numbers are anchored to the import-dependency-graph
audit (HEAD 7eac88c6) — the *reference* line for humans.  The assertions
match on ``(module, imported names)`` structure, not line or formatting,
so legitimate line shifts (e.g. module-docstring additions) do not
false-fail, while a hoist, rename, or deletion of a break-point fails
loudly and forces a conscious re-pin.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

import pytest

_TEST_DIR = Path(__file__).resolve().parent  # src/exo/tests
_EXO_ROOT = _TEST_DIR.parent  # src/exo (the "exo" package root)

# The guard is worthless if it scans the wrong directory: assert the root
# actually contains the package we mean to protect.
assert (_EXO_ROOT / "main.py").is_file(), f"guard root {_EXO_ROOT} is not src/exo"

# The 9 lazy break-points that dissolve the 21-module import SCC.
# Each entry: (file rel to src/exo, ref line @ audit 7eac88c6, module, names).
# Matched by structure — (module, imported names) — not by line or layout.
LAZY_BREAK_POINTS: tuple[tuple[str, int, str, tuple[str, ...]], ...] = (
    (
        "worker/runner/bootstrap.py",
        154,
        "exo.worker.runner.runner",
        ("Runner",),
    ),
    (
        "worker/runner/bootstrap.py",
        158,
        "exo.worker.engines.image.builder",
        ("MfluxBuilder",),
    ),
    (
        "worker/runner/bootstrap.py",
        168,
        "exo.worker.engines.mlx.builder",
        ("MlxBuilder",),
    ),
    (
        "worker/engines/mlx/utils_mlx.py",
        138,
        "exo.worker.engines.mlx.auto_parallel",
        ("_get_tcp_relay",),
    ),
    (
        "worker/engines/mlx/utils_mlx.py",
        267,
        "exo.worker.engines.mlx.vision",
        ("VisionProcessor",),
    ),
    (
        "worker/engines/mlx/auto_parallel.py",
        277,
        "exo.worker.runner.bootstrap",
        ("logger",),
    ),
    (
        "worker/engines/mlx/auto_parallel.py",
        340,
        "exo.worker.runner.bootstrap",
        ("logger",),
    ),
    (
        "worker/engines/mlx/cache.py",
        1163,
        "exo.worker.engines.mlx.auto_parallel",
        ("PipelineFirstLayer", "PipelineLastLayer"),
    ),
    (
        "worker/runner/llm_inference/batch_generator.py",
        95,
        "exo.worker.engines.mlx.utils_mlx",
        ("mlx_force_oom",),
    ),
)

# A break-point whose import legitimately sits at module scope *today*
# because a FIX card owns removing the inversion: the guard tolerates the
# current documented state but fails if it silently changes (reverts to
# lazy, or disappears entirely without a deliberate re-pin).  Once the
# FIX card is done, this entry moves to LAZY_BREAK_POINTS.
#
# NOTE: the shared->worker inversion (shared/tracing.py importing
# exo.worker.runner.bootstrap.logger, audit t_8e9ece9b @ HEAD 7eac88c6) was
# FIXED while this guard was being written: tracing.py now imports the leaf
# logger from exo.utils.logging, deleting the edge entirely.  That is the
# *resolved* state the FIX card targeted — the guard below pins it: the
# bootstrap import must stay gone (tracked eagerly, fail-closed if the
# override table is emptied unintentionally or the edge reappears).
TRACKED_EAGER_OVERRIDES: tuple[tuple[str, int, str, tuple[str, ...]], ...] = (
    (
        "shared/tracing.py",
        14,
        "exo.worker.runner.bootstrap",
        ("logger",),
    ),
)


@lru_cache(maxsize=None)
def _module_source(rel_path: str) -> str:
    """Read (never import) a module under src/exo, caching its source."""
    full_path = _EXO_ROOT / rel_path
    if not full_path.is_file():
        raise FileNotFoundError(
            f"guard target {rel_path} missing; was it moved or deleted?",
        )
    return full_path.read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _module_ast(rel_path: str) -> ast.Module:
    """Parse (never import) a module under src/exo, caching the tree."""
    return ast.parse(_module_source(rel_path), filename=str(_EXO_ROOT / rel_path))


def _import_signature(node: ast.ImportFrom) -> tuple[str, tuple[str, ...]] | None:
    """(module, sorted imported names) for an ImportFrom node.

    ``asname`` aliases are ignored: ``from m import X as Y`` still imports X.
    Plain ``import`` statements and relative imports return None.
    """
    if node.level or not node.module:
        return None
    names = tuple(sorted(alias.name for alias in node.names))
    return (node.module, names)


def _all_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Every function/method node in the tree (methods live in class bodies)."""
    functions: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node)
    return functions


def _within_function(tree: ast.Module, node: ast.ImportFrom) -> bool:
    """True if *node* occurs inside any function/method body."""
    return any(node in ast.walk(fn) for fn in _all_functions(tree))


@pytest.mark.parametrize(
    ("rel_path", "ref_line", "module", "names"),
    LAZY_BREAK_POINTS,
    ids=[f"{p}:{ln}" for p, ln, _, _ in LAZY_BREAK_POINTS],
)
def test_break_point_stays_function_scoped(
    rel_path: str,
    ref_line: int,
    module: str,
    names: tuple[str, ...],
) -> None:
    """A pinned lazy import must not be hoisted to module scope."""
    tree = _module_ast(rel_path)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and _import_signature(node) == (module, names)
    ]
    # The signature may legitimately exist at module scope too (a different
    # edge in the same file, e.g. inside a TYPE_CHECKING guard): the guard
    # is about the *pinned lazy occurrence* — the one inside a function.
    lazy_nodes = [node for node in matches if _within_function(tree, node)]
    assert lazy_nodes, (
        f"{rel_path} break-point {module} {sorted(names)} has no "
        "function-scoped occurrence — it was hoisted to module scope, "
        "renamed, or removed; re-pin the guard to the current break-point. "
        f"(ref line {ref_line}, current lines "
        + ", ".join(str(n.lineno) for n in matches)
        + ")",
    )


@pytest.mark.parametrize(
    ("rel_path", "ref_line", "module", "names"),
    TRACKED_EAGER_OVERRIDES,
    ids=[f"{p}:{ln}" for p, ln, _, _ in TRACKED_EAGER_OVERRIDES],
)
def test_tracked_inversion_stays_documented(
    rel_path: str,
    ref_line: int,
    module: str,
    names: tuple[str, ...],
) -> None:
    """A known inversion owned by a FIX card must stay resolved.

    The audit recorded this edge at module scope (pre-FIX); the FIX card
    (t_46d744d7) removed it (tracing.py now imports the leaf logger from
    exo.utils.logging).  The guard pins the RESOLVED state: the eager
    bootstrap edge must remain gone.  If it reappears (re-inversion) or
    the override table is emptied unintentionally, this fails loudly.
    """
    tree = _module_ast(rel_path)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and _import_signature(node) == (module, names)
    ]
    assert not matches, (
        f"{rel_path} tracked inversion {module} {sorted(names)} REAPPEARED "
        f"(ref line {ref_line}) — the FIX card t_46d744d7 removed this eager "
        "edge; if it is back, the inversion was reintroduced. Re-pin the "
        "guard to the current break-point.",
    )


def _iter_import_froms(node: ast.AST) -> list[ast.ImportFrom]:
    """Every ImportFrom inside *node*, skipping TYPE_CHECKING subtrees.

    Subtrees under a module-level ``if TYPE_CHECKING:`` (or its
    ``exec("if TYPE_CHECKING: ...")`` alias) do not execute at import time
    and are excluded even when nested inside a walkable statement.
    """
    result: list[ast.ImportFrom] = []
    for child in ast.iter_child_nodes(node):
        if (
            isinstance(child, ast.If)
            and isinstance(child.test, ast.Name)
            and child.test.id == "TYPE_CHECKING"
        ):
            continue
        result.extend(_iter_import_froms(child))
    if isinstance(node, ast.ImportFrom) and node.module and not node.level:
        result.append(node)
    return result


def _iter_eager_imports(node: ast.AST) -> list[str]:
    """Dotted ``exo.*`` module names imported at *module scope* inside node.

    Descends into module-level compound statements (``try``, ``with``,
    loops, runtime ``if``) because those execute at import time, but stops
    at ``FunctionDef``/``AsyncFunctionDef``/``ClassDef`` boundaries — code
    inside a function or class body runs later, not at import.  Also stops
    at nested ``if TYPE_CHECKING`` (not executed at import), including the
    ``exec("if TYPE_CHECKING: ...")`` alias.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return []  # function/class bodies are lazy by definition
    names: set[str] = set()
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # function/class bodies are lazy by definition
        if (
            isinstance(child, ast.If)
            and isinstance(child.test, ast.Name)
            and child.test.id == "TYPE_CHECKING"
        ):
            continue
        names.update(_iter_eager_imports(child))
    if isinstance(node, ast.ImportFrom) and node.module and not node.level:
        names.add(node.module)
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name:
                names.add(alias.name)
    return sorted(names)


def _eager_import_edges(exo_root: Path) -> dict[str, set[str]]:
    """Eager (import-time) edges: dotted module name -> set of dotted names.

    Edges that do NOT execute at import time are excluded:
    - ``if TYPE_CHECKING`` blocks (incl. the ``exec("if TYPE_CHECKING...")``
      alias used by a few modules to hide the block from importers),
    - imports inside any function / class / other nested body.
    """
    edges: dict[str, set[str]] = {}
    for py_file in sorted(exo_root.rglob("*.py")):
        rel = py_file.relative_to(exo_root).as_posix()
        if rel.endswith("__init__.py"):
            dotted = "exo." + rel[: -len("__init__.py")].rstrip("/").replace("/", ".")
        else:
            dotted = "exo." + rel[:-3].replace("/", ".")
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError:
            continue
        module_imports: set[str] = set()
        for stmt in tree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                module_imports.update(_iter_eager_imports(stmt))
                continue
            if (
                isinstance(stmt, ast.If)
                and isinstance(stmt.test, ast.Name)
                and stmt.test.id == "TYPE_CHECKING"
            ):
                continue  # TYPE_CHECKING guard: not executed at import.
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                and stmt.value.func.id == "exec"
                and stmt.value.args
            ):
                arg = stmt.value.args[0]
                is_typecheck_exec = False
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    is_typecheck_exec = "TYPE_CHECKING" in arg.value
                elif isinstance(arg, ast.JoinedStr):
                    is_typecheck_exec = any(
                        isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                        and "TYPE_CHECKING" in value.value
                        for value in arg.values
                    )
                if is_typecheck_exec:
                    continue  # exec("if TYPE_CHECKING: ..."): not eager either.
                module_imports.update(_iter_eager_imports(arg))
                continue
            # Other module-level statements (try/except fallback imports,
            # with-blocks, runtime-if imports) all execute at import time:
            module_imports.update(_iter_eager_imports(stmt))
        module_imports = {name for name in module_imports if name.startswith("exo.")}
        if module_imports:
            edges[dotted] = module_imports
    return edges


def _eager_cycle(edges: dict[str, set[str]]) -> tuple[str, ...] | None:
    """Return a module-name cycle reachable via eager edges, if any.

    Predecessor-aware DFS: only the path *to* the repeated module is
    returned, so a self-loop reports the single node and a genuine cycle
    reports its members in order.
    """
    visited: set[str] = set()

    def dfs(node: str, path: list[str]) -> tuple[str, ...] | None:
        visited.add(node)
        path.append(node)
        for nxt in sorted(edges.get(node, ())):
            if nxt in path:
                return tuple(path[path.index(nxt) :])
            if nxt not in visited:
                cycle = dfs(nxt, path)
                if cycle is not None:
                    return cycle
        path.pop()
        return None

    for start in sorted(edges):
        if start not in visited:
            cycle = dfs(start, [])
            if cycle is not None:
                return cycle
    return None


def test_eager_import_graph_is_acyclic() -> None:
    """The whole eager (import-time) graph of src/exo must stay acyclic."""
    edges = _eager_import_edges(_EXO_ROOT)
    assert len(edges) > 100, (
        f"eager scan found only {len(edges)} modules — the scan is broken; "
        "a healthy src/exo has ~200+",
    )
    cycle = _eager_cycle(edges)
    assert cycle is None, (
        "Eager module-level import cycle detected: "
        + " -> ".join(cycle)
        + ". Every edge on this cycle must be a function-level (lazy) "
        "import or a TYPE_CHECKING guard; module-level imports would make "
        "exo fail with a circular import at startup.",
    )
