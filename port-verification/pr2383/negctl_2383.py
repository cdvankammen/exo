"""Per-half negative control for PR #2383 (placement failure reasons).

Reverts ONE half of the port in place, runs the placement suite, restores the
file, and prints the failure set. The failure set names the behaviour that half
is responsible for.

Usage (from the worktree root):
  PYTHONPATH=<tree>/src EXO_DASHBOARD_DIR=exo/dashboard \
    /Users/chris/exo/.venv/bin/python negctl_2383.py <half>

  half = none | no-nodes | too-few | not-connected | memory-msg
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

PLACEMENT = Path("src/exo/master/placement.py")
TESTFILE = "src/exo/master/tests/test_placement.py"

# half -> (text present after the port, text to restore). Revert = swap back.
REVERTS = {
    # "no cycles at all" used to fall through to the memory filter.
    "no-nodes": (
        '    cycles = topology.get_cycles()\n'
        '    if not cycles:\n'
        '        raise ValueError("No nodes are available")\n',
        "    cycles = topology.get_cycles()\n",
    ),
    # min_nodes shortfall used to fall through too (the common 2-node case).
    "too-few": (
        "    if not candidate_cycles:\n"
        "        largest = max(len(cycle) for cycle in cycles)\n"
        "        raise ValueError(\n"
        '            f"Needs {command.min_nodes} connected nodes, but at most {largest} '
        'are connected to each other"\n'
        "        )\n",
        "",
    ),
    "not-connected": (
        "        if not candidate_cycles:\n"
        '            raise ValueError("The chosen nodes aren\'t all connected to each other")\n',
        "",
    ),
    # the single message upstream replaced
    "memory-msg": (
        "        raise ValueError(_not_enough_memory(command, candidate_cycles, node_memory))",
        '        raise ValueError("No cycles found with sufficient memory")',
    ),
}

HELPER = re.compile(
    r"\ndef _not_enough_memory\(\n.*?\n    \)\n\n", re.DOTALL
)


def revert(half: str) -> str:
    """Apply the revert in place; return the original file content."""
    path = PLACEMENT
    original = path.read_text()
    new, old = REVERTS[half]
    assert new in original, f"{half}: post-port marker not found"
    assert original.count(new) == 1, f"{half}: marker occurs {original.count(new)}x"
    text = original.replace(new, old)
    if half == "memory-msg":
        # the helper becomes dead code; drop it so the file stays lint-clean
        stripped, n = HELPER.subn("", text)
        assert n == 1, f"{half}: helper removal matched {n}x"
        text = stripped
    ast.parse(text)
    path.write_text(text)
    return original


def run_tests() -> tuple[int, list[str]]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    env["EXO_DASHBOARD_DIR"] = "exo/dashboard"
    proc = subprocess.run(
        ["/Users/chris/exo/.venv/bin/python", "-m", "pytest", TESTFILE, "-q"],
        capture_output=True,
        text=True,
        env=env,
    )
    fails = sorted(
        {m.group(1) for m in re.finditer(r"^(?:FAILED|ERROR) (\S+)", proc.stdout, re.M)}
    )
    m = re.search(r"(\d+) passed", proc.stdout)
    passed = int(m.group(1)) if m else 0
    return passed, fails


def main() -> int:
    half = sys.argv[1] if len(sys.argv) > 1 else "none"
    if half == "none":
        passed, fails = run_tests()
        label = "PORTED (no revert)"
    else:
        original = revert(half)
        try:
            passed, fails = run_tests()
        finally:
            PLACEMENT.write_text(original)
        label = f"REVERTED half={half}"
    print(f"{label}: passed={passed} failures={len(fails)}")
    for f in fails:
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())