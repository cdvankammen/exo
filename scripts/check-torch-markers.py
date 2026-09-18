#!/usr/bin/env python3
"""
Guard against uv.lock torch marker regeneration trap (uv 0.12.x quirk).

PROBLEM: `uv lock` on macOS regenerates uv.lock and re-introduces broken
markers on torch/torchaudio/torchvision edges in the exo package's
requires-dist.  The broken form is:

    marker = "sys_platform == 'linux' and extra == 'mlx'
              and extra != 'mlx-cpu' and extra != 'mlx-cuda12'
              and extra != 'mlx-cuda13'"

The `extra !=` clauses cause uv to treat the edge as FALSE even when only
`mlx` is selected (confirmed: uv 0.12.1/0.12.2 mis-evaluate multi-extra
negation clauses).  The fixed form is simply:

    marker = "sys_platform == 'linux' and extra == 'mlx'"

This script checks for the broken markers and either reports or fixes them.

Origin: commit f48bdf86 (2026-08-06) fixed the markers once; commit 7546fbbd
(2026-09-01) re-introduced them via a lock refresh.  Every `uv lock` run
on macOS will re-introduce the broken markers until uv itself is patched.

Usage:
    python scripts/check-torch-markers.py              # check only (exit 1 if broken)
    python scripts/check-torch-markers.py --fix        # fix in-place
    python scripts/check-torch-markers.py --dry-run    # fix but only print diff
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BROKEN_MARKER_RE = re.compile(
    r"""(?P<prefix>\{ name = "(?P<pkg>torch|torchaudio|torchvision)",\s*marker = ")"""
    r"""(?P<marker>sys_platform == 'linux' and extra == 'mlx'"""
    r""" and extra != 'mlx-cpu' and extra != 'mlx-cuda12'"""
    r""" and extra != 'mlx-cuda13')""",
    re.MULTILINE,
)

FIXED_FORM = "sys_platform == 'linux' and extra == 'mlx'"

EXPECTED_LINES = [
    '    { name = "torch", marker = "sys_platform == \'linux\' and extra == \'mlx\'", specifier = "==2.10.0" },',
    '    { name = "torchaudio", marker = "sys_platform == \'linux\' and extra == \'mlx\'", specifier = "==2.10.0" },',
    '    { name = "torchvision", marker = "sys_platform == \'linux\' and extra == \'mlx\'", specifier = "==0.25.0" },',
]


def find_uv_lock(start: Path) -> Path:
    """Walk up from start to find uv.lock."""
    for parent in [start] + list(start.parents):
        candidate = parent / "uv.lock"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No uv.lock found starting from {start}")


def check_markers(lock_path: Path) -> list[tuple[int, str, str]]:
    """Return list of (line_no, pkg_name, broken_line) for each broken marker."""
    content = lock_path.read_text()
    broken = []
    for line_no, line in enumerate(content.splitlines(), 1):
        for m in BROKEN_MARKER_RE.finditer(line):
            broken.append((line_no, m.group("pkg"), line.rstrip()))
    return broken


def fix_markers(content: str) -> str:
    """Replace all broken markers with the correct form."""
    return BROKEN_MARKER_RE.sub(
        lambda m: m.group("prefix") + FIXED_FORM,
        content,
    )


def verify_fix(lock_path: Path) -> list[str]:
    """After fixing, verify the corrected lines are present."""
    content = lock_path.read_text()
    missing = []
    for expected in EXPECTED_LINES:
        if expected not in content:
            missing.append(expected.strip())
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix", action="store_true", help="Fix broken markers in-place")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without modifying the file")
    args = parser.parse_args()

    lock_path = find_uv_lock(Path.cwd())
    broken = check_markers(lock_path)

    if not broken:
        print(f"OK: {lock_path} — all torch markers are correct.")
        return 0

    # Report broken markers
    print(f"BROKEN: {lock_path} — {len(broken)} broken torch marker(s):")
    for line_no, pkg, line in broken:
        print(f"  line {line_no}: {pkg}")
        print(f"    {line}")
    print()
    print(f"  Expected (fixed form):")
    for expected in EXPECTED_LINES:
        print(f"    {expected}")
    print()

    if not args.fix and not args.dry_run:
        print("Run with --fix to repair, or --dry-run to preview.")
        return 1

    # Apply fix
    original = lock_path.read_text()
    fixed = fix_markers(original)

    if args.dry_run:
        # Show unified diff
        import difflib
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            fixed.splitlines(keepends=True),
            fromfile=str(lock_path),
            tofile=f"{lock_path} (fixed)",
        )
        sys.stdout.writelines(diff)
        return 1  # exit 1 = markers were broken (even in dry-run)

    lock_path.write_text(fixed)

    # Verify the fix landed
    missing = verify_fix(lock_path)
    if missing:
        print(f"ERROR: Fix applied but verification failed. Missing:")
        for m in missing:
            print(f"  {m}")
        return 1

    print(f"FIXED: {lock_path} — {len(broken)} marker(s) repaired.")
    print("Next steps:")
    print("  1. Run: uv lock --check  (verify lock is still valid)")
    print("  2. On Linux box: uv pip install -e '.[mlx]' --dry-run  (verify torch resolves)")
    print("  3. Commit the fixed uv.lock")
    return 0


if __name__ == "__main__":
    sys.exit(main())