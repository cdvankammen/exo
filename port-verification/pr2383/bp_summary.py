"""Summarise a basedpyright --outputjson file: rule multiset per file.

Usage: python bp_summary.py <outputjson> [<outputjson> ...]
A line-number shift after a port's first hunk makes absolute-line comparison
meaningless, so diagnostics are keyed by RULE per FILE (skill §9f).
"""

import json
import sys
from collections import Counter


def load(path: str) -> Counter:
    with open(path) as fh:
        data = json.load(fh)
    out: Counter = Counter()
    for diag in data.get("generalDiagnostics", []):
        f = diag.get("file", "?").rsplit("/", 1)[-1]
        rule = diag.get("rule", diag.get("severity", "?"))
        out[(f, rule)] += 1
    return out


def main() -> int:
    counters = [load(p) for p in sys.argv[1:]]
    names = [p.rsplit("/", 1)[-1].replace(".json", "") for p in sys.argv[1:]]
    if len(counters) == 1:
        for (f, rule), n in sorted(counters[0].items()):
            print(f"{n}x {f} [{rule}]")
        print(f"total diagnostics: {sum(counters[0].values())}")
        return 0
    base, changed = counters[0], counters[1]
    added = changed - base
    removed = base - changed
    print(f"base total:   {sum(base.values())}")
    print(f"changed total:{sum(changed.values())}")
    print(f"ADDED (new):   {dict(added) or '{}'}")
    print(f"REMOVED (gone):{dict(removed) or '{}'}")
    print("VERDICT:", "NO NEW DIAGNOSTIC RULES" if not added else "REGRESSION?")
    return 1 if added else 0


if __name__ == "__main__":
    sys.exit(main())