#!/bin/bash
# Chunked pytest for PR #2383: enumerate collectable test files explicitly
# (skill §9j: --ignore does not stop conftest collection aborts) and compare
# the base-vs-changed failure SETS, not the counts.
set -u
TREE="$1"
OUT="$2"
cd "$TREE" || exit 1
export PYTHONPATH="$TREE/src"
export EXO_DASHBOARD_DIR=exo/dashboard
PY=/Users/chris/exo/.venv/bin/python

FILES=$(find src/exo -name 'test_*.py' \
  -not -path '*/worker/tests/unittests/test_runner/*' \
  -not -path '*/worker/engines/mlx/*' \
  -not -path '*/worker/tests/unittests/test_mlx/*' \
  -not -name 'test_cancellation.py' \
  | sort)

: > "$OUT"
for f in $FILES; do
  res=$($PY -m pytest "$f" -q 2>&1)
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "$res" | grep -E '^(FAILED|ERROR)' | sed 's/ - .*//' | sort -u >> "$OUT"
  fi
done
sort -u "$OUT" -o "$OUT"
echo "failure-set lines: $(wc -l < "$OUT")"