#!/usr/bin/env bash
set -euo pipefail

# Check whether the latest results.tsv row is a new ROI best.
# If yes, stage files, commit, and push the current branch.
#
# Usage:
#   scripts/checkpoint_best.sh [description]
#
# Env vars:
#   RESULTS_FILE      path to results file (default: results.tsv)
#   CHECKPOINT_FILES  space-separated files to stage on checkpoint
#                     (default: "train.py results.tsv experiment_log.md")

RESULTS_FILE="${RESULTS_FILE:-results.tsv}"
CHECKPOINT_FILES="${CHECKPOINT_FILES:-train.py results.tsv experiment_log.md}"
EXTRA_DESC="${1:-}"

if [[ ! -f "$RESULTS_FILE" ]]; then
  echo "No ${RESULTS_FILE}; skipping checkpoint."
  exit 0
fi
CHECK_RESULT="$(python3 - "$RESULTS_FILE" <<'PY'
import csv
import math
import sys

path = sys.argv[1]
rows = []
with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = list(reader)

if not rows:
    print("NO_ROWS")
    sys.exit(0)

latest = rows[-1]
try:
    latest_roi = float(latest["val_roi"])
except Exception:
    print("LATEST_INVALID")
    sys.exit(0)

if math.isnan(latest_roi):
    print("LATEST_NAN")
    sys.exit(0)

prev_best = float("-inf")
for r in rows[:-1]:
    try:
        v = float(r["val_roi"])
    except Exception:
        continue
    if math.isnan(v):
        continue
    if v > prev_best:
        prev_best = v

if latest_roi > prev_best:
    latest_brier = latest.get("val_brier", "nan")
    print(f"IMPROVED\t{latest_roi:.6f}\t{latest_brier}")
else:
    print(f"NOT_IMPROVED\t{latest_roi:.6f}\t{prev_best:.6f}")
PY
)"

IFS=$'\t' read -r status roi metric <<<"$CHECK_RESULT"
if [[ "$status" != "IMPROVED" ]]; then
  echo "No new best ROI (${CHECK_RESULT}); skipping commit/push."
  exit 0
fi

for f in $CHECKPOINT_FILES; do
  if [[ -e "$f" ]]; then
    git add "$f"
  fi
done

if git diff --cached --quiet; then
  echo "New best ROI found, but no staged changes; skipping commit/push."
  exit 0
fi

branch="$(git rev-parse --abbrev-ref HEAD)"
msg="checkpoint: new best roi=${roi} brier=${metric}"
if [[ -n "$EXTRA_DESC" ]]; then
  msg="${msg} | ${EXTRA_DESC}"
fi

git commit -m "$msg"
git push -u origin "$branch"
echo "Checkpoint pushed on ${branch}: ${msg}"
