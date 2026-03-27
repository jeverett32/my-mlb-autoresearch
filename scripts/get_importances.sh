#!/usr/bin/env bash
# Extract LGB/XGB feature importances from run.log.
# Usage: scripts/get_importances.sh [run.log]
LOG="${1:-run.log}"
grep -A 80 -i "feature importance\|importance" "$LOG" | head -100
