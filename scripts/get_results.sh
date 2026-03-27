#!/usr/bin/env bash
# Extract key metrics from run.log. Run after every training run.
# Usage: scripts/get_results.sh [run.log]
LOG="${1:-run.log}"
grep -E "val_roi|val_brier|n_bets|Mean ROI|Mean Brier|Fold [0-9]|ERROR|Traceback" "$LOG" | tail -60
