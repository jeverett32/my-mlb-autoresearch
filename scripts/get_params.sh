#!/usr/bin/env bash
# Show all tunable hyperparameters from train.py.
# Usage: scripts/get_params.sh
echo "=== Tunable parameters ==="
grep -E "^(MODEL|CALIBRATE|EARLY_CUTOFF|BEST_W|MOMENTUM_W|CONFIDENCE_THRESHOLD|KELLY_FRACTION|MAX_BET_FRAC|PROB_CAP|EARLY_SEASON_GAMES)\s*=" train.py

echo ""
echo "=== Walk-forward folds ==="
python3 - <<'PY'
import re
with open("train.py") as f:
    src = f.read()
m = re.search(r'WALK_FORWARD_FOLDS\s*=\s*\[(.+?)\]', src, re.DOTALL)
if m:
    print(m.group(0))
PY

echo ""
echo "=== LGB_PARAMS ==="
python3 - <<'PY'
import re
with open("train.py") as f:
    src = f.read()
m = re.search(r'LGB_PARAMS\s*=\s*\{(.+?)\}', src, re.DOTALL)
if m:
    print("LGB_PARAMS = {" + m.group(1) + "}")
PY

echo ""
echo "=== XGB_PARAMS ==="
python3 - <<'PY'
import re
with open("train.py") as f:
    src = f.read()
m = re.search(r'XGB_PARAMS\s*=\s*\{(.+?)\}', src, re.DOTALL)
if m:
    print("XGB_PARAMS = {" + m.group(1) + "}")
PY
