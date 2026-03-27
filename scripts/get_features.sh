#!/usr/bin/env bash
# Extract FEATURE_COLUMNS and EARLY_FEATURE_COLUMNS counts and names from train.py.
# Usage: scripts/get_features.sh
python3 - <<'PY'
import re

with open("train.py") as f:
    src = f.read()

for label in ("FEATURE_COLUMNS", "EARLY_FEATURE_COLUMNS"):
    m = re.search(rf'{label}\s*=\s*\[(.+?)\]', src, re.DOTALL)
    if m:
        items = re.findall(r'"([^"]+)"', m.group(1))
        print(f"{label} ({len(items)} features):")
        for item in items:
            print(f"  {item}")
    else:
        print(f"{label} not found in train.py")
    print()
PY
