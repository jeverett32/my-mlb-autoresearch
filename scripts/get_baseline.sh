#!/usr/bin/env bash
# Show current best ROI, latest row, and plateau count from results.tsv.
# Usage: scripts/get_baseline.sh

echo "=== Latest row ==="
tail -n 1 results.tsv

echo ""
echo "=== Best ROI across all ok rows ==="
python3 - <<'PY'
import csv, math, sys

rows = []
with open("results.tsv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = [r for r in reader if r.get("status") == "ok"]

if not rows:
    print("No ok rows found.")
    sys.exit(0)

best = max(rows, key=lambda r: float(r["val_roi"]) if not math.isnan(float(r["val_roi"])) else float("-inf"))
print(f"Description : {best.get('description', 'N/A')}")
print(f"Commit      : {best.get('commit', 'N/A')}")
print(f"val_roi     : {best['val_roi']}")
print(f"val_brier   : {best.get('val_brier', 'N/A')}")

# Plateau count: consecutive non-improving runs from the end
prev_best = float("-inf")
for r in rows[:-1]:
    try:
        v = float(r["val_roi"])
        if not math.isnan(v) and v > prev_best:
            prev_best = v
    except Exception:
        pass

plateau = 0
for r in reversed(rows):
    try:
        v = float(r["val_roi"])
        if math.isnan(v) or v <= prev_best:
            plateau += 1
        else:
            break
    except Exception:
        plateau += 1

print(f"\nConsecutive non-improving runs (plateau): {plateau}/5")
PY
