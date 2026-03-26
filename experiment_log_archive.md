# Experiment Log Archive — autoresearch/mlb-mar26 (Runs baseline–Run10)

---

## Run: baseline (commit 2335522)
- W=31, threshold=0.03, 26 features, no anchor
- val_roi=-0.53%, val_brier=0.341, n_bets=3,965
- **Finding:** No market anchor = completely miscalibrated

---

## Run1 (commit c94be5a) — RESET
- Residual MLP (market_logit + small correction), threshold=0.025
- val_roi=-2.37%, val_brier=0.347, n_bets=3,965
- **Finding:** Residual arch doesn't self-anchor

---

## Run2 (commit a977741) ✓ KEPT
- Expanded features, sharp_x_fip, weight_decay=3e-4, threshold=0.04
- val_roi=+0.47%, val_brier=0.352, n_bets=3,922
- **Finding:** Calibration still broken without anchor

---

## Run3 (commit a33c2ec) — RESET
- threshold=0.07 from run2 base
- val_roi=-0.99%, val_brier=0.363, n_bets=3,762
- **Finding:** Threshold alone can't fix calibration

---

## Run4 (commit 12c4d98) ✓ KEPT
- BCE + anchor_loss lambda=1.0
- val_roi=+0.70%, val_brier=0.240, n_bets=803
- **Finding:** Market anchor essential — Brier 0.35→0.240

---

## Run5 (commit 483311c) — RESET
- lambda=2.0, threshold=0.03
- val_roi=-6.94%, val_brier=0.240, n_bets=427
- **Finding:** lambda=1.0 is optimal; tighter over-constrains

---

## Run6 (commit 8bd6097) — RESET
- lambda=0.5, threshold=0.04
- val_roi=-4.96%, val_brier=0.245, n_bets=2,005
- **Finding:** lambda=0.5 drifts from market

---

## Run7 (commit 5dcbd99) — RESET
- Interactions (sharp_x_woba, line_x_fip, rest_x_form), wider MLP (256,192,128,64)
- val_roi=-0.86%, val_brier=0.241, n_bets=1,383
- **Finding:** Interactions + wider net = overfitting

---

## Run8 (commit c7fee1f) ✓ KEPT
- W=20, removed noisy features (war, k_pct, bb_pct, wind_dir)
- val_roi=+2.80%, val_brier=0.240, n_bets=748
- **Finding:** Shorter window + cleaner features improve ROI

---

## Run9 (commit 837cbfa) ✓ KEPT
- W=15
- val_roi=+5.09%, val_brier=0.240, n_bets=819
- **Finding:** W=15 is the optimal rolling window

---

## Run10 (commit bfa2dc4) — RESET
- W=10
- val_roi=-2.50%, val_brier=0.240, n_bets=796
- **Finding:** W=10 too noisy
