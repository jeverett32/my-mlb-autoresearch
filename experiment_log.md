# Experiment Log — autoresearch/mlb-mar26

---

## Run: baseline (commit 2335522)
**What was tried:** Baseline MLP (256, 128, 64), W=31 rolling window, threshold=0.03. 26 features including market, pitcher FIP/K9, wRC+, wOBA, WAR, weather, context.

**Result:** val_roi=-0.53%, val_brier=0.341, n_bets=3,965. Nearly every game got a bet — model probabilities far from market, very poor calibration. Slight negative ROI.

**Next step:** Model needs to be anchored to the market price. Miscalibration is the core problem.

---

## Run1 (commit c94be5a) — RESET
**What was tried:** Residual MLP architecture: `output = market_logit_unscaled + small_correction`. Added `market_logit` feature. Threshold lowered to 0.025.

**Result:** val_roi=-2.37%, val_brier=0.347, n_bets=3,965. Worse across the board. The small init on the correction layer didn't prevent it from learning noisy corrections.

**Next step:** Scrap residual architecture. Try direct market-anchoring via an auxiliary loss term in training.

---

## Run2 (commit a977741)
**What was tried:** Expanded features (sp_era, sp_bb9, rolling_era, rolling_whip, k_pct, bb_pct, wind_speed), sharp_x_fip interaction. Higher weight_decay=3e-4, threshold=0.04.

**Result:** val_roi=+0.47%, val_brier=0.352, n_bets=3,922. Positive ROI but still ~94% of games get bets — calibration still broken, Brier got worse.

**Next step:** The calibration problem is persistent. Need to anchor model to market price during training.

---

## Run3 (commit a33c2ec) — RESET
**What was tried:** Same as run2 but threshold=0.07 to reduce bet count.

**Result:** val_roi=-0.99%, val_brier=0.363, n_bets=3,762. Even at 7% edge threshold, 90% of games still bet. Model confidently wrong everywhere.

**Next step:** Raising threshold alone doesn't fix calibration. Must use a training-time fix.

---

## Run4 (commit 12c4d98) ✓ KEPT
**What was tried:** Added market-logit anchor loss during training: `loss = BCE + 1.0 * (logit - market_logit)^2`. This pulls model predictions toward the market price during training.

**Result:** val_roi=+0.70%, val_brier=0.240 (near-market!), n_bets=803. Huge calibration improvement. Brier dropped from ~0.35 to 0.240 — essentially at market level. Bets reduced to 19% of games.

**Next step:** Calibration is solved. Now explore anchor lambda and threshold tuning.

---

## Run5 (commit 483311c) — RESET
**What was tried:** Lambda=2.0 (tighter anchor) + threshold=0.03.

**Result:** val_roi=-6.94%, val_brier=0.240, n_bets=427. Tighter anchor + lower threshold = fewer bets but all losing. Over-constrained the model.

**Next step:** Lambda=1.0 is optimal. Try looser lambda next.

---

## Run6 (commit 8bd6097) — RESET
**What was tried:** Lambda=0.5 (looser anchor) + threshold=0.04.

**Result:** val_roi=-4.96%, val_brier=0.245, n_bets=2,005. Too many bets, worse calibration. Model drifted from market.

**Next step:** Lambda=1.0 is the sweet spot. Don't deviate.

---

## Run7 (commit 5dcbd99) — RESET
**What was tried:** Added interaction features (sharp_x_woba, line_x_fip, rest_x_form) + wider network (256, 192, 128, 64).

**Result:** val_roi=-0.86%, val_brier=0.241, n_bets=1,383. More interactions and capacity hurt — overfitting or noise.

**Next step:** Keep simple 3-layer MLP. Try W parameter.

---

## Run8 (commit c7fee1f) ✓ KEPT
**What was tried:** W=20 (shorter rolling window vs W=31 baseline), removed war/k_pct/bb_pct/wind_dir (noisy features).

**Result:** val_roi=+2.80%, val_brier=0.240, n_bets=748. Big improvement. Cleaner feature set + more reactive rolling stats works better.

**Next step:** Try W=15.

---

## Run9 (commit 837cbfa) ✓ KEPT — **BEST SO FAR**
**What was tried:** W=15 rolling window.

**Result:** val_roi=+5.09%, val_brier=0.240, n_bets=819. Best ROI so far. W=15 is more reactive to recent form and the market may not fully price in recent streaks.

**Next step:** Try W=10 and threshold tuning.

---

## Run10 (commit bfa2dc4) — RESET
**What was tried:** W=10 (even shorter window).

**Result:** val_roi=-2.50%, val_brier=0.240, n_bets=796. Too short — W=10 is noisy. W=15 is the sweet spot.

**Next step:** Stay at W=15. Try threshold=0.05.

---

## Run11 (commit 788ef84) — RESET
**What was tried:** Dual rolling windows W=15 + W=7 (short-term hot streak features).

**Result:** val_roi=-1.35%, val_brier=0.240, n_bets=772. Dual windows hurt. W=7 adds noise. Single W=15 window is cleaner.

**Next step:** Try threshold=0.05 from run9 base.

---

## Run12 (commit 6ea056f) — RESET
**What was tried:** threshold=0.05 (more selective) from run9 base (W=15).

**Result:** val_roi=-7.97%, val_brier=0.240, n_bets=305. Far too few bets, all losing. Raising threshold kills ROI — the model's best edges are in the 0.03–0.05 range.

**Next step:** Keep threshold=0.04 (run9 default). Try home/road-specific rolling win% as new feature.

---

## Summary of Key Findings

| What works | What doesn't |
|---|---|
| Market anchor loss (lambda=1.0) | Residual architecture |
| W=15 rolling window | W≤10 or W≥20 rolling windows |
| Clean feature set (27 features) | Extra interaction features |
| Simple 3-layer MLP | Wider/deeper networks |
| | Dual rolling windows |
| | Lambda ≠ 1.0 |

**Current best:** run9 — ROI=+5.09%, Brier=0.240, 819 bets

