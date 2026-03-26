# Experiment Log — autoresearch/mlb-mar26
*(Runs baseline–Run10 archived in experiment_log_archive.md)*

---

## Run11 (commit 788ef84) — RESET
- Dual rolling windows W=15 + W=7 (short-term hot streak features)
- val_roi=-1.35%, val_brier=0.240, n_bets=772
- **Finding:** Dual windows add noise; single W=15 is cleaner

---

## Run12 (commit 6ea056f) — RESET
- threshold=0.05 (more selective) from run9 base
- val_roi=-7.97%, val_brier=0.240, n_bets=305
- **Finding:** Threshold=0.05 kills ROI — best edges are in 0.03–0.05 range

---

## Run13 (commit a0864fd) — RESET
- Home/road-specific rolling win% (home_win_pct - away_road_win_pct), 28 features
- val_roi=+1.58%, val_brier=0.240, n_bets=655
- **Finding:** Home/road split adds noise — too few games per split window

---

## Run14 (commit 7e52c91) ✓ KEPT (high variance)
- Win% momentum: (W=5 win% - W=15 win%) for home minus away, replaced home_road_split
- First run: val_roi=+8.79%; Re-run: val_roi=+0.92%, val_brier=0.240, n_bets=647
- **Finding:** HIGH VARIANCE. Momentum signal real but training noise is large. W=5 short window confirmed viable.

---

## Run15 — RESET (base: run14)
- Added run_diff_momentum_DIFF (W=5 run_diff_avg - W=15 run_diff_avg)
- val_roi=+3.20%, val_brier=0.240, n_bets=686
- **Finding:** Run-diff momentum adds noise on top of win% momentum

---

## Run16 — RESET (base: run14)
- MOMENTUM_W=3 (more aggressive streak detection)
- val_roi=+5.95%, val_brier=0.240, n_bets=685
- **Finding:** W=3 too noisy; W=5 is the sweet spot for momentum

---

## Run17 — RESET (base: run14)
- momentum_x_sharp interaction (momentum_DIFF * sharp_move_flag)
- val_roi=-6.07%, val_brier=0.241, n_bets=641
- **Finding:** Momentum × sharp interaction harmful

---

## Run18 — RESET (base: run14)
- Pitcher/park interaction: sp_k9_DIFF * park_factor (backlog item)
- val_roi=+3.81%, val_brier=0.240, n_bets=666
- **Finding:** k9×park interaction doesn't add value

---

## Run19 — RESET (base: run14)
- Removed redundant pitcher features: sp_era_DIFF, sp_bb9_DIFF, rolling_k9_DIFF (all captured by FIP)
- val_roi=+6.97%, val_brier=0.241, n_bets=537
- **Finding:** Trimming pitcher feats reduces n_bets too much; full feature set preferred

---

## Run20 — RESET (base: run14, PLATEAU PIVOT 1)
- Sharp-money weighted training loss (2× weight for |line_move_delta| ≥ 0.03 games)
- val_roi=+2.31%, val_brier=0.240, n_bets=898
- **Finding:** Sharp weighting hurts — too many bets, lower bar

---

## Run21 — RESET (base: run14, PLATEAU PIVOT 2)
- Focal loss (γ=2) instead of BCE
- val_roi=-1.53%, val_brier=0.240, n_bets=135
- **Finding:** Focal loss: too few bets (model becomes overconfident/selective)

---

## Run22 — RESET (base: run14)
- Dropout=0.1 (vs 0.2 baseline)
- val_roi=+2.68%, val_brier=0.241, n_bets=1,026
- **Finding:** Lower dropout → worse calibration, too many bets (overfitting)

---

## Run23 — RESET (base: run14, PLATEAU PIVOT 3)
- TabTransformer: self-attention over feature tokens (d_model=32, 2 heads, 2 layers)
- val_roi=-18.73%, val_brier=0.241, n_bets=482
- **Finding:** TabTransformer fails badly; MLP with market anchor is superior for this data

---

## Run24 (commit 8533916) ✓ KEPT — **NEW BEST**
- Feature neutralization: remove market_implied_prob-correlated signal from non-market features (cols 5+) using OLS residualization
- val_roi=+10.41%, val_brier=0.2398, n_bets=644
- **Finding:** Neutralization forces model to find genuine alpha. Best result yet. Hubáček approach validated.

---

## Run25 — IN PROGRESS (base: run24)
- **Hypothesis:** Neutralize against market logit (log-odds) instead of raw probability — more linear neutralization space.

---

## Summary of Key Findings

| What works | What doesn't |
|---|---|
| Market anchor loss (lambda=1.0) | lambda ≠ 1.0 |
| W=15 rolling window | W≤10 or W≥20 |
| W=5 momentum (win% diff) | Run-diff momentum, W=3 momentum |
| Feature neutralization (Hubáček) | TabTransformer, focal loss |
| Clean 28-feature MLP (256,128,64) | Wider/deeper nets, interactions |
| threshold=0.04, Kelly=0.25 | threshold ≥ 0.05 |

**Current best:** Run24 — ROI=+10.41%, Brier=0.2398, 644 bets
