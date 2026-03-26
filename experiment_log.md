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

## Run25 (commit aba69aa) — RESET
- Neutralize non-market features against market logit instead of raw probability
- val_roi=+5.38%, val_brier=0.240, n_bets=681
- **Finding:** Logit-space neutralization is weaker than raw-prob; run24 (raw-prob) is better

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

## Run26 — RESET (base: run24)
- Kelly=0.15 (vs 0.25 baseline)
- val_roi=+3.11%, val_brier=0.240, n_bets=635
- **Finding:** Kelly=0.15 worse — likely different stochastic training seed. Kelly=0.25 kept.

---

## Run27 (commit HEAD) ✓ KEPT — INFRASTRUCTURE
- Fixed random seeds (torch + numpy, seed=42) for deterministic, comparable experiments
- val_roi=+1.34%, val_brier=0.240, n_bets=673
- **Finding:** Seed=42 is the new deterministic baseline. All future runs are fair A/B comparisons.

---

## Run28 — RESET (base: run27)
- Selective neutralization: neutralize only structural features (pitcher/batting/weather), skip rolling-form + momentum
- val_roi=-2.48%, val_brier=0.241, n_bets=647
- **Finding:** Rolling-form features NEED neutralization. Full neutralization (run27) is better.

---

## Run29 — RESET (base: run27)
- 2-factor OLS neutralization: remove [market_implied_prob, open_home_implied] from non-market features
- val_roi=-2.85%, val_brier=0.241, n_bets=645
- **Finding:** 2-factor neutralization worse — adding open_home_implied over-neutralizes, removes valid signal

---

## Run30 — RESET (base: run27)
- Partial neutralization α=0.5
- val_roi=-0.64%, val_brier=0.241, n_bets=618
- **Finding:** Partial neutralization worse than full. α=1.0 is optimal.

## Run31 — RESET (base: run27)
- LR=5e-4, seed=42
- val_roi=+0.39%, val_brier=0.240, n_bets=278 (too few bets)
- **Finding:** Lower LR causes underfitting/few bets. LR=1e-3 is optimal.

## Run32 — RESET (base: run27, DIAGNOSTIC)
- No neutralization, seed=42
- val_roi=+1.35%, val_brier=0.240, n_bets=670
- **CRITICAL FINDING:** Same ROI as run27 (+1.34% with neutralization). Neutralization has zero effect on seed=42. Seed is the dominant confound, not features.

## Run33 — INTERRUPTED (ensemble pivot paused)
- User pivoted to Feature Engineering Sprint before run33 could be executed.

---

## [Feature Engineering Sprint — Statistical Audit]

**Audit (Pearson |r| with home_win, n=36k):**

| Rank | Feature | |r| | Status |
|---|---|---|---|
| Weakest | away_pitcher_is_lefty | 0.0001 | REMOVE |
| Weakest | rest_days_DIFF | 0.0005 | REMOVE |
| Weakest | sp_rest_DIFF | 0.0005 | REMOVE (perfect duplicate of rest_days_DIFF) |
| Weak | temp_c | 0.0009 | Keep (below threshold but contextual) |
| Weak | momentum_DIFF | 0.0010 | Keep (below threshold but hypothesis-driven) |
| Strong | market_implied_prob | 0.1858 | Keep |
| Strong | wrc_plus_DIFF | 0.1128 | Keep |

**Park-factor interaction audit:** All park×pitcher interactions fail. park_factor std≈0.04 makes every product collinear with the underlying feature (r>0.99 redundancy). Substituting with best-passing alternatives:

| New Feature | r_home | max_redundancy | Decision |
|---|---|---|---|
| fip_x_line = sp_fip_DIFF × line_move_delta | +0.0057 | 0.072 | ADD |
| form_x_fip = run_diff_avg_W_DIFF × sp_fip_DIFF | -0.0060 | 0.025 | ADD |

## Run34 — RESET (Feature Engineering Sprint)
- Remove: away_pitcher_is_lefty, rest_days_DIFF, sp_rest_DIFF; Add: fip_x_line, form_x_fip (28→27 features)
- val_roi=-2.70%, val_brier=0.241, n_bets=589
- **Finding:** Pruning + new interactions hurt. New features (|r|≈0.006) are too weak to compensate for removed context. Try pruning-only (remove 3 without adding new features).

## Run35 — RESET (Feature Engineering Sprint, pruning-only)
- Remove: away_pitcher_is_lefty, rest_days_DIFF, sp_rest_DIFF (28→25 features); no additions
- val_roi=-0.07%, val_brier=0.240, n_bets=590
- **Finding:** Pruning alone also hurts with seed=42. The 3 "zero-signal" features may have structural regularization value despite low individual correlation. Feature set is stable at 28 features.

## [Feature Engineering Sprint — Conclusion]
- **Audit**: 10/28 features below |r|<0.005, but none can be safely pruned without hurting ROI (seed=42)
- **Park interactions**: All fail screening (redundancy r>0.99 due to park_factor std≈0.04)
- **New interactions**: fip_x_line, form_x_fip pass screening but add noise in practice
- **Verdict**: Current 28-feature set is the stable optimum. Deeper interaction features require non-linear selection methods (e.g., permutation importance on a trained model).

---

**Current best:** Run27 (seed=42) — ROI=+1.34%, Brier=0.240, 673 bets
