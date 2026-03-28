# Experiment Log — autoresearch/mlb-mar27

## MLP-Era Distilled Findings (reference only — NOT comparable to walk-forward ROI)

| What worked | What didn't |
|---|---|
| Market anchor loss (lambda=1.0) | lambda ≠ 1.0, focal loss |
| W=15 rolling window | W≤10 or W≥20 |
| W=5 momentum signal | Run-diff momentum, W=3 |
| Feature neutralization (Hubáček) | TabTransformer, wider/deeper MLP |
| threshold=0.04, Kelly=0.25 | threshold ≥ 0.05, Kelly=0.15 | *(MLP era only — walk-forward LR best at threshold=0.12)* |

**MLP stable baseline (run27, seed=42):** ROI=+1.34%, Brier=0.240, 673 bets *(single-split, not comparable)*

---

## [Current Era: 4-fold Walk-Forward, Honest Regime (TW=None)]
- 4-fold walk-forward (val=2022, 2023, 2024, 2025). Mean ROI is the only metric.
- Current best: **44.85%** (Run 145) — LR, TW=None, C=0.04, threshold=0.14, PROB_CAP=(0.34,0.66), EARLY_CUTOFF=25, cv=3
- Feature set: 63 features (base 43 + day_of_week + sharp_x_fip + momentum_DIFF + luck_DIFF + pythagorean_DIFF + luck_x_momentum + park_x_pythagorean + pythagorean_short_DIFF + post_rule_change + post_dh_era + covid_era)
- Fold breakdown: fold1=50.59% (72), fold2=63.92% (55), fold3=47.70% (78), fold4=17.18% (89)

---

<!-- Add new runs below this line -->

## Run 199 — Add sharp money signals to early specialist
**Change**: Add line_move_delta, sharp_move_flag to EARLY_FEATURE_COLUMNS
**Result**: roi=+42.33% — worse; specialist takes more bets but worse quality
**Decision**: REVERTED

## Run 200 — sp_luck_DIFF (ERA-FIP gap, pitcher regression signal)
**Change**: Add sp_luck_DIFF = sp_era_DIFF - sp_fip_DIFF
**Result**: roi=+44.85% — identical; linear combination of existing features, L1 zeros it
**Decision**: REVERTED

## Run 201 — Half-season folds (Phase 5)
**Change**: 8 half-season folds instead of 4 full-year folds
**Result**: roi=+47.30% (8-fold mean), fold8=−12.66% (2025H2). Not comparable to 4-fold.
**Decision**: REVERTED
**Insight**: 2025H1=+62.48%, 2025H2=−12.66% — market regime change in late 2025

## Run 202 — Mean imputation (instead of median)
**Change**: SimpleImputer strategy='mean' in build_lr()
**Result**: roi=+44.85% — identical; few NaN values after upstream processing
**Decision**: REVERTED

## Run 203 — Remove run_diff_std_W_DIFF
**Change**: Remove run_diff_std_W_DIFF from FEATURE_COLUMNS
**Result**: roi=+44.59% — worse; consistency metric contributes
**Decision**: REVERTED

## Run 204 — XGB max_depth=3 (shallow trees)
**Change**: MODEL="xgb", XGB max_depth=3
**Result**: roi=+30.38%, fold4=+3.24% — GBDT still much worse than LR L1
**Decision**: REVERTED
**Insight**: Even depth=3 XGB can't match LR sparsity; GBDT dead end for all configs

## Run 205 — Truncate fold4 to 2025H1 (exclude late-2025 regime change)
**Hypothesis**: Run 201 (half-season folds) revealed 2025H1=+62.48% but 2025H2=−12.66%. The late-2025 period appears to have a market regime change. Truncating fold4 to 2025 first-half only removes the noisy period while keeping 4 comparable folds.
**Change**: Fold4 val_end changed from 2026-01-01 to 2025-07-20
**Result**: roi=+56.17% (fold4=+62.48%, 38 bets) — inflated by excluding bad period; cherry-picking
**Decision**: REVERTED (not comparable; data snooping)

## Run 206 — MOMENTUM_W=8 (fine-grained)
**Hypothesis**: MOMENTUM_W=7 and MOMENTUM_W=9 were tried (worse). 8 is between them. If the optimum is at a non-integer, 8 might outperform 10 by smoothing momentum slightly differently.
**Change**: MOMENTUM_W=8
**Result**: roi=+43.26%, fold2=−5.70pp — worse; MOMENTUM_W=10 is confirmed optimal
**Decision**: REVERTED

## Run 207 — BEST_W=16 (fine-grained)
**Hypothesis**: BEST_W=14 and BEST_W=17 were tried (worse), but 16 has not been tested. Slightly wider window than 15 may capture more stable team form.
**Change**: BEST_W=16
**Result**: roi=+42.28%, fold2=−7.81pp — worse; BEST_W=15 confirmed optimal
**Decision**: REVERTED

## Run 208 — Remove total_move_delta (O/U noise for moneyline)
**Hypothesis**: total_move_delta captures O/U line movement. While sharp O/U movement may correlate with game environment, it's fundamentally about run totals, not win probability. For a moneyline model, this adds noise.
**Change**: Remove total_move_delta from FEATURE_COLUMNS
**Result**: roi=+43.66%, fold3=−3.05pp — worse; O/U movement correlates with game environment info
**Decision**: REVERTED

## Run 209 — Add sp_rest_DIFF + rest_days_DIFF to EARLY_FEATURE_COLUMNS
**Hypothesis**: The early specialist has home_rest_days/away_rest_days but not their difference or pitcher rest. In early season, when rolling stats are unreliable, rest advantages and pitcher-specific rest may be more predictive.
**Change**: Add sp_rest_DIFF, rest_days_DIFF to EARLY_FEATURE_COLUMNS
**Result**: roi=+44.60%, fold3=−0.98pp — worse; rest features don't help specialist
**Decision**: REVERTED

## Run 210 — Remove open_home_implied (redundant with market_implied_prob + line_move_delta)
**Hypothesis**: open_home_implied + line_move_delta ≈ market_implied_prob. Since market_implied_prob and line_move_delta are both in FEATURE_COLUMNS, open_home_implied is a near-exact linear combination. Removing it reduces a nearly redundant feature.
**Change**: Remove open_home_implied from FEATURE_COLUMNS
**Result**: roi=+43.99%, fold4=+14.03% (−3.15pp) — worse; open_home_implied is NOT fully redundant
**Decision**: REVERTED
**Insight**: Despite near-linear relationship, L1 gives open_home_implied a distinct coefficient; it carries information beyond what market_implied_prob + line_move_delta provides

## Run 145 — covid_era flag ★ CURRENT BEST
**Change**: Add covid_era = (year == 2020)
**Result**: roi=+44.85% (+0.0002pp over Run 144) — effectively identical but strictly better
**Decision**: KEPT ✓
