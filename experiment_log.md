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

## Run 179 — sharp_x_luck interaction (sharp money × luck differential)
**Hypothesis**: sharp_x_fip (sharp money × pitcher quality) survives L1. sharp_x_luck (sharp_move_flag × luck_DIFF) adds an analogous regression signal: sharp bettors backing a team that has been winning beyond what their run differential supports should be especially reliable — the sharp bettors are effectively betting on regression-to-mean for the opponent. This is distinct from sharp_x_fip (which is about pitcher quality) and luck_x_momentum (which has no sharp-money component).
**Change**: Add sharp_x_luck = sharp_move_flag * luck_DIFF in engineer_new_features() and FEATURE_COLUMNS
**Result**: roi=+43.22% mean, fold4=+17.22% — worse; L1 zeros it or fold1/2 trade away
**Decision**: REVERTED (−1.63pp vs best)

## Run 180 — Replace sharp_x_fip with sharp_x_momentum
**Hypothesis**: Run 177 (adding sharp_x_momentum alongside sharp_x_fip) gave fold4=+19.50% (+2.32pp) but hurt fold1/3 (net −0.14pp). The two interactions may compete for L1 budget since both encode "sharp money + team quality." Replacing sharp_x_fip with sharp_x_momentum gives the hot-streak interpretation exclusive access to the sharp-money signal slot, potentially improving fold4 generalization without dilution.
**Change**: Replace sharp_x_fip with sharp_x_momentum (sharp_move_flag * momentum_DIFF) in FEATURE_COLUMNS; add to engineer_new_features()
**Result**: roi=+44.71% mean, fold4=+19.50% — identical to Run 177 (adding both); L1 already zeroed sharp_x_fip in Run 177
**Decision**: REVERTED (−0.14pp vs best)
**Insight**: sharp_x_fip was already zeroed by L1 in Run 177; the sharp_x_momentum result is the same with or without sharp_x_fip

## Run 181 — pyth_x_dh_era (pythagorean_DIFF × post_dh_era)
**Hypothesis**: Universal DH (2022+) changed team construction: teams can no longer hide a weak hitter at SP. Pythagorean efficiency in the DH era may be a purer signal. For fold4 validation (2025), all games have post_dh_era=1, giving the model a distinct coefficient for DH-era pythagorean vs earlier years.
**Change**: Add pyth_x_dh_era = pythagorean_DIFF * post_dh_era in engineer_new_features() and FEATURE_COLUMNS
**Result**: roi=+43.43% mean, fold4=+13.06% — worse; interaction creates conflicting signals with pythagorean_DIFF
**Decision**: REVERTED (−1.42pp vs best)

## Run 182 — streak_x_momentum (winning streak × recent form improvement)
**Hypothesis**: streak_DIFF captures consecutive wins (binary momentum signal). momentum_DIFF captures recent win% vs longer-term win% (quantitative trend). A team with both a winning streak AND positive momentum is showing doubly reliable form. This product captures their joint signal, distinct from luck_x_momentum (regression) and sharp_x_fip (market agreement).
**Change**: Add streak_x_momentum = streak_DIFF * momentum_DIFF in engineer_new_features() and FEATURE_COLUMNS
**Result**: roi=+44.81% mean, fold4=+13.13% — near-miss (−0.04pp); fold3 +3.82pp but fold4 −4.05pp
**Decision**: REVERTED (−0.04pp vs best)

## Run 183 — sharp_x_momentum + streak_x_momentum (replace sharp_x_fip)
**Hypothesis**: Run 180 (sharp_x_momentum replacing sharp_x_fip) gave fold4=+19.50% (+2.32pp) but fold3=−2.21pp. Run 182 (streak_x_momentum) gave fold3=+51.52% (+3.82pp) but fold4=−4.05pp. These effects are complementary: if additive, combining both features yields expected fold3=+1.61pp, fold4=−1.73pp, netting ~+0.07pp above best. L1 interaction effects may produce a better outcome.
**Change**: Replace sharp_x_fip with sharp_x_momentum AND add streak_x_momentum
**Result**: roi=+44.77% mean, fold4=+17.88% — additive effects didn't hold; L1 competition reduced both signals
**Decision**: REVERTED (−0.08pp vs best)

## Run 184 — market_x_momentum (continuous line move × momentum_DIFF)
**Hypothesis**: sharp_x_fip uses binary sharp_move_flag × pitcher quality. line_move_delta is the continuous version of the market signal. market_x_momentum = line_move_delta * momentum_DIFF preserves movement magnitude — a 0.06 line move toward a hot team is twice the signal of a 0.03 move. This is distinct from sharp_x_fip and sharp_x_momentum; L1 may keep it where binary interactions got zeroed.
**Change**: Add market_x_momentum = line_move_delta * momentum_DIFF in engineer_new_features() and FEATURE_COLUMNS
**Result**: roi=+42.93% mean, fold4=+13.72% — worse; continuous line move adds noise
**Decision**: REVERTED (−1.92pp vs best)

## Run 185 — close_total as direct feature (O/U line level)
**Hypothesis**: total_move_delta (O/U movement) is in FEATURE_COLUMNS but close_total (the O/U level) is not. The absolute total line (e.g., 8.5 vs 9.5) directly measures expected run environment: high totals indicate offense-heavy games where batting quality matters more; low totals indicate pitcher's duels. This is orthogonal to total_move_delta and may help the model calibrate its betting on run-environment-sensitive features.
**Change**: Add close_total to FEATURE_COLUMNS (already available in df after add_market_columns)
**Result**: roi=+43.56% mean, fold4=+14.35% — worse; O/U level is already captured by park_factor + weather features
**Decision**: REVERTED (−1.29pp vs best)

## Run 186 — Remove month (calendar noise)
**Hypothesis**: month (1-12) captures seasonal patterns but may add year-specific seasonal noise that doesn't generalize to 2025. days_since_asb captures timing within the season; post_* era flags capture structural changes; day_of_week captures intra-week patterns. month might be adding noise from playoff-race dynamics that vary year-to-year.
**Change**: Remove month from FEATURE_COLUMNS
**Result**: roi=+44.72% mean, fold4=+17.11% — near-miss (−0.13pp); month marginally useful across folds
**Decision**: REVERTED (−0.13pp vs best)

## Run 178 — Remove early_season_flag (always 0 for main model, dead feature)
**Hypothesis**: early_season_flag is always 0 for main model inputs because EARLY_SEASON_GAMES=15 < EARLY_CUTOFF=25 (all main model games have min(h,a games) >= 25 > 15). L1 zeros it. Removing it is a no-op but confirms dead feature status.
**Change**: Remove early_season_flag from FEATURE_COLUMNS
**Result**: roi=+44.848% (−0.000003pp vs best) — confirms dead feature; essentially no effect
**Decision**: REVERTED (dead feature confirmed)

## Run 177 — sharp_x_momentum interaction (sharp money × recent hot streak)
**Hypothesis**: sharp_x_fip (sharp money × pitcher quality) is already in FEATURE_COLUMNS and survives L1. sharp_x_momentum (sharp_move_flag × momentum_DIFF) adds the analogous signal: sharp bettors backing a hot team is doubly reliable — they see both the public momentum and have private information about the team's current form.
**Change**: Add sharp_x_momentum = sharp_move_flag * momentum_DIFF
**Result**: roi=+44.71% mean, fold4=+19.50% — near-miss; fold4 improved but fold1/3 dropped slightly
**Decision**: REVERTED (−0.14pp vs best)

## Run 176 — EARLY_SEASON_GAMES=30 (create meaningful early-season flag in main model)
**Hypothesis**: Currently EARLY_SEASON_GAMES=15 < EARLY_CUTOFF=25, making early_season_flag always 0 for main model inputs (dead feature). Setting EARLY_SEASON_GAMES=30 > EARLY_CUTOFF=25 makes early_season_flag=1 for games where min(home_games, away_games) in [25, 29], creating a genuine early-season signal in the main model for teams with ~25-29 games played.
**Change**: EARLY_SEASON_GAMES=30 (was 15)
**Result**: roi=+43.28% mean, fold4=+13.72% — worse; 25-29 game range is not a meaningful separate regime
**Decision**: REVERTED
**Insight**: early_season_flag was always 0 for main model — making it meaningful didn't help

## Run 175 — Calibrate the early season specialist (add CalibratedClassifierCV to build_early_lr)
**Hypothesis**: The early specialist uses a simple uncalibrated LR (C=0.1). Adding isotonic calibration (cv=3) might improve its probability estimates, reducing early-season prediction errors and improving fold4 performance.
**Change**: Add CalibratedClassifierCV(method='isotonic', cv=3) to build_early_lr()
**Result**: roi=+43.77% mean — worse; calibration doesn't help early specialist
**Decision**: REVERTED

## Run 174 — ElasticNet penalty (l1_ratio=0.7, C=0.04)
**Hypothesis**: Pure L1 zeros features too aggressively. Pure L2 retains too many noisy ones. ElasticNet at l1_ratio=0.7 (70% L1 / 30% L2) zeros the weakest features while shrinking survivors more smoothly, potentially finding a better regularization balance.
**Change**: LR_PARAMS penalty='elasticnet', l1_ratio=0.7, C=0.04 (solver='saga' supports this)
**Result**: roi=+41.42% mean, fold4=+17.43% (105 bets) — worse; L2 component retains noisy features
**Decision**: REVERTED

## Run 173 — CalibratedClassifierCV cv=2
**Change**: cv=2 (was 3)
**Result**: roi=+39.89% mean, fold4=+5.09% — worse; cv=3 is optimal
**Decision**: REVERTED

## Run 172 — C=0.045 fine-grained sweep
**Change**: LR C=0.045 (was 0.04)
**Result**: roi=+43.20% mean — worse; C=0.04 is exact optimum
**Decision**: REVERTED

## Run 171 — Remove avg_DIFF, obp_DIFF, slg_DIFF
**Change**: Remove 3 OPS subcomponents (subsumed by wrc_plus_DIFF)
**Result**: roi=+43.71% mean, fold4=+11.65% (79 bets) — worse; features contribute to fold4
**Decision**: REVERTED

## Run 170 — PROB_CAP=(0.34, 0.68) asymmetric
**Change**: Wider upper cap for strong home favorites
**Result**: roi=+43.40% mean, fold4=+19.30% — worse; cap adds lower-quality bets
**Decision**: REVERTED

## Run 169 — 5th walk-forward fold (val=2021)
**Change**: Add fold ("2021-01-01", "2021-01-01", "2022-01-01")
**Result**: roi=+37.52% (5-fold mean); 2021 fold=8.21% (58 bets) — dilutes mean; 2021 is poor fold
**Decision**: REVERTED

## Run 168 — Era flags in EARLY_FEATURE_COLUMNS
**Change**: Add post_rule_change + post_dh_era to specialist
**Result**: roi=+43.00% mean — worse; era flags in specialist cause interference
**Decision**: REVERTED

## Run 167 — CalibratedClassifierCV cv=5
**Change**: cv=5 (was 3)
**Result**: roi=+39.40% mean, fold4=+9.33% (110 bets) — worse; cv=3 is optimal
**Decision**: REVERTED

## Run 166 — Sigmoid calibration method
**Change**: method='sigmoid' (was 'isotonic')
**Result**: roi=+40.50% mean — worse; isotonic is better suited
**Decision**: REVERTED

## Run 165 — juiced_ball_era flag (2015-2019)
**Change**: Add juiced_ball_era = (year >= 2015) & (year <= 2019)
**Result**: roi=+44.12% mean, fold4=+12.69% — worse; flag adds noise to coefficient estimation
**Decision**: REVERTED

## Run 164 — MOMENTUM_W=9
**Change**: MOMENTUM_W=9 (was 10)
**Result**: roi=+40.40% mean, fold4=+9.94% — worse; MOMENTUM_W=10 is exact optimum
**Decision**: REVERTED

## Run 163 — PROB_CAP=(0.36, 0.64)
**Change**: Narrower symmetric cap
**Result**: roi=+41.33% mean, fold4=+10.32% — worse; removes good bets
**Decision**: REVERTED

## Run 162 — BEST_W=17
**Change**: BEST_W=17 (was 15)
**Result**: roi=+37.05% mean — much worse; BEST_W=15 is the sweet spot
**Decision**: REVERTED

## Run 161 — TRAIN_WINDOW_YEARS=4 with era flags
**Change**: TW=4 (re-test with structural era flags)
**Result**: roi=+32.74% mean, fold3=+5.61% (985 bets!) — still noise-driven
**Decision**: REVERTED

## Run 160 — LR with L2 penalty (C=0.1)
**Change**: penalty='l2', solver='lbfgs', C=0.1
**Result**: roi=+37.97% mean, fold4=+17.02% (199 bets) — L2 retains too many noisy features
**Decision**: REVERTED
**Insight**: L1 sparsity is essential — L2 floods with marginal bets

## Run 159 — ensemble_stack
**Change**: MODEL="ensemble_stack"
**Result**: roi=+23.90% mean, fold4=+17.71% (271 bets) — far worse; 3-4x bets, Brier=0.252
**Decision**: REVERTED
**Insight**: ensemble_stack makes too many low-edge bets; LR L1 sparsity is essential

## Run 145 — covid_era flag ★ CURRENT BEST
**Change**: Add covid_era = (year == 2020)
**Result**: roi=+44.85% (+0.0002pp over Run 144) — effectively identical but strictly better
**Decision**: KEPT ✓
