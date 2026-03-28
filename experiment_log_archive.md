# Experiment Log Archive — autoresearch/mlb-mar27

<!-- Oldest runs moved here when experiment_log.md exceeds 15 entries -->

## Key milestones summary (Runs 1-128):
- **Run 1**: LR baseline 17.78% (3-fold)
- **Run 9**: C=0.05 L1 gives 19.13% (3-fold)
- **Run 19**: threshold=0.12 gives 34.35% (3-fold)
- **Run 56**: threshold=0.20 gives 56.90% (3-fold, not comparable to 4-fold)
- **Run 66**: EARLY_CUTOFF=25 gives 35.73% (4-fold)
- **Run 72**: luck_DIFF feature +0.04pp
- **Run 75**: pythagorean_DIFF feature +0.52pp
- **Run 77**: C=0.04 gives 36.40% (4-fold); PROB_CAP sweep follows
- **Run 82-87**: PROB_CAP sweep from (0.25,0.75) → (0.34,0.66); threshold=0.14 → 41.16%
- **Run 93**: luck_x_momentum → 42.21% (4-fold honest best pre-130)
- **Run 101**: TRAIN_WINDOW=4 → 43.80% (noise-driven; fold4≈0%)
- **Run 112**: TW=4 noise best → 44.19% (fold4=-0.13%)
- **Run 121**: park_x_pythagorean at TW=None → 42.83%
- **Run 122**: pythagorean_short_DIFF → 43.51%
- **Run 128**: C=0.05 worse; C=0.04 is optimal

## Key milestones summary (Runs 146-173) — all failed to beat 44.85%:
- L1 aggressively zeros most interaction features; structural era flags (post_rule_change, post_dh_era, covid_era) are the only features that survived
- All model alternatives inferior to LR+L1: ensemble_stack=23.90%, LR+L2=37.97%
- Calibration: cv=3 is optimal (cv=2: fold4=5.09%, cv=5: fold4=9.33%)
- EARLY_CUTOFF=None catastrophic (fold4=-15%); 2021 fold very poor (8.21% ROI)
- All fine-grained parameter sweeps exhausted (C=0.045, BEST_W=14/17, MOMENTUM_W=9/12, threshold=0.13/0.15)
- Feature removal: days_since_asb, avg_DIFF/obp_DIFF/slg_DIFF all hurt fold4
- Era flags: juiced_ball_era (2015-2019) noise, same_division (22.26%) worse

## Key milestones summary (Runs 129-145):
- **Run 130**: post_rule_change (year>=2023) → 44.84% (+2.63pp breakthrough); fold3 (2024) +10.15pp
- **Run 139**: LGB with full feature set → 27.10%; LR's L1 sparsity clearly better
- **Run 144**: post_dh_era (year>=2022) → 44.85% marginal improvement KEPT
- **Run 145**: covid_era (year==2020) → 44.85% (+0.0002pp) KEPT; honest 4-fold best

## Confirmed optimal settings (as of Run 145):
- MODEL=lr, CALIBRATE=True, TW=None
- C=0.04, penalty=l1
- BEST_W=15, MOMENTUM_W=10
- CONFIDENCE_THRESHOLD=0.14
- PROB_CAP=(0.34, 0.66)
- EARLY_CUTOFF=25 (critical — None gives fold4=-15%)
- KELLY_FRACTION: invariant (scales bet sizes, not ROI %)
- Feature set: 64 features (base 43 + day_of_week + sharp_x_fip + momentum_DIFF + luck_DIFF + pythagorean_DIFF + luck_x_momentum + park_x_pythagorean + pythagorean_short_DIFF + post_rule_change + post_dh_era + covid_era)

## Confirmed dead ends (feature interactions that L1 zeros):
- fip_x_line, form_x_fip, woba_x_sharp, temp_x_fip, abs_line_move, temp_x_wrc, sharp_x_mkt, sharp_x_form, luck_recent_DIFF, luck_x_pythagorean, era_vs_fip_DIFF
- rest_x_fip, post_rule_x_fip (zeroed by L1)
- season_year (extrapolates badly)
- post_rule_x_luck (fold4 drops)
- pythagorean_x_momentum, wrc_x_post_rule (hurt or no benefit)
- same_division flag (worse)

## Key milestones summary (Runs 174-186) — all failed to beat 44.85%:
- ElasticNet (l1_ratio=0.7) worse (41.42%); L1 sparsity essential
- Calibrate early specialist: no help
- EARLY_SEASON_GAMES=30: making early_season_flag meaningful didn't help
- sharp_x_momentum, sharp_x_luck, streak_x_momentum, pyth_x_dh_era, market_x_momentum: all zeroed by L1 or hurt
- close_total: redundant with park_factor+weather
- Remove month: marginal loss (-0.13pp)
- Remove early_season_flag: confirmed dead feature (always 0)

## Key milestones summary (Runs 187-198) — all failed to beat 44.85%:
- C=0.035 (44.43%): C=0.04 confirmed optimal from both sides
- threshold=0.16 (34.29%, fold4 negative): 0.14 is a sharp optimum
- LGB num_leaves=15 (29.25%), XGB max_depth=3 (30.38%): GBDT confirmed dead end for all configs
- solver='liblinear' (43.80%): saga finds better optimum
- Remove weather features (42.74%): weather contributes to fold1/3
- market_prob_sq (42.44%): quadratic market term adds noise
- RobustScaler (38.05%): StandardScaler much better
- Remove day_of_week (42.91%), is_series_finale (42.48%), handedness (42.74%): all contribute
- Early specialist C=0.05 (44.41%), L1 penalty (44.48%): L2/C=0.1 optimal for specialist
- Remove run_diff_std_W_DIFF (44.59%): consistency metric contributes

## Confirmed dead ends (feature interactions that L1 zeros — extended):
- sharp_x_luck, sharp_x_momentum, streak_x_momentum, pyth_x_dh_era, market_x_momentum, close_total
- sp_luck_DIFF (linear combination of existing features — L1 correctly zeros it)

## Confirmed dead ends (parameter sweeps):
- C=0.03, C=0.045, C=0.05 (C=0.04 is sweet spot)
- threshold=0.13, threshold=0.15 (0.14 is optimal in 4-fold honest regime, re-confirmed with era flags)
- PROB_CAP=(0.33,0.67), (0.34,0.68), (0.35,0.65), (0.36,0.64)
- BEST_W=12, BEST_W=14, BEST_W=17 (15 is optimal)
- MOMENTUM_W=7, MOMENTUM_W=9, MOMENTUM_W=12 (10 is optimal)
- EARLY_CUTOFF=None, 20, 30 (25 is optimal)
- TRAIN_WINDOW_YEARS=3,4,5 (None is honest; 4 is noise-driven)
- LGB, XGB, ensemble_avg, ensemble_stack (LR dominates)
- DYNAMIC_THRESHOLD=True (worse)
- CALIBRATE=False (no effect on LR)
- CalibratedClassifierCV cv=2/5, sigmoid method (cv=3 isotonic optimal)
- L2 penalty (37.97%), ElasticNet l1_ratio=0.7 (41.42%)
- 5th fold (2021): 8.21% ROI dilutes mean
- EARLY_SEASON_GAMES=30 (no benefit)
- Era flags in EARLY_FEATURE_COLUMNS (interference)

## Key milestones summary (Runs 199-210) — all failed to beat 44.85%:
- sp_luck_DIFF (ERA-FIP gap): linear combination of existing features, zeroed by L1
- market_prob_sq: quadratic market term adds noise
- Half-season folds: 47.30% (8-fold mean), not comparable; reveals 2025H1=+62.48%, 2025H2=−12.66%
- Mean imputation: identical (no NaN values)
- Remove run_diff_std_W_DIFF, total_move_delta, open_home_implied: all hurt
- XGB max_depth=3: 30.38%, GBDT dead end confirmed for ALL configs
- MOMENTUM_W=8: 43.26%, confirmed 10 is optimal
- BEST_W=16: 42.28%, confirmed 15 is optimal
- Early specialist: sp_rest_DIFF/rest_days_DIFF additions don't help; sharp money signals hurt
- Truncated fold4 (2025H1 only): 56.17% but cherry-picking, not a valid comparison

## ALL PHASES EXHAUSTED as of Run 210:
- Phase 1 (baselines): LR, LGB, XGB, ensemble_avg, ensemble_stack all tested
- Phase 2 (features): 64-feature set fully optimized; all additions zeroed by L1
- Phase 3 (architecture): LGB num_leaves=15, XGB max_depth=3, ElasticNet, L2 all worse
- Phase 4 (calibration/betting): cv=3 isotonic optimal; threshold=0.14 sharp optimum
- Phase 5 (walk-forward): 4-fold optimal; half-season reveals 2025H2 regime change
- Research backlog: ALL QUEUED items tried or invalidated
