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

## Confirmed dead ends (parameter sweeps):
- C=0.03, C=0.05 (C=0.04 is sweet spot)
- threshold=0.13, threshold=0.15 (0.14 is optimal in 4-fold honest regime, re-confirmed with era flags)
- PROB_CAP=(0.33,0.67), (0.35,0.65)
- BEST_W=12, BEST_W=14 (15 is optimal)
- MOMENTUM_W=7, MOMENTUM_W=12 (10 is optimal)
- EARLY_CUTOFF=None, 20, 30 (25 is optimal)
- TRAIN_WINDOW_YEARS=3,4,5 (None is honest; 4 is noise-driven)
- LGB, XGB, ensemble_avg, ensemble_stack (LR dominates)
- DYNAMIC_THRESHOLD=True (worse)
- CALIBRATE=False (no effect on LR)
