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

## Confirmed optimal settings (as of Run 130):
- MODEL=lr, CALIBRATE=True, TW=None
- C=0.04, penalty=l1
- BEST_W=15, MOMENTUM_W=10
- CONFIDENCE_THRESHOLD=0.14
- PROB_CAP=(0.34, 0.66)
- EARLY_CUTOFF=25
- Feature set: 61 features (base 43 + day_of_week + sharp_x_fip + momentum_DIFF + luck_DIFF + pythagorean_DIFF + luck_x_momentum + park_x_pythagorean + pythagorean_short_DIFF + post_rule_change)

## Confirmed dead ends (feature interactions that L1 zeros):
- fip_x_line, form_x_fip, woba_x_sharp, temp_x_fip, abs_line_move, temp_x_wrc, sharp_x_mkt, sharp_x_form, luck_recent_DIFF, luck_x_pythagorean, era_vs_fip_DIFF
- season_year (extrapolates badly)
- post_rule_x_luck (fold4 drops)

## Confirmed dead ends (parameter sweeps):
- C=0.03, C=0.05 (C=0.04 is sweet spot)
- threshold=0.13, threshold=0.15 (0.14 is optimal in 4-fold honest regime)
- PROB_CAP=(0.33,0.67), (0.35,0.65)
- BEST_W=12 (15 is optimal)
- MOMENTUM_W=7 (10 is optimal)
- EARLY_CUTOFF=20, 30 (25 is optimal)
- TRAIN_WINDOW_YEARS=3,4,5 (None is honest; 4 is noise-driven)
- LGB, XGB, ensemble_avg, ensemble_stack (LR dominates)
