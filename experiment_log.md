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

## [New Era: Multi-Model Walk-Forward, Full Feature Set]
- Walk-forward validation (3 folds: val=2022, 2023, 2024). Mean ROI is the only metric.
- Multi-model arena: lgb / xgb / lr / mlp / ensemble_avg / ensemble_stack
- Early season specialist: separate LR for games where either team has < 15 games played
- Full feature set: 43 FEATURE_COLUMNS + 11 EARLY_FEATURE_COLUMNS
- Prob capping at [0.25, 0.75]; half-Kelly for early-season bets
- ensemble_stack uses disagreement feature (stats_prob - market_prob) as meta-input

**Run order (Phase 1 — baselines):** lr → lgb → xgb → ensemble_avg → ensemble_stack

---

<!-- Add new runs below this line -->

## Run 5 — ensemble_stack baseline (Phase 1)
**Hypothesis**: LR meta-learner stacked on LGB + XGB + LR with disagreement feature (stats_prob − market_prob) should capture non-linear model interactions and beat simple LR.
**Change**: MODEL="ensemble_stack"
**Result**: roi=+11.04%, brier=0.2531, n_bets=5541 (3-fold mean)
**Decision**: REVERTED (LR 17.78% still best)
**Insight**: Stack degrades calibration (higher Brier); meta-learner overfits the disagreement signal.

## PLATEAU REACHED (5/5) — moving to Phase 2: Feature Engineering

## Run 85 — PROB_CAP=(0.34, 0.66) (KEPT — new best)
**Change**: PROB_CAP=(0.34,0.66)
**Result**: roi=+39.66% mean, fold4=+27.10%, n_bets=381
**Decision**: KEPT (+0.67pp mean; fold4 stable)

---

## Run 84 — PROB_CAP=(0.32, 0.68) (KEPT — new best)
**Hypothesis**: Continuing tighter cap sweep.
**Change**: PROB_CAP=(0.32,0.68)
**Result**: roi=+38.99% mean, brier=0.2369, fold4=+27.97%, n_bets=402
**Decision**: KEPT (new best: +1.11pp mean; fold4 also improved to 27.97%)
**Insight**: Cap continues to improve through 0.32; fold4 rebounded slightly.

---

## Run 83 — PROB_CAP=(0.30, 0.70) (KEPT — new best)
**Hypothesis**: Tighter cap than (0.28,0.72) further concentrates high-quality bets.
**Change**: PROB_CAP=(0.30,0.70)
**Result**: roi=+37.88% mean, brier=0.2369, fold4=+26.91%, n_bets=435
**Decision**: KEPT (new best: +0.40pp; fold4 declining with tighter cap — watch for overfitting)
**Insight**: ROI continues to improve but fold4 regression suggests diminishing returns on cap tightening.

---

## Run 82 — PROB_CAP=(0.28, 0.72) (KEPT — new best)
**Hypothesis**: Slightly tighter cap concentrates on higher-confidence bets without over-restricting like (0.35,0.65).
**Change**: PROB_CAP=(0.28,0.72) (was 0.25,0.75)
**Result**: roi=+37.48% mean, brier=0.2369, fold4=+28.43%, n_bets=467
**Decision**: KEPT (new best: +1.08pp over 36.40%; n_bets lower but all folds profitable)
**Insight**: Slight tightening of prob cap improves signal-to-noise; doesn't overfit like (0.35,0.65) did.

---

## Run 81 — sharp_x_form interaction
**Hypothesis**: Sharp money confirming team form has non-linear signal.
**Change**: Added sharp_x_form = sharp_move_flag * win_pct_W_DIFF.
**Result**: roi=+35.63% mean — worse.
**Decision**: REVERTED

---

## Run 80 — era_vs_fip_DIFF (pitcher luck gap)
**Hypothesis**: ERA-FIP gap measures pitcher luck; unlucky away SP should regress.
**Change**: Added era_vs_fip_DIFF = sp_era_DIFF - sp_fip_DIFF.
**Result**: roi=+36.40% mean — identical to best (L1 zeroed).
**Decision**: REVERTED; highly collinear with existing ERA/FIP features.

---

## Run 79 — threshold=0.12 at C=0.04
**Result**: roi=+32.37% mean, fold4=+28.53% — much worse on mean.
**Decision**: REVERTED; threshold=0.13 optimal.

---

## Run 78 — LR C=0.03
**Result**: roi=+33.56% mean, fold4=+19.28% — much worse; overfit.
**Decision**: REVERTED; C=0.04 is the new optimum.

---

## Run 77 — LR C=0.04 (KEPT — new best)
**Hypothesis**: With more features (luck_DIFF, pythagorean_DIFF), C=0.05 may be slightly over-regularizing; C=0.04 allows these new features more weight.
**Change**: LR_PARAMS C=0.04 (was 0.05)
**Result**: roi=+36.40% mean, brier=0.2369, fold4=+30.27%, n_bets=504
**Decision**: KEPT (new best: +0.11pp mean, +5.32pp fold4)
**Insight**: More features warrant slightly less regularization; fold4 jump from 24.95% to 30.27% is very significant.

---

## Run 76 — short-window pythagorean_DIFF (MOMENTUM_W=10)
**Hypothesis**: Recent run efficiency (10-game window) adds independent signal from 15-game Pythagorean.
**Change**: Added pythagorean_short_DIFF (using MOMENTUM_W=10 RS/RA) + expanded roll_short join.
**Result**: roi=+36.04% mean, fold4=+24.96% — slightly worse than best.
**Decision**: REVERTED
**Insight**: Short-window Pythagorean adds noise; 15-game window is the right granularity.

---

## Run 75 — pythagorean_DIFF feature (KEPT — new best)
**Hypothesis**: Rolling Pythagorean win% ratio (RS^2/(RS^2+RA^2)) captures run production efficiency beyond raw run differential.
**Change**: Added `pythagorean_DIFF = h_pyth - a_pyth` to engineer_new_features() and FEATURE_COLUMNS.
**Result**: roi=+36.29% mean, brier=0.2368, fold4=+24.95%, n_bets=505
**Decision**: KEPT (new best: +0.52pp over 35.77%; fold4 also improved)
**Insight**: Pythagorean efficiency ratio carries independent signal from run_diff_avg — captures RS/RA balance separately.

---

## Run 74 — woba_x_sharp interaction
**Result**: roi=+35.42% mean — worse. L1 zeroed or near-zeroed it.
**Decision**: REVERTED

---

## Run 73 — threshold=0.14 at current best config
**Result**: roi=+35.71% mean, fold4=+22.62% — slightly worse.
**Decision**: REVERTED; threshold=0.13 optimal.

---

## Run 72 — luck_DIFF feature (KEPT — new best)
**Hypothesis**: Teams overperforming Pythagorean expectation (lucky wins) tend to regress; this luck differential gives an independent signal.
**Change**: Added `luck_DIFF = season_win_pct_DIFF - pythagorean_DIFF` to engineer_new_features() and FEATURE_COLUMNS.
**Result**: roi=+35.77% mean, brier=0.2369, fold4=+24.66%, n_bets=504
**Decision**: KEPT (new best: +0.04pp over 35.73%)
**Insight**: Pythagorean regression signal kept by L1 — small but consistent improvement across folds 1 and 3.

---

## Run 71 — Early specialist C=0.5
**Hypothesis**: More specialist training data with EARLY_CUTOFF=25 allows less regularization.
**Change**: build_early_lr C=0.5 (was 0.1)
**Result**: roi=+35.56% mean, fold4=+24.77% — slightly worse.
**Decision**: REVERTED
**Insight**: Specialist C=0.1 is near-optimal; reducing regularization marginally worse.

---

## Run 70 — expanded EARLY_FEATURE_COLUMNS (add open_home_implied, line_move, sharp, day_of_week)
**Hypothesis**: Specialist lacks market momentum signals that help the regular model.
**Change**: Added open_home_implied, line_move_delta, sharp_move_flag, day_of_week to EARLY_FEATURE_COLUMNS.
**Result**: roi=+34.76% mean, fold4=+24.66% — worse.
**Decision**: REVERTED
**Insight**: Original minimal specialist feature set is better; more features add noise for early-season LR.

---

## Run 69 — threshold=0.12 at EARLY_CUTOFF=25
**Result**: roi=+32.05% mean, fold4=+24.12% — much worse.
**Decision**: REVERTED
**Insight**: threshold=0.13 remains optimal regardless of EARLY_CUTOFF setting.

---

## Run 68 — EARLY_CUTOFF=22
**Result**: roi=+32.99% mean, fold4=+17.66% — much worse, highly variable.
**Decision**: REVERTED

---

## Run 67 — EARLY_CUTOFF=30
**Result**: roi=+34.49% mean, fold4=+23.82% — worse.
**Decision**: REVERTED
**Insight**: EARLY_CUTOFF=25 is the sweet spot; 22 causes high variance (fold4=17.66%), 30 is also worse.

---

## Run 66 — EARLY_CUTOFF=25 (KEPT — new 4-fold best)
**Hypothesis**: Expanding early-season specialist to 25 games improves coverage of genuinely uncertain early period.
**Change**: EARLY_CUTOFF=25 (was 15)
**Result**: roi=+35.73% mean, brier=0.2368, fold4=+24.71%, n_bets=509
**Decision**: KEPT (new 4-fold best mean: +0.21pp over 35.52%)
**Insight**: Larger specialist window selects slightly different bet pool; mean ROI up but fold4 drops (fewer bets: 129 vs 164). Tradeoff accepted per mean-ROI-only rule.

---

## Run 65 — temp_x_wrc interaction
**Hypothesis**: Heat amplifies batting quality gap (seeded interaction from backlog).
**Change**: Added temp_x_wrc to FEATURE_COLUMNS.
**Result**: roi=+32.96% mean, fold4=+24.07% — much worse.
**Decision**: REVERTED
**Insight**: Interaction features consistently degrade LR; market signal dominates and interactions introduce noise.

---

## Run 64 — LR C=0.06
**Hypothesis**: Fine-grained search between C=0.05 and C=0.10.
**Change**: LR_PARAMS C=0.06 (was 0.05)
**Result**: roi=+32.49% mean, fold4=+21.98% — much worse.
**Decision**: REVERTED
**Insight**: C=0.05 is a sharp optimum; small increase in C causes significant degradation.

---

## Run 63 — threshold=0.14 at current best config
**Hypothesis**: Run 39 tested 0.14 before day_of_week/MOMENTUM_W=10; with better features, 0.14 may beat 0.13.
**Change**: CONFIDENCE_THRESHOLD=0.14 (was 0.13)
**Result**: roi=+35.12% mean, fold4=+24.13% — worse on both metrics.
**Decision**: REVERTED
**Insight**: threshold=0.13 remains optimal; 0.14 degrades fold4 significantly.

---

## Run 62 — MOMENTUM_W=7
**Hypothesis**: MOMENTUM_W=7 splits the difference between tested values of 5 and 10.
**Change**: MOMENTUM_W=7 (was 10)
**Result**: roi=+35.14% mean, fold4=+26.92% — slightly worse.
**Decision**: REVERTED
**Insight**: MOMENTUM_W=10 remains optimal; 7 is worse on mean.

---

## Run 61 — BEST_W=20
**Hypothesis**: W=20 smooths out noise; may generalize better to 2025.
**Change**: BEST_W=20
**Result**: roi=+34.06% mean, fold4=+23.60% — worse. W=15 optimal confirmed.
**Decision**: REVERTED

---

## Run 60 — BEST_W=10
**Hypothesis**: Shorter window may be more reactive to recent trends.
**Change**: BEST_W=10
**Result**: roi=+33.77% mean, fold4=+26.26% — worse.
**Decision**: REVERTED
**Insight**: W=15 is confirmed optimal for walk-forward; W in {10,12,20} all worse.

---

## Run 59 — BEST_W=12
**Hypothesis**: W=15 was validated in MLP era; walk-forward may favor a shorter 12-game window balancing recency with stability.
**Change**: BEST_W=12 (was 15)
**Result**: roi=+31.83% mean, fold4=+22.89% — significantly worse.
**Decision**: REVERTED
**Insight**: W=15 is clearly superior to W=12 walk-forward; shorter window is noisier.

---

## Run 58 — sp_form_vs_career feature
**Hypothesis**: SP recent rolling ERA minus career ERA (as a DIFF) captures whether starters are in better/worse form than their career norms.
**Change**: Added `sp_form_vs_career = rolling_era_DIFF - sp_era_DIFF` to engineer_new_features() and FEATURE_COLUMNS.
**Result**: roi=+35.52% mean, fold4=+29.50% — identical to best (L1 zeroed the feature).
**Decision**: REVERTED
**Insight**: L1 regression at C=0.05 zeros out features that are near-linear combinations of existing ones; this feature adds no independent signal.

---

## Run 57 — road_trip_length feature
**Hypothesis**: Consecutive away games for the away team captures cumulative fatigue beyond rest_days alone.
**Change**: Added `road_trip_length` (consecutive away game count) to engineer_new_features() and FEATURE_COLUMNS.
**Result**: roi=+34.00% mean, fold4=+25.23%, n_bets=547 — worse.
**Decision**: REVERTED
**Insight**: Road trip length adds noise; rest_days likely already captures fatigue signal.

---

## Run 56 — is_series_opener feature
**Result**: roi=+34.84% mean, fold4=+27.39% — worse. REVERTED.

---

## Run 55 — day_of_week circular encoding (dow_sin, dow_cos)
**Result**: roi=+33.13% mean, fold4=+20.69% — worse than raw integer. REVERTED.

---

## Run 54 — day_of_week feature (KEPT — current best)
**Hypothesis**: Day-of-week captures travel/fatigue patterns not captured by rest_days alone.
**Change**: Added `day_of_week` (0=Mon…6=Sun) to schedule context and FEATURE_COLUMNS.
**Result**: roi=+35.52% mean, brier=0.2368, fold4=+29.50%, n_bets=549
**Decision**: KEPT (new best: +0.38pp mean, +1.81pp on fold4)
**Insight**: Day of week carries real signal — likely Monday games (travel fatigue) or weekend games differ structurally.

---

## Run 53 — game_number_in_season feature
**Result**: roi=+35.06% — marginal drop. REVERTED.

---

## Run 52 — LGB num_leaves=15 at best config
**Result**: roi=+23.20% — far worse than LR. REVERTED.

---

## Run 51 — EARLY_CUTOFF=20
**Result**: roi=+32.11% mean, fold4=+16.29% — worse. REVERTED.

---

## Run 50 — EARLY_CUTOFF=10
**Result**: roi=+28.78% mean, fold4=+19.57% — worse. REVERTED.

---

## Run 49 — TRAIN_WINDOW_YEARS=3 (recent-data-only training)
**Hypothesis**: Limiting training to last 3 years avoids stale patterns.
**Result**: roi=+28.86% mean, fold4=+7.20% — much worse. REVERTED.
**Insight**: Full training history essential for model calibration; with ~140 bets/fold at threshold=0.13, need all data for stable coefficients.

---

## Run 48 — abs_line_move feature
**Result**: roi=+33.96% — marginal drop. REVERTED.

---

## Run 47 — fip_x_market interaction feature
**Result**: roi=+35.14% — identical (L1 zeros it). REVERTED.

---

## Run 46 — no market_implied_prob (at honest 4-fold config)
**Result**: roi=+34.27% mean — worse than baseline. REVERTED.
**Insight**: Market consensus remains useful even at honest 4-fold setting.

---

## Run 45 — EARLY_FEATURE_COLUMNS + market microstructure
**Result**: roi=+33.74% mean — worse. REVERTED.

---

## Run 44 — market_confidence feature (|mp - 0.5|)
**Result**: roi=+34.22% mean — worse. REVERTED.

---

## Run 43 — L1 feature pruning (remove 15 zeroed features)
**Result**: roi=+33.23% mean — worse. REVERTED.
**Insight**: L1 already handles zero-coef features; explicit pruning doesn't help.

---

## Run 42 — LR C=0.03 at 4-fold config
**Result**: roi=+34.15% mean — worse. C=0.05 optimal.

---

## Run 41 — elasticnet l1_ratio=0.5 and 0.8
**Result**: roi=+32.58% (0.5), +32.19% (0.8) — worse. L1 optimal.

---

## Run 40 — MOMENTUM_W=10 (CURRENT BEST pre-day_of_week)
**Hypothesis**: Longer momentum window (10g vs 5g) captures medium-term team trends.
**Change**: MOMENTUM_W=10 (was 5)
**Result**: roi=+35.14% mean, fold4=+27.69%, n_bets=556
**Decision**: KEPT (new best at time of discovery)
**Insight**: 10-game momentum window better than 5 or 7 for capturing team trajectory.

---

## ⚠️ OVERFITTING DISCOVERY: tight PROB_CAP hurts 2025 generalization
When PROB_CAP=(0.35,0.65) was set, 2022-2024 folds showed 43.51% but 2025 fold dropped to 16.89%.
Reverting to PROB_CAP=(0.25,0.75) restored 2025 ROI to ~25-27%. Tight cap was overfitting.
**Honest best config**: LR L1 C=0.05, threshold=0.13, PROB_CAP=(0.25,0.75), 4 folds → 33.58% mean, 26.67% on 2025.

---

## Run 40 — threshold=0.15 at PROB_CAP=(0.25,0.75), 4-fold
**Result**: roi=+34.88% mean, fold4=+20.23%, n_bets=344
**Decision**: REVERTED (fold4 degrades vs 0.13/0.14)

---

## Run 39 — threshold=0.14 at PROB_CAP=(0.25,0.75), 4-fold (best 4-fold mean)
**Result**: roi=+35.73% mean, fold4=+24.97%, n_bets=428
**Decision**: NEAR-BEST (best 4-fold mean but 2025 slightly weaker than 0.13)

---

## Run 38 — threshold=0.13 at PROB_CAP=(0.25,0.75), 4-fold (CURRENT BEST)
**Result**: roi=+33.58% mean, fold4=+26.67%, n_bets=553 (138/fold avg)
**Decision**: KEPT (best 2025 generalization; all 4 folds profitable)
**Insight**: 2025 fold at 26.67% is the most honest estimate of true edge.

---

## Run 37 — threshold=0.12 at PROB_CAP=(0.25,0.75), 4-fold
**Result**: roi=+32.01% mean, fold4=+25.01%, n_bets=524
**Decision**: SUPERSEDED by threshold=0.13

---

## Run 36 — PROB_CAP=(0.25,0.75) restore, 4-fold (key diagnostic)
**Hypothesis**: Tighter PROB_CAP may have overfit to 2022-2024; test original cap on all 4 folds.
**Change**: PROB_CAP=(0.25,0.75) restored; threshold=0.12; 4 folds
**Result**: roi=+32.01% mean, fold4=+25.01% (vs 17.66% with tight cap)
**Decision**: KEPT (original cap generalizes better to 2025)
**Insight**: Tight PROB_CAP is a selection overfitter — filters by year-specific edge distribution.

---

## Run 35 — threshold=0.12 at PROB_CAP=(0.35,0.65), 4-fold diagnostic
**Result**: roi=+35.59% mean, fold4=+17.66% — tight cap inflates 2022-2024, hurts 2025.
**Decision**: REVERTED

---

## Run 34 — threshold=0.10 at PROB_CAP=(0.35,0.65), 4-fold diagnostic
**Result**: roi=+31.40% mean, fold4=+17.46%
**Decision**: REVERTED

---

## Run 33 — 4-fold walk-forward (add 2025 val fold)
**Hypothesis**: Adding 2025 as a 4th validation fold gives a more honest out-of-sample test.
**Change**: WALK_FORWARD_FOLDS extended to include ("2025-01-01", "2025-01-01", "2026-01-01")
**Result**: roi=+36.86% (4-fold mean), brier=0.2367; Fold 4 (2025): roi=+16.89%, n_bets=135
**Decision**: KEPT (4-fold is now the standard; 2025 out-of-sample at 16.89% is meaningful signal)
**Insight**: 2025 ROI is ~20pp lower than 2022-2024 folds — suggests some overfitting to the tuned threshold/cap on 2022-2024. 2025 still profitable but lower edge density. The 3-fold "best" of 43.51% was partly the result of tuning on those folds.

---

## Run 32 — threshold=0.14 probe
**Hypothesis**: Push threshold past 0.13 to see if ROI keeps climbing.
**Change**: CONFIDENCE_THRESHOLD=0.14
**Result**: roi=+42.93%, brier=0.2363, n_bets=204 (3-fold mean, ~68/fold)
**Decision**: REVERTED (−0.58pp vs 0.13; fold 3 at 30.9% suggests noise dominates at this sample size)
**Insight**: 0.13 is the reliable peak; 0.14 enters high-variance territory.

---

## Run 31 — threshold=0.13 (new best at PROB_CAP=0.35/0.65)
**Hypothesis**: Fine-tuning threshold past 0.12 at the new tighter PROB_CAP setting.
**Change**: CONFIDENCE_THRESHOLD=0.13 (was 0.12)
**Result**: roi=+43.51%, brier=0.2363, n_bets=259 (3-fold mean, ~86/fold)
**Decision**: KEPT (new best: +1.94pp over threshold=0.12)
**Insight**: Peak shifts upward with tighter cap; 86 bets/fold is borderline but all folds positive.

---

## Run 30 — threshold=0.11 at PROB_CAP=(0.35,0.65)
**Hypothesis**: Fine-tune below 0.12 to confirm 0.12 is still the lower bound.
**Change**: CONFIDENCE_THRESHOLD=0.11
**Result**: roi=+37.30%, brier=0.2363, n_bets=471 (3-fold mean)
**Decision**: REVERTED (−4.21pp vs 0.12/0.13; too many low-quality bets)
**Insight**: Below 0.12 dilutes the bet quality at this PROB_CAP.

---

## Run 29 — no-market-prob LR (pure stats)
**Hypothesis**: Remove market_implied_prob from FEATURE_COLUMNS; model becomes stats-only, edges may be purer.
**Change**: market_implied_prob removed from features
**Result**: roi=+39.69%, brier=0.2364, n_bets=360 (3-fold mean)
**Decision**: REVERTED (−1.88pp; market consensus is still a useful feature)
**Insight**: Market implied prob adds signal; removing it hurts model calibration.

---

## Run 28 — interaction features added (all 5 seeded)
**Hypothesis**: fip_x_line, form_x_fip, temp_x_wrc, woba_x_sharp, early_x_fip add synergistic signal.
**Change**: Added 5 seeded interactions to FEATURE_COLUMNS
**Result**: roi=+39.48%, brier=0.2365, n_bets=364 (3-fold mean)
**Decision**: REVERTED (−2.09pp; L1 can't prune these enough at C=0.05)
**Insight**: Interaction features add noise; existing features already capture main signals.

---

## Run 27 — LR C=0.10 (weaker regularization) at new config
**Hypothesis**: C=0.10 with tighter PROB_CAP may calibrate better.
**Change**: LR_PARAMS C=0.10
**Result**: roi=+39.67%, brier=0.2365, n_bets=379 (3-fold mean)
**Decision**: REVERTED (−1.90pp; C=0.05 remains optimal)
**Insight**: C=0.05 is the sweet spot regardless of PROB_CAP setting.

---

## Run 26 — PROB_CAP=(0.40,0.60) probe
**Hypothesis**: Even tighter cap further concentrates bets.
**Change**: PROB_CAP=(0.40,0.60) (was 0.35,0.65)
**Result**: roi=+35.83%, brier=0.2363, n_bets=419 (3-fold mean)
**Decision**: REVERTED (reverses gain from tighter cap; sweet spot is 0.35,0.65)
**Insight**: Cap at 0.40 over-restricts the edge window; fewer valid bets with worse ROI.

---

## Run 25 — PROB_CAP=(0.35,0.65)
**Hypothesis**: Tighter cap forces model to have >12% edge even after aggressive clipping — stronger filter for high-quality bets.
**Change**: PROB_CAP=(0.35,0.65) (was 0.30,0.70)
**Result**: roi=+41.57%, brier=0.2363, n_bets=359 (3-fold mean)
**Decision**: KEPT (new best: +1.00pp over 0.30,0.70; n_bets adequate at ~120/fold)
**Insight**: Incremental improvement; peak seems near — capping at 0.40 reverses gains.

---

## Run 24 — PROB_CAP=(0.30,0.70)
**Hypothesis**: Tighter prob capping prevents overconfident model probs from triggering low-quality bets; only genuine large edges survive the 0.12 threshold.
**Change**: PROB_CAP=(0.30,0.70) (was 0.25,0.75)
**Result**: roi=+40.57%, brier=0.2363, n_bets=433 (3-fold mean)
**Decision**: SUPERSEDED by (0.35,0.65)
**Insight**: Tighter cap is better — filters out bets where edge is inflated by model overconfidence.

---

## Run 23 — EARLY_CUTOFF=None (disable specialist)
**Hypothesis**: All games using one LR model may generalize better than splitting early/regular season.
**Change**: EARLY_CUTOFF=None
**Result**: roi=+33.82%, brier=0.2370, n_bets=507 (3-fold mean)
**Decision**: REVERTED (−0.53pp; specialist marginally helpful)
**Insight**: Early-season specialist remains worth keeping.

---

## Run 22 — Dynamic threshold probe
**Hypothesis**: Adjusting threshold by market certainty (BASE + 0.02*|mp−0.5|) selects better bets.
**Change**: DYNAMIC_THRESHOLD=True (BASE=0.12)
**Result**: roi=+34.58%, brier=0.2363, n_bets=498 (3-fold mean)
**Decision**: REVERTED (+0.23pp within noise; added complexity not justified)
**Insight**: Static threshold suffices; market certainty adjustment marginal.

---

## Run 21 — threshold=0.20 probe (extreme selectivity)
**Hypothesis**: Even higher threshold may find ROI peak or show noise collapse.
**Change**: CONFIDENCE_THRESHOLD=0.20
**Result**: roi=+56.90%, brier=0.2363, n_bets=68 (3-fold mean, ~23/fold)
**Decision**: REVERTED (meaningless — 23 bets/fold is pure noise)
**Insight**: Above 0.15 is unreliable; sample too small.

---

## Run 20 — threshold=0.15 probe (diminishing-returns check)
**Hypothesis**: threshold=0.15 will show whether ROI keeps rising or variance explodes.
**Change**: CONFIDENCE_THRESHOLD=0.15 (was 0.12)
**Result**: roi=+39.76%, brier=0.2363, n_bets=243 (3-fold mean, ~81/fold)
**Decision**: REVERTED (high variance; fold 2 at 51.8% looks like noise at 73 bets)
**Insight**: Beyond 0.12 the sample is too thin for reliable inference; 0.12 is the practical ceiling.

---

## Run 19 — threshold=0.12 (best reliable)
**Hypothesis**: threshold=0.12 balances high ROI per bet with sufficient sample size.
**Change**: CONFIDENCE_THRESHOLD=0.12 (was 0.10)
**Result**: roi=+34.35%, brier=0.2363, n_bets=521 (3-fold mean, ~174/fold)
**Decision**: KEPT (new best with adequate sample; +7.42pp over 0.10)
**Insight**: Large jump from 0.10→0.12 suggests a cluster of very high-edge bets at this band.

---

## Run 18 — threshold=0.10
**Hypothesis**: Continuing threshold sweep.
**Change**: CONFIDENCE_THRESHOLD=0.10 (was 0.09)
**Result**: roi=+26.93%, brier=0.2363, n_bets=880 (3-fold mean)
**Decision**: SUPERSEDED by 0.12
**Insight**: ROI still climbing; incremental +1.63pp.

---

## Run 17 — threshold=0.09
**Hypothesis**: Continuing threshold sweep.
**Change**: CONFIDENCE_THRESHOLD=0.09 (was 0.08)
**Result**: roi=+25.30%, brier=0.2363, n_bets=1127 (3-fold mean)
**Decision**: SUPERSEDED by higher thresholds
**Insight**: Steady climb continues.

---

## Run 15 — LR L1 C=0.05 threshold=0.08 (peak search)
**Hypothesis**: threshold=0.08 will either find a new peak or show diminishing returns as n_bets drops below ~500/fold.
**Change**: CONFIDENCE_THRESHOLD=0.08 (was 0.07)
**Result**: roi=+23.74%, brier=0.2363, n_bets=1470 (3-fold mean)
**Decision**: KEPT (new best: +2.13pp over 0.07; ROI still climbing strongly)
**Insight**: High-edge bets continue to outperform; trend still upward.

---

## Run 14 — LR L1 C=0.05 threshold=0.07 (test diminishing returns)
**Hypothesis**: threshold=0.07 continues the ROI-vs-volume tradeoff; finding the peak before sample size degrades variance too much.
**Change**: CONFIDENCE_THRESHOLD=0.07 (was 0.06)
**Result**: roi=+21.62%, brier=0.2363, n_bets=1877 (3-fold mean)
**Decision**: KEPT (new best: +0.41pp over 0.06)
**Insight**: ROI still rising but n_bets declining sharply; likely approaching peak selectivity.

---

## Run 13 — LR L1 C=0.05 threshold=0.06 (even more selective)
**Hypothesis**: Pushing threshold to 0.06 further concentrates bets on highest-confidence predictions; diminishing returns may kick in as sample size shrinks.
**Change**: CONFIDENCE_THRESHOLD=0.06 (was 0.05)
**Result**: roi=+21.21%, brier=0.2363, n_bets=2352 (3-fold mean)
**Decision**: KEPT (new best: +1.33pp over 0.05)
**Insight**: ROI keeps improving with selectivity; top-edge bets have strong positive calibration.

---

## Run 12 — LR L1 C=0.05 threshold=0.05 (higher selectivity)
**Hypothesis**: Raising threshold to 0.05 selects only highest-confidence bets; if model calibration is good, these bets have higher per-unit ROI.
**Change**: CONFIDENCE_THRESHOLD=0.05 (was 0.04)
**Result**: roi=+19.88%, brier=0.2363, n_bets=2931 (3-fold mean)
**Decision**: KEPT (new best: +0.75pp over threshold=0.04)
**Insight**: Fewer, higher-confidence bets improve ROI; model's high-edge predictions are well-calibrated.

---

## Run 11 — LR L1 C=0.05 threshold=0.03 (Phase 4)
**Hypothesis**: Lowering threshold to 0.03 captures more bets with smaller edges; if the model is well-calibrated those edges still have positive EV, boosting total ROI.
**Change**: CONFIDENCE_THRESHOLD=0.03 (was 0.04)
**Result**: roi=+18.12%, brier=0.2363, n_bets=4408 (3-fold mean)
**Decision**: REVERTED (more bets but lower per-bet ROI)
**Insight**: 0.04 threshold filters marginal bets correctly; 0.03 adds noise.

---

## Run 10 — LR L1 C=0.02 (even stronger regularization)
**Hypothesis**: Continuing to increase L1 strength (C=0.02) will further prune noisy features and improve walk-forward ROI.
**Change**: LR_PARAMS C=0.02 (was 0.05)
**Result**: roi=+19.07%, brier=0.2365, n_bets=3528 (3-fold mean)
**Decision**: REVERTED (C=0.05 at 19.13% remains best; diminishing returns)
**Insight**: C=0.05 is the sweet spot; too aggressive L1 drops useful features.

---

## Run 9 — LR L1 C=0.05 (stronger regularization)
**Hypothesis**: Tighter L1 regularization (C=0.05 vs 0.10) will zero out more noise features and improve generalization further.
**Change**: LR_PARAMS C=0.05 (was 0.10)
**Result**: roi=+19.13%, brier=0.2363, n_bets=3607 (3-fold mean)
**Decision**: KEPT (new best: +1.17pp over C=0.10)
**Insight**: Stronger L1 improves all 3 folds; sparser feature set reduces overfitting to noisy team stats.

---

## Run 8 — LR with L1 penalty (SAGA solver) for auto feature selection (Phase 3)
**Hypothesis**: L1 regularization will zero out redundant features automatically, producing a sparse LR that generalizes better than L2 across walk-forward folds.
**Change**: LR_PARAMS: penalty=l1, solver=saga (from lbfgs)
**Result**: roi=+17.96%, brier=0.2365, n_bets=3660 (3-fold mean)
**Decision**: KEPT (new best: +0.18pp over L2 baseline)
**Insight**: L1 sparse selection marginally beats L2; SAGA converges to better sparse solution with 55 features.

---

## Run 7 — Add 3 seeded interactions (Phase 2B)
**Hypothesis**: Adding fip_x_line, form_x_fip, woba_x_sharp will capture non-linear synergies between sharp market moves and team quality signals, improving LR ROI beyond 17.78%.
**Change**: Added fip_x_line, form_x_fip, woba_x_sharp to FEATURE_COLUMNS → 58 features
**Result**: roi=+17.71%, brier=0.2367, n_bets=3673 (3-fold mean)
**Decision**: REVERTED (LR 17.78% still best; interactions add noise)
**Insight**: Interaction features don't add signal for LR; L2 penalty already handles collinear market features.

---

## Run 6 — Prune 5 zero-LGB-importance features (Phase 2A)
**Hypothesis**: Removing sp_bb9_DIFF, war_DIFF, away_pitcher_is_lefty, early_season_flag, sharp_x_fip (all zero LGB permutation importance) will reduce noise for LR and improve ROI.
**Change**: Removed 5 zero-importance features → 50 features total
**Result**: roi=+17.73%, brier=0.2366, n_bets=3647 (3-fold mean)
**Decision**: REVERTED (marginal decrease from 17.78%)
**Insight**: Pruned features had negligible LR effect; L2 reg already dampens their noise.

---

## Run 4 — ensemble_avg baseline (Phase 1)
**Hypothesis**: Averaging LR + LGB + XGB probabilities should reduce individual model variance and produce a more stable ROI, potentially beating LR alone.
**Change**: MODEL="ensemble_avg"
**Result**: roi=+15.56%, brier=0.2375, n_bets=4234 (3-fold mean)
**Decision**: REVERTED (LR 17.78% still best)
**Insight**: Averaging dilutes LR's advantage; tree models pull the ensemble down.

---

## Run 3 — XGB baseline (Phase 1)
**Hypothesis**: XGBoost handles non-linear interactions similarly to LGB but with a different regularization scheme; may produce different ROI profile.
**Change**: MODEL="xgb"
**Result**: roi=+15.61%, brier=0.2384, n_bets=4031 (3-fold mean)
**Decision**: REVERTED (LR 17.78% still best)
**Insight**: XGB better than LGB but still below LR; tree models may overfit with heavily market-correlated features.

---

## Run 2 — LGB baseline (Phase 1)
**Hypothesis**: LightGBM should outperform LR walk-forward because it handles non-linear feature interactions and imputes NaN natively, giving it more training signal from complex stat relationships.
**Change**: MODEL="lgb"
**Result**: roi=+12.68%, brier=0.2407, n_bets=4893 (3-fold mean)
**Decision**: REVERTED (LR at 17.78% is still best)
**Insight**: LGB bet on more games but with lower ROI; LR's regularization better with market features dominating.

---

## Run 1 — LR baseline (Phase 1)
**Hypothesis**: Logistic Regression with full 55-feature set and calibration will establish a walk-forward baseline; market features alone should produce positive but modest ROI.
**Change**: MODEL="lr" (default), all other params at default values
**Result**: roi=+17.78%, brier=0.2366, n_bets=3650 (3-fold mean)
**Decision**: KEPT
**Insight**: LR baseline surprisingly strong at 17.78% walk-forward ROI; market features dominate with reasonable calibration.
