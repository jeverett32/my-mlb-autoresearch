# Experiment Log — autoresearch/mlb-mar27

## MLP-Era Distilled Findings (reference only — NOT comparable to walk-forward ROI)

| What worked | What didn't |
|---|---|
| Market anchor loss (lambda=1.0) | lambda ≠ 1.0, focal loss |
| W=15 rolling window | W≤10 or W≥20 |
| W=5 momentum signal | Run-diff momentum, W=3 |
| Feature neutralization (Hubáček) | TabTransformer, wider/deeper MLP |
| threshold=0.04, Kelly=0.25 | threshold ≥ 0.05, Kelly=0.15 |

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
