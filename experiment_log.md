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

## Run 1 — LR baseline (Phase 1)
**Hypothesis**: Logistic Regression with full 55-feature set and calibration will establish a walk-forward baseline; market features alone should produce positive but modest ROI.
**Change**: MODEL="lr" (default), all other params at default values
**Result**: roi=+17.78%, brier=0.2366, n_bets=3650 (3-fold mean)
**Decision**: KEPT
**Insight**: LR baseline surprisingly strong at 17.78% walk-forward ROI; market features dominate with reasonable calibration.
