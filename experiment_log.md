# Experiment Log — autoresearch/mlb-mar26

*(MLP single-split era: Runs baseline–Run35 archived in experiment_log_archive.md)*

---

## MLP-Era Distilled Findings (reference only — NOT comparable to walk-forward ROI)

| What worked | What didn't |
|---|---|
| Market anchor loss (lambda=1.0) | lambda ≠ 1.0, focal loss |
| W=15 rolling window | W≤10 or W≥20 |
| W=5 momentum signal | Run-diff momentum, W=3 |
| Feature neutralization (Hubáček) | TabTransformer, wider/deeper MLP |
| threshold=0.04, Kelly=0.25 | threshold ≥ 0.05, Kelly=0.15 |

**MLP stable baseline (run27, seed=42):** ROI=+1.34%, Brier=0.240, 673 bets *(single-split)*

---

## [New Era: Multi-Model Walk-Forward, Full Feature Set]

**Key architecture changes from MLP era:**
- Walk-forward validation (3 folds: val=2022, 2023, 2024). Mean ROI is the only metric.
- Multi-model arena: lgb / xgb / lr / mlp / ensemble_avg / ensemble_stack
- Early season specialist: separate LR for games where either team has < 15 games played
- Full feature set restored: 43 features including all FanGraphs DIFFs, streak, season-to-date, `month`, `days_since_asb`, `wind_dir_sin/cos`, individual `home/away_rest_days`
- `SimpleImputer` in LR/MLP paths — no more aggressive dropna; more rows survive
- Prob capping at [0.25, 0.75] before edge calc; half-Kelly for early-season bets
- `ensemble_stack` uses disagreement feature (stats_prob - market_prob) as extra meta-input
- MLP-era audit finding (low |r| features) treated as a prior, not a conclusion — GBDT permutation importance will re-adjudicate

**Starting feature count:** 43 (FEATURE_COLUMNS) + 11 (EARLY_FEATURE_COLUMNS)

**Run order (Phase 1 — baselines):** lr → lgb → xgb → ensemble_avg → ensemble_stack

---

<!-- Add new runs below this line -->

## Run 1 — LR baseline (Phase 1)
**Hypothesis**: LR (CalibratedClassifierCV) anchors near market probs and should yield near-zero or slightly positive walk-forward ROI; will miss non-linear interactions that GBDTs capture.
**Change**: MODEL="lr" (no other changes; default 43-feature set)