# autoresearch: MLB Kalshi Betting Edition
An autonomous experiment loop to maximize walk-forward ROI on MLB home/away Kalshi contracts.

---

## Setup
1. **Agree on a run tag** based on today's date (e.g., `mlb-mar26`). If the branch exists, check it out.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from master.
3. **Read context** (in this order):
   - `experiment_log.md` + `experiment_log_archive.md` — history; read before every run
   - `train.py` — the only file you modify
   - `research_backlog.md` — what to try next
   - `feature_engineering.md` — screening rules for new features
   - `data_dictionary.md` — directional definitions for all features
   - `scripts/checkpoint_best.sh` — best-run auto commit/push logic
4. **Initialize `results.tsv`** with header: `commit  val_roi  val_brier  status  description`
5. **Initialize `experiment_log.md`** if empty.
6. **Ensure checkpoint script is executable:** `chmod +x scripts/checkpoint_best.sh`

---

## The Model

`train.py` is a **multi-model, walk-forward system with an early-season specialist**. Only modify the designated sections.

### What you can change

| Section | Variable / Function | What to do |
|---|---|---|
| Model selection | `MODEL` | `"lgb"`, `"xgb"`, `"lr"`, `"mlp"`, `"ensemble_avg"`, `"ensemble_stack"` |
| Regular features | `FEATURE_COLUMNS` | Add/remove names. Only names present in the DataFrame are used. |
| Early specialist features | `EARLY_FEATURE_COLUMNS` | Features for the early-season LR (< EARLY_CUTOFF games). Keep to pitcher + market + context. |
| New feature engineering | `engineer_new_features(df)` | Add computed columns here. Document in `data_dictionary.md` Section 9. |
| Model hyperparameters | `LGB_PARAMS`, `XGB_PARAMS`, `LR_PARAMS`, `MLP_*` | Tune per-model. |
| Rolling windows | `BEST_W`, `MOMENTUM_W` | Change window lengths. |
| Early specialist threshold | `EARLY_CUTOFF` | Games-played cutoff; set to `None` to disable the specialist entirely. |
| Betting logic | `CONFIDENCE_THRESHOLD`, `KELLY_FRACTION`, `MAX_BET_FRAC`, `PROB_CAP` | Betting parameters. |
| Walk-forward folds | `WALK_FORWARD_FOLDS` | Add/remove folds or shift dates. |
| Calibration | `CALIBRATE` | Toggle isotonic post-hoc calibration for GBDT/MLP. |

### What you must NOT change
- `evaluate()` — immutable ground truth
- `run_walk_forward()` — immutable engine (fold *values* in `WALK_FORWARD_FOLDS` are fine to change)
- `master_mlb.csv`

---

## The Thinker Protocol
Before every experiment:
1. Read `experiment_log.md` — identify failed patterns. Do not repeat them.
2. Read `research_backlog.md` — check for Phase 1 items first.
3. Write a **falsifiable hypothesis** in `experiment_log.md` **before** editing `train.py`.
   > *"LightGBM should outperform LR walk-forward because it handles non-linear FG stat interactions and imputes NaN natively, giving it more training rows."*

---

## The Experiment Loop

```
1. Hypothesize  →  Thinker Protocol
2. Edit         →  Modify train.py
3. Run          →  uv run train.py > run.log 2>&1
4. Compare      →  uv run check_improvement.py
5. Log          →  Append to experiment_log.md (BEFORE any git action)
6. Checkpoint (mandatory):
     scripts/checkpoint_best.sh "[short description of this run]"
     - ROI improved   → script commits + pushes current branch
     - ROI same/worse → script exits without commit/push
7. Repeat
```

**Never stop.**

---

## Model Strategy Guide

### Phase 1 — Model Baselines (run once each, default 43-feature set)
Establish walk-forward ROI for each model. Run in order:

`lr` → `lgb` → `xgb` → `ensemble_avg` → `ensemble_stack`

The best single result is the **floor** for all future experiments.

Important: MLP-era single-split ROI (~+1% to +10%) is NOT comparable to walk-forward mean ROI. Expect the scale to differ. Treat all MLP-era numbers as prior intuitions only.

### Phase 2 — Feature Engineering
Once baselines are set:

**A. Read LGB feature importances** (auto-printed on last fold). Cross-reference with `feature_engineering.md` Phase 1 screening. The 43-feature set is a fresh slate — let GBDT permutation importance drive pruning decisions, not MLP-era Pearson correlations.

**B. Add seeded interactions** from `engineer_new_features()`. Each candidate must pass `feature_engineering.md` Phase 1 screening (|r| > 0.003, redundancy < 0.90) before a training run.

**C. Rolling window tuning** — MLP-era established W=15 as optimal, but that was under a single split. Re-validate under walk-forward: try W in {10, 15, 20}.

**D. Calendar features** — `month` and `days_since_asb` are new. If LGB importance shows them near-zero, prune them.

**E. Early specialist tuning** — try `EARLY_CUTOFF` in {10, 15, 20} or `None`. A smaller cutoff means fewer games go to the specialist (which may be undertrained).

### Phase 3 — Architecture & Hyperparameters
If feature engineering stalls:
- Tune LGB: `num_leaves=15`, halve `learning_rate`
- Tune XGB: `max_depth=3`
- Try `ensemble_stack` — the disagreement meta-feature (stats_prob − market_prob) is the key signal from the notebook's two-stage model
- `MODEL="mlp"` lowest priority

### Phase 4 — Calibration & Betting Logic
- Toggle `CALIBRATE` — LR already calibrates via CalibratedClassifierCV; toggling affects GBDT/MLP
- Try `CONFIDENCE_THRESHOLD` in {0.02, 0.03, 0.05}
- Try `EARLY_CUTOFF=None` to test if the specialist helps or hurts overall ROI
- Try dynamic threshold (see `research_backlog.md`)

### Phase 5 — Walk-Forward Tuning
- Add 2025 fold if data available
- Shrink to half-season folds if fold variance > 10pp

---

## The Plateau Rule
5 consecutive runs without ROI improvement:
1. Log "PLATEAU REACHED" in `experiment_log.md`
2. Move to the next phase
3. If all phases exhausted: reset `FEATURE_COLUMNS` to 43-feature set and try the next untested model

---

## experiment_log.md Format
```markdown
## Run [N] — [short description]
**Hypothesis**: one sentence
**Change**: what was modified
**Result**: roi=X, brier=Y, n_bets=Z (N-fold mean)
**Decision**: KEPT / REVERTED
**Insight**: one sentence
```

---

## Key Reminders
- **Walk-forward mean ROI is the only valid metric.** Single-fold ROI is not trustworthy.
- High ROI + high Brier → check for data leakage immediately.
- Low Brier + negative ROI → calibrated market parrot; raise threshold or reduce market features.
- GBDTs handle NaN natively — do not pre-impute before `build_lgb()` or `build_xgb()`.
- LR and MLP have `SimpleImputer` built in — do not pre-impute for those either.
- `engineer_new_features()` runs after all lag/rolling/FanGraphs features — the full lagged dataset is available.
- The early specialist uses `EARLY_FEATURE_COLUMNS`, not `FEATURE_COLUMNS`. Features only in the early set don't need to be in the regular set.
- MLP-era history is in `experiment_log_archive.md`. Read it before claiming something hasn't been tried.
- Never use `git reset --hard HEAD~1` in this workflow.
