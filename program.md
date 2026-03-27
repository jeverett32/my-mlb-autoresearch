# autoresearch: MLB Kalshi Betting Edition
An autonomous experiment loop to maximize walk-forward ROI on MLB home/away Kalshi contracts.

---

## AUTONOMY RULE — Read This First
**Never stop. Never ask the user for permission, direction, or confirmation.** Every decision point has a deterministic rule below. If something is ambiguous, pick the most conservative option and keep going. The loop runs until the user manually stops it.

---

## Setup

### Step 1 — Branch
Pick a run tag from today's date (e.g., `mlb-mar27`). Then:
```bash
git checkout autoresearch/mlb-<tag> 2>/dev/null || git checkout -b autoresearch/mlb-<tag> master
```
- Branch **already existed** → check it out and skip to Step 3. Do not re-initialize.
- Branch **did not exist** → created from master. Continue to Step 2.

### Step 2 — Initialize files (new branch only)
`results.tsv` is created automatically by `train.py` if missing — no action needed.

Initialize `experiment_log.md` with exactly this content (replace `<tag>` with the run tag):
```markdown
# Experiment Log — autoresearch/mlb-<tag>

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
```

Initialize `experiment_log_archive.md` with exactly this content:
```markdown
# Experiment Log Archive — autoresearch/mlb-<tag>

<!-- Oldest runs moved here when experiment_log.md exceeds 15 entries -->
```

### Step 3 — Prepare
```bash
chmod +x scripts/*.sh
```

### Step 4 — Read context
- `experiment_log.md` — current run history
- `research_backlog.md` — what to try next (check phase and QUEUED items)
- Read `feature_engineering.md` and `data_dictionary.md` only when adding new features
- Read `experiment_log_archive.md` only to verify whether a specific approach was tried before the current log window

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
1. Run `scripts/get_baseline.sh` — check current best ROI and plateau count.
2. Read `experiment_log.md` — identify failed patterns. Do not repeat them. Only read `experiment_log_archive.md` if you need to verify whether a specific approach was tried before the current log window.
3. Read `research_backlog.md` — pick the top QUEUED item for the current phase.
4. Write a **falsifiable hypothesis** in `experiment_log.md` **before** editing `train.py`.
   > *"LightGBM should outperform LR walk-forward because it handles non-linear FG stat interactions and imputes NaN natively, giving it more training rows."*

---

## The Experiment Loop

```
1. Hypothesize  →  Thinker Protocol (above)
2. Inspect      →  scripts/get_features.sh + scripts/get_params.sh
3. Edit         →  Modify train.py only
4. Run          →  uv run train.py > run.log 2>&1
5. Inspect      →  scripts/get_results.sh  (metrics)
                   scripts/get_importances.sh  (if lgb/xgb)
6. Log          →  Append to experiment_log.md (BEFORE any git action)
7. Decide       →  See Decision Tree below
8. Checkpoint   →  scripts/checkpoint_best.sh "[short description]"
9. Repeat       →  Never stop
```

### Decision Tree (step 7)

```
Did the run crash?
  YES → scripts/get_results.sh to find error; fix train.py; re-run. Do NOT log a crashed run.
  NO  → continue

Did val_roi improve vs. previous best?
  YES → Decision: KEPT. Run checkpoint. Continue loop.
  NO  → Decision: REVERTED. Run: scripts/revert_train.sh
        Run scripts/get_baseline.sh — plateau count is computed automatically from results.tsv.
        If plateau count >= 5: log "PLATEAU REACHED", advance to next phase (see Phase Guide).
        Continue loop with next hypothesis.
```

**Never ask for permission at any step of this tree.**

---

## Information Scripts (use these — never read raw log/csv files)

| Need | Command |
|---|---|
| Current metrics from last run | `scripts/get_results.sh` |
| Best ROI, latest row, plateau count | `scripts/get_baseline.sh` |
| Current feature lists | `scripts/get_features.sh` |
| Current tunable params | `scripts/get_params.sh` |
| LGB/XGB feature importances | `scripts/get_importances.sh` |
| Revert train.py after failed run | `scripts/revert_train.sh` |
| Commit/push if new best | `scripts/checkpoint_best.sh "[desc]"` |

---

## Model Strategy Guide

### Phase 1 — Model Baselines (run once each, default 43-feature set)
Establish walk-forward ROI for each model. Run in order:

`lr` → `lgb` → `xgb` → `ensemble_avg` → `ensemble_stack`

The best single result is the **floor** for all future experiments.

Important: MLP-era single-split ROI (~+1% to +10%) is NOT comparable to walk-forward mean ROI. Expect the scale to differ. Treat all MLP-era numbers as prior intuitions only.

### Phase 2 — Feature Engineering
Once baselines are set:

**A. Read LGB feature importances** via `scripts/get_importances.sh`. Cross-reference with `feature_engineering.md` Phase 1 screening. The 43-feature set is a fresh slate — let GBDT permutation importance drive pruning decisions, not MLP-era Pearson correlations.

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
1. Log "PLATEAU REACHED — moving to Phase N" in `experiment_log.md`
2. Advance to next phase automatically. Do NOT wait for user input.
3. If all phases exhausted: reset `FEATURE_COLUMNS` to 43-feature set and try the next untested model from Phase 1.

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
- MLP-era history is in `experiment_log_archive.md`. Only read it to verify a specific approach — not on every run.
- Never use `git reset --hard HEAD~1` in this workflow.
