# MLB Research Backlog

Status tags: `[QUEUED]` | `[RETRY]` — tried under MLP, retest under GBDT walk-forward | `[INVALIDATED]` — do not retry without new rationale

---

## 1. Feature Engineering

### Already in train.py (new additions vs MLP era)
- `[ACTIVE]` `month`, `days_since_asb` — calendar context
- `[ACTIVE]` `streak_DIFF` — win/loss streak signal
- `[ACTIVE]` `season_win_pct_DIFF`, `season_run_diff_avg_DIFF` — expanding season mean
- `[ACTIVE]` `run_diff_std_W_DIFF` — team consistency (variance of run margin)
- `[ACTIVE]` `runs_allowed_avg_W_DIFF` — pitching side of rolling form
- `[ACTIVE]` `sp_whip_DIFF`, `rolling_era_DIFF`, `rolling_whip_DIFF` — more pitcher signals
- `[ACTIVE]` `rookie_DIFF` — pitcher experience flag
- `[ACTIVE]` Full FanGraphs batting set: `avg_DIFF`, `obp_DIFF`, `slg_DIFF`, `k_pct_DIFF`, `bb_pct_DIFF`, `k_per_9_DIFF`, `bb_per_9_DIFF`, `hr_per_9_DIFF`, `era_DIFF`, `fip_DIFF`, `owar_DIFF`
- `[ACTIVE]` `wind_dir_sin`, `wind_dir_cos` — circular wind encoding replacing raw degrees
- `[ACTIVE]` `home_rest_days`, `away_rest_days` — individual rest signals (not just diff)
- `[ACTIVE]` Early season specialist (EARLY_CUTOFF=15, separate LR)

### Seeded in engineer_new_features() — screen before adding to FEATURE_COLUMNS
- `[QUEUED]` `fip_x_line` — sharp money + FIP edge synergy
- `[QUEUED]` `form_x_fip` — hot team + good SP compounding
- `[QUEUED]` `temp_x_wrc` — heat amplifies batting quality gap
- `[QUEUED]` `woba_x_sharp` — sharp money + batting edge synergy
- `[QUEUED]` `early_x_fip` — FIP reliability drops early in season

### Additional candidates
- `[QUEUED]` **Opener flag** (`is_opener`): 1 if SP career_starts/appearances ratio suggests opener. Requires `sp_gs_lag1` in CSV.
- `[QUEUED]` **Road trip duration**: Consecutive away games counter for away team.
- `[QUEUED]` **abs(line_move_delta)**: Market volatility independent of direction.
- `[RETRY]` **Park-factor interactions**: MLP-era collinearity (r>0.99, park_factor std≈0.04). GBDTs may extract non-linear signal — check LGB importance first.

---

## 2. Market Decorrelation

- `[ACTIVE]` `ensemble_stack` includes disagreement (stats_prob − market_prob) as meta-feature — cleaner version of the notebook's two-stage model.
- `[INVALIDATED]` Residual MLP (run1), sharp-weighted loss (run20).
- `[QUEUED]` **Two-stage LR only**: Train LR without `market_implied_prob`, blend with market in meta-LR. Test via `ensemble_stack` with only LR as base model.

---

## 3. Architectural Experiments

- `[QUEUED]` `MODEL="lr"` — Phase 1 baseline
- `[QUEUED]` `MODEL="lgb"` — Phase 1 baseline
- `[QUEUED]` `MODEL="xgb"` — Phase 1 baseline
- `[QUEUED]` `MODEL="ensemble_avg"` — Phase 1 baseline
- `[QUEUED]` `MODEL="ensemble_stack"` — Phase 1 baseline
- `[RETRY]` `MODEL="mlp"` — lowest priority; MLP-era history suggests GBDTs dominate
- `[INVALIDATED]` TabTransformer (run23), focal loss (run21), wider MLP (run7)

### GBDT tuning (Phase 3)
- `[QUEUED]` LGB: try `num_leaves=15`, halve `learning_rate` with more `n_estimators`
- `[QUEUED]` XGB: try `max_depth=3`
- `[QUEUED]` LR: try `penalty="l1"` (SAGA solver) for automatic feature selection

---

## 4. Betting Logic & Calibration

- `[QUEUED]` `CALIBRATE=False` — test whether isotonic helps or hurts each model
- `[QUEUED]` `EARLY_CUTOFF=None` — test disabling specialist (all games → regular model)
- `[QUEUED]` `CONFIDENCE_THRESHOLD` in {0.02, 0.03, 0.05} — MLP best was 0.04
- `[QUEUED]` `KELLY_FRACTION` in {0.15, 0.20} — MLP best was 0.25
- `[QUEUED]` **Dynamic threshold**: `BASE + 0.02 * abs(market_implied_prob - 0.5)`
- `[INVALIDATED]` Kelly=0.15 (run26), threshold=0.05 (run12)

---

## 5. Walk-Forward Configuration

- `[QUEUED]` 4th fold: `("2025-01-01", "2025-01-01", "2026-01-01")` if 2025 data in CSV
- `[QUEUED]` Half-season folds if fold-to-fold ROI variance > 10pp