# MLB Research Backlog

Status tags: `[QUEUED]` | `[RETRY]` | `[INVALIDATED]` — do not retry | `[ACTIVE]` — in train.py

---

## 1. Feature Engineering

### Already in train.py
- `[ACTIVE]` `month`, `days_since_asb` — calendar context
- `[ACTIVE]` `streak_DIFF`, `season_win_pct_DIFF`, `season_run_diff_avg_DIFF`
- `[ACTIVE]` `run_diff_std_W_DIFF`, `runs_allowed_avg_W_DIFF`
- `[ACTIVE]` `sp_whip_DIFF`, `rolling_era_DIFF`, `rolling_whip_DIFF`, `rookie_DIFF`
- `[ACTIVE]` Full FanGraphs batting set (avg/obp/slg/woba/wrc_plus/k_pct/bb_pct/k_per_9/bb_per_9/hr_per_9/era/fip/owar)
- `[ACTIVE]` `wind_dir_sin`, `wind_dir_cos`, `home_rest_days`, `away_rest_days`
- `[ACTIVE]` Early season specialist (EARLY_CUTOFF=25) — REQUIRED; disabling = 27% ROI (Run 215)

### Seeded interactions — all tried, all INVALIDATED at C=0.04
- `[INVALIDATED]` `fip_x_line` — Run 218: L1 zeroed
- `[INVALIDATED]` `form_x_fip` — Run 214: L1 zeroed
- `[INVALIDATED]` `temp_x_wrc` — Run 217: adds noise
- `[INVALIDATED]` `woba_x_sharp` — Run 220: slightly worse
- `[INVALIDATED]` `early_x_fip` — Run 221: L1 zeroed
- `[INVALIDATED]` `market_x_sp_era` — Run 238: L1 zeroed
- `[INVALIDATED]` `abs(line_move_delta)` — Run 213: adds noise

### Feature removal — all tried, all INVALIDATED
- `[INVALIDATED]` Remove `war_DIFF` — Run 233: L1 already zeroed it; identical ROI
- `[INVALIDATED]` Remove `era_DIFF` — Run 236: carries independent signal; ROI drops
- `[INVALIDATED]` Remove `days_since_asb` — Run 232: carries independent signal
- `[INVALIDATED]` Remove `run_diff_std_W_DIFF` — Run 203: contributes; worse
- `[INVALIDATED]` Remove `open_home_implied` — Run 210: carries independent signal
- `[INVALIDATED]` Remove `total_move_delta` — Run 208: correlates with game environment

### Additional feature candidates
- `[INVALIDATED]` **Road trip duration** — Run 241: L1 zeroed; rest_days_DIFF already captures inter-game recovery
- `[INVALIDATED]` **Opener flag** — requires `sp_gs_lag1` in CSV; not available in 69-col dataset
- `[INVALIDATED]` **Park-factor interactions** — redundant with existing park_x_pythagorean

### New candidates tried 2026-03-28
- `[INVALIDATED]` **LR batting multicollinearity / iso_DIFF** — Run 211: removing slash-line diffs loses fidelity
- `[INVALIDATED]` **Pythagorean exponent 1.83** — Run 212: negligible feature value change
- `[INVALIDATED]` **Cold-start SP shrinkage** — Run 202: NaN rate too low to matter
- `[INVALIDATED]` **year_norm** — Run 240: hurts fold 4
- `[INVALIDATED]` **is_series_opener** — Run 237: adds noise
- `[INVALIDATED]` **`sp_rest_DIFF` ordering audit** — data integrity check only; not an experiment

---

## 2. Market Decorrelation

- `[INVALIDATED]` `ensemble_stack` — Run 234: GBDTs drag down meta-LR; 23.90% ROI
- `[INVALIDATED]` Residual MLP (run1), sharp-weighted loss (run20)
- `[INVALIDATED]` **Two-stage LR** — Run 245: roi=38.02% (-6.8pp). L1 joint optimization superior; forcing market/stats separation destroys signal.

---

## 3. Architectural Experiments

- `[ACTIVE]` `MODEL="lr"` — current best (44.85%, Run 145)
- `[INVALIDATED]` `MODEL="lgb"`, `"xgb"`, `"ensemble_stack"`, `"ensemble_avg"` — all < 35% ROI; GBDT dead end
- `[INVALIDATED]` `MODEL="ensemble_avg"` — excluded (LGB+XGB drag it down)
- `[INVALIDATED]` `MODEL="mlp"` — Run 242: roi=5.40%, brier=0.38; catastrophic overfitting in walk-forward

### LR tuning — exhausted
- `[INVALIDATED]` C=0.03 — Run 235: 44.09%
- `[INVALIDATED]` C=0.04 — ✓ CURRENT BEST
- `[INVALIDATED]` C=0.05 — Run 219: 42.51%
- `[INVALIDATED]` L2 penalty — Run 239: 38.32%; L1 sparsity essential
- `[INVALIDATED]` CalibratedClassifierCV cv=5 — Run 227: 39.40%
- `[INVALIDATED]` CalibratedClassifierCV cv=2 — (cv=3 confirmed optimal)
- `[INVALIDATED]` Remove outer calibration — Run 226: 39.73%
- `[INVALIDATED]` Calibrate early specialist — Run 231: 43.77%

---

## 4. Betting Logic & Calibration — exhausted

- `[INVALIDATED]` `CALIBRATE=False` — N/A for LR (hardcoded calibration); skip
- `[INVALIDATED]` `EARLY_CUTOFF=None` — Run 215: catastrophic
- `[INVALIDATED]` `EARLY_CUTOFF=20` — Run 229: 40.22%
- `[INVALIDATED]` `EARLY_CUTOFF=30` — Run 230: 41.97%; EARLY_CUTOFF=25 confirmed optimal
- `[INVALIDATED]` `CONFIDENCE_THRESHOLD=0.13` — Run 224: 38.70%
- `[INVALIDATED]` `CONFIDENCE_THRESHOLD=0.15` — Run 225: 37.10%; 0.14 confirmed optimal
- `[INVALIDATED]` `KELLY_FRACTION` variants — Run 216: ROI%-invariant; never test Kelly
- `[INVALIDATED]` `DYNAMIC_THRESHOLD=True` — Run 228: 43.79%
- `[INVALIDATED]` `PROB_CAP=(0.32, 0.68)` — Run 222: 43.51%
- `[INVALIDATED]` `PROB_CAP=(0.36, 0.64)` — Run 223: 41.33%; (0.34,0.66) confirmed optimal

---

## 5. Walk-Forward Configuration — exhausted

- `[ACTIVE]` 4-fold (2022/2023/2024/2025) — current
- `[INVALIDATED]` Half-season folds — Run 201: not comparable
- `[INVALIDATED]` Truncating fold4 — Run 205: data snooping

---

## Remaining genuinely QUEUED experiments
All backlog items exhausted as of 2026-03-28. Every candidate feature, architecture, and betting-logic variant has been tried and invalidated. Current best: Run 145 at 44.85% ROI (4-fold walk-forward, L1-LR, C=0.04).
