"""
MLB betting model training script. Single-file, modifiable.
Usage: uv run train.py
"""

import os
import math
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Hyperparameters (edit these for experiments)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300          # training time budget in seconds
CSV_INPUT_PATH = "master_mlb.csv"

# Feature engineering
BEST_W = 15              # rolling window for win%, run-diff, runs-scored averages
EARLY_SEASON_GAMES = 15  # games before this index are flagged as early-season

# Model architecture
HIDDEN_DIMS = (256, 128, 64)
DROPOUT = 0.2

# Optimization
LR = 1e-3
WEIGHT_DECAY = 3e-4
BATCH_SIZE = 512
WARMUP_RATIO = 0.05
WARMDOWN_RATIO = 0.4
FINAL_LR_FRAC = 0.01

# Betting
CONFIDENCE_THRESHOLD = 0.04   # minimum edge (model_prob - market_implied_prob) to place a bet
KELLY_FRACTION = 0.25          # fractional Kelly multiplier

# Market anchoring: pulls model logits toward market logit during training
# Prevents model from drifting far from market; forces learning of residual edge
MARKET_ANCHOR_LAMBDA = 1.0

# ---------------------------------------------------------------------------
# Feature Engineering (ported from feature_engineering.ipynb)
# ---------------------------------------------------------------------------

# 2022–2024 park factors (100 = league average; stored as value/100)
PARK_FACTORS = {
    "ARI": 101, "ATL": 100, "ATH": 97,  "BAL": 99,  "BOS": 107,
    "CHC": 97,  "CHW": 99,  "CIN": 105, "CLE": 97,  "COL": 112,
    "DET": 98,  "HOU": 100, "KCR": 104, "LAA": 100, "LAD": 100,
    "MIA": 101, "MIL": 97,  "MIN": 102, "NYM": 97,  "NYY": 100,
    "PHI": 101, "PIT": 101, "SDP": 96,  "SEA": 91,  "SFG": 97,
    "STL": 100, "TBR": 96,  "TEX": 101, "TOR": 100, "WSN": 101,
}

DROP_SUBSTR = {"game_pk", "odds_source", "starter_id"}

FEATURE_COLUMNS = [
    # --- Market (Closing & Opening Prices) ---
    "market_implied_prob",      # Closing home win probability (de-vigged). 0.5 = even money.
    "open_home_implied",        # Opening home win probability. Baseline market expectation.
    "line_move_delta",          # (Market - Open). Positive means the market moved toward the Home team.
    "sharp_move_flag",          # Binary (1 if |line_move| > 0.03). Indicates significant market conviction.
    "total_move_delta",         # Change in Over/Under total. Can signal weather or lineup shifts.

    # --- Pitcher Quality (DIFF = Home - Away, or similar) ---
    "sp_fip_DIFF",              # (Away FIP - Home FIP). Positive favors Home (lower FIP is better).
    "sp_era_DIFF",              # (Away ERA - Home ERA). Positive favors Home (lower ERA is better).
    "sp_k9_DIFF",               # (Home K/9 - Away K/9). Positive favors Home (more strikeouts).
    "sp_bb9_DIFF",              # (Away BB/9 - Home BB/9). Positive favors Home (fewer walks).
    "rolling_k9_DIFF",          # Difference in K/9 over the last few starts (recent form).

    # --- Batting Performance ---
    "wrc_plus_DIFF",            # (Home wRC+ - Away wRC+). Standardized offensive power (100 = Avg).
    "woba_DIFF",                # (Home wOBA - Away wOBA). Overall offensive contribution.

    # --- Handedness Logic ---
    "pitcher_handedness_diff",  # Categorical delta between SP handedness.
    "home_pitcher_is_lefty",    # Binary flag (1 = Left, 0 = Right).
    "away_pitcher_is_lefty",    # Binary flag (1 = Left, 0 = Right).

    # --- Rolling Form (W-game window) ---
    "win_pct_W_DIFF",           # (Home Win% - Away Win%) over window W.
    "run_diff_avg_W_DIFF",      # (Home Run Diff - Away Run Diff) over window W.
    "runs_scored_avg_W_DIFF",   # (Home Avg Runs - Away Avg Runs) over window W.

    # --- Weather & Venue ---
    "temp_c",                   # Temperature in Celsius. High temp = more home runs/offense.
    "wind_speed_kmh",           # Wind speed. High wind can be volatile depending on direction.
    "park_factor",              # Venue-specific scoring index (e.g., 1.12 for Coors, 0.91 for Seattle).
    "is_night_game",            # Binary flag. Night games often have lower run environments.

    # --- Context & Fatigue ---
    "rest_days_DIFF",           # (Home Rest - Away Rest). Positive = Home is better rested.
    "sp_rest_DIFF",             # (Home SP Rest - Away SP Rest). Specifically for the starting pitchers.
    "is_series_finale",         # Binary flag. Often associated with "getaway" lineups or different motivation.
    "early_season_flag",        # Binary flag (first 15 games). Used to weight historical vs. current season stats.

    # --- Interactions & Splits ---
    "sharp_x_fip",              # Interaction: sharp_move_flag * sp_fip_DIFF. Signal synergy.
    "home_road_split_DIFF",     # (Home Team Win% at Home - Away Team Win% on Road).
]


def _gcol(df, name):
    """Get a DataFrame column by name, or a NaN Series if missing."""
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def american_to_raw_implied(ml):
    if ml is None:
        return np.nan
    try:
        if pd.isna(ml):
            return np.nan
    except TypeError:
        pass
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return np.nan
    return abs(ml) / (abs(ml) + 100.0) if ml < 0 else 100.0 / (ml + 100.0)


def american_to_decimal(ml):
    if ml is None:
        return np.nan
    try:
        if pd.isna(ml):
            return np.nan
    except TypeError:
        pass
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return np.nan
    return 1.0 + 100.0 / abs(ml) if ml < 0 else 1.0 + ml / 100.0


def _devig_home(p_h_raw, p_a_raw):
    s = p_h_raw + p_a_raw
    out_h = p_h_raw / s
    bad = (s <= 0) | pd.isna(s)
    return out_h.where(~bad, np.nan)


def add_market_columns(df):
    df = df.copy()
    oh_ml = pd.to_numeric(df["open_home_ml"], errors="coerce")
    oa_ml = pd.to_numeric(df["open_away_ml"], errors="coerce")
    ch_ml = pd.to_numeric(df["close_home_ml"], errors="coerce")
    ca_ml = pd.to_numeric(df["close_away_ml"], errors="coerce")

    oh = oh_ml.map(american_to_raw_implied)
    oa = oa_ml.map(american_to_raw_implied)
    ch = ch_ml.map(american_to_raw_implied)
    ca = ca_ml.map(american_to_raw_implied)

    # Fall back to close when open is missing
    oh = oh.where(oh.notna(), ch)
    oa = oa.where(oa.notna(), ca)

    df["open_home_implied"] = _devig_home(oh, oa)
    df["market_implied_prob"] = _devig_home(ch, ca)
    df["line_move_delta"] = df["market_implied_prob"] - df["open_home_implied"]
    df["sharp_move_flag"] = (df["line_move_delta"].abs() >= 0.03).astype(float)

    ot = pd.to_numeric(df["open_total"], errors="coerce")
    ct = pd.to_numeric(df["close_total"], errors="coerce")
    df["total_move_delta"] = (ct - ot).fillna(0.0)
    df["dec_close_home"] = ch_ml.map(american_to_decimal)
    return df


def add_schedule_context(df):
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    df["park_factor"] = df["home_team"].map(
        lambda t: float(PARK_FACTORS.get(t, 100)) / 100.0
    )

    deg = np.deg2rad(pd.to_numeric(df["wind_dir_deg"], errors="coerce").fillna(0))
    df["wind_dir_sin"] = np.sin(deg)
    df["wind_dir_cos"] = np.cos(deg)

    # Rest days per team per season (via long format)
    rows = []
    for _, r in df.iterrows():
        for side, team in [("home", r["home_team"]), ("away", r["away_team"])]:
            rows.append({"game_id": r["game_id"], "game_date": r["game_date"],
                         "season": r["season"], "team": team, "side": side})
    g = pd.DataFrame(rows).sort_values(["team", "season", "game_date"])
    g["prev_date"] = g.groupby(["team", "season"], sort=False)["game_date"].shift(1)
    g["rest_days"] = (g["game_date"] - g["prev_date"]).dt.days.fillna(4).clip(lower=0)

    rh = g[g["side"] == "home"][["game_id", "rest_days"]].rename(columns={"rest_days": "home_rest"})
    ra = g[g["side"] == "away"][["game_id", "rest_days"]].rename(columns={"rest_days": "away_rest"})
    df = df.merge(rh, on="game_id", how="left").merge(ra, on="game_id", how="left")
    df["rest_days_DIFF"] = df["home_rest"] - df["away_rest"]
    df["sp_rest_DIFF"] = df["rest_days_DIFF"]  # proxy (no SP-specific rest in CSV)

    # Series finale flag
    tmp = df.sort_values(["home_team", "season", "game_date"]).copy()
    nxt = tmp.groupby(["home_team", "season"])["away_team"].shift(-1)
    tmp["is_series_finale"] = ((nxt != tmp["away_team"]) | nxt.isna()).astype(float)
    df = df.drop(columns=["is_series_finale"], errors="ignore")
    df = df.merge(tmp[["game_id", "is_series_finale"]], on="game_id", how="left")
    return df


def shift_team_perf(df):
    """Lag all per-game team stats by 1 (prevent lookahead)."""
    base_ids = ["game_id", "game_date", "season", "home_team", "away_team",
                "home_win", "home_score", "away_score"]
    home_cols = [
        c for c in df.columns
        if c.startswith("home_")
        and c not in {"home_team", "home_win", "home_score", "home_starter_id", "home_pitcher_is_lefty"}
        and not any(s in c for s in DROP_SUBSTR)
    ]
    away_cols = [
        c for c in df.columns
        if c.startswith("away_")
        and c not in {"away_team", "away_score", "away_starter_id", "away_pitcher_is_lefty"}
        and not any(s in c for s in DROP_SUBSTR)
    ]

    h = df[base_ids + ["home_pitcher_is_lefty"] + home_cols].copy()
    h = h.rename(columns={c: c.replace("home_", "", 1) for c in home_cols})
    h["team"], h["opp"], h["is_home"] = df["home_team"], df["away_team"], 1.0
    h["win"] = df["home_win"].astype(float)
    h["runs_for"] = df["home_score"].astype(float)
    h["runs_against"] = df["away_score"].astype(float)

    a = df[base_ids + ["away_pitcher_is_lefty"] + away_cols].copy()
    a = a.rename(columns={c: c.replace("away_", "", 1) for c in away_cols})
    a["team"], a["opp"], a["is_home"] = df["away_team"], df["home_team"], 0.0
    a["win"] = 1.0 - df["home_win"].astype(float)
    a["runs_for"] = df["away_score"].astype(float)
    a["runs_against"] = df["home_score"].astype(float)

    long = pd.concat([h, a], ignore_index=True)
    long = long.sort_values(["team", "season", "game_date", "game_id"])

    exclude = set(base_ids) | {"team", "opp", "is_home", "win", "runs_for", "runs_against"}
    stat_cols = [c for c in long.columns if c not in exclude]
    for c in stat_cols:
        long[c] = pd.to_numeric(long[c], errors="coerce")
    for c in stat_cols:
        long[c + "_lag1"] = long.groupby(["team", "season"], sort=False)[c].shift(1)

    lag_cols = [c for c in long.columns if c.endswith("_lag1")]
    long = long.drop(columns=stat_cols, errors="ignore")

    hm = long[long["is_home"] == 1][["game_id"] + lag_cols].copy()
    hm.columns = ["game_id"] + ["h_" + c for c in lag_cols]
    aw = long[long["is_home"] == 0][["game_id"] + lag_cols].copy()
    aw.columns = ["game_id"] + ["a_" + c for c in lag_cols]

    out = df.merge(hm, on="game_id", how="left").merge(aw, on="game_id", how="left")
    return out, long


def add_rolling_form(long_games, W):
    """Add W-game rolling win%, run-diff avg, runs-scored avg, and home/road-specific win%."""
    g = long_games.sort_values(["team", "season", "game_date", "game_id"]).copy()

    def rmean(ser, window):
        return ser.shift(1).rolling(window, min_periods=max(2, window // 3)).mean()

    g[f"win_pct_{W}"] = g.groupby(["team", "season"], sort=False)["win"].transform(
        lambda s: rmean(s, W))
    g["run_margin"] = g["runs_for"] - g["runs_against"]
    g[f"run_diff_avg_{W}"] = g.groupby(["team", "season"], sort=False)["run_margin"].transform(
        lambda s: rmean(s, W))
    g[f"runs_scored_avg_{W}"] = g.groupby(["team", "season"], sort=False)["runs_for"].transform(
        lambda s: rmean(s, W))

    # Home-specific rolling win% (only over each team's home games)
    mask_home = g["is_home"] == 1
    mask_away = g["is_home"] == 0
    g.loc[mask_home, f"home_win_pct_{W}"] = (
        g[mask_home].groupby(["team", "season"], sort=False)["win"]
        .transform(lambda s: rmean(s, W))
    )
    # Road-specific rolling win% (only over each team's away games)
    g.loc[mask_away, f"road_win_pct_{W}"] = (
        g[mask_away].groupby(["team", "season"], sort=False)["win"]
        .transform(lambda s: rmean(s, W))
    )

    roll_cols = [f"win_pct_{W}", f"run_diff_avg_{W}", f"runs_scored_avg_{W}",
                 f"home_win_pct_{W}", f"road_win_pct_{W}"]
    hm = g[mask_home][["game_id"] + roll_cols].rename(
        columns={c: f"h_{c}" for c in roll_cols})
    aw = g[mask_away][["game_id"] + roll_cols].rename(
        columns={c: f"a_{c}" for c in roll_cols})
    return hm.merge(aw, on="game_id")


def load_and_engineer_features():
    print("Loading CSV...")
    df = pd.read_csv(CSV_INPUT_PATH, low_memory=False)
    print(f"  {len(df):,} rows loaded")

    print("Adding market columns...")
    df = add_market_columns(df)

    print("Adding schedule context (rest days, park factors, wind)...")
    df = add_schedule_context(df)

    print("Lagging team performance stats...")
    df_shift, long_games = shift_team_perf(df)

    print(f"Adding rolling form (W={BEST_W})...")
    roll_merge = add_rolling_form(long_games, BEST_W)
    df_feat = df_shift.merge(roll_merge, on="game_id", how="left")

    # Computed diff features
    df_feat["sp_fip_DIFF"]             = _gcol(df_feat, "a_fip_lag1")             - _gcol(df_feat, "h_fip_lag1")
    df_feat["sp_era_DIFF"]             = _gcol(df_feat, "a_sp_era_lag1")          - _gcol(df_feat, "h_sp_era_lag1")
    df_feat["sp_k9_DIFF"]              = _gcol(df_feat, "h_sp_k9_lag1")           - _gcol(df_feat, "a_sp_k9_lag1")
    df_feat["sp_bb9_DIFF"]             = _gcol(df_feat, "a_sp_bb9_lag1")          - _gcol(df_feat, "h_sp_bb9_lag1")
    df_feat["rolling_k9_DIFF"]         = _gcol(df_feat, "h_rolling_k9_lag1")      - _gcol(df_feat, "a_rolling_k9_lag1")
    df_feat["rolling_era_DIFF"]        = _gcol(df_feat, "a_rolling_era_lag1")     - _gcol(df_feat, "h_rolling_era_lag1")
    df_feat["rolling_whip_DIFF"]       = _gcol(df_feat, "a_rolling_whip_lag1")    - _gcol(df_feat, "h_rolling_whip_lag1")
    df_feat["wrc_plus_DIFF"]           = _gcol(df_feat, "h_wrc_plus_lag1")        - _gcol(df_feat, "a_wrc_plus_lag1")
    df_feat["woba_DIFF"]               = _gcol(df_feat, "h_woba_lag1")            - _gcol(df_feat, "a_woba_lag1")
    df_feat["war_DIFF"]                = _gcol(df_feat, "h_war_lag1")             - _gcol(df_feat, "a_war_lag1")
    df_feat["k_pct_DIFF"]              = _gcol(df_feat, "a_k_pct_lag1")           - _gcol(df_feat, "h_k_pct_lag1")
    df_feat["bb_pct_DIFF"]             = _gcol(df_feat, "h_bb_pct_lag1")          - _gcol(df_feat, "a_bb_pct_lag1")
    df_feat["pitcher_handedness_diff"] = _gcol(df_feat, "home_pitcher_is_lefty")  - _gcol(df_feat, "away_pitcher_is_lefty")
    df_feat["win_pct_W_DIFF"]          = _gcol(df_feat, f"h_win_pct_{BEST_W}")   - _gcol(df_feat, f"a_win_pct_{BEST_W}")
    df_feat["run_diff_avg_W_DIFF"]     = _gcol(df_feat, f"h_run_diff_avg_{BEST_W}") - _gcol(df_feat, f"a_run_diff_avg_{BEST_W}")
    df_feat["runs_scored_avg_W_DIFF"]  = _gcol(df_feat, f"h_runs_scored_avg_{BEST_W}") - _gcol(df_feat, f"a_runs_scored_avg_{BEST_W}")
    # Weather
    df_feat["wind_speed_kmh"]          = pd.to_numeric(df_feat["wind_speed_kmh"], errors="coerce").fillna(0.0)
    # Interaction: sharp money aligning with pitcher edge
    df_feat["sharp_x_fip"]            = df_feat["sharp_move_flag"] * df_feat["sp_fip_DIFF"]
    # Home/road split: home team's home win% minus away team's road win%
    df_feat["home_road_split_DIFF"]   = _gcol(df_feat, f"h_home_win_pct_{BEST_W}") - _gcol(df_feat, f"a_road_win_pct_{BEST_W}")

    df_feat = df_feat.sort_values("game_date").reset_index(drop=True)
    df_feat["hg"] = df_feat.groupby(["home_team", "season"]).cumcount()
    df_feat["ag"] = df_feat.groupby(["away_team", "season"]).cumcount()
    df_feat["early_season_flag"] = (df_feat[["hg", "ag"]].min(axis=1) < EARLY_SEASON_GAMES).astype(float)

    return df_feat

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BettingMLP(nn.Module):
    def __init__(self, n_features, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ---------------------------------------------------------------------------
# Betting logic
# ---------------------------------------------------------------------------

def kelly_stake(prob, decimal_odds, fraction=KELLY_FRACTION):
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    kelly = (b * prob - (1.0 - prob)) / b
    return float(max(0.0, min(kelly * fraction, 0.25)))


def evaluate(model, X_sc, y, mkt_probs, device):
    """
    Returns (brier_score, roi, n_bets).

    Kalshi ROI: contracts are priced 0–1 (same as implied probability).
      - Buy Home (Yes): price = mkt_prob  → decimal odds = 1 / mkt_prob
      - Buy Away (No):  price = 1-mkt_prob → decimal odds = 1 / (1-mkt_prob)
    Bet home when model_prob - mkt_prob > threshold,
    bet away when mkt_prob - model_prob > threshold.
    """
    model.eval()
    X_t = torch.tensor(X_sc, dtype=torch.float32, device=device)
    with torch.no_grad():
        probs = torch.sigmoid(model(X_t)).cpu().numpy()

    brier = float(np.mean((probs - y) ** 2))

    total_staked = 0.0
    total_profit = 0.0
    n_bets = 0
    for i in range(len(probs)):
        if np.isnan(mkt_probs[i]):
            continue
        mp = float(mkt_probs[i])
        pp = float(probs[i])
        edge_home = pp - mp
        edge_away = mp - pp  # model thinks away is undervalued

        if edge_home >= CONFIDENCE_THRESHOLD and mp > 1e-6:
            dec_odds = 1.0 / mp
            stake = kelly_stake(pp, dec_odds)
            if stake > 0:
                n_bets += 1
                total_staked += stake
                total_profit += stake * (dec_odds - 1.0) if y[i] == 1.0 else -stake

        elif edge_away >= CONFIDENCE_THRESHOLD and (1.0 - mp) > 1e-6:
            prob_away = 1.0 - pp
            dec_odds = 1.0 / (1.0 - mp)
            stake = kelly_stake(prob_away, dec_odds)
            if stake > 0:
                n_bets += 1
                total_staked += stake
                total_profit += stake * (dec_odds - 1.0) if y[i] == 0.0 else -stake

    roi = total_profit / total_staked if total_staked > 0 else float('nan')
    return brier, roi, n_bets

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown + (1 - cooldown) * FINAL_LR_FRAC

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()

df = load_and_engineer_features()

active_feats = [c for c in FEATURE_COLUMNS if c in df.columns]
print(f"Active features ({len(active_feats)}): {active_feats}")

req_cols = active_feats + ["home_win", "market_implied_prob", "game_date"]
df = df.dropna(subset=[c for c in req_cols if c in df.columns])

train_df = df[df["game_date"] < "2024-01-01"]
val_df   = df[df["game_date"] >= "2024-01-01"]
print(f"Train rows: {len(train_df):,} | Val rows: {len(val_df):,}")

X_train = train_df[active_feats].values.astype(np.float32)
y_train = train_df["home_win"].values.astype(np.float32)
X_val   = val_df[active_feats].values.astype(np.float32)
y_val   = val_df["home_win"].values.astype(np.float32)

mkt_probs_val = val_df["market_implied_prob"].values.astype(np.float32)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = BettingMLP(len(active_feats)).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

X_t = torch.tensor(X_train_sc, dtype=torch.float32, device=device)
y_t = torch.tensor(y_train,    dtype=torch.float32, device=device)

# Market anchor: logit(market_implied_prob) for training set
mkt_train = train_df["market_implied_prob"].clip(1e-4, 1 - 1e-4).values.astype(np.float32)
mkt_logit_train = np.log(mkt_train / (1.0 - mkt_train))
mkt_logit_t = torch.tensor(mkt_logit_train, dtype=torch.float32, device=device)

# ---------------------------------------------------------------------------
# Training loop (time-budgeted)
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0.0
step = 0
smooth_loss = 0.0

while True:
    t0 = time.time()
    model.train()

    perm = torch.randperm(len(X_t), device=device)
    epoch_loss = 0.0
    n_batches = 0
    for i in range(0, len(X_t), BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        logits = model(X_t[idx])
        bce_loss = F.binary_cross_entropy_with_logits(logits, y_t[idx])
        anchor_loss = MARKET_ANCHOR_LAMBDA * ((logits - mkt_logit_t[idx]) ** 2).mean()
        loss = bce_loss + anchor_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    t1 = time.time()
    dt = t1 - t0

    if step > 2:
        total_training_time += dt

    # LR schedule
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    avg_loss = epoch_loss / max(n_batches, 1)
    ema = 0.9
    smooth_loss = ema * smooth_loss + (1 - ema) * avg_loss
    debiased = smooth_loss / (1 - ema ** (step + 1))

    remaining = max(0, TIME_BUDGET - total_training_time)
    print(
        f"\repoch {step:04d} ({100*progress:.1f}%) | loss: {debiased:.4f} | "
        f"lrm: {lrm:.2f} | remaining: {remaining:.0f}s    ",
        end="", flush=True,
    )

    step += 1
    if step > 2 and total_training_time >= TIME_BUDGET:
        break

print()

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

brier, roi, n_bets = evaluate(model, X_val_sc, y_val, mkt_probs_val, device)

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0

print("---")
print(f"val_roi:          {roi:.6f}")
print(f"val_brier:        {brier:.6f}")
print(f"n_bets:           {n_bets}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_params_M:     {n_params / 1e6:.3f}")
print(f"n_features:       {len(active_feats)}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

results_file = "results.tsv"
header = "commit\tval_roi\tval_brier\tstatus\tdescription\n"
if not os.path.exists(results_file):
    with open(results_file, "w") as f:
        f.write(header)

commit = os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip() or "HEAD"
status = "ok" if not (roi != roi) else "fail"  # nan check
desc = f"run13 W={BEST_W} threshold={CONFIDENCE_THRESHOLD} anchor={MARKET_ANCHOR_LAMBDA} home_road_split"
row = f"{commit}\t{roi:.6f}\t{brier:.6f}\t{status}\t{desc}\n"

with open(results_file, "a") as f:
    f.write(row)
print(f"Results saved to {results_file}")
