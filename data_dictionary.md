# MLB Feature Data Dictionary
All `DIFF` features: **positive value = home team advantage**.
When adding a feature in `engineer_new_features()`, document it here in Section 8 before committing.

---

## 1. Market Features
| Feature | Definition | Direction |
|---|---|---|
| `market_implied_prob` | De-vigged closing home win probability | >0.5 = Home Favorite |
| `open_home_implied` | De-vigged opening home win probability | Baseline expectation |
| `line_move_delta` | Closing − Opening probability | (+) = sharp money → Home |
| `sharp_move_flag` | 1 if \|line_move_delta\| ≥ 0.03 | High-conviction move |
| `total_move_delta` | Closing − Opening O/U total | (+) = market expects more runs |

## 2. Pitcher Quality — Lagged (shift+1 within team-season)
| Feature | Definition | Direction |
|---|---|---|
| `sp_fip_DIFF` | Away FIP − Home FIP | (+) = Home SP has better FIP |
| `sp_era_DIFF` | Away ERA − Home ERA | (+) = Home SP has better ERA |
| `sp_k9_DIFF` | Home K/9 − Away K/9 | (+) = Home SP strikes out more |
| `sp_bb9_DIFF` | Away BB/9 − Home BB/9 | (+) = Home SP walks fewer |
| `sp_whip_DIFF` | Away WHIP − Home WHIP | (+) = Home SP has better WHIP |
| `rolling_k9_DIFF` | Home rolling K/9 − Away rolling K/9 | (+) = Home SP better recent form |
| `rolling_era_DIFF` | Away rolling ERA − Home rolling ERA | (+) = Home SP better recent ERA |
| `rolling_whip_DIFF` | Away rolling WHIP − Home rolling WHIP | (+) = Home SP better recent WHIP |
| `rookie_DIFF` | Away is_rookie − Home is_rookie | (+) = Home pitcher more experienced |
| `sp_rest_DIFF` | Home SP rest days − Away SP rest days (capped 14d) | (+) = Home SP more rested |

## 3. Batting — FanGraphs Season-to-Date, Lagged 1 Game
| Feature | Definition | Direction |
|---|---|---|
| `wrc_plus_DIFF` | Home wRC+ − Away wRC+ | (+) = Home offense stronger |
| `woba_DIFF` | Home wOBA − Away wOBA | (+) = Home contributes more offense |
| `avg_DIFF` | Home AVG − Away AVG | (+) = Home hits better |
| `obp_DIFF` | Home OBP − Away OBP | (+) = Home gets on base more |
| `slg_DIFF` | Home SLG − Away SLG | (+) = Home hits for more power |
| `k_pct_DIFF` | Away K% − Home K% | (+) = Home strikes out less |
| `bb_pct_DIFF` | Home BB% − Away BB% | (+) = Home walks more |
| `k_per_9_DIFF` | Away K/9 (team) − Home K/9 (team) | (+) = Home team K's opponents more |
| `bb_per_9_DIFF` | Home BB/9 − Away BB/9 | (+) = Home issues fewer walks |
| `hr_per_9_DIFF` | Away HR/9 − Home HR/9 | (+) = Home allows fewer HRs |
| `era_DIFF` | Away team ERA − Home team ERA | (+) = Home pitching staff better |
| `fip_DIFF` | Away team FIP − Home team FIP | (+) = Home pitching staff better |
| `owar_DIFF` | Home oWAR − Away oWAR | (+) = Home offense more valuable |
| `war_DIFF` | Home WAR − Away WAR | (+) = Home overall more valuable |

## 4. Handedness
| Feature | Definition | Direction |
|---|---|---|
| `pitcher_handedness_diff` | home_is_lefty − away_is_lefty | L/R matchup delta |
| `home_pitcher_is_lefty` | 1 = Home SP is LHP | — |
| `away_pitcher_is_lefty` | 1 = Away SP is LHP | — |

## 5. Rolling Form (Window = BEST_W games, default 15)
| Feature | Definition | Direction |
|---|---|---|
| `win_pct_W_DIFF` | Home W-game win% − Away | (+) = Home has better recent record |
| `run_diff_avg_W_DIFF` | Home W-game run diff avg − Away | (+) = Home winning more decisively |
| `run_diff_std_W_DIFF` | Home W-game run diff std − Away | (+) = Home more consistent |
| `runs_scored_avg_W_DIFF` | Home W-game runs scored avg − Away | (+) = Home offense hotter |
| `runs_allowed_avg_W_DIFF` | Home W-game runs allowed avg − Away | (+) = Home pitching stingier |
| `momentum_DIFF` | (Home 5g win% − Home 15g win%) − same for Away | (+) = Home on relative hot streak |

## 6. Season-to-Date (Expanding Mean)
| Feature | Definition | Direction |
|---|---|---|
| `season_win_pct_DIFF` | Home season win% − Away season win% | (+) = Home better overall |
| `season_run_diff_avg_DIFF` | Home season run diff avg − Away | (+) = Home more dominant overall |
| `streak_DIFF` | Home streak − Away streak | (+) = Home on winning streak |

## 7. Weather, Venue & Schedule
| Feature | Definition | Direction |
|---|---|---|
| `temp_c` | Temperature in Celsius | Higher = more offense |
| `wind_speed_kmh` | Wind speed at stadium | Affects ball flight |
| `wind_dir_sin` | sin(wind_dir_deg) — circular encoding | — |
| `wind_dir_cos` | cos(wind_dir_deg) — circular encoding | — |
| `park_factor` | Scoring index / 100 (1.13=Coors, 0.94=Oakland) | (+) = Hitter-friendly |
| `is_night_game` | 1 = Night game | Night = lower run environment |
| `home_rest_days` | Home team rest days (capped 7d) | Individual rest signal |
| `away_rest_days` | Away team rest days (capped 7d) | Individual rest signal |
| `rest_days_DIFF` | Home rest − Away rest | (+) = Home more rested |
| `is_series_finale` | 1 = Last game of homestand series | Getaway lineup risk |
| `early_season_flag` | 1 if either team < 15 games played | Low-confidence period |
| `month` | Calendar month (1–12) | Season structure |
| `days_since_asb` | Days elapsed since All-Star break (0 before) | Second-half context |

## 8. Interactions (base set)
| Feature | Definition | Direction |
|---|---|---|
| `sharp_x_fip` | `sharp_move_flag` × `sp_fip_DIFF` | Sharp money + pitcher edge synergy |

## 9. Engineered Interactions (from `engineer_new_features()`)
| Feature | Definition | Direction | Status |
|---|---|---|---|
| `fip_x_line` | `sp_fip_DIFF` × `line_move_delta` | Sharp money validating FIP edge | Seeded, not in FEATURE_COLUMNS |
| `form_x_fip` | `run_diff_avg_W_DIFF` × `sp_fip_DIFF` | Hot team + good SP compounding | Seeded, not in FEATURE_COLUMNS |
| `temp_x_wrc` | `temp_c` × `wrc_plus_DIFF` | Hot weather amplifies batting gap | Seeded, not in FEATURE_COLUMNS |
| `woba_x_sharp` | `woba_DIFF` × `sharp_move_flag` | Sharp money + batting edge synergy | Seeded, not in FEATURE_COLUMNS |
| `early_x_fip` | `early_season_flag` × `sp_fip_DIFF` | FIP less reliable early in season | Seeded, not in FEATURE_COLUMNS |

---

## Early Specialist Features (EARLY_FEATURE_COLUMNS)
Used by the early-season LR specialist for games where either team has < 15 games played.
Pitcher + market + context only — rolling team stats unreliable this early.

`sp_era_DIFF`, `sp_whip_DIFF`, `sp_k9_DIFF`, `sp_bb9_DIFF`, `sp_fip_DIFF`,
`market_implied_prob`, `park_factor`, `home_rest_days`, `away_rest_days`,
`month`, `is_night_game`

---

## Removed / Deprecated Features
| Feature | Reason |
|---|---|
| `home_road_split_DIFF` | Too few games per split window; noise (run13) |
| `run_diff_momentum_DIFF` | Duplicate of win% momentum signal (run15) |
| `war_DIFF`, `k_pct_DIFF`, `bb_pct_DIFF`, `wind_dir_deg` (raw) | Removed run8 — noisy or circular-encoded instead |
| Park-factor interactions (`park_x_wind`, `fip_x_park`, etc.) | MLP-era: collinearity r>0.99, park_factor std≈0.04. Re-test with GBDT importance before restoring. |