# MLB Feature Data Dictionary

This document defines the features used in the `train.py` model. Most 'DIFF' features are calculated such that a **positive value favors the Home Team**.

## 1. Market Features (Closing & Opening Prices)
| Feature | Definition | Directional Bias |
| :--- | :--- | :--- |
| `market_implied_prob` | Closing home win probability (de-vigged). | 0.5 = Even; >0.5 = Home Favorite. |
| `open_home_implied` | Opening home win probability. | Baseline market expectation. |
| `line_move_delta` | `market_implied_prob` - `open_home_implied`. | (+) = Money moved toward Home. |
| `sharp_move_flag` | Binary (1 if \|line_move\| > 0.03). | Signals high-conviction market move. |
| `total_move_delta` | Change in Over/Under total from open to close. | (+) = Market expects more runs. |

## 2. Pitcher Quality (DIFF = Away - Home or Home - Away)
| Feature | Definition | Directional Bias |
| :--- | :--- | :--- |
| `sp_fip_DIFF` | `Away_FIP` - `Home_FIP`. | (+) = Home has better FIP (lower is better). |
| `sp_era_DIFF` | `Away_ERA` - `Home_ERA`. | (+) = Home has better ERA (lower is better). |
| `sp_k9_DIFF` | `Home_K9` - `Away_K9`. | (+) = Home strikes out more batters. |
| `sp_bb9_DIFF` | `Away_BB9` - `Home_BB9`. | (+) = Home issues fewer walks. |
| `rolling_k9_DIFF` | Difference in K/9 over the last few starts. | (+) = Home SP in better recent form. |

## 3. Offensive Performance
| Feature | Definition | Directional Bias |
| :--- | :--- | :--- |
| `wrc_plus_DIFF` | `Home_wRC+` - `Away_wRC+`. | (+) = Home offense is more powerful. |
| `woba_DIFF` | `Home_wOBA` - `Away_wOBA`. | (+) = Home contributes more offense. |

## 4. Handedness & Splits
| Feature | Definition | Directional Bias |
| :--- | :--- | :--- |
| `pitcher_handedness_diff` | Categorical delta between SP handedness. | Interaction between L/R matchups. |
| `home_pitcher_is_lefty` | Binary (1 = Left, 0 = Right). | Home SP handedness. |
| `away_pitcher_is_lefty` | Binary (1 = Left, 0 = Right). | Away SP handedness. |
| `home_road_split_DIFF` | `Home_Win%_at_Home` - `Away_Win%_on_Road`. | (+) = Home has stronger home-field edge. |

## 5. Rolling Form (Window W)
| Feature | Definition | Directional Bias |
| :--- | :--- | :--- |
| `win_pct_W_DIFF` | `Home_Win%` - `Away_Win%`. | (+) = Home has better recent record. |
| `run_diff_avg_W_DIFF` | `Home_Run_Diff` - `Away_Run_Diff`. | (+) = Home winning games more decisively. |
| `runs_scored_avg_W_DIFF` | `Home_Avg_Runs` - `Away_Avg_Runs`. | (+) = Home offense is currently "hotter". |

## 6. Weather, Venue & Context
| Feature | Definition | Directional Bias |
| :--- | :--- | :--- |
| `temp_c` | Temperature in Celsius. | (+) = Higher run environment. |
| `wind_speed_kmh` | Wind speed at the stadium. | Affects ball flight/volatility. |
| `park_factor` | Scoring index (e.g., 1.12=Coors, 0.91=Seattle). | (+) = High scoring venue. |
| `is_night_game` | Binary (1 = Night, 0 = Day). | Often lower run environments. |
| `rest_days_DIFF` | `Home_Rest` - `Away_Rest`. | (+) = Home is better rested. |
| `sp_rest_DIFF` | `Home_SP_Rest` - `Away_SP_Rest`. | (+) = Home SP is more rested. |
| `is_series_finale` | Binary (1 = Yes). | Associated with "getaway" lineups. |
| `early_season_flag` | Binary (1 = First 15 games). | Flags lower-confidence historical data. |

## 7. Interactions
| Feature | Definition | Directional Bias |
| :--- | :--- | :--- |
| `sharp_x_fip` | `sharp_move_flag` * `sp_fip_DIFF`. | Synergy between market move and FIP. |