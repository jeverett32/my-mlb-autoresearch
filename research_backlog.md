# MLB Research Backlog: Ideas to Explore

This document serves as the strategic menu for the Autoresearch agent. When the model hits a plateau or needs a new hypothesis, it should pull from these categories.

## 1. Feature Engineering & Signal Extraction
* **Travel & Fatigue (Rest-Diff)**: 
    * *Hypothesis*: Teams on the road for long stretches with fewer rest days underperform their baseline.
    * *Implementation*: `home_rest - away_rest`. Also consider a `road_trip_duration` counter.
* **Pitcher/Park Interaction**:
    * *Hypothesis*: High-strikeout pitchers (K/9) are less affected by "hitters' parks" (high park factors) than contact pitchers.
    * *Implementation*: Create an interaction term: `sp_k9_DIFF * park_factor`.
* **The "Opener" Correction**:
    * *Hypothesis*: Starting pitcher stats are misleading when a team uses an "opener."
    * *Implementation*: Add a binary flag `is_opener` if `sp_gs_lag1` (games started) is significantly lower than appearances.
* **Weather Sensitivities**:
    * *Hypothesis*: High temperatures (`temp_c`) and wind blowing out (`wind_dir_cos`) increase run environments, favoring the team with the higher `wrc_plus`.
    * *Implementation*: `temp_c * wrc_plus_DIFF`.

## 2. Market Decorrelation (Hubáček-style)
* **Residual Loss Head**:
    * *Hypothesis*: The market is highly efficient; the model should only learn what the market *doesn't* know.
    * *Implementation*: Modify the loss function to penalize predictions that are too close to `market_implied_prob` unless they are correct, or explicitly train on the residual `home_win - market_implied_prob`.
* **Market Price Volatility**:
    * *Hypothesis*: Large moves between `open_home_implied` and `market_implied_prob` signal "sharp" money that the model should either follow or contrarian-fade.
    * *Implementation*: Use `line_move_delta` as a weighting factor in the loss function.

## 3. Architectural Shifts
* **XGBoost/LightGBM Pivot**:
    * *Strategy*: If Neural Nets are failing to find non-linearities, switch to Gradient Boosted Decision Trees (GBDT).
    * *Goal*: Use `val_roi` as the evaluation metric for early stopping rather than log-loss.
* **Self-Attention / TabTransformer**:
    * *Strategy*: Replace the middle MLP layers with a Multi-Head Attention block.
    * *Goal*: Allow the model to learn complex interactions between categorical features (like `park_factor` and `pitcher_handedness`) that fixed dense layers might miss.
* **Feature Neutralization**:
    * *Strategy*: Explicitly "neutralize" features that are already heavily baked into the market price to force the model to find alternative signals.

## 4. Betting Logic & Calibration
* **Dynamic Confidence Threshold ($\phi$)**:
    * *Hypothesis*: A static 0.025 edge is too rigid. 
    * *Implementation*: Scale $\phi$ based on `market_implied_prob`. (e.g., require a larger edge for heavy favorites vs. even-money games).
* **Platt Scaling / Isotonic Regression**:
    * *Hypothesis*: Raw sigmoid outputs are often over/under confident.
    * *Implementation*: Add a post-processing calibration step to the `evaluate()` pipeline to ensure `val_brier` is minimized before calculating ROI.
* **Kelly Fraction Optimization**:
    * *Hypothesis*: 0.25 Kelly might be too aggressive for the high variance of MLB.
    * *Implementation*: Test `0.1 Kelly` (Deci-Kelly) vs `Flat 1%` of bankroll to see which preserves capital during "cold streaks" while maintaining ROI.