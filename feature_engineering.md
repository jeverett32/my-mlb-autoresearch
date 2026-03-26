# MLB Feature Engineering Protocol

Goal: Maximize signal-to-noise ratio by iteratively adding, transforming, or pruning features.

**Where to add new features:** `engineer_new_features(df)` in `train.py`. This runs after all lag/rolling/FanGraphs features are computed — the full lagged dataset is available. Document every new feature in `data_dictionary.md` Section 9 before committing.

**Important reset:** The 43-feature set is a fresh slate vs. the MLP era. MLP-era Pearson correlations (|r| < 0.005 = noise) are a prior, not a conclusion. Let GBDT permutation importance drive pruning. Only remove features that the GBDT also ranks near-zero AND whose removal holds or improves walk-forward ROI.

---

## Phase 1: Statistical Screening (before any training run)

Every new feature **must** pass these checks first:

1. **Correlation Check**: Pearson |r| with `home_win` on the full dataset. If |r| < 0.003, it's likely noise — do not proceed without a strong domain reason.
2. **Redundancy Check**: Max Pearson |r| with every existing `FEATURE_COLUMN`. If > 0.90, merge or choose the stronger one. If > 0.99, the feature is collinear — drop it. (This is why park-factor interaction features failed in the MLP era: park_factor std ≈ 0.04 made every product have r > 0.99 with the underlying feature.)
3. **Lookahead Bias Check**: Confirm the feature uses only lagged data. Features in `engineer_new_features()` can safely use any column already produced by `build_long_format()`, `add_rolling_features()`, or the FanGraphs lag block — those are all shift(1)-lagged.

Quick screening script pattern:
```python
import pandas as pd, numpy as np
df = pd.read_csv("master_mlb.csv")  # load after full feature engineering
new_feat = df["sp_fip_DIFF"] * df["line_move_delta"]
print(abs(new_feat.corr(df["home_win"])))          # Phase 1 check
print(df[existing_feats].corrwith(new_feat).abs().max())  # Phase 2 check
```

---

## Phase 2: Pruning Sessions (every 5 runs)

1. Run `MODEL="lgb"` and read the printed top-20 + bottom-5 feature importances (auto-printed on last fold).
2. Cross-reference bottom features with Phase 1 correlation scores.
3. Run a "negative experiment": remove bottom 3–5 features and check if walk-forward mean ROI holds or improves.
4. If ROI holds or improves: remove permanently from `FEATURE_COLUMNS` and `data_dictionary.md`.
5. If ROI drops: they have structural regularization value despite low importance — keep and note in `experiment_log.md`.

**GBDT vs. MLP pruning:** GBDTs rank features by split gain — a feature with low Pearson correlation can still have high importance if it captures non-linear interactions. Don't prune based on Pearson alone. Always run the negative experiment.

---

## Phase 3: Transformation Experiments

If new raw features don't screen well, try transforming existing ones:

- **Binning continuous variables**: Convert `temp_c` into Cold/Mild/Hot (< 10°C, 10–25°C, > 25°C). GBDTs already find these splits — low priority. More useful for LR.
- **Ratio instead of difference**: Instead of `wrc_plus_DIFF`, try `home_wrc_plus / away_wrc_plus` — ratios can capture asymmetric effects.
- **Log-transform skewed features**: Pitcher ERAs in first few starts can be extreme — `log(1 + era)` before differencing.
- **Absolute value**: `abs(line_move_delta)` captures market uncertainty independent of direction.

---

## Interaction Feature Checklist

Before adding any interaction `A × B`:
- [ ] Both A and B individually pass Phase 1 screening
- [ ] Interaction |r| with `home_win` > 0.003
- [ ] Max redundancy with existing features < 0.90
- [ ] Direction is consistent with `data_dictionary.md` (positive = home advantage)
- [ ] Documented in `data_dictionary.md` Section 9 with status "Seeded"
- [ ] Added to `FEATURE_COLUMNS` only after screening passes (not before)