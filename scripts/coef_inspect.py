"""Print L1 LR feature coefficients for fold 3 (train < 2024)."""
import warnings; warnings.filterwarnings('ignore')
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np

src = open('train.py').read()
split_marker = "\ndf = load_and_engineer_features()"
exec(src[:src.index(split_marker)], globals())

df = load_and_engineer_features()
active_feats = [c for c in FEATURE_COLUMNS if c in df.columns]

train_df = df[df['game_date'] < '2024-01-01'].copy()
train_df = train_df.dropna(subset=['home_win', 'market_implied_prob'])
early_tr = ((train_df['home_games_played'] < EARLY_CUTOFF) |
            (train_df['away_games_played'] < EARLY_CUTOFF))
X_tr = train_df.loc[~early_tr, active_feats]
y_tr = train_df.loc[~early_tr, 'home_win'].values

clf = build_lr(X_tr, y_tr)

coefs = [est.estimator.named_steps['mdl'].coef_[0]
         for est in clf.calibrated_classifiers_]
mean_coef = np.mean(coefs, axis=0)

feat_imp = sorted(zip(active_feats, mean_coef),
                  key=lambda x: abs(x[1]), reverse=True)

print("\nActive (|coef| > 0.001):")
for f, c in feat_imp:
    if abs(c) > 0.001:
        print(f"  {f:35s} {c:+.4f}")

print("\nZeroed out:")
for f, c in feat_imp:
    if abs(c) <= 0.001:
        print(f"  {f}")
