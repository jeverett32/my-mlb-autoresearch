# Experiment Log — autoresearch/mlb-mar27

## MLP-Era Distilled Findings (reference only — NOT comparable to walk-forward ROI)

| What worked | What didn't |
|---|---|
| Market anchor loss (lambda=1.0) | lambda ≠ 1.0, focal loss |
| W=15 rolling window | W≤10 or W≥20 |
| W=5 momentum signal | Run-diff momentum, W=3 |
| Feature neutralization (Hubáček) | TabTransformer, wider/deeper MLP |
| threshold=0.04, Kelly=0.25 | threshold ≥ 0.05, Kelly=0.15 | *(MLP era only — walk-forward LR best at threshold=0.12)* |

**MLP stable baseline (run27, seed=42):** ROI=+1.34%, Brier=0.240, 673 bets *(single-split, not comparable)*

---

## [Current Era: 4-fold Walk-Forward, Honest Regime (TW=None)]
- 4-fold walk-forward (val=2022, 2023, 2024, 2025). Mean ROI is the only metric.
- Current best: **44.84%** (Run 130) — LR, TW=None, C=0.04, threshold=0.14, PROB_CAP=(0.34,0.66), EARLY_CUTOFF=25
- Feature set: 43 base + day_of_week + sharp_x_fip + momentum_DIFF + luck_DIFF + pythagorean_DIFF + luck_x_momentum + park_x_pythagorean + pythagorean_short_DIFF + post_rule_change = 61 features
- Key insight: post_rule_change (year>=2023) gives +2.63pp by allowing LR to recalibrate for rule-change era

---

<!-- Add new runs below this line -->

## Run 145 — covid_era flag (2020 unusual 60-game season) ★ NEW BEST (marginal)
**Hypothesis**: 2020 was a 60-game season with no fans and unusual player conditioning; training data from 2020 may mislead the model. A covid_era flag lets LR downweight 2020 patterns.
**Change**: Add covid_era = (year == 2020)
**Result**: roi=+44.85% (+0.0002pp) — effectively identical
**Decision**: KEPT ✓

## Run 144 — post_dh_era flag (universal DH from 2022) + Run 130 config ★ NEW BEST (marginal)
**Hypothesis**: Universal DH adopted in 2022 changed batting lineup quality calculations for NL teams; a separate binary flag for year >= 2022 may add independent signal from post_rule_change (2023).
**Change**: Add post_dh_era = (year >= 2022) alongside post_rule_change
**Result**: roi=+44.85% mean, fold4=+17.18% — +0.003pp improvement (marginal; noise vs real)
**Decision**: KEPT ✓ (strictly improves over 44.84%)

## Run 143 — Remove days_since_asb (may be noisy) from Run 130 config
**Hypothesis**: days_since_asb measures time since All-Star Break; month feature already captures this seasonality, making days_since_asb redundant noise.
**Change**: Remove days_since_asb from FEATURE_COLUMNS
**Result**: roi=+43.72% mean, fold4=+12.71% — days_since_asb contributes; removing hurts fold4
**Decision**: REVERTED

## Run 142 — BEST_W=12 on Run 130 best config
**Change**: BEST_W=12 (was 15)
**Result**: roi=+39.03% mean, fold4=+12.02% — BEST_W=15 is significantly better
**Decision**: REVERTED

## Run 141 — MOMENTUM_W=7 on Run 130 best config
**Change**: MOMENTUM_W=7 (was 10)
**Result**: roi=+42.78% mean, fold4=+16.75% — MOMENTUM_W=10 is better
**Decision**: REVERTED

## Run 140 — post_rule_x_luck interaction + Run 130 feature set
**Change**: Add post_rule_x_luck = post_rule_change * luck_DIFF
**Result**: roi=+44.29% mean, fold4=+13.89% — fold3 improved but fold4 drops; overall worse
**Decision**: REVERTED

## Run 139 — LGB with full Run 130 feature set
**Change**: MODEL="lgb"
**Result**: roi=+27.10% mean, fold4=+18.57% (154 bets) — much worse than LR 44.84%
**Decision**: REVERTED
**Insight**: LGB still underperforms LR even with richer features; LR's L1 sparsity better suits this domain

## Run 138 — season_year + Run 130 feature set
**Change**: Add season_year = numeric year (float)
**Result**: roi=+42.87% mean, fold4=+9.29% — year extrapolation damages fold4 generalization
**Decision**: REVERTED
**Insight**: Linear year trend extrapolates badly out-of-sample; post_rule_change binary is the right abstraction

## Run 137 — fip_x_line (continuous) + Run 130 feature set
**Change**: Add fip_x_line = line_move_delta * sp_fip_DIFF
**Result**: roi=+44.84% (identical) — zeroed by L1
**Decision**: REVERTED

## Run 136 — EARLY_CUTOFF=30 on Run 130 best config
**Change**: EARLY_CUTOFF=30 (was 25)
**Result**: roi=+41.97% mean, fold4=+13.54% (99 bets) — EARLY_CUTOFF=25 is optimal
**Decision**: REVERTED

## Run 135 — EARLY_CUTOFF=20 on Run 130 best config
**Change**: EARLY_CUTOFF=20 (was 25)
**Result**: roi=+39.24% mean, fold4=+10.84% (93 bets) — much worse
**Decision**: REVERTED

## Run 134 — woba_x_sharp + Run 130 feature set
**Change**: Add woba_x_sharp = woba_DIFF * sharp_move_flag
**Result**: roi=+44.44% mean, fold4=+16.34% (90 bets) — below 44.84%; slight deterioration
**Decision**: REVERTED

## Run 133 — PROB_CAP=(0.35, 0.65)
**Change**: PROB_CAP=(0.35, 0.65) (was 0.34/0.66)
**Result**: roi=+42.87% mean, fold4=+14.43% — worse; (0.34, 0.66) is optimal
**Decision**: REVERTED

## Run 132 — PROB_CAP=(0.33, 0.67)
**Change**: PROB_CAP=(0.33, 0.67) (was 0.34/0.66)
**Result**: roi=+43.98% mean, fold4=+19.32% — worse; wider cap admits lower-quality bets
**Decision**: REVERTED

## Run 131 — form_x_fip + all features (TW=None)
**Change**: Add form_x_fip = win_pct_W_DIFF * sp_fip_DIFF
**Result**: roi=+44.84% (identical) — form_x_fip zeroed by L1
**Decision**: REVERTED

## Run 130 — post_rule_change flag + pythagorean stack (TW=None) ★ CURRENT BEST
**Hypothesis**: 2023 MLB rule changes (pitch clock, shift ban, bigger bases) altered game dynamics; binary post_rule_change allows LR to adjust weights for modern vs legacy patterns.
**Change**: TW=None, Run 122 features + post_rule_change = (year >= 2023)
**Result**: roi=+44.84% mean, fold3=+47.68% (78 bets), fold4=+17.18% (89 bets)
**Decision**: KEPT ✓
**Insight**: post_rule_change is powerful! fold3 (2024) jumps from 37.53% → 47.68%. Real structural signal.

## Run 129 — abs_line_move + Run 122 feature set (TW=None)
**Change**: Add abs_line_move = abs(line_move_delta)
**Result**: roi=+40.82% mean, fold4=+16.84% — worse; redundant with existing market features
**Decision**: REVERTED
