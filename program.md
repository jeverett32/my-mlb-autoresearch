autoresearch: MLB Kalshi Betting Edition

This is an autonomous experiment to optimize a model for predicting high-ROI MLB bets on Kalshi.

Setup

Agree on a run tag: Propose a tag based on today's date (e.g., mlb-mar26). The branch autoresearch/<tag> must not already exist.

Create the branch: git checkout -b autoresearch/<tag> from master.

Read context:

master_mlb.csv: The core dataset.

feature_engineering.ipynb: Contains the cells for feature creation, scaling, and ROI calculations.

prepare.py: Handles data loading, date-based splitting (training on historical seasons, validating on recent ones), and the ROI/Brier score evaluation harness. Do not modify.

train.py: The file you modify. Model architecture, training loop, and betting logic (thresholds, stake sizing).

Initialize results.tsv: Create results.tsv with the header:
commit	val_roi	val_brier	status	description

Initialize experiment_log.md: Create a file to track qualitative insights across runs.

Experimentation

Each run has a 5-minute wall clock budget for training. Launch with: uv run train.py.

What you CAN do:

Modify train.py. You are encouraged to experiment with:

Model architectures (MLPs, Transformers, etc.).

Feature selection/weighting.

Decorrelation techniques (Hubáček-style market price decorrelation).

Betting logic: confidence thresholds ($\phi$), Kelly Criterion variants, or Sharpe-based stake sizing.

Hyperparameters (LR, Batch Size, Weight Decay).

What you CANNOT do:

Modify prepare.py or the evaluation metric code.

Install new packages not in pyproject.toml.

Metrics of Success:

ROI (Primary): The goal is to maximize Return on Investment on the validation set.

Brier Score (Secondary): Ensure the probabilities are well-calibrated.

Simplicity: If ROI is similar, favor the simpler model.

Output & Logging

The script prints:

---
val_roi:        0.152
val_brier:      0.211
training_secs:  300.0
num_params_M:   1.2


1. results.tsv (Tab-Separated)

Log every run:
commit	val_roi	val_brier	status	description

2. experiment_log.md

After each run, append a short entry:

What was tried: (e.g., "Added rolling pitcher ERA features")

Result: (e.g., "ROI increased by 2%, but Brier score worsened. Model is overconfident.")

Next Step: (e.g., "Try adding temperature scaling to calibrate probabilities.")

The Experiment Loop

Check current git state.

Edit train.py based on the previous entries in experiment_log.md.

git commit -m "..."

Run: uv run train.py > run.log 2>&1

Extract val_roi and val_brier. If empty, it's a crash; check tail -n 50 run.log.

Update results.tsv and experiment_log.md.

Advance or Reset:

If val_roi improved: Keep the commit and continue.

If val_roi stayed same or worsened: git reset --hard HEAD~1 and try a different approach.

NEVER STOP: Continue iterating until manually interrupted. If you hit a plateau in ROI, revisit the feature_engineering.ipynb notes to see if you can implement a more complex feature directly into the train.py pipeline.