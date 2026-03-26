# autoresearch: MLB Kalshi Betting Edition

This is an autonomous experiment to optimize a model for predicting high-ROI MLB bets on Kalshi.

## Setup

1.  **Agree on a run tag**: Propose a tag based on today's date (e.g., `mlb-mar26`). If the run tag already exists, enter that branch because it is the same day.
2.  **Create the branch**: `git checkout -b autoresearch/<tag>` from master.
3.  **Read context**:
    * `master_mlb.csv`: The core dataset.
    * `train.py`: The file you modify. It contains the full pipeline: feature engineering, model architecture, training loop, and the `evaluate()` function.
    * `research_backlog.md`: A list of high-level strategic ideas and advanced features to implement.
4.  **Initialize results.tsv**: Create `results.tsv` (tab-separated) with the header:
    `commit	val_roi	val_brier	status	description`
5.  **Initialize experiment_log.md**: Create this file to track qualitative insights and hypotheses.

## Experimentation

Each run has a **5-minute wall clock budget** for training. Launch with: `uv run train.py`.

### What you CAN do:
* **Modify `train.py`**: You are encouraged to experiment with architectures (MLP, XGBoost, Transformers), feature selection, decorrelation techniques, and betting logic (thresholds, Kelly sizing).
* **Consult the Backlog**: Use `research_backlog.md` for inspiration when you need a new direction.

### What you CANNOT do:
* **Modify the evaluation metric**: The `evaluate()` function in `train.py` is the ground truth. Do not change it.
* **Install new packages**: Use only what is provided in `pyproject.toml`.

## The "Thinker" Protocol

Before every experiment, you **must**:
1.  Read `experiment_log.md` to identify failed patterns and avoid repetition.
2.  Consult `research_backlog.md` and follow the procedures in `feature_engineering.md` to determine if the next step should be adding a new signal or pruning an old one.
3.  Formulate a **hypothesis** (e.g., "The model ignores travel fatigue; adding a rest-diff interaction term should improve ROI").
4.  Record this hypothesis in `experiment_log.md` **before** running the code.

## The Experiment Loop

1.  **Hypothesize**: Follow the "Thinker Protocol."
2.  **Edit**: Modify `train.py` based on your hypothesis.
3.  **Run**: `uv run train.py > run.log 2>&1`
4.  **Compare**: Execute `uv run check_improvement.py`.
5.  **Log**: Update `experiment_log.md` with results (`val_roi`, `val_brier`, `n_bets`). 
    * *Crucial*: Log the result **before** committing or resetting so the record is permanent.
6.  **Advance or Reset**:
    * **If ROI improved**: `git add .` and `git commit -m "KEPT: [Hypothesis Name] - ROI: [Value]"`
    * **If ROI stayed same or worsened**: Revert ONLY the code using `git checkout HEAD -- train.py`. Do NOT use `reset --hard`.
7.  **The Plateau Rule**: If `val_roi` does not improve for 5 consecutive runs, pivot to a major strategy in `research_backlog.md`.

**NEVER STOP**: Continue iterating indefinitely until manually interrupted.