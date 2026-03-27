autoresearch: MLB Kalshi Betting Edition (Autonomous + Cloud-Resilient)

Goal

Run an autonomous experiment loop to maximize validation ROI for Kalshi MLB betting while preserving progress in remote git so session expiry does not lose work.

Key rule

Whenever a run produces a new best `val_roi`, immediately checkpoint with:

`scripts/checkpoint_best.sh "<short description>"`

This script is responsible for:
- detecting whether the most recent `results.tsv` row is a new ROI best
- committing the relevant files
- pushing to the current branch

Do not manually re-implement that logic in ad hoc commands during the loop.

Files

- `master_mlb.csv`: dataset
- `train.py`: ONLY model/feature/betting logic to iterate on
- `results.tsv`: per-run numeric history
- `experiment_log.md`: qualitative run notes
- `scripts/checkpoint_best.sh`: checkpoint-on-improvement script

Hard constraints

- Do not modify `evaluate()` in `train.py`.
- Do not install new dependencies not already present.
- Keep each training run bounded by the existing time budget in `train.py`.
- Never use `git reset --hard HEAD~1` in this workflow.

One-time initialization (start of autonomous session)

1) Verify branch and sync:
   - `git rev-parse --abbrev-ref HEAD`
   - `git fetch origin <current-branch>`
   - `git pull origin <current-branch>`

2) Ensure tracking files exist:
   - If `results.tsv` missing, create with header:
     `commit	val_roi	val_brier	status	description`
   - If `experiment_log.md` missing, create it.

3) Ensure checkpoint script is executable:
   - `chmod +x scripts/checkpoint_best.sh`

Main autonomous loop (repeat until manually interrupted)

1) Read recent history
   - Read last entries from `results.tsv` and `experiment_log.md`.
   - Pick ONE concrete next change to `train.py`.

2) Implement one experiment
   - Edit `train.py` (feature engineering, architecture, optimization, or betting logic).
   - Keep changes focused and attributable.

3) Run training
   - `uv run train.py > run.log 2>&1`
   - If run crashes, inspect log, fix, and retry.

4) Verify metrics logged
   - Confirm latest row appended to `results.tsv`.
   - Ensure `val_roi` and `val_brier` are present.

5) Update qualitative log
   - Append a short entry to `experiment_log.md` with:
     - What changed
     - Observed result
     - Next hypothesis

6) Attempt checkpoint push (mandatory each run)
   - Run:
     - `scripts/checkpoint_best.sh "<what changed in this run>"`
   - If latest run is not a new best, script exits without commit/push.
   - If latest run is a new best, script commits and pushes automatically.

7) Continue loop
   - Move to next single-change experiment.
   - If plateauing, prioritize new feature interactions and market-decorrelation ideas.

Failure handling

- If training fails: debug from `run.log`, fix `train.py`, rerun.
- If push fails due to transient network issue: retry push with backoff.
- If checkpoint script reports no staged changes on an improved result, ensure `train.py`, `results.tsv`, and `experiment_log.md` were actually updated before rerunning the script.

Success criteria

- Primary: maximize `val_roi`.
- Secondary: maintain/improve `val_brier`.
- Tie-breaker: prefer simpler models/settings when ROI is similar.

Termination behavior

When the loop is manually stopped, the branch should already contain pushed commits for every newly discovered best pipeline, preserving progress across cloud/session interruptions.