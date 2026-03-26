# my-mlb-autoresearch

Autonomous experiment loop for optimizing an MLB game-outcome model targeting **Kalshi moneyline markets**.

An AI agent reads `program.md`, modifies `train.py`, trains for a fixed 5-minute budget, evaluates on ROI and Brier score, logs the result, and keeps or discards the change — repeating until interrupted.

## How it works

Three files drive everything:

| File | Role |
|------|------|
| `train.py` | **Agent modifies this.** Feature engineering pipeline, model architecture (default: MLP), training loop, Kalshi ROI/Brier evaluation, and betting logic. |
| `program.md` | **Human modifies this.** Instructions and context for the autonomous agent. |

### Metrics

- **`val_roi`** (primary) — Return on Investment on the 2024–present validation set using fractional Kelly sizing against Kalshi-derived implied probabilities.
- **`val_brier`** (secondary) — Brier score; ensures probabilities are well-calibrated.

### Data split

- **Train:** all games before 2024-01-01
- **Val:** 2024-01-01 onward

### Kalshi odds model

Kalshi MLB contracts are priced as implied probabilities (0–1). The pipeline derives these from `close_home_ml` via standard devig:

- **Home bet:** decimal odds = `1 / market_implied_prob`
- **Away bet:** decimal odds = `1 / (1 - market_implied_prob)`

Bets are placed on either side when `|model_prob − market_implied_prob| ≥ CONFIDENCE_THRESHOLD`, sized by fractional Kelly.

## Quick start

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Run a single experiment
uv run train.py
```

## Running the autonomous agent

Point Claude Code (or any capable agent) at this repo and prompt:

```
Let's start a new experiment. Find your instructions in program.md.
```

The agent will loop — editing `train.py`, training, evaluating, logging to `results.tsv` and `experiment_log.md` — until you interrupt it.

## Project structure

```
train.py            — model + feature engineering (agent edits this)
program.md          — agent instructions (human edits this)
master_mlb.csv      — core dataset
analysis.ipynb      — visualize results.tsv (val_roi / val_brier over time)
pyproject.toml      — dependencies
```

## License

MIT
