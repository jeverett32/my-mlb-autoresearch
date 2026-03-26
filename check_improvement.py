import pandas as pd
import sys

try:
    df = pd.read_csv('results.tsv', sep='\t')
    if len(df) < 2:
        print("Initial baseline established.")
        sys.exit(0)
    
    # Filter for successful runs only
    valid_runs = df[df['status'] == 'ok'].copy()
    if len(valid_runs) < 2:
        sys.exit(0)

    best_roi = valid_runs.iloc[:-1]['val_roi'].max()
    current_roi = valid_runs.iloc[-1]['val_roi']
    diff = current_roi - best_roi
    
    print(f"Current ROI: {current_roi:.4f} | Previous Best: {best_roi:.4f}")
    print(f"Improvement: {diff:.4f}")
except Exception as e:
    print(f"Metrics not ready: {e}")