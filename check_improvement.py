import pandas as pd
df = pd.read_csv('results.tsv', sep='\t')
if len(df) < 2: exit(0)
best_roi = df.iloc[:-1]['val_roi'].max()
current_roi = df.iloc[-1]['val_roi']
print(f"Improvement: {current_roi - best_roi:.4f}")