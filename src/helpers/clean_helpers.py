# src/helpers/clean_helpers.py
import pandas as pd

def quick_null_like_check(df, null_like=None, sample_frac=0.01):
    hits = {}
    for col in df.columns:
        s = df[col].astype(str).str.lower()
        if sample_frac < 1.0:
            s = s.sample(frac=sample_frac, random_state=0)
        mask = s.isin(null_like)
        if mask.any():
            hits[col] = mask.sum() / len(s)
    return pd.Series(hits, name="null_like_fraction").sort_values(ascending=False)