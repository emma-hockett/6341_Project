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


def apply_exempt_split(df: pd.DataFrame, exempt_cols: list[str]) -> list[str]:
    """
    For each column in exempt_cols:
      - Create a new column <col>_exempt (boolean, True where value == 'exempt', case-insensitive)
      - Replace those 'exempt' values in the base column with <NA>

    Returns a list of the new flag column names.
    """
    created = []

    for col in exempt_cols:
        if col not in df.columns:
            continue

        s = df[col].astype("string")
        mask = s.str.strip().str.casefold() == "exempt"

        new_col = f"{col}_exempt"
        df[new_col] = mask.astype("boolean")
        df.loc[mask, col] = pd.NA
        created.append(new_col)

    return created