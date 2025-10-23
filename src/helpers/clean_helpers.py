# src/helpers/clean_helpers.py
import pandas as pd
from pandas.api.types import is_string_dtype
import src.utils.file_utils as fu


def null_like_check(df: pd.DataFrame, null_like_values) -> pd.Series:
    """
    For each column in df:
      - verify it is of type string
      - force the string values to lowercase
      - check if any column values are null-like
      - calculate the proportion of column values that are null-like, if any
    Returns pd.Series with null-like value proportions greater than zero
    """

    # Parse list of values representing null values
    null_like_tokens = [str(t).strip().casefold() for t in (null_like_values or []) if t is not None]
    null_like_tokens = list(dict.fromkeys(null_like_tokens))
    if not null_like_tokens:
        return pd.Series(dtype="float64", name="null_like_fraction")

    columns_with_null_like = {}

    # We want to iterate through all columns for null-like values
    for col in df.columns:
        s = df[col]
        if not is_string_dtype(s.dtype): # We're only checking stringy values
            continue

        # For efficiency, take a sample of the column's values. We're looking for patterns of token use.
        s_sample = s.sample(frac=0.01, random_state=0)

        # We force everything to lowercase so that we don't have to test variations based on case
        s_sample_lowercase = s_sample.str.casefold()
        non_null_s = s_sample_lowercase[s_sample_lowercase.notna()]
        if non_null_s.empty:
            continue

        # If the value is in our list of null-like strings, we have a match
        match = non_null_s.isin(null_like_tokens)

        # If the column has any matches, calculate the proportion
        if match.any():
            columns_with_null_like[col] = float(match.sum()) / float(len(s_sample))

    return pd.Series(columns_with_null_like, name="null_like_fraction").sort_values(ascending=False)


def apply_exempt_split(df: pd.DataFrame, exempt_cols: list[str]) -> list[str]:
    """
    For each column in exempt_cols:
      - create <col>_exempt (nullable boolean) True where value == 'exempt' (case-insensitive, after strip)
      - set base column cells to <NA> where exempt
    Returns list of created flag column names.
    """
    created = [] # The new _exempt columns created
    for col in exempt_cols:
        if col not in df.columns:
            continue
        s = df[col]

        # Find the values within the column with a value of exempt
        non_null_s = s.str.strip().str.casefold()
        mask = non_null_s.isin(["exempt", "1111"])

        flag_col = f"{col}_exempt"
        df[flag_col] = mask.astype("boolean[pyarrow]")

        # Fill the cells that had been exempt with pd.NA
        if mask.any():
            df.loc[mask, col] = pd.NA

        created.append(flag_col)

    return created

def convert_by_schema(df: pd.DataFrame, cfg_schema: dict) -> pd.DataFrame:
    """
    Convert columns to the exact dtypes specified in schema.yaml
    - If column is string, uses vectorized string ops, then parses
    - Emits a warning count of new nulls created by coercion
    """
    out = df
    cols_spec = (cfg_schema.get("columns") or {})

    for col, spec in cols_spec.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("role") == "drop": # Skip the dropped columns
            continue
        if col not in out.columns: # Don't convert anything we don't have a definition for
            continue

        target = str(spec.get("dtype", "")) # Get the to-be dtype
        if not target:
            continue

        s = out[col]
        before_na = s.isna().sum() # Used to make sure we aren't accidentally creating null values during conversion

        try:
            # In this dataset, everything is string by default.
            if target.startswith(("Int", "Float")):
                out[col] = pd.to_numeric(s, errors="coerce").astype(target)
            elif target.startswith("Bool"):
                out[col] = s.map({"1": True, "2": False})
                out[col] = out[col].astype("boolean")

            # Report new nulls introduced by coercion
            after_na = out[col].isna().sum()
            n_fail = int(after_na - before_na)
            if n_fail > 0:
                print(f"{col}: {n_fail} values could not be converted to {target}")

        except Exception as e:
            print(f"{col}: failed to convert to {target} ({type(e).__name__}: {e})")

    return out


# This avoids converting to Python objects and performs the operation as PyArrow strings for faster execution
def strip_string_columns_inplace(df: pd.DataFrame) -> None:
    for col in df.columns:
        s = df[col]
        if is_string_dtype(s.dtype):
            df[col] = s.str.strip()


def to_pandas_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> None:
    """
    In-place: turn columns into pandas 'category'.
    """
    for col in cat_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].astype("category")


def apply_action_taken_flag(df: pd.DataFrame, cfg_clean: dict) -> pd.DataFrame:
    """
    Create binary target variable 'approved_flag' from action_taken codes.
    Reads approved/denied/exclude codes from clean.yaml.
    Returns filtered DataFrame with approved_flag (Int8: 1=approved, 0=denied).
    """
    action_cfg = cfg_clean["clean"]["action_taken"]
    approved = set(action_cfg["approved"])
    denied = set(action_cfg["denied"])
    valid = approved | denied

    # Select only the observations that have codes matching approved or denied actions
    mask = df["action_taken"].isin(valid)
    df_out = df.loc[mask].copy()

    # We don't overwrite action_taken in case we want to further analyze the data later
    df_out["denied_flag"] = df_out["action_taken"].isin(denied).astype("bool[pyarrow]")

    return df_out


def generate_schema_summary(df: pd.DataFrame, cfg_schema: dict, path_key: str = "schema_summary") -> pd.DataFrame:
    """
    Build column-level metadata for ALL columns currently in df (including derived ones like *_exempt, approved_flag).
    Skips columns with role: drop (per schema). Writes CSV to the path resolved by `path_key` in paths.yaml.
    Columns: column_name, data_type, missing_pct, unique_count, sample_value, notes
    """
    cols_spec = (cfg_schema.get("columns") or {})
    rows = []
    n_rows = len(df)

    for col in df.columns:
        spec = cols_spec.get(col, {}) if isinstance(cols_spec, dict) else {}
        if spec.get("role") == "drop":
            continue

        s = df[col]
        data_type = str(s.dtype)
        missing_pct = float(s.isna().mean() * 100.0) if n_rows else 0.0
        unique_count = int(s.nunique(dropna=True))

        # sample value from first non-null (stringify to be CSV-safe)
        if unique_count > 0:
            idx = s.first_valid_index()
            sample_value = s.loc[idx] if idx is not None else None
        else:
            sample_value = None

        # normalize sample to a short string for CSV readability
        if pd.isna(sample_value):
            sample_value_str = ""
        else:
            sv = sample_value
            # shorten long strings
            sv = str(sv)
            sample_value_str = sv if len(sv) <= 120 else (sv[:117] + "...")

        rows.append({
            "column_name": col,
            "data_type": data_type,
            "missing_pct": round(missing_pct, 6),
            "unique_count": unique_count,
            "sample_value": sample_value_str,
            "notes": spec.get("notes", ""),
        })

    summary = pd.DataFrame(rows, columns=[
        "column_name", "data_type", "missing_pct", "unique_count", "sample_value", "notes"
    ])

    out_path = fu.get_path(path_key)
    summary.to_csv(out_path, index=False)
    print(f"Wrote schema summary to {out_path}  ({len(summary)} columns)")

    return summary