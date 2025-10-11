# src/helpers/clean_helpers.py
import pandas as pd
from pandas.api.types import is_string_dtype, is_object_dtype
import src.utils.file_utils as fu


def quick_null_like_check(df: pd.DataFrame, null_like, sample_frac: float = 0.01, random_state: int = 0) -> pd.Series:
    tokens = [str(t).strip().casefold() for t in (null_like or []) if t is not None]
    tokens = list(dict.fromkeys(tokens))
    if not tokens:
        return pd.Series(dtype="float64", name="null_like_fraction")

    hits = {}
    for col in df.columns:
        s = df[col]
        if not is_string_dtype(s.dtype):
            continue

        ss = s.sample(frac=sample_frac, random_state=random_state) if 0 < sample_frac < 1.0 else s
        norm = ss.str.casefold()

        nn = norm[norm.notna()]
        if nn.empty:
            continue

        match = nn.isin(tokens)

        if match.any():
            hits[col] = float(match.sum()) / float(len(ss))

    return pd.Series(hits, name="null_like_fraction").sort_values(ascending=False)


def apply_exempt_split(df: pd.DataFrame, exempt_cols: list[str]) -> list[str]:
    """
    For each column in exempt_cols:
      - create <col>_exempt (nullable boolean) True where value == 'exempt' (case-insensitive, after strip)
      - set base column cells to <NA> where exempt
    Returns list of created flag column names.
    """
    created = []
    for col in exempt_cols:
        if col not in df.columns:
            continue
        s = df[col]

        norm = s.str.strip().str.casefold()
        mask = norm.eq("exempt")

        flag_col = f"{col}_exempt"
        df[flag_col] = mask.astype("boolean[pyarrow]")

        # Fill the cells that had been exempt with pd.NA
        if mask.any():
            df.loc[mask, col] = pd.NA

        created.append(flag_col)

    return created

def convert_by_schema(df: pd.DataFrame, cfg_schema: dict, *, in_place: bool = True) -> pd.DataFrame:
    """
    Convert columns to the exact dtypes specified in schema.yaml
    - If column is string, uses vectorized string ops, then parses
    - Emits a warning count of new nulls created by coercion
    """
    out = df if in_place else df.copy()
    cols_spec = (cfg_schema.get("columns") or {})

    for col, spec in cols_spec.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("role") == "drop":
            continue
        if col not in out.columns:
            continue

        target = str(spec.get("dtype", ""))
        if not target:
            continue

        s = out[col]
        before_na = s.isna().sum()

        try:
            if target.startswith(("Int", "Float")):
                out[col] = pd.to_numeric(s, errors="coerce").astype(target)

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
    df_out["approved_flag"] = df_out["action_taken"].isin(approved).astype("bool[pyarrow]")

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