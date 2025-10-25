# src/helpers/eda_helpers.py
import pandas as pd
from typing import Dict, List, Optional
from sklearn.model_selection import train_test_split
import src.utils.file_utils as fu

def generate_multi_hot_features(df: pd.DataFrame, cfg: Dict, prefix: str) -> pd.DataFrame:
    # Derive the code map key from the prefix
    map_key = f"{prefix}code_map"
    code_map: Dict[str, List[int]] = cfg["feature_engineering"][map_key]

    # Build the slot column names
    slot_cols = [f"{prefix}{i}" for i in range(1, 6)]

    # Compute multi-hot columns
    slots = df[slot_cols]
    out_cols: List[str] = []

    for label, codes in code_map.items():
        col_name = f"{prefix}{label}"
        mask = slots.isin(codes).any(axis=1)
        df[col_name] = mask.astype("int8[pyarrow]").fillna(-1)
        out_cols.append(col_name)

    return df


def impute_income(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing income values using median (income / loan_amount) ratio stratified by loan_type.
    """
    # Compute ratio
    df["income_to_loan_ratio"] = df["income"] / df["loan_amount"]

    # Median ratio by loan_type
    ratio_medians = (
        df.loc[df["income_to_loan_ratio"].notna()]
            .groupby("loan_type")["income_to_loan_ratio"]
            .median()
    )

    # Map median ratio back to rows
    df["median_ratio"] = df["loan_type"].map(ratio_medians)

    # Impute missing income
    missing_mask = df["income"].isna() & df["loan_amount"].notna()
    df.loc[missing_mask, "income"] = (
        df.loc[missing_mask, "loan_amount"] * df.loc[missing_mask, "median_ratio"]
    )

    # Drop temporary columns
    df = df.drop(columns=["income_to_loan_ratio", "median_ratio"])
    return df


def impute_property_value(df: pd.DataFrame) -> pd.DataFrame:
    # Compute LTV ratio
    df["loan_to_value_ratio"] = df["loan_amount"] / df["property_value"]

    # Calculate median LTV ratio by loan_type
    ltv_medians = (
        df.loc[df["loan_to_value_ratio"].notna()]
            .groupby("loan_type")["loan_to_value_ratio"]
            .median()
    )

    # Map median ratios back to each row
    df["median_ltv_ratio"] = df["loan_type"].map(ltv_medians)

    # Impute property_value where missing
    mask = df["property_value"].isna() & df["loan_amount"].notna()
    df.loc[mask, "property_value"] = (
        df.loc[mask, "loan_amount"] / df.loc[mask, "median_ltv_ratio"]
    )

    # Drop temporary columns
    df = df.drop(columns=["loan_to_value_ratio", "median_ltv_ratio"])
    return df


def create_train_test_splits(df: pd.DataFrame):
    # Stratified 85/15 split
    train_idx, test_idx = train_test_split(
        df.index,
        stratify=df["denied_flag"],
        test_size=0.15,
        random_state=42
    )

    # Convert to DataFrames for consistent output
    train_df = pd.DataFrame(train_idx, columns=["index"])
    test_df = pd.DataFrame(test_idx, columns=["index"])

    # Output directory
    train_output_path = fu.get_path("train_index")
    test_output_path = fu.get_path("test_index")

    # Save index lists
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    print(f"Train indices saved to: {train_output_path / 'train_index.csv'}")
    print(f"Test indices saved to:  {test_output_path / 'test_index.csv'}")


def one_hot_encode_columns(df: pd.DataFrame, cfg_feature_engineering: dict) -> pd.DataFrame:
    fe_cfg = cfg_feature_engineering["feature_engineering"]
    df_out = df.copy()

    for col_key, code_map in fe_cfg.items():
        if not col_key.endswith("_ohe_map"):
            continue

        base_col = col_key.replace("_ohe_map", "")
        if base_col not in df_out.columns:
            continue

        s = df_out[base_col]
        s_int = s.astype("Int64") if pd.api.types.is_integer_dtype(s) else s

        # missing mask
        na_mask = s.isna() | (s == -1)
        df_out[f"{base_col}_NA"] = na_mask.astype("int8[pyarrow]")

        # track mapped codes
        mapped_codes = set()

        # create columns for all defined mappings
        for label, codes in code_map.items():
            mask = s_int.isin(codes) & ~na_mask
            df_out[f"{base_col}_{label}"] = mask.astype("int8[pyarrow]")
            mapped_codes.update(codes)

        # catch any unmapped, non-missing codes
        other_mask = ~s_int.isin(mapped_codes) & ~na_mask
        df_out[f"{base_col}_other"] = other_mask.astype("int8[pyarrow]")

        # drop the original column
        df_out.drop(columns=[base_col], inplace=True)

    return df_out