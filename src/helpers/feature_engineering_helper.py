# src/helpers/eda_helpers.py
import pandas as pd
from typing import Dict, List

import pandas as pd
from typing import Dict, List, Tuple

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
        df[col_name] = mask.astype("boolean[pyarrow]")
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