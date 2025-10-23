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