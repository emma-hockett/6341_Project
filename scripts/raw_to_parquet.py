#!/usr/bin/env python3
"""
Convert HMDA 2024 pipe-delimited (.txt) dataset to Parquet format.

Usage:
    python scripts/raw_to_parquet.py --input data/raw/2024_combined_mlar_header.txt --output data/interim/2024_combined_mlar_header.parquet
"""

import argparse
import pandas as pd
from pathlib import Path


def main(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"Reading raw file: {input_path}")
    df = pd.read_csv(
        input_path,
        sep="|",
        dtype=str,              # read everything as string initially
        na_values=["", "NA", "NULL", None],
        low_memory=False,
    )

    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")

    # Save to Parquet (snappy compression)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing Parquet file to: {output_path}")
    df.to_parquet(output_path, index=False, compression="snappy")

    print("Conversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HMDA TXT to Parquet")
    parser.add_argument("--input", required=True, help="Path to input .txt file")
    parser.add_argument("--output", required=True, help="Path to output .parquet file")
    args = parser.parse_args()

    main(args.input, args.output)