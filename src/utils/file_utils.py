# src/utils/conf.py
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import os
import yaml
import pandas as pd

def _find_project_root(start: Path | None = None) -> Path:
    """
    Walk upwards to find a reasonable project root.
    Prefers an explicit env var; otherwise looks for common sentinels.
    """
    # 1) Allow override via env var for notebooks, tests, CI, etc.
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    # 2) Auto-detect from current location
    p = (start or Path.cwd()).resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists() or (parent / "configs").exists():
            return parent
    return p  # fallback

@lru_cache
def project_root() -> Path:
    return _find_project_root()

def config_path(name: str) -> Path:
    """
    Map 'paths' -> <root>/configs/paths.yaml, etc.
    """
    return project_root() / "configs" / f"{name}.yaml"

@lru_cache
def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}

@lru_cache
def load_config(name: str) -> dict:
    """
    load_config("paths") -> dict from configs/paths.yaml
    load_config("clean") -> dict from configs/clean.yaml
    """
    return load_yaml(config_path(name))

def get_path(key: str) -> Path:
    """
    Resolve a key from paths.yaml into an absolute Path under project_root.
    Example: paths.yaml -> { hmda_raw: "data/interim/2024_combined_mlar_header.parquet" }
    """
    paths_cfg = load_config("paths")
    try:
        rel = Path(paths_cfg[key])
    except KeyError as e:
        raise KeyError(f"paths.yaml is missing key: {key!r}") from e
    return (project_root() / rel).resolve()


def save_parquet(df, key: str) -> Path:
    """
    Save a DataFrame as a Parquet file using the PyArrow engine.
    The output path is resolved from configs/paths.yaml via the given key.

    Example:
        save_parquet(my_df, "hmda_2024_typed")

    Args:
        df: Pandas DataFrame to save
        key: Key in paths.yaml that maps to the output file path
    """
    output_path = get_path(key)

    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"Saved to {output_path}")

    return output_path


def load_parquet(key: str) -> pd.DataFrame:
    """
    Load a Parquet file using the configured path key and PyArrow backend.

    Example:
        df = load_parquet("hmda_raw")

    Args:
        key: Key in paths.yaml that maps to the Parquet file path

    Returns:
        Pandas DataFrame
    """

    input_path = get_path(key)
    print(f"Loading dataset from {input_path}")
    return pd.read_parquet(input_path, dtype_backend="pyarrow")