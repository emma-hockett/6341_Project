# src/utils/conf.py
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import os

import yaml
import pandas as pd

def _find_project_root(start: Path | None = None) -> Path:
    # Allow override via env var for notebooks, tests, CI, etc.
    env_root = os.getenv("PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    # Or auto-detect from current location
    p = (start or Path.cwd()).resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists() or (parent / "configs").exists():
            return parent
    return p

@lru_cache
def project_root() -> Path:
    return _find_project_root()

def config_path(name: str) -> Path:
    return project_root() / "configs" / f"{name}.yaml"

@lru_cache
def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}

@lru_cache
def load_config(name: str) -> dict:
    return load_yaml(config_path(name))

def get_path(key: str) -> Path:
    paths_cfg = load_config("paths")
    try:
        rel = Path(paths_cfg[key])
    except KeyError as e:
        raise KeyError(f"paths.yaml is missing key: {key!r}") from e
    return (project_root() / rel).resolve()


def save_parquet(df, key: str) -> Path:
    output_path = get_path(key)

    df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")
    print(f"Saved to {output_path}")

    return output_path


def load_parquet(key: str) -> pd.DataFrame:
    input_path = get_path(key)
    print(f"Loading dataset from {input_path}")
    return pd.read_parquet(input_path, dtype_backend="pyarrow")