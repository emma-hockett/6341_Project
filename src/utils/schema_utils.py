# src/utils/schema_utils.py

def get_columns_by_attribute(cfg_schema: dict, attr: str, value) -> list[str]:
    """
    Return column names where spec[attr] == value (strict equality).
    Works for booleans and strings as-is.
    """
    cols = cfg_schema.get("columns", {}) or {}
    return [
        name for name, spec in cols.items()
        if isinstance(spec, dict) and spec.get(attr) == value
    ]