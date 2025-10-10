# src/utils/schema_utils.py

def get_columns_by_attribute(cfg_schema: dict, attr: str, value: str) -> list[str]:
    """
    Generic schema helper: return column names where a given attribute (role, type, dtype, etc.)
    matches a value (case-insensitive).

    Example:
        get_columns_by_attribute(cfg_schema, "role", "drop")
        get_columns_by_attribute(cfg_schema, "type", "numeric")
        get_columns_by_attribute(cfg_schema, "dtype", "Int16")
    """
    if not cfg_schema:
        return []
    columns = cfg_schema.get("columns", {}) or {}
    attr_lower = str(attr).lower()
    value_lower = str(value).lower()

    matches = []
    for name, spec in columns.items():
        if not isinstance(spec, dict):
            continue
        val = spec.get(attr_lower) or spec.get(attr)
        if isinstance(val, str) and val.lower() == value_lower:
            matches.append(name)
    return matches