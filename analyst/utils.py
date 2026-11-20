import pandas as pd
import re

_RAW_CONTRIB = re.compile(r"внесок [^()]*\([^()]*від загальної зміни\); ")

def hide_raw_contribution(text: str) -> str:
    return _RAW_CONTRIB.sub("", text)

def looks_like_id(name: str, s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return False
    non_null = int(s.notna().sum())
    if not non_null:
        return False
    return name.lower().endswith("_id") or s.nunique(dropna=True) > 0.5 * non_null
