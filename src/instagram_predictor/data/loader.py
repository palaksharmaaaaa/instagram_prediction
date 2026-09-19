import ast
import json
import threading
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from ..config import settings
from .generator import ensure_dataset_exists
from .feature_engineering import compute_derived_metrics

_LOADER_LOCK = threading.Lock()
_DATA_CACHE: Optional[pd.DataFrame] = None


def parse_list_field(val: Any) -> List[str]:
    """
    Robustly deserializes a multi-label list field into a genuine Python list of strings.
    Handles:
    - Existing list/tuple/iterable of strings/objects
    - String representations of lists: "['a', 'b']", '["a", "b"]'
    - Nested stringified lists: "['[\"a\"]']" or "['[\\'a\\']']"
    - Comma-separated strings: "a, b"
    - Single string: "a"
    - Missing / NaN / None: []
    """
    if val is None or (isinstance(val, float) and np.isnan(val)) or val == "":
        return []

    if isinstance(val, (list, tuple, np.ndarray, set)):
        items = list(val)
    elif isinstance(val, str):
        val_clean = val.strip()
        if not val_clean:
            return []
        if (val_clean.startswith("[") and val_clean.endswith("]")) or (
            val_clean.startswith("(") and val_clean.endswith(")")
        ):
            try:
                parsed = ast.literal_eval(val_clean)
                if isinstance(parsed, (list, tuple, set)):
                    items = list(parsed)
                else:
                    items = [parsed]
            except (ValueError, SyntaxError):
                try:
                    parsed = json.loads(val_clean)
                    if isinstance(parsed, (list, tuple, set)):
                        items = list(parsed)
                    else:
                        items = [parsed]
                except Exception:
                    inner = val_clean[1:-1].strip()
                    if not inner:
                        return []
                    items = [s.strip().strip("'\"") for s in inner.split(",") if s.strip()]
        elif "," in val_clean:
            items = [s.strip().strip("'\"") for s in val_clean.split(",") if s.strip()]
        else:
            items = [val_clean]
    else:
        items = [str(val)]

    result: List[str] = []
    for item in items:
        if item is None or (isinstance(item, float) and np.isnan(item)):
            continue
        item_str = str(item).strip()
        if item_str.startswith("[") and item_str.endswith("]"):
            nested = parse_list_field(item_str)
            result.extend(nested)
        else:
            cleaned = item_str.strip("'\"")
            if cleaned:
                result.append(cleaned)
    return result


def load_dataset(reload: bool = False) -> pd.DataFrame:
    """
    Loads and caches the multi-dimensional Instagram dataset with computed derived metrics.
    Ensures multi-label list fields are cleanly deserialized into genuine Python lists.
    Thread-safe implementation protected by _LOADER_LOCK.
    """
    global _DATA_CACHE
    with _LOADER_LOCK:
        if _DATA_CACHE is None or reload:
            df = ensure_dataset_exists()
            for col in ["account_categories", "categorizations"]:
                if col in df.columns:
                    df[col] = df[col].apply(parse_list_field)
            df = compute_derived_metrics(df)
            _DATA_CACHE = df
        return _DATA_CACHE.copy()


def clear_loader_cache() -> None:
    """
    Clears the in-memory dataset cache in a thread-safe manner.
    """
    global _DATA_CACHE
    with _LOADER_LOCK:
        _DATA_CACHE = None
