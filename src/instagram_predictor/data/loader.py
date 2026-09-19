import pandas as pd
from typing import Optional
from ..config import settings
from .generator import ensure_dataset_exists
from .feature_engineering import compute_derived_metrics

_DATA_CACHE: Optional[pd.DataFrame] = None


def load_dataset(reload: bool = False) -> pd.DataFrame:
    """
    Loads and caches the multi-dimensional Instagram dataset with computed derived metrics.
    """
    global _DATA_CACHE
    if _DATA_CACHE is None or reload:
        df = ensure_dataset_exists()
        df = compute_derived_metrics(df)
        _DATA_CACHE = df
    return _DATA_CACHE.copy()
