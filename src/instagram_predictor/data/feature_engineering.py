"""
Feature construction. Features are deterministic transformations of observed values only.
There are no default fills: a missing optional input stays NaN and is handled inside the
fitted model (median / native-NaN), and the forecaster reports it as imputed.
"""

from typing import Dict, List
import numpy as np
import pandas as pd

from .loader import OPTIONAL_NUMERIC_COLUMNS, OPTIONAL_BOOL_COLUMNS

# An optional column is used for training only if enough rows carry it and it actually varies.
MIN_OPTIONAL_COVERAGE = 0.80
CATEGORICAL_FEATURES = ["platform", "media_type", "posted_day_of_week"]
ALWAYS_NUMERIC = ["log_followers", "hour_sin", "hour_cos"]


def select_feature_spec(posts: pd.DataFrame) -> Dict:
    """Decides, from the data itself, which optional features are usable."""
    used: List[str] = []
    dropped: Dict[str, str] = {}
    for col in OPTIONAL_NUMERIC_COLUMNS + OPTIONAL_BOOL_COLUMNS:
        if col not in posts.columns:
            dropped[col] = "column not present"
            continue
        cov = float(posts[col].notna().mean())
        if cov < MIN_OPTIONAL_COVERAGE:
            dropped[col] = f"only {cov:.0%} of rows have it (need {MIN_OPTIONAL_COVERAGE:.0%})"
        elif posts[col].nunique(dropna=True) < 2:
            dropped[col] = "constant in the data"
        else:
            used.append(col)
    return {
        "numeric": ALWAYS_NUMERIC + used,
        "categorical": list(CATEGORICAL_FEATURES),
        "optional_used": used,
        "optional_dropped": dropped,
    }


def build_features(rows: pd.DataFrame, spec: Dict) -> pd.DataFrame:
    """Builds the model matrix. Missing optional values remain NaN."""
    out = pd.DataFrame(index=rows.index)
    out["log_followers"] = np.log(rows["total_followers"].astype(float))
    angle = 2.0 * np.pi * rows["posted_hour_of_day"].astype(float) / 24.0
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)
    for col in spec["optional_used"]:
        out[col] = pd.to_numeric(rows[col], errors="coerce") if col in rows.columns else np.nan
    for col in spec["categorical"]:
        out[col] = rows[col].astype(str)
    return out[spec["numeric"] + spec["categorical"]]
