"""
Legacy Predictor Module (v1 Prototype Architecture)
===================================================
DEPRECATION NOTICE:
This module represents the v1 prototype architecture and is maintained solely for
backward compatibility with legacy scripts and tests (e.g., tests/test_end_to_end.py).

Production systems should use the enterprise-grade `instagram_predictor` package:
- Inference & Simulation: `instagram_predictor.models` (e.g. `predict_batch`, `simulate_post_performance`)
- Registry & Pipelines: `instagram_predictor.models.registry` (`get_reach_pipeline`, `get_impressions_pipeline`)
- Query Filtering: `instagram_predictor.services.analytics_service` (`apply_query_filters`)
"""

import os
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
REACH_MODEL_PATH = os.path.join(MODEL_DIR, "reach_model.pkl")
IMPRESSION_MODEL_PATH = os.path.join(MODEL_DIR, "impression_model.pkl")

NUMERIC_FEATURES = [
    "Followers",
    "Posts",
    "Likes Avg.",
    "Comments Avg.",
    "Views Avg.",
    "Boost Index",
    "Engagement Rate",
    "Engagement Rate (60 Days)",
    "Avg. 7 Day",
    "Avg. 14 Day",
    "Avg. 30 Day"
]

CATEGORICAL_FEATURES = [
    "Country",
    "Main topic",
    "Main video category"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# In-memory model cache
_MODEL_CACHE = {
    "reach": None,
    "impression": None
}


def get_reach_model():
    if _MODEL_CACHE["reach"] is None:
        if not os.path.exists(REACH_MODEL_PATH):
            raise FileNotFoundError(f"Reach model not found at {REACH_MODEL_PATH}. Please run train.py first.")
        _MODEL_CACHE["reach"] = joblib.load(REACH_MODEL_PATH)
    return _MODEL_CACHE["reach"]


def get_impression_model():
    if _MODEL_CACHE["impression"] is None:
        if not os.path.exists(IMPRESSION_MODEL_PATH):
            raise FileNotFoundError(f"Impression model not found at {IMPRESSION_MODEL_PATH}. Please run train.py first.")
        _MODEL_CACHE["impression"] = joblib.load(IMPRESSION_MODEL_PATH)
    return _MODEL_CACHE["impression"]


def apply_filter(df, filters):
    """
    Applies extracted filters (category, country, username, numeric ranges, top-N) to dataframe.
    """
    if df.empty or not filters:
        return df.copy()

    result = df.copy()

    for column, condition in filters.items():
        if column.startswith("_"):
            continue  # Post-filter modifiers like _top_n handled after

        # 1. Category / Topic Filter
        if column == "category":
            cat_query = str(condition).lower()
            mask = (
                result["Main topic"].astype(str).str.lower().str.contains(cat_query, regex=False, na=False)
                | result["Main video category"].astype(str).str.lower().str.contains(cat_query, regex=False, na=False)
            )
            result = result[mask]
            continue

        # 2. Country Filter
        if column == "Country":
            c_code = str(condition).upper()
            result = result[result["Country"].astype(str).str.upper() == c_code]
            continue

        # 3. Username / Handle Filter
        if column == "Username":
            u_query = str(condition).lower().lstrip("@")
            mask = (
                result["Username"].astype(str).str.lower().str.contains(u_query, regex=False, na=False)
                | result["Channel Name"].astype(str).str.lower().str.contains(u_query, regex=False, na=False)
            )
            result = result[mask]
            continue

        # 4. Generic Numeric Conditions
        if column in result.columns and isinstance(condition, dict):
            col_series = pd.to_numeric(result[column], errors="coerce")
            op = condition.get("operator")

            if op == "between":
                min_v = condition.get("min", -np.inf)
                max_v = condition.get("max", np.inf)
                result = result[(col_series >= min_v) & (col_series <= max_v)]
            elif op == ">":
                result = result[col_series > condition["value"]]
            elif op == ">=":
                result = result[col_series >= condition["value"]]
            elif op == "<":
                result = result[col_series < condition["value"]]
            elif op == "<=":
                result = result[col_series <= condition["value"]]
            elif op == "==":
                result = result[col_series == condition["value"]]

    # 5. Handle Top-N sorting / limiting
    if "_top_n" in filters:
        top_spec = filters["_top_n"]
        sort_col = top_spec.get("sort_by", "Followers")
        ascending = top_spec.get("ascending", False)
        limit = top_spec.get("limit", 10)

        if sort_col in result.columns:
            result = result.sort_values(by=sort_col, ascending=ascending)
        result = result.head(limit)

    return result.reset_index(drop=True)


def predict(df, predict_reach=True, predict_impressions=True):
    """
    Runs model inference on input dataframe for requested targets.
    """
    if df.empty:
        return df.copy()

    df = df.copy()
    X = df[FEATURES]

    if predict_reach:
        reach_model = get_reach_model()
        df["Predicted Reach"] = np.round(reach_model.predict(X)).astype(int)

    if predict_impressions:
        impression_model = get_impression_model()
        df["Predicted Impressions"] = np.round(impression_model.predict(X)).astype(int)

    return df