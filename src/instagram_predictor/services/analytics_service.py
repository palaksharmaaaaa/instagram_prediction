from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd

from ..data import load_dataset
from ..nlp import parse_query
from ..models import predict_batch
from ..schemas import ParsedQuery


def apply_query_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Applies extracted query filters to the DataFrame.
    """
    if df.empty or not filters:
        return df.copy()

    result = df.copy()

    for col, cond in filters.items():
        if col.startswith("_"):
            continue

        # Category
        if col == "category":
            cat_query = str(cond).lower()
            acc_cats = result["account_categories_str"].astype(str).str.lower() if "account_categories_str" in result.columns else pd.Series("", index=result.index)
            mask = (
                result["category"].astype(str).str.lower().str.contains(cat_query, regex=False, na=False)
                | result["account_category"].astype(str).str.lower().str.contains(cat_query, regex=False, na=False)
                | acc_cats.str.contains(cat_query, regex=False, na=False)
            )
            result = result[mask]
            continue

        # Categorization / Content Style
        if col == "categorization":
            style_query = str(cond).lower()
            mask = result["categorization"].astype(str).str.lower().str.contains(style_query, regex=False, na=False)
            result = result[mask]
            continue

        # Media Type
        if col == "media_type":
            m_query = str(cond).lower()
            result = result[result["media_type"].astype(str).str.lower() == m_query]
            continue

        # Country
        if col == "country":
            c_query = str(cond).upper()
            mask = (
                (result["country"].astype(str).str.upper() == c_query)
                | (result["top_country"].astype(str).str.upper() == c_query)
            )
            result = result[mask]
            continue

        # Username / Handle
        if col == "username":
            u_query = str(cond).lower().lstrip("@")
            mask = (
                result["username"].astype(str).str.lower().str.contains(u_query, regex=False, na=False)
                | result["full_name"].astype(str).str.lower().str.contains(u_query, regex=False, na=False)
            )
            result = result[mask]
            continue

        # Numeric bounds
        if col in result.columns and isinstance(cond, dict):
            series = pd.to_numeric(result[col], errors="coerce")
            op = cond.get("operator")

            if op == "between":
                min_v = cond.get("min", -np.inf)
                max_v = cond.get("max", np.inf)
                result = result[(series >= min_v) & (series <= max_v)]
            elif op == ">":
                result = result[series > cond["value"]]
            elif op == ">=":
                result = result[series >= cond["value"]]
            elif op == "<":
                result = result[series < cond["value"]]
            elif op == "<=":
                result = result[series <= cond["value"]]
            elif op == "==":
                result = result[series == cond["value"]]

    # Top-N sorting
    if "_top_n" in filters:
        spec = filters["_top_n"]
        sort_by = spec.get("sort_by", "total_followers")
        ascending = spec.get("ascending", False)
        limit = spec.get("limit", 10)

        if sort_by in result.columns:
            result = result.sort_values(by=sort_by, ascending=ascending)
        result = result.head(limit)

    return result.reset_index(drop=True)


def run_analytics_pipeline(
    prompt: str,
    df: Optional[pd.DataFrame] = None
) -> Tuple[ParsedQuery, pd.DataFrame]:
    """
    End-to-end service coordinating query parsing, filtering, and predictive modeling.
    """
    parsed_query = parse_query(prompt)
    if df is None:
        df = load_dataset()

    filtered_df = apply_query_filters(df, parsed_query.filters)

    if parsed_query.predict_reach or parsed_query.predict_impressions:
        if not filtered_df.empty:
            filtered_df = predict_batch(
                filtered_df,
                predict_reach=parsed_query.predict_reach,
                predict_impressions=parsed_query.predict_impressions
            )

    return parsed_query, filtered_df
