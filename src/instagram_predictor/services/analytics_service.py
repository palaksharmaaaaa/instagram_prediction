"""
Creator search over OBSERVED data. Nothing here predicts or estimates: every returned value is a
value from the data file, and every filter that could not be applied is reported by the parser.
"""

import operator
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from ..nlp import parse_query
from ..schemas import ParsedQuery

_OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> Tuple[pd.DataFrame, list]:
    """Applies parsed filters conjunctively. Returns (rows, notes). Missing values never match a numeric filter."""
    notes = []
    result = df.copy()
    for col, cond in filters.items():
        if col.startswith("_"):
            continue
        if col == "username":
            result = result[result["username"].astype(str).str.contains(str(cond), case=False, regex=False, na=False)]
        elif col == "category_contains":
            if "account_category" not in result.columns:
                notes.append("The data has no category column; category filter not applied.")
                continue
            result = result[result["account_category"].astype(str).str.contains(str(cond), case=False, regex=False, na=False)]
        elif isinstance(cond, list):
            if col not in result.columns:
                notes.append(f"The data has no '{col}' column; that filter was not applied.")
                continue
            series = pd.to_numeric(result[col], errors="coerce")
            mask = pd.Series(True, index=result.index)
            for c in cond:
                mask &= _OPS[c["operator"]](series, c["value"]).fillna(False)
            result = result[mask]
    top = filters.get("_top_n")
    if top:
        col = top["sort_by"]
        if col in result.columns:
            result = result.sort_values(col, ascending=top["ascending"], na_position="last").head(top["limit"])
        else:
            notes.append(f"Cannot rank by '{col}': column missing.")
    return result.reset_index(drop=True), notes


def run_creator_query(prompt: str, profiles: pd.DataFrame) -> Tuple[ParsedQuery, pd.DataFrame, list]:
    parsed = parse_query(prompt)
    rows, notes = apply_filters(profiles, parsed.filters)
    return parsed, rows, notes


def observed_post_summary(posts: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Per-creator descriptive statistics of the supplied posts (medians of observed values only)."""
    if posts is None or posts.empty:
        return None
    g = posts.groupby("username")
    out = g.agg(posts=("per_media_reach", "size"), median_reach=("per_media_reach", "median"),
                median_followers=("total_followers", "median")).reset_index()
    out["median_reach_per_follower"] = out["median_reach"] / out["median_followers"]
    return out
