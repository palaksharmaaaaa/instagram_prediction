import os
import pandas as pd
import numpy as np

NUMERIC_COLUMNS = [
    "Likes",
    "Likes Avg.",
    "Posts",
    "Followers",
    "Boost Index",
    "Comments Avg.",
    "Views Avg.",
    "Avg. 1 Day",
    "Avg. 3 Day",
    "Avg. 7 Day",
    "Avg. 14 Day",
    "Avg. 30 Day",
    "Engagement Rate",
    "Engagement Rate (60 Days)"
]

CATEGORICAL_COLUMNS = [
    "Country",
    "Main topic",
    "Main video category"
]


def load_data(path):
    """
    Loads dataset, removes duplicates, and standardizes data types
    without pre-imputing global medians (preventing data leakage).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    df = pd.read_csv(path)

    # Remove duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Convert numeric columns to proper float/int types
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean string columns
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": np.nan, "": np.nan, "None": np.nan})

    return df