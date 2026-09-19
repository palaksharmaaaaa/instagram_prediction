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


def load_data(path):
    df = pd.read_csv(path)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Convert numeric columns
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill numerical missing values
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values
    categorical_columns = [
        "Country",
        "Main topic",
        "Main video category"
    ]

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df