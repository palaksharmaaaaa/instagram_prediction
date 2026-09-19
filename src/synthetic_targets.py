import numpy as np
import pandas as pd


def create_synthetic_targets(df, random_state=42):
    """
    Generates realistic prototype synthetic target variables ('synthetic_reach'
    and 'synthetic_impressions') based on followers, engagement, and view signals.
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()

    followers = df["Followers"].fillna(0)
    views = df["Views Avg."].fillna(0)
    likes = df["Likes Avg."].fillna(0)
    comments = df["Comments Avg."].fillna(0)
    engagement_rate = df["Engagement Rate"].fillna(0)

    # Combined engagement interaction signal
    engagement_signal = likes + comments

    # Synthetic Reach formulation: combination of follower base, video views, and engagement
    reach = (
        0.55 * followers
        + 0.25 * views
        + 0.15 * engagement_signal
        + 0.05 * (followers * engagement_rate)
    )

    # Controlled multiplicative noise (std dev = 8%)
    noise = rng.normal(loc=1.0, scale=0.08, size=len(df))
    reach = np.maximum(reach * noise, 1000.0)

    # Synthetic Impressions: reach multiplied by frequency factor driven by engagement
    frequency_multiplier = 1.15 + 0.5 * engagement_rate
    impressions = np.maximum(reach * frequency_multiplier, reach)

    df["synthetic_reach"] = reach
    df["synthetic_impressions"] = impressions

    return df