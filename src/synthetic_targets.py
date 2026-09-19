import numpy as np


def create_synthetic_targets(df, random_state=42):

    rng = np.random.default_rng(random_state)

    # Basic engagement signal
    engagement_signal = (
        df["Likes Avg."]
        + df["Comments Avg."]
    )

    # Normalized signals
    followers_signal = np.log1p(df["Followers"])
    views_signal = np.log1p(df["Views Avg."])
    engagement_rate = df["Engagement Rate"]

    # Synthetic Reach
    reach = (
        0.55 * df["Followers"]
        + 0.25 * df["Views Avg."]
        + 0.15 * engagement_signal
        + 0.05 * df["Followers"] * engagement_rate
    )

    # Add controlled noise
    noise = rng.normal(
        loc=1.0,
        scale=0.08,
        size=len(df)
    )

    reach = np.maximum(
        reach * noise,
        1000
    )

    # Synthetic Impressions
    impressions = (
        reach * (
            1.15
            + 0.5 * engagement_rate
        )
    )

    impressions = np.maximum(
        impressions,
        reach
    )

    df["synthetic_reach"] = reach
    df["synthetic_impressions"] = impressions

    return df