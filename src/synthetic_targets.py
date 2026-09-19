"""
Legacy Synthetic Targets Module (v1 Prototype Architecture)
===========================================================
DEPRECATION NOTICE:
This module represents the v1 prototype architecture and is maintained solely for
backward compatibility with legacy training scripts and tests (e.g., tests/test_end_to_end.py).

Production systems should use the enterprise-grade `instagram_predictor` package:
- Synthetic Data Generation: `instagram_predictor.data.generator` (`generate_synthetic_posts`)
- Target Formulations: See `generate_synthetic_posts` for modern 2026 reach and impressions modeling.
"""

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