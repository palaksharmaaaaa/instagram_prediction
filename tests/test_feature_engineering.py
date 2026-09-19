import numpy as np
import pandas as pd
from instagram_predictor.data.feature_engineering import compute_derived_metrics


def test_derived_metrics_computation():
    raw_df = pd.DataFrame([{
        "total_followers": 100_000,
        "total_following": 500,
        "total_media_posts": 200,
        "per_media_likes": 4000,
        "per_media_comments": 200,
        "per_media_shares": 500,
        "per_media_saves": 300,
        "per_media_reach": 50000,
        "per_media_impressions": 75000,
        "gender_female_pct": 0.55,
        "media_type": "Reel"
    }])

    computed = compute_derived_metrics(raw_df)

    assert computed["total_engagement"].iloc[0] == 5000
    assert computed["engagement_rate"].iloc[0] == 0.05
    assert abs(computed["virality_score"].iloc[0] - (500 / 4001)) < 1e-5
    assert computed["reach_efficiency"].iloc[0] == 0.5
    assert computed["frequency_ratio"].iloc[0] == 1.5
    assert computed["save_rate"].iloc[0] == (300 / 75000)
    assert computed["share_rate"].iloc[0] == (500 / 50000)

    # Test new log-scale and interaction features
    assert abs(computed["log_followers"].iloc[0] - np.log1p(100000)) < 1e-5
    assert computed["reach_potential"].iloc[0] == 5000
    assert computed["virality_momentum"].iloc[0] > 0
    assert computed["save_efficiency"].iloc[0] > 0
    assert computed["interaction_density"].iloc[0] == (1000 / 4001)

    # Test granular 2026 features
    assert computed["follower_following_ratio"].iloc[0] == (100000 / 501.0)
    assert computed["explore_discovery_potential"].iloc[0] > 0
    assert computed["call_to_action_boost"].iloc[0] > 0
    assert computed["hashtag_density"].iloc[0] > 0
    assert abs(computed["gender_male_pct"].iloc[0] - 0.45) < 1e-5
