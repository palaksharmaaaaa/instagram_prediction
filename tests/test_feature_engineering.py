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
    assert "creator_scale_engagement" in computed.columns
    expected_cse = np.log1p(100000) * (5000 / 100001.0)
    assert abs(computed["creator_scale_engagement"].iloc[0] - expected_cse) < 1e-5
    assert computed["virality_momentum"].iloc[0] > 0
    assert computed["save_efficiency"].iloc[0] > 0
    assert computed["interaction_density"].iloc[0] == (1000 / 4001)

    # Test granular 2026 features
    assert computed["follower_following_ratio"].iloc[0] == (100000 / 501.0)
    assert computed["explore_discovery_potential"].iloc[0] > 0
    assert computed["call_to_action_boost"].iloc[0] > 0
    assert computed["hashtag_density"].iloc[0] > 0
    assert abs(computed["gender_male_pct"].iloc[0] - 0.45) < 1e-5


def test_dynamic_temporal_features_derivation():
    """ML-02: Verify is_weekend and is_peak_posting_hour are dynamically derived from day and hour."""
    test_cases = pd.DataFrame([
        # Saturday peak hour
        {"posted_day_of_week": "Saturday", "posted_hour_of_day": 18, "is_weekend": 0.0, "is_peak_posting_hour": 0.0},
        # Sunday non-peak hour
        {"posted_day_of_week": "Sunday", "posted_hour_of_day": 3, "is_weekend": 0.0, "is_peak_posting_hour": 1.0},
        # Monday peak hour
        {"posted_day_of_week": "Monday", "posted_hour_of_day": 12},
        # Wednesday peak hour (11, 13, 19, 20, 21)
        {"posted_day_of_week": "Wednesday", "posted_hour_of_day": 20},
        # Friday non-peak hour
        {"posted_day_of_week": "Friday", "posted_hour_of_day": 15},
    ])

    computed = compute_derived_metrics(test_cases)

    # Row 0: Saturday (weekend=1.0), 18:00 (peak=1.0) - even though inputs had 0.0
    assert computed["is_weekend"].iloc[0] == 1.0
    assert computed["is_peak_posting_hour"].iloc[0] == 1.0

    # Row 1: Sunday (weekend=1.0), 03:00 (peak=0.0) - even though inputs had wrong flags
    assert computed["is_weekend"].iloc[1] == 1.0
    assert computed["is_peak_posting_hour"].iloc[1] == 0.0

    # Row 2: Monday (weekend=0.0), 12:00 (peak=1.0)
    assert computed["is_weekend"].iloc[2] == 0.0
    assert computed["is_peak_posting_hour"].iloc[2] == 1.0

    # Row 3: Wednesday (weekend=0.0), 20:00 (peak=1.0)
    assert computed["is_weekend"].iloc[3] == 0.0
    assert computed["is_peak_posting_hour"].iloc[3] == 1.0

    # Row 4: Friday (weekend=0.0), 15:00 (peak=0.0)
    assert computed["is_weekend"].iloc[4] == 0.0
    assert computed["is_peak_posting_hour"].iloc[4] == 0.0


def test_feature_columns_numeric_integrity():
    """ML-05: Verify creator_scale_engagement is present and reach_potential is removed."""
    from instagram_predictor.data.feature_engineering import FEATURE_COLUMNS_NUMERIC

    assert "creator_scale_engagement" in FEATURE_COLUMNS_NUMERIC
    assert "reach_potential" not in FEATURE_COLUMNS_NUMERIC


def test_load_dataset_nano_tier_coverage():
    """ML-04: Verify that load_dataset contains accounts across all 4 tiers, specifically Nano (<10k)."""
    from instagram_predictor.data import load_dataset

    df = load_dataset(reload=True)

    nano_accounts = df[df["total_followers"] < 10_000]
    micro_accounts = df[(df["total_followers"] >= 10_000) & (df["total_followers"] < 100_000)]
    macro_accounts = df[(df["total_followers"] >= 100_000) & (df["total_followers"] < 1_000_000)]
    mega_accounts = df[df["total_followers"] >= 1_000_000]

    # Verify Nano tier is populated with at least 15-25 unique profiles
    assert len(nano_accounts) > 0, "Dataset must contain Nano accounts (< 10,000 followers)"
    assert nano_accounts["username"].nunique() >= 15, "Nano tier must have at least 15 unique profiles"
    assert micro_accounts["username"].nunique() >= 15, "Micro tier must have at least 15 unique profiles"
    assert macro_accounts["username"].nunique() >= 15, "Macro tier must have at least 15 unique profiles"
    assert mega_accounts["username"].nunique() >= 15, "Mega tier must have at least 15 unique profiles"

    # Verify follower bounds for Nano tier
    assert nano_accounts["total_followers"].min() >= 500
    assert nano_accounts["total_followers"].max() < 10_000


def test_load_dataset_multilabel_list_deserialization():
    """DATA-01: Verify that multi-label list columns are true Python list objects with valid strings."""
    from instagram_predictor.data import load_dataset

    df = load_dataset()

    assert "categorizations" in df.columns
    assert "account_categories" in df.columns
    assert "categorization" in df.columns
    assert "account_categories_str" in df.columns

    # Verify all rows have genuine Python lists with clean str elements
    for _, row in df.iterrows():
        # categorizations
        cats = row["categorizations"]
        assert isinstance(cats, list), f"categorizations must be a list, got {type(cats)}"
        assert len(cats) > 0, "categorizations list must not be empty"
        for item in cats:
            assert isinstance(item, str) and type(item) is str, f"item must be str, got {type(item)}"
            assert not item.startswith("[") and not item.endswith("]"), f"item contains raw list brackets: {item}"

        # account_categories
        acc_cats = row["account_categories"]
        assert isinstance(acc_cats, list), f"account_categories must be a list, got {type(acc_cats)}"
        assert len(acc_cats) > 0, "account_categories list must not be empty"
        for item in acc_cats:
            assert isinstance(item, str) and type(item) is str, f"item must be str, got {type(item)}"
            assert not item.startswith("[") and not item.endswith("]"), f"item contains raw list brackets: {item}"

        # String counterparts
        assert isinstance(row["categorization"], str)
        assert isinstance(row["account_categories_str"], str)


def test_parse_list_field_robustness():
    """DATA-01: Verify robust deserialization across diverse input formats and edge cases."""
    from instagram_predictor.data import parse_list_field

    assert parse_list_field("['Sports', 'Health & Fitness']") == ["Sports", "Health & Fitness"]
    assert parse_list_field('["Sports", "Health & Fitness"]') == ["Sports", "Health & Fitness"]
    assert parse_list_field("['[\"Sports\"]']") == ["Sports"]
    assert parse_list_field("['Sports']") == ["Sports"]
    assert parse_list_field("Sports, Health & Fitness") == ["Sports", "Health & Fitness"]
    assert parse_list_field("Sports") == ["Sports"]
    assert parse_list_field(["Sports", np.str_("Health & Fitness")]) == ["Sports", "Health & Fitness"]
    assert parse_list_field([]) == []
    assert parse_list_field("[]") == []
    assert parse_list_field(None) == []
    assert parse_list_field(np.nan) == []
    assert parse_list_field("") == []
    assert parse_list_field("['Women\\'s Health', 'Sports']") == ["Women's Health", "Sports"]

