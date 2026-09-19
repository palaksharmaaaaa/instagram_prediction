"""
Verification tests for the consolidated dataset:
- data/consolidated_profiles_posts.csv (profile-centric with nested posts)
- data/consolidated_profiles_posts_flat.csv (flat post-centric)

Verifies:
1. Schema deduplication: exactly 64 canonical snake_case columns with zero duplicate
   columns (case-insensitive check: len(cols) == len(set(c.lower() for c in cols))).
2. Elimination of redundant casing/naming variants (e.g. Followers, Url, Posts, Channel Name, etc.).
3. Redundant file top_200_instagrammers.csv is deleted and not required for tests or runtime.
4. Platform segregation (grouped contiguously: Instagram, YouTube, Snapchat).
5. Follower sorting within each platform tier.
6. Completeness: exactly 442 profiles and 1,326 nested posts.
7. Domain invariant: Reach <= Impressions across all 442 profiles and 1,326 nested posts.
"""

import json
from pathlib import Path
from typing import List
import pandas as pd
import pytest

from instagram_predictor.config import settings

CONSOLIDATED_PATH = Path("data/consolidated_profiles_posts.csv")
FLAT_PATH = Path("data/consolidated_profiles_posts_flat.csv")
RAW_POSTS_PATH = Path("data/raw/instagram_profiles_posts.csv")
TOP_200_PATH = Path("data/top_200_instagrammers.csv")

# 33 Canonical Profile Columns
EXPECTED_PROFILE_COLUMNS: List[str] = [
    "platform",
    "username",
    "full_name",
    "country",
    "total_followers",
    "total_following",
    "total_media_posts",
    "is_verified",
    "account_category",
    "profile_url",
    "boost_index",
    "engagement_rate",
    "engagement_rate_60d",
    "avg_likes",
    "avg_comments",
    "avg_video_views",
    "avg_1d",
    "avg_3d",
    "avg_7d",
    "avg_14d",
    "avg_30d",
    "account_age_years",
    "posting_frequency_per_week",
    "follower_growth_rate_30d",
    "account_bio_has_link",
    "account_categories",
    "account_categories_str",
    "top_country",
    "secondary_country",
    "primary_age_group",
    "gender_female_pct",
    "gender_male_pct",
    "audience_activity_score",
]

# 30 Canonical Post Columns
EXPECTED_POST_COLUMNS: List[str] = [
    "post_id",
    "media_type",
    "category",
    "categorization",
    "categorizations",
    "caption_length_chars",
    "hashtags_count",
    "mentions_count",
    "has_call_to_action",
    "video_duration_seconds",
    "carousel_slide_count",
    "video_title_length",
    "thumbnail_has_face",
    "screenshot_count",
    "posted_day_of_week",
    "posted_hour_of_day",
    "is_weekend",
    "is_peak_posting_hour",
    "per_media_likes",
    "per_media_comments",
    "per_media_shares",
    "per_media_saves",
    "per_media_video_views",
    "per_media_completion_rate",
    "reach_from_home_pct",
    "reach_from_explore_pct",
    "reach_from_hashtags_pct",
    "reach_from_other_pct",
    "per_media_reach",
    "per_media_impressions",
]

EXPECTED_CANONICAL_COLUMNS: List[str] = EXPECTED_PROFILE_COLUMNS + EXPECTED_POST_COLUMNS + ["nested_posts"]

REDUNDANT_VARIANTS: List[str] = [
    "Username",
    "Channel Name",
    "Country",
    "Url",
    "Main topic",
    "Main video category",
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
    "Engagement Rate (60 Days)",
]


def test_consolidated_files_exist():
    assert CONSOLIDATED_PATH.exists(), f"Missing {CONSOLIDATED_PATH}"
    assert FLAT_PATH.exists(), f"Missing {FLAT_PATH}"


def test_top_200_file_removed_and_settings_updated():
    """Verify that top_200_instagrammers.csv is deleted and not required."""
    assert not TOP_200_PATH.exists(), f"Redundant file {TOP_200_PATH} should be deleted"
    assert hasattr(settings, "CONSOLIDATED_DATA_PATH"), "settings must include CONSOLIDATED_DATA_PATH"
    assert not hasattr(settings, "LEGACY_DATA_PATH"), "settings must NOT include LEGACY_DATA_PATH"
    assert settings.CONSOLIDATED_DATA_PATH.exists(), f"Configured {settings.CONSOLIDATED_DATA_PATH} must exist"


def test_schema_deduplication_and_canonical_columns():
    """Verify exactly 64 canonical snake_case columns with zero duplicate columns (case-insensitive)."""
    df_cons = pd.read_csv(CONSOLIDATED_PATH)
    df_flat = pd.read_csv(FLAT_PATH)

    assert len(EXPECTED_PROFILE_COLUMNS) == 33
    assert len(EXPECTED_POST_COLUMNS) == 30
    assert len(EXPECTED_CANONICAL_COLUMNS) == 64

    # 1. Total column count is exactly 64
    assert len(df_cons.columns) == 64, f"Expected 64 columns in consolidated file, got {len(df_cons.columns)}"
    assert len(df_flat.columns) == 64, f"Expected 64 columns in flat file, got {len(df_flat.columns)}"

    # 2. Zero duplicate columns (case-insensitive check)
    cols_cons = list(df_cons.columns)
    cols_flat = list(df_flat.columns)
    assert len(cols_cons) == len(set(c.lower() for c in cols_cons)), "Duplicate or case-clash columns found in consolidated CSV"
    assert len(cols_flat) == len(set(c.lower() for c in cols_flat)), "Duplicate or case-clash columns found in flat CSV"

    # 3. Exact column list match
    assert cols_cons == EXPECTED_CANONICAL_COLUMNS, "Consolidated columns do not match expected canonical schema"
    assert cols_flat == EXPECTED_CANONICAL_COLUMNS, "Flat columns do not match expected canonical schema"

    # 4. Redundant variants and duplicate casing are completely absent
    for redundant in REDUNDANT_VARIANTS:
        assert redundant not in df_cons.columns, f"Redundant column '{redundant}' still present in consolidated CSV"
        assert redundant not in df_flat.columns, f"Redundant column '{redundant}' still present in flat CSV"


def test_platform_segregation_and_ordering():
    """Verify platform segregation and descending follower ordering within each platform tier."""
    df_cons = pd.read_csv(CONSOLIDATED_PATH)
    df_flat = pd.read_csv(FLAT_PATH)

    platforms = df_cons["platform"].tolist()
    unique_platforms = list(df_cons["platform"].unique())

    # All 3 platforms present
    assert set(unique_platforms) == {"Instagram", "YouTube", "Snapchat"}

    # Verified segregated: each platform's records appear contiguously
    seen = set()
    prev = None
    for p in platforms:
        if p != prev:
            assert p not in seen, f"Platform {p} appears non-contiguously (broken segregation)"
            seen.add(p)
            prev = p

    # Counts
    counts = df_cons["platform"].value_counts().to_dict()
    assert counts["Instagram"] == 320
    assert counts["YouTube"] == 63
    assert counts["Snapchat"] == 59
    assert len(df_cons) == 442

    flat_counts = df_flat["platform"].value_counts().to_dict()
    assert flat_counts["Instagram"] == 960
    assert flat_counts["YouTube"] == 189
    assert flat_counts["Snapchat"] == 177
    assert len(df_flat) == 1326

    # Verify follower sorting within each platform tier
    for platform in ["Instagram", "YouTube", "Snapchat"]:
        plat_df = df_cons[df_cons["platform"] == platform]
        assert plat_df["total_followers"].is_monotonic_decreasing, f"Followers not sorted descending in {platform}"


def test_nested_posts_structure_and_invariants():
    """Verify all 442 profiles and 1,326 nested posts maintain reach <= impressions."""
    df_cons = pd.read_csv(CONSOLIDATED_PATH)
    df_flat = pd.read_csv(FLAT_PATH)

    total_nested_posts = 0

    for idx, row in df_cons.iterrows():
        raw_nested = row["nested_posts"]
        assert isinstance(raw_nested, str), f"Row {idx} nested_posts is not string"

        posts = json.loads(raw_nested)
        assert isinstance(posts, list), f"Row {idx} parsed nested_posts is not a list"
        assert len(posts) >= 1, f"Row {idx} ({row['username']}) has empty posts list"

        total_nested_posts += len(posts)

        for post in posts:
            for col in EXPECTED_POST_COLUMNS:
                assert col in post, f"Post {post.get('post_id')} missing post column {col}"

            # Fundamental Domain Invariant: Reach <= Impressions
            reach = post["per_media_reach"]
            impressions = post["per_media_impressions"]
            assert reach <= impressions, (
                f"Row {idx} ({row['username']}) Post {post['post_id']}: "
                f"Reach {reach} > Impressions {impressions}"
            )

    # Invariant: exactly 442 profiles and 1,326 nested posts
    assert len(df_cons) == 442
    assert total_nested_posts == 1326

    # Flat dataset invariant: reach <= impressions across all 1326 rows
    assert len(df_flat) == 1326
    assert (df_flat["per_media_reach"] > df_flat["per_media_impressions"]).sum() == 0


def test_profile_fields_populated_canonical():
    """Verify canonical profile fields are populated for representative creators."""
    df_cons = pd.read_csv(CONSOLIDATED_PATH)

    # Top Instagrammer check (e.g. cristiano)
    cristiano_row = df_cons[df_cons["username"].str.lower() == "cristiano"].iloc[0]
    assert cristiano_row["platform"] == "Instagram"
    assert cristiano_row["total_followers"] > 400_000_000
    assert "instagram.com" in str(cristiano_row["profile_url"])
    assert cristiano_row["boost_index"] > 80
    assert cristiano_row["engagement_rate"] > 0.0
    assert cristiano_row["avg_likes"] > 0.0
    assert cristiano_row["avg_comments"] > 0.0

    # YouTube creator check
    yt_rows = df_cons[df_cons["platform"] == "YouTube"]
    assert len(yt_rows) == 63
    first_yt = yt_rows.iloc[0]
    assert "youtube.com" in str(first_yt["profile_url"])
    assert first_yt["total_followers"] > 0

    # Snapchat creator check
    sc_rows = df_cons[df_cons["platform"] == "Snapchat"]
    assert len(sc_rows) == 59
    first_sc = sc_rows.iloc[0]
    assert "snapchat.com" in str(first_sc["profile_url"])
    assert first_sc["total_followers"] > 0
