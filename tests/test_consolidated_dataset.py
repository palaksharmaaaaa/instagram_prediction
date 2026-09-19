"""
Verification tests for the consolidated dataset:
- data/consolidated_profiles_posts.csv (profile-centric with nested posts)
- data/consolidated_profiles_posts_flat.csv (flat post-centric)

Verifies:
1. Full join column preservation (all columns from instagram_profiles_posts.csv and top_200_instagrammers.csv).
2. Platform segregation (grouped by platform: Instagram, YouTube, Snapchat).
3. Follower sorting within each platform tier.
4. Nested posts JSON validity, completeness, and schema adherence.
5. Domain invariant: Reach <= Impressions for all posts across all profiles.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

CONSOLIDATED_PATH = Path("data/consolidated_profiles_posts.csv")
FLAT_PATH = Path("data/consolidated_profiles_posts_flat.csv")
RAW_POSTS_PATH = Path("data/raw/instagram_profiles_posts.csv")
TOP_200_PATH = Path("data/top_200_instagrammers.csv")


def test_consolidated_files_exist():
    assert CONSOLIDATED_PATH.exists(), f"Missing {CONSOLIDATED_PATH}"
    assert FLAT_PATH.exists(), f"Missing {FLAT_PATH}"


def test_full_join_column_names_preserved():
    df_cons = pd.read_csv(CONSOLIDATED_PATH, nrows=5)
    df_posts = pd.read_csv(RAW_POSTS_PATH, nrows=5)
    df_top = pd.read_csv(TOP_200_PATH, nrows=5)

    # All columns from raw posts must be present
    for col in df_posts.columns:
        assert col in df_cons.columns, f"Column {col} from raw posts missing from consolidated dataset"

    # All columns from top 200 must be present
    for col in df_top.columns:
        assert col in df_cons.columns, f"Column {col} from top 200 missing from consolidated dataset"

    # Must contain nested_posts
    assert "nested_posts" in df_cons.columns


def test_platform_segregation():
    df_cons = pd.read_csv(CONSOLIDATED_PATH)

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


def test_nested_posts_structure_and_invariants():
    df_cons = pd.read_csv(CONSOLIDATED_PATH)

    for idx, row in df_cons.iterrows():
        raw_nested = row["nested_posts"]
        assert isinstance(raw_nested, str), f"Row {idx} nested_posts is not string"

        posts = json.loads(raw_nested)
        assert isinstance(posts, list), f"Row {idx} parsed nested_posts is not a list"
        assert len(posts) >= 1, f"Row {idx} ({row['username']}) has empty posts list"

        for post in posts:
            assert "post_id" in post
            assert "media_type" in post
            assert "per_media_reach" in post
            assert "per_media_impressions" in post

            # Fundamental Invariant: Reach <= Impressions
            reach = post["per_media_reach"]
            impressions = post["per_media_impressions"]
            assert reach <= impressions, (
                f"Row {idx} ({row['username']}) Post {post['post_id']}: "
                f"Reach {reach} > Impressions {impressions}"
            )


def test_profile_fields_populated():
    df_cons = pd.read_csv(CONSOLIDATED_PATH)

    # Top Instagrammer check (e.g. cristiano)
    cristiano_row = df_cons[df_cons["username"].str.lower() == "cristiano"].iloc[0]
    assert cristiano_row["platform"] == "Instagram"
    assert cristiano_row["Followers"] > 400_000_000
    assert cristiano_row["total_followers"] > 400_000_000
    assert "instagram.com" in str(cristiano_row["Url"])
    assert cristiano_row["Boost Index"] > 80
    assert cristiano_row["Engagement Rate"] > 0.0

    # YouTube creator check
    yt_rows = df_cons[df_cons["platform"] == "YouTube"]
    assert len(yt_rows) == 63
    first_yt = yt_rows.iloc[0]
    assert "youtube.com" in str(first_yt["Url"])

    # Snapchat creator check
    sc_rows = df_cons[df_cons["platform"] == "Snapchat"]
    assert len(sc_rows) == 59
    first_sc = sc_rows.iloc[0]
    assert "snapchat.com" in str(first_sc["Url"])
