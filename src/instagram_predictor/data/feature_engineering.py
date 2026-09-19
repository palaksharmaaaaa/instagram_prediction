import numpy as np
import pandas as pd
from typing import List


# Production ML feature definitions (High-Dimensional 2026 SOTA Feature Space)
FEATURE_COLUMNS_NUMERIC = [
    # Account-level features
    "total_followers",
    "total_following",
    "total_media_posts",
    "account_age_years",
    "posting_frequency_per_week",
    "follower_growth_rate_30d",
    "follower_following_ratio",
    "log_followers",
    "log_following",
    "log_posts",
    # Post engagement metrics & video
    "per_media_likes",
    "per_media_comments",
    "per_media_shares",
    "per_media_saves",
    "per_media_video_views",
    "per_media_completion_rate",
    # Content structure & semantics
    "caption_length_chars",
    "hashtags_count",
    "mentions_count",
    "has_call_to_action",
    "video_duration_seconds",
    "carousel_slide_count",
    "posted_hour_of_day",
    "is_weekend",
    "is_peak_posting_hour",
    # Demographics & audience
    "gender_female_pct",
    "gender_male_pct",
    "audience_activity_score",
    # Algorithmic discovery
    "reach_from_home_pct",
    "reach_from_explore_pct",
    "reach_from_hashtags_pct",
    # Multi-label content style indicators
    "is_educational",
    "is_entertaining",
    "is_promotional",
    "is_behind_scenes",
    "is_inspirational",
    # Domain interaction features
    "reach_potential",
    "virality_momentum",
    "save_efficiency",
    "interaction_density",
    "explore_discovery_potential",
    "call_to_action_boost",
    "watch_efficiency",
    "hashtag_density"
]

FEATURE_COLUMNS_CATEGORICAL = [
    "media_type",
    "category",
    "categorization",
    "top_country",
    "secondary_country",
    "primary_age_group",
    "posted_day_of_week"
]

ALL_FEATURE_COLUMNS = FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL


def _get_numeric_series(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _get_categorical_series(df: pd.DataFrame, col: str, default: str) -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype=str)


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes industry-standard Instagram analytics, log-scale metrics,
    and high-dimensional domain interaction features.
    """
    df = df.copy()

    # Raw counts & defaults using robust series getters
    followers = _get_numeric_series(df, "total_followers", 1.0).replace(0, 1.0)
    following = _get_numeric_series(df, "total_following", 0.0)
    posts = _get_numeric_series(df, "total_media_posts", 0.0)
    acc_age = _get_numeric_series(df, "account_age_years", 3.0)
    posting_freq = _get_numeric_series(df, "posting_frequency_per_week", 3.5)
    growth_rate = _get_numeric_series(df, "follower_growth_rate_30d", 0.02)

    likes = _get_numeric_series(df, "per_media_likes", 0.0)
    comments = _get_numeric_series(df, "per_media_comments", 0.0)
    shares = _get_numeric_series(df, "per_media_shares", 0.0)
    saves = _get_numeric_series(df, "per_media_saves", 0.0)
    video_views = _get_numeric_series(df, "per_media_video_views", 0.0)
    completion_rate = _get_numeric_series(df, "per_media_completion_rate", 0.0)

    caption_len = _get_numeric_series(df, "caption_length_chars", 250.0)
    hashtags = _get_numeric_series(df, "hashtags_count", 5.0)
    mentions = _get_numeric_series(df, "mentions_count", 0.0)
    has_cta = _get_numeric_series(df, "has_call_to_action", 1.0)
    video_sec = _get_numeric_series(df, "video_duration_seconds", 0.0)
    slide_count = _get_numeric_series(df, "carousel_slide_count", 1.0)
    hour_of_day = _get_numeric_series(df, "posted_hour_of_day", 18.0)
    is_wknd = _get_numeric_series(df, "is_weekend", 0.0)
    is_peak = _get_numeric_series(df, "is_peak_posting_hour", 1.0)

    female_pct = _get_numeric_series(df, "gender_female_pct", 0.50)
    male_pct = _get_numeric_series(df, "gender_male_pct", 1.0 - female_pct)
    audience_activity = _get_numeric_series(df, "audience_activity_score", 0.75)

    explore_pct = _get_numeric_series(df, "reach_from_explore_pct", 0.25)
    hashtag_pct = _get_numeric_series(df, "reach_from_hashtags_pct", 0.05)
    home_pct = _get_numeric_series(df, "reach_from_home_pct", 0.60)

    # Fill back into DataFrame
    df["account_age_years"] = acc_age
    df["posting_frequency_per_week"] = posting_freq
    df["follower_growth_rate_30d"] = growth_rate
    df["follower_following_ratio"] = followers / (following + 1.0)
    df["per_media_video_views"] = video_views
    df["per_media_completion_rate"] = completion_rate
    df["caption_length_chars"] = caption_len
    df["hashtags_count"] = hashtags
    df["mentions_count"] = mentions
    df["has_call_to_action"] = has_cta
    df["video_duration_seconds"] = video_sec
    df["carousel_slide_count"] = slide_count
    df["posted_hour_of_day"] = hour_of_day
    df["is_weekend"] = is_wknd
    df["is_peak_posting_hour"] = is_peak
    df["gender_female_pct"] = female_pct
    df["gender_male_pct"] = male_pct
    df["audience_activity_score"] = audience_activity
    df["reach_from_home_pct"] = home_pct
    df["reach_from_explore_pct"] = explore_pct
    df["reach_from_hashtags_pct"] = hashtag_pct

    df["secondary_country"] = _get_categorical_series(df, "secondary_country", "US")
    df["posted_day_of_week"] = _get_categorical_series(df, "posted_day_of_week", "Wednesday")
    df["media_type"] = _get_categorical_series(df, "media_type", "Reel")
    df["category"] = _get_categorical_series(df, "category", "Sports")

    # Multi-label categorization extraction
    cat_raw = _get_categorical_series(df, "categorization", "Educational / How-To")
    cat_lower = cat_raw.str.lower()
    df["is_educational"] = cat_lower.str.contains("educational").astype(float)
    df["is_entertaining"] = cat_lower.str.contains("entertaining").astype(float)
    df["is_promotional"] = cat_lower.str.contains("promotional").astype(float)
    df["is_behind_scenes"] = cat_lower.str.contains("behind the scenes").astype(float)
    df["is_inspirational"] = cat_lower.str.contains("inspirational").astype(float)

    # Clean primary categorization for categorical encoder
    df["categorization"] = cat_raw.apply(lambda s: s.split(",")[0].strip() if "," in s else s.strip())

    df["top_country"] = _get_categorical_series(df, "top_country", "US")
    df["primary_age_group"] = _get_categorical_series(df, "primary_age_group", "25-34")

    # 1. Total Engagement & Rates
    total_eng = likes + comments + shares + saves
    df["total_engagement"] = total_eng
    df["engagement_rate"] = total_eng / followers
    df["virality_score"] = shares / (likes + 1.0)

    # 2. Log-scale features (Power-Law distributions)
    df["log_followers"] = np.log1p(followers)
    df["log_following"] = np.log1p(following)
    df["log_posts"] = np.log1p(posts)

    # 3. Domain Interaction Terms
    df["reach_potential"] = followers * (total_eng / followers)
    is_reel = (df.get("media_type", "").astype(str) == "Reel").astype(float)
    is_carousel = (df.get("media_type", "").astype(str) == "Carousel").astype(float)

    df["virality_momentum"] = (shares / (likes + 1.0)) * (1.0 + is_reel)
    df["save_efficiency"] = (saves / (likes + 1.0)) * (1.0 + is_carousel + df["is_educational"] * 0.5)
    df["interaction_density"] = (comments + shares + saves) / (likes + 1.0)
    df["explore_discovery_potential"] = df["virality_momentum"] * (1.0 + explore_pct)
    df["call_to_action_boost"] = has_cta * (1.0 + is_carousel * 0.5)
    df["watch_efficiency"] = completion_rate * is_reel
    df["hashtag_density"] = hashtags / (np.log1p(caption_len) + 1.0)

    # 4. Save Rate & Share Rate (if reach/impressions exist)
    if "per_media_reach" in df.columns and "per_media_impressions" in df.columns:
        reach = pd.to_numeric(df["per_media_reach"], errors="coerce").fillna(1).replace(0, 1)
        impressions = pd.to_numeric(df["per_media_impressions"], errors="coerce").fillna(1).replace(0, 1)

        df["reach_efficiency"] = reach / followers
        df["frequency_ratio"] = impressions / reach
        df["save_rate"] = saves / impressions
        df["share_rate"] = shares / reach

    return df
