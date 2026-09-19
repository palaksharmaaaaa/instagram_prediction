import pytest
import numpy as np
import pandas as pd
from pydantic import ValidationError

from instagram_predictor.schemas import (
    PlatformType,
    MediaType,
    ContentCategory,
    ContentStyle,
    Demographics,
    PostMetrics,
    ProfileInput,
    PostInput,
)
from instagram_predictor.data.feature_engineering import (
    compute_derived_metrics,
    FEATURE_COLUMNS_CATEGORICAL,
    PRE_PUBLISH_FEATURE_COLUMNS_CATEGORICAL,
    FEATURE_COLUMNS_NUMERIC,
    PRE_PUBLISH_FEATURE_COLUMNS_NUMERIC,
)
from instagram_predictor.models import (
    simulate_post_performance,
    explain_post_prediction,
    predict_batch,
)


# =============================================================================
# 1. Schema Parsing & Platform Extension Tests
# =============================================================================

def test_schema_parsing_instagram_youtube_snapchat():
    """Verify ProfileInput and PostInput correctly parse Instagram, YouTube, and Snapchat schemas."""
    # 1. Instagram Profile & Post
    ig_prof = ProfileInput(
        platform=PlatformType.INSTAGRAM,
        username="ig_creator",
        total_followers=150_000,
        total_following=450,
        total_media_posts=300,
        account_category=ContentCategory.FASHION_BEAUTY,
    )
    ig_post = PostInput(
        platform=PlatformType.INSTAGRAM,
        media_type=MediaType.REEL,
        category=ContentCategory.FASHION_BEAUTY,
        caption_length_chars=280,
    )
    assert ig_prof.platform == PlatformType.INSTAGRAM
    assert ig_post.platform == PlatformType.INSTAGRAM
    assert ig_post.media_type == MediaType.REEL

    # 2. YouTube Profile & Post with YouTube-specific fields
    yt_prof = ProfileInput(
        platform=PlatformType.YOUTUBE,
        username="tech_reviewer",
        total_followers=1_200_000,
        total_following=120,
        total_media_posts=450,
        account_category=ContentCategory.SCIENCE_TECHNOLOGY,
    )
    yt_post = PostInput(
        platform=PlatformType.YOUTUBE,
        media_type=MediaType.YOUTUBE_VIDEO,
        category=ContentCategory.SCIENCE_TECHNOLOGY,
        video_title_length=72,
        thumbnail_has_face=True,
        video_duration_seconds=720.0,
    )
    assert yt_prof.platform == PlatformType.YOUTUBE
    assert yt_post.platform == PlatformType.YOUTUBE
    assert yt_post.media_type == MediaType.YOUTUBE_VIDEO
    assert yt_post.video_title_length == 72
    assert yt_post.thumbnail_has_face is True

    # 3. Snapchat Profile & Post with Snapchat-specific fields
    snap_prof = ProfileInput(
        platform=PlatformType.SNAPCHAT,
        username="snap_lens_creator",
        total_followers=85_000,
        total_following=200,
        total_media_posts=1200,
        account_category=ContentCategory.MUSIC_ENTERTAINMENT,
    )
    snap_post = PostInput(
        platform=PlatformType.SNAPCHAT,
        media_type=MediaType.SNAPCHAT_SPOTLIGHT,
        category=ContentCategory.MUSIC_ENTERTAINMENT,
        screenshot_count=45,
        video_duration_seconds=18.0,
    )
    assert snap_prof.platform == PlatformType.SNAPCHAT
    assert snap_post.platform == PlatformType.SNAPCHAT
    assert snap_post.media_type == MediaType.SNAPCHAT_SPOTLIGHT
    assert snap_post.screenshot_count == 45


def test_schema_backward_compatibility_defaults():
    """Verify that omitting platform defaults safely to INSTAGRAM."""
    prof = ProfileInput(
        username="legacy_creator",
        total_followers=50_000,
        total_following=300,
        total_media_posts=150,
    )
    assert prof.platform == PlatformType.INSTAGRAM

    post = PostInput(
        caption_length_chars=200,
    )
    assert post.platform == PlatformType.INSTAGRAM
    assert post.media_type == MediaType.REEL
    # Platform-specific defaults
    assert post.video_title_length == 60
    assert post.thumbnail_has_face is True
    assert post.screenshot_count == 0


def test_platform_specific_fields_validation_bounds():
    """Verify bounds enforcement for YouTube and Snapchat specific fields."""
    # YouTube title length: ge=0, le=100
    valid_yt = PostInput(media_type=MediaType.YOUTUBE_SHORT, video_title_length=100)
    assert valid_yt.video_title_length == 100

    with pytest.raises(ValidationError):
        PostInput(media_type=MediaType.YOUTUBE_SHORT, video_title_length=101)

    with pytest.raises(ValidationError):
        PostInput(media_type=MediaType.YOUTUBE_SHORT, video_title_length=-1)

    # Snapchat screenshot_count: ge=0
    valid_snap = PostInput(media_type=MediaType.SNAPCHAT_POST, screenshot_count=0)
    assert valid_snap.screenshot_count == 0

    with pytest.raises(ValidationError):
        PostInput(media_type=MediaType.SNAPCHAT_POST, screenshot_count=-5)


# =============================================================================
# 2. Auto-Inference of Platform from Media Type Tests
# =============================================================================

def test_auto_inference_of_platform_from_media_type():
    """Verify platform is automatically inferred when platform is omitted."""
    # YouTube formats auto-infer YouTube
    assert PostInput(media_type=MediaType.YOUTUBE_SHORT).platform == PlatformType.YOUTUBE
    assert PostInput(media_type=MediaType.YOUTUBE_VIDEO).platform == PlatformType.YOUTUBE
    assert PostInput(media_type=MediaType.COMMUNITY_POST).platform == PlatformType.YOUTUBE
    assert PostInput(media_type="YouTube Short").platform == PlatformType.YOUTUBE

    # Snapchat formats auto-infer Snapchat
    assert PostInput(media_type=MediaType.SNAPCHAT_SPOTLIGHT).platform == PlatformType.SNAPCHAT
    assert PostInput(media_type=MediaType.SNAPCHAT_STORY).platform == PlatformType.SNAPCHAT
    assert PostInput(media_type=MediaType.SNAPCHAT_POST).platform == PlatformType.SNAPCHAT
    assert PostInput(media_type="Snapchat Spotlight").platform == PlatformType.SNAPCHAT

    # Instagram formats auto-infer Instagram
    assert PostInput(media_type=MediaType.REEL).platform == PlatformType.INSTAGRAM
    assert PostInput(media_type=MediaType.CAROUSEL).platform == PlatformType.INSTAGRAM
    assert PostInput(media_type=MediaType.STATIC_IMAGE).platform == PlatformType.INSTAGRAM
    assert PostInput(media_type=MediaType.STORY).platform == PlatformType.INSTAGRAM
    assert PostInput(media_type=MediaType.VIDEO).platform == PlatformType.INSTAGRAM
    assert PostInput(media_type="Video").platform == PlatformType.INSTAGRAM


def test_platform_media_type_mismatch_raises_validation_error():
    """Verify that explicitly mismatched platform and media_type raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        PostInput(platform=PlatformType.INSTAGRAM, media_type=MediaType.YOUTUBE_SHORT)
    assert "does not match" in str(exc_info.value) or "not supported" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info2:
        PostInput(platform=PlatformType.YOUTUBE, media_type=MediaType.REEL)
    assert "does not match" in str(exc_info2.value) or "not supported" in str(exc_info2.value)

    with pytest.raises(ValidationError) as exc_info3:
        PostInput(platform=PlatformType.SNAPCHAT, media_type=MediaType.YOUTUBE_VIDEO)
    assert "does not match" in str(exc_info3.value) or "not supported" in str(exc_info3.value)


# =============================================================================
# 3. Feature Engineering Cross-Platform Derived Metrics Tests
# =============================================================================

def test_feature_engineering_derived_metrics_multi_platform():
    """Verify is_short_form, is_video_content, watch_efficiency, and platform in feature engineering."""
    # Ensure platform is in categorical definitions
    assert "platform" in FEATURE_COLUMNS_CATEGORICAL
    assert "platform" in PRE_PUBLISH_FEATURE_COLUMNS_CATEGORICAL
    assert "is_short_form" in FEATURE_COLUMNS_NUMERIC
    assert "is_short_form" in PRE_PUBLISH_FEATURE_COLUMNS_NUMERIC
    assert "is_video_content" in FEATURE_COLUMNS_NUMERIC
    assert "is_video_content" in PRE_PUBLISH_FEATURE_COLUMNS_NUMERIC

    test_df = pd.DataFrame([
        # Row 0: Instagram Reel
        {"media_type": "Reel", "platform": "Instagram", "per_media_completion_rate": 0.75, "total_followers": 50000},
        # Row 1: Instagram Carousel
        {"media_type": "Carousel", "platform": "Instagram", "per_media_completion_rate": 0.0, "total_followers": 50000},
        # Row 2: Instagram Static Image
        {"media_type": "Static Image", "platform": "Instagram", "per_media_completion_rate": 0.0, "total_followers": 50000},
        # Row 3: Instagram Video
        {"media_type": "Video", "platform": "Instagram", "per_media_completion_rate": 0.50, "total_followers": 50000},
        # Row 4: YouTube Short
        {"media_type": "YouTube Short", "platform": "YouTube", "per_media_completion_rate": 0.85, "total_followers": 50000},
        # Row 5: YouTube Video (Long-form)
        {"media_type": "YouTube Video", "platform": "YouTube", "per_media_completion_rate": 0.40, "total_followers": 50000},
        # Row 6: YouTube Community Post
        {"media_type": "Community Post", "platform": "YouTube", "per_media_completion_rate": 0.0, "total_followers": 50000},
        # Row 7: Snapchat Spotlight
        {"media_type": "Snapchat Spotlight", "platform": "Snapchat", "per_media_completion_rate": 0.80, "total_followers": 50000},
        # Row 8: Snapchat Story
        {"media_type": "Snapchat Story", "platform": "Snapchat", "per_media_completion_rate": 0.90, "total_followers": 50000},
        # Row 9: Snapchat Post
        {"media_type": "Snapchat Post", "platform": "Snapchat", "per_media_completion_rate": 0.0, "total_followers": 50000},
    ])

    computed = compute_derived_metrics(test_df)

    # 1. is_short_form verification: 1.0 for Reel, YouTube Short, Snapchat Spotlight
    expected_short_form = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    np.testing.assert_array_equal(computed["is_short_form"].values, expected_short_form)

    # 2. is_video_content verification: 1.0 for Reel, Video, YouTube Video, YouTube Short, Snapchat Spotlight, Snapchat Story
    expected_video = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    np.testing.assert_array_equal(computed["is_video_content"].values, expected_video)

    # 3. watch_efficiency verification: per_media_completion_rate * is_video_content
    expected_watch_eff = [
        0.75 * 1.0,  # Reel
        0.0 * 0.0,   # Carousel
        0.0 * 0.0,   # Static Image
        0.50 * 1.0,  # Video
        0.85 * 1.0,  # YouTube Short
        0.40 * 1.0,  # YouTube Video
        0.0 * 0.0,   # Community Post
        0.80 * 1.0,  # Snapchat Spotlight
        0.90 * 1.0,  # Snapchat Story
        0.0 * 0.0,   # Snapchat Post
    ]
    np.testing.assert_allclose(computed["watch_efficiency"].values, expected_watch_eff)


def test_feature_engineering_platform_auto_inference_from_media():
    """Verify platform is correctly inferred during feature derivation when column is absent."""
    df_no_platform = pd.DataFrame([
        {"media_type": "YouTube Short", "total_followers": 20000},
        {"media_type": "Snapchat Spotlight", "total_followers": 30000},
        {"media_type": "Reel", "total_followers": 40000},
        {"media_type": "YouTube Video", "total_followers": 50000},
        {"media_type": "Snapchat Story", "total_followers": 60000},
    ])

    computed = compute_derived_metrics(df_no_platform)
    assert computed["platform"].tolist() == ["YouTube", "Snapchat", "Instagram", "YouTube", "Snapchat"]


# =============================================================================
# 4. Simulation Execution Tests (Instagram, YouTube, Snapchat)
# =============================================================================

@pytest.mark.parametrize(
    "platform,media_type,duration,slides",
    [
        (PlatformType.INSTAGRAM, MediaType.REEL, 30.0, 1),
        (PlatformType.INSTAGRAM, MediaType.CAROUSEL, 0.0, 5),
        (PlatformType.INSTAGRAM, MediaType.VIDEO, 120.0, 1),
        (PlatformType.YOUTUBE, MediaType.YOUTUBE_SHORT, 25.0, 1),
        (PlatformType.YOUTUBE, MediaType.YOUTUBE_VIDEO, 600.0, 1),
        (PlatformType.YOUTUBE, MediaType.COMMUNITY_POST, 0.0, 1),
        (PlatformType.SNAPCHAT, MediaType.SNAPCHAT_SPOTLIGHT, 15.0, 1),
        (PlatformType.SNAPCHAT, MediaType.SNAPCHAT_STORY, 10.0, 1),
        (PlatformType.SNAPCHAT, MediaType.SNAPCHAT_POST, 0.0, 1),
    ]
)
def test_simulation_execution_across_platforms(platform, media_type, duration, slides):
    """Verify simulate_post_performance successfully forecasts for all cross-platform formats."""
    profile = ProfileInput(
        platform=platform,
        username=f"creator_{platform.value.lower()}",
        total_followers=120_000,
        total_following=450,
        total_media_posts=250,
        account_category=ContentCategory.SPORTS,
    )
    post = PostInput(
        platform=platform,
        media_type=media_type,
        category=ContentCategory.SPORTS,
        categorizations=[ContentStyle.ENTERTAINING, ContentStyle.EDUCATIONAL],
        video_duration_seconds=duration,
        carousel_slide_count=slides,
        video_title_length=65,
        thumbnail_has_face=True,
        screenshot_count=20,
    )

    result = simulate_post_performance(profile, post, confidence_level=0.80)

    assert result is not None
    assert result.projected_reach.point_estimate > 0
    assert result.projected_impressions.point_estimate > 0
    # Invariant: reach <= impressions
    assert result.projected_reach.point_estimate <= result.projected_impressions.point_estimate
    # Conformal interval bounds
    assert result.projected_reach.lower <= result.projected_reach.point_estimate <= result.projected_reach.upper
    assert result.projected_impressions.lower <= result.projected_impressions.point_estimate <= result.projected_impressions.upper
    # Calibration tier and coverage
    assert result.calibration_tier in ["Nano (<10k followers)", "Micro (10k-100k followers)", "Macro (100k-1M followers)", "Mega (1M+ followers)"]
    assert "80% Mondrian Conformal Coverage" in result.prediction_interval_coverage


# =============================================================================
# 5. TreeSHAP Explainability Tests (Cross-Platform)
# =============================================================================

@pytest.mark.parametrize(
    "platform,media_type,vid_sec,slides",
    [
        (PlatformType.YOUTUBE, MediaType.YOUTUBE_SHORT, 30.0, 1),
        (PlatformType.YOUTUBE, MediaType.YOUTUBE_VIDEO, 600.0, 1),
        (PlatformType.SNAPCHAT, MediaType.SNAPCHAT_SPOTLIGHT, 15.0, 1),
        (PlatformType.INSTAGRAM, MediaType.REEL, 30.0, 1),
    ]
)
def test_treeshap_explainability_cross_platform(platform, media_type, vid_sec, slides):
    """Verify TreeSHAP interventional explainability handles YouTube, Snapchat, and Instagram formats."""
    profile = ProfileInput(
        platform=platform,
        username="shaptester",
        total_followers=250_000,
        total_following=500,
        total_media_posts=350,
        account_category=ContentCategory.SCIENCE_TECHNOLOGY,
    )
    post = PostInput(
        platform=platform,
        media_type=media_type,
        category=ContentCategory.SCIENCE_TECHNOLOGY,
        categorizations=[ContentStyle.EDUCATIONAL, ContentStyle.ENTERTAINING],
        caption_length_chars=350,
        hashtags_count=8,
        has_call_to_action=True,
        video_duration_seconds=vid_sec,
        carousel_slide_count=slides,
        posted_hour_of_day=19,
        posted_day_of_week="Wednesday",
    )

    explanations = explain_post_prediction(post=post, profile=profile)

    assert explanations is not None
    assert "base_reach" in explanations
    assert "final_reach" in explanations
    assert "drivers" in explanations
    assert len(explanations["drivers"]) == 6

    # Verify Shapley value conservation: base_reach + sum(impacts) == final_reach
    base_reach = explanations["base_reach"]
    final_reach = explanations["final_reach"]
    impact_sum = sum(d["impact"] for d in explanations["drivers"])
    assert base_reach + impact_sum == final_reach

    # Verify format driver accurately names the media type
    format_driver = next(d for d in explanations["drivers"] if "Media Format" in d["name"])
    assert media_type.value in format_driver["name"]


# =============================================================================
# 6. Batch Inference Multi-Platform Routing Tests
# =============================================================================

def test_batch_prediction_multi_platform_routing():
    """Verify predict_batch cleanly handles mixed Instagram, YouTube, and Snapchat rows."""
    batch_df = pd.DataFrame([
        {
            "username": "ig_star",
            "platform": "Instagram",
            "media_type": "Reel",
            "category": "Sports",
            "total_followers": 200_000,
            "total_following": 400,
            "total_media_posts": 500,
            "caption_length_chars": 200,
            "hashtags_count": 5,
        },
        {
            "username": "yt_creator",
            "platform": "YouTube",
            "media_type": "YouTube Short",
            "category": "Science & Technology",
            "total_followers": 500_000,
            "total_following": 100,
            "total_media_posts": 300,
            "caption_length_chars": 150,
            "hashtags_count": 3,
        },
        {
            "username": "snap_influencer",
            "platform": "Snapchat",
            "media_type": "Snapchat Spotlight",
            "category": "Fashion & Beauty",
            "total_followers": 75_000,
            "total_following": 250,
            "total_media_posts": 800,
            "caption_length_chars": 100,
            "hashtags_count": 2,
        },
    ])

    results = predict_batch(batch_df, predict_reach=True, predict_impressions=True)

    assert "predicted_reach" in results.columns
    assert "predicted_impressions" in results.columns
    assert (results["predicted_reach"] > 0).all()
    assert (results["predicted_impressions"] >= results["predicted_reach"]).all()
