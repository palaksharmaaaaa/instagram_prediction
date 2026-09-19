import pytest
from pydantic import ValidationError
from instagram_predictor.schemas import (
    ProfileInput,
    PostInput,
    MediaType,
    ContentCategory,
    ContentStyle,
    Demographics,
    PostMetrics,
    ConfidenceInterval
)


def test_valid_profile_schema():
    profile = ProfileInput(
        username="@Cristiano",
        full_name="Cristiano Ronaldo",
        country="ES",
        total_followers=465000000,
        total_following=520,
        total_media_posts=3328,
        is_verified=True,
        account_category=ContentCategory.SPORTS
    )
    assert profile.username == "cristiano"
    assert profile.total_followers == 465000000
    assert profile.total_following == 520


def test_invalid_following_exceeds_platform_limit():
    with pytest.raises(ValidationError):
        ProfileInput(
            username="bot_account",
            total_followers=1000,
            total_following=7501,  # Platform limit is 7,500
            total_media_posts=10
        )


def test_negative_counts_raise_validation_error():
    with pytest.raises(ValidationError):
        ProfileInput(
            username="bad_account",
            total_followers=-50,
            total_following=100,
            total_media_posts=10
        )


def test_demographics_bounds():
    demo = Demographics(top_country="us", primary_age_group="25-34", gender_female_pct=0.65)
    assert demo.top_country == "US"
    assert demo.gender_female_pct == 0.65

    with pytest.raises(ValidationError):
        Demographics(gender_female_pct=1.5)  # Out of [0, 1] range


def test_confidence_interval_schema():
    ci = ConfidenceInterval(lower=1000, point_estimate=1500, upper=2000, confidence_level=0.80)
    assert ci.lower <= ci.point_estimate <= ci.upper


def test_demographics_gender_sum_validation():
    # Sum must be approximately 1.0 within 0.05 tolerance
    valid_demo = Demographics(gender_female_pct=0.52, gender_male_pct=0.48)
    assert valid_demo.gender_female_pct == 0.52
    assert valid_demo.gender_male_pct == 0.48

    # Sum significantly deviating from 1.0 (e.g. 0.90 + 0.90 = 1.80) raises ValidationError
    with pytest.raises(ValidationError) as exc_info:
        Demographics(gender_female_pct=0.90, gender_male_pct=0.90)
    assert "Demographics female and male percentages must sum to 1.0" in str(exc_info.value)

    # Sum significantly less than 1.0 (e.g. 0.30 + 0.30 = 0.60) raises ValidationError
    with pytest.raises(ValidationError) as exc_info2:
        Demographics(gender_female_pct=0.30, gender_male_pct=0.30)
    assert "Demographics female and male percentages must sum to 1.0" in str(exc_info2.value)


def test_post_caption_length_platform_limit():
    # Valid caption length up to 2200
    post = PostInput(caption_length_chars=2200)
    assert post.caption_length_chars == 2200

    # Exceeding 2200 raises ValidationError
    with pytest.raises(ValidationError) as exc_info:
        PostInput(caption_length_chars=2201)
    assert "less than or equal to 2200" in str(exc_info.value)

