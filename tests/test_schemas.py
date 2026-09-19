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
