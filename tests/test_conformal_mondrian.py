import json
import pytest
from instagram_predictor.config import settings
from instagram_predictor.models.engine import simulate_post_performance
from instagram_predictor.models.hpo import run_bayesian_hpo
from instagram_predictor.schemas import (
    ProfileInput,
    PostInput,
    MediaType,
    ContentCategory,
    ContentStyle,
    Demographics,
    PostMetrics,
)


def test_mondrian_metadata_structure():
    """Verify that metadata has valid Mondrian tier-conditional conformal quantiles."""
    assert settings.MODEL_METADATA_PATH.exists(), "Metadata file must exist"
    with open(settings.MODEL_METADATA_PATH, "r") as f:
        meta = json.load(f)

    reach_eval = meta["evaluation"]["reach"]
    imp_eval = meta["evaluation"]["impressions"]

    assert "tier_conformal_quantiles" in reach_eval
    assert "tier_conformal_quantiles" in imp_eval

    for tier in ["nano", "micro", "macro", "mega"]:
        assert tier in reach_eval["tier_conformal_quantiles"]
        assert tier in imp_eval["tier_conformal_quantiles"]
        r_tier = reach_eval["tier_conformal_quantiles"][tier]
        assert 0.0 < r_tier["q80"] <= r_tier["q90"]
        i_tier = imp_eval["tier_conformal_quantiles"][tier]
        assert 0.0 < i_tier["q80"] <= i_tier["q90"]


@pytest.mark.parametrize(
    "followers,expected_tier_substr",
    [
        (5_000, "Nano"),
        (50_000, "Micro"),
        (500_000, "Macro"),
        (5_000_000, "Mega"),
    ],
)
def test_simulation_tier_assignment(followers, expected_tier_substr):
    """Verify that simulate_post_performance correctly assigns the Mondrian calibration tier."""
    profile = ProfileInput(
        username="test_creator",
        full_name="Test Creator",
        country="US",
        total_followers=followers,
        total_following=300,
        total_media_posts=150,
        is_verified=False,
        account_category=ContentCategory.SCIENCE_TECHNOLOGY,
    )
    post = PostInput(
        media_type=MediaType.REEL,
        category=ContentCategory.SCIENCE_TECHNOLOGY,
        categorization=ContentStyle.EDUCATIONAL,
        demographics=Demographics(top_country="US", primary_age_group="25-34", gender_female_pct=0.5),
        metrics=PostMetrics(likes=max(int(followers * 0.03), 10), comments=5, shares=5, saves=5),
    )

    sim = simulate_post_performance(profile, post, confidence_level=0.80)
    assert expected_tier_substr in sim.calibration_tier
    assert "80% Mondrian Conformal Coverage" in sim.prediction_interval_coverage
    assert sim.projected_reach.lower <= sim.projected_reach.point_estimate <= sim.projected_reach.upper
    assert sim.projected_impressions.lower <= sim.projected_impressions.point_estimate <= sim.projected_impressions.upper
    assert sim.projected_reach.point_estimate <= sim.projected_impressions.point_estimate


def test_epistemic_ood_detector():
    """Verify in-distribution vs out-of-distribution epistemic uncertainty flags."""
    # In-distribution creator
    profile_normal = ProfileInput(
        username="normal_creator",
        full_name="Normal Creator",
        country="US",
        total_followers=100_000,
        total_following=500,
        total_media_posts=200,
        is_verified=False,
        account_category=ContentCategory.FASHION_BEAUTY,
    )
    post_normal = PostInput(
        media_type=MediaType.CAROUSEL,
        category=ContentCategory.FASHION_BEAUTY,
        categorization=ContentStyle.INSPIRATIONAL,
        demographics=Demographics(top_country="US", primary_age_group="25-34", gender_female_pct=0.6),
        metrics=PostMetrics(likes=3_000, comments=100, shares=150, saves=200),
    )
    sim_normal = simulate_post_performance(profile_normal, post_normal)
    assert "Calibrated (In-Distribution" in sim_normal.uncertainty_rating

    # Extreme OOD creator (e.g. 50M followers)
    profile_ood = ProfileInput(
        username="mega_celebrity",
        full_name="Mega Celebrity",
        country="US",
        total_followers=50_000_000,
        total_following=10,
        total_media_posts=1_000,
        is_verified=True,
        account_category=ContentCategory.MUSIC_ENTERTAINMENT,
    )
    sim_ood = simulate_post_performance(profile_ood, post_normal)
    assert "Extrapolation Alert" in sim_ood.uncertainty_rating


def test_hpo_module():
    """Verify that Bayesian HPO executes and returns best parameters and cross-validated MAE."""
    results = run_bayesian_hpo("per_media_reach", n_trials=2)
    assert "best_params" in results
    assert "best_cv_mae" in results
    assert results["best_cv_mae"] > 0
    assert "learning_rate" in results["best_params"]
