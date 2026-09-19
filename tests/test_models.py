from instagram_predictor.models import (
    get_reach_pipeline,
    get_impressions_pipeline,
    predict_batch,
    simulate_post_performance,
    get_model_metadata
)
from instagram_predictor.data import load_dataset
from instagram_predictor.schemas import (
    ProfileInput,
    PostInput,
    MediaType,
    ContentCategory,
    ContentStyle,
    Demographics
)


def test_pipelines_loaded():
    reach_p = get_reach_pipeline()
    imp_p = get_impressions_pipeline()
    assert reach_p is not None
    assert imp_p is not None


def test_model_conformal_metadata():
    meta = get_model_metadata()
    assert "conformal_quantile_80" in meta["evaluation"]["reach"]
    assert "conformal_quantile_90" in meta["evaluation"]["reach"]
    assert "conformal_quantile_80" in meta["evaluation"]["impressions"]
    assert "group_cv_r2_mean" in meta["evaluation"]["reach"]


def test_batch_prediction_invariants():
    df = load_dataset().head(10)
    preds = predict_batch(df, predict_reach=True, predict_impressions=True)

    assert "predicted_reach" in preds.columns
    assert "predicted_impressions" in preds.columns
    # Invariant: reach <= impressions
    assert (preds["predicted_reach"] <= preds["predicted_impressions"]).all()


def test_simulate_post_performance_confidence_intervals():
    profile = ProfileInput(
        username="test_athlete",
        full_name="Test Athlete",
        country="US",
        total_followers=1_000_000,
        total_following=300,
        total_media_posts=450,
        is_verified=True,
        account_category=ContentCategory.SPORTS
    )
    post = PostInput(
        media_type=MediaType.REEL,
        category=ContentCategory.SPORTS,
        categorization=ContentStyle.ENTERTAINING,
        demographics=Demographics(top_country="US", primary_age_group="18-24", gender_female_pct=0.45)
    )

    sim = simulate_post_performance(profile, post, confidence_level=0.80)

    # Check confidence bounds
    assert sim.projected_reach.lower <= sim.projected_reach.point_estimate <= sim.projected_reach.upper
    assert sim.projected_impressions.lower <= sim.projected_impressions.point_estimate <= sim.projected_impressions.upper
    assert sim.projected_reach.point_estimate <= sim.projected_impressions.point_estimate
    assert len(sim.optimization_tips) > 0


def test_simulate_post_performance_creative_sensitivity():
    """ML-01: Verify simulation shows realistic sensitivity to creative parameters for unpublished posts."""
    profile = ProfileInput(
        username="creative_tester",
        full_name="Creative Tester",
        country="US",
        total_followers=100_000,
        total_following=400,
        total_media_posts=250,
        is_verified=False,
        account_category=ContentCategory.EDUCATION_CAREERS
    )

    # 1. Media format sensitivity: Reel vs Static Image
    post_static = PostInput(
        media_type=MediaType.STATIC_IMAGE,
        category=ContentCategory.EDUCATION_CAREERS,
        categorization=ContentStyle.EDUCATIONAL,
        has_call_to_action=False,
        posted_hour_of_day=15,
        posted_day_of_week="Tuesday",
    )
    post_reel = PostInput(
        media_type=MediaType.REEL,
        category=ContentCategory.EDUCATION_CAREERS,
        categorization=ContentStyle.EDUCATIONAL,
        has_call_to_action=False,
        posted_hour_of_day=15,
        posted_day_of_week="Tuesday",
    )
    sim_static = simulate_post_performance(profile, post_static)
    sim_reel = simulate_post_performance(profile, post_reel)

    assert sim_reel.projected_reach.point_estimate > sim_static.projected_reach.point_estimate, (
        f"Reel reach ({sim_reel.projected_reach.point_estimate}) should exceed Static ({sim_static.projected_reach.point_estimate})"
    )
    assert sim_reel.projected_impressions.point_estimate > sim_static.projected_impressions.point_estimate

    # 2. Call to Action sensitivity: CTA=True vs CTA=False
    post_with_cta = PostInput(
        media_type=MediaType.REEL,
        category=ContentCategory.EDUCATION_CAREERS,
        categorization=ContentStyle.EDUCATIONAL,
        has_call_to_action=True,
        posted_hour_of_day=15,
        posted_day_of_week="Tuesday",
    )
    sim_cta = simulate_post_performance(profile, post_with_cta)
    assert sim_cta.projected_reach.point_estimate >= sim_reel.projected_reach.point_estimate
    assert sim_cta.projected_save_rate >= sim_reel.projected_save_rate

    # 3. Timing sensitivity: Peak hour (18:00) vs Off-peak hour (03:00)
    post_off_peak = PostInput(
        media_type=MediaType.REEL,
        category=ContentCategory.EDUCATION_CAREERS,
        categorization=ContentStyle.EDUCATIONAL,
        has_call_to_action=True,
        posted_hour_of_day=3,
        posted_day_of_week="Tuesday",
    )
    post_peak = PostInput(
        media_type=MediaType.REEL,
        category=ContentCategory.EDUCATION_CAREERS,
        categorization=ContentStyle.EDUCATIONAL,
        has_call_to_action=True,
        posted_hour_of_day=18,
        posted_day_of_week="Tuesday",
    )
    sim_off_peak = simulate_post_performance(profile, post_off_peak)
    sim_peak = simulate_post_performance(profile, post_peak)
    assert sim_peak.projected_reach.point_estimate >= sim_off_peak.projected_reach.point_estimate


def test_simulate_post_performance_extreme_numerical_bounds():
    """ML-07: Verify conformal bounds do not overflow on extreme values."""
    profile_extreme = ProfileInput(
        username="extreme_account",
        full_name="Extreme Account",
        country="US",
        total_followers=500_000_000,
        total_following=100,
        total_media_posts=5000,
        is_verified=True,
        account_category=ContentCategory.SPORTS
    )
    post_extreme = PostInput(
        media_type=MediaType.REEL,
        category=ContentCategory.SPORTS,
        categorization=ContentStyle.ENTERTAINING,
        posted_hour_of_day=18,
        posted_day_of_week="Sunday",
    )

    sim = simulate_post_performance(profile_extreme, post_extreme, confidence_level=0.90)

    # Bounds must be finite positive integers without overflow
    assert isinstance(sim.projected_reach.lower, int)
    assert isinstance(sim.projected_reach.upper, int)
    assert isinstance(sim.projected_impressions.lower, int)
    assert isinstance(sim.projected_impressions.upper, int)
    assert sim.projected_reach.lower <= sim.projected_reach.point_estimate <= sim.projected_reach.upper
    assert sim.projected_impressions.lower <= sim.projected_impressions.point_estimate <= sim.projected_impressions.upper
    assert sim.projected_reach.upper < float("inf")
    assert sim.projected_impressions.upper < float("inf")
