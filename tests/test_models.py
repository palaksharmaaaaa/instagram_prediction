from instagram_predictor.models import (
    get_reach_pipeline,
    get_impressions_pipeline,
    get_pre_publish_reach_pipeline,
    get_pre_publish_impressions_pipeline,
    get_diagnostic_reach_pipeline,
    get_diagnostic_impressions_pipeline,
    predict_batch,
    simulate_post_performance,
    explain_post_prediction,
    get_model_metadata
)
from instagram_predictor.data import (
    load_dataset,
    compute_derived_metrics,
    ALL_PRE_PUBLISH_FEATURE_COLUMNS,
    ALL_DIAGNOSTIC_FEATURE_COLUMNS,
    PRE_PUBLISH_FEATURE_COLUMNS_NUMERIC,
    PRE_PUBLISH_FEATURE_COLUMNS_CATEGORICAL,
)
from instagram_predictor.schemas import (
    ProfileInput,
    PostInput,
    MediaType,
    ContentCategory,
    ContentStyle,
    Demographics,
    PostMetrics,
)
import numpy as np
import pandas as pd


def test_pipelines_loaded():
    reach_p = get_reach_pipeline()
    imp_p = get_impressions_pipeline()
    pre_reach_p = get_pre_publish_reach_pipeline()
    pre_imp_p = get_pre_publish_impressions_pipeline()
    diag_reach_p = get_diagnostic_reach_pipeline()
    diag_imp_p = get_diagnostic_impressions_pipeline()

    assert reach_p is not None
    assert imp_p is not None
    assert pre_reach_p is not None
    assert pre_imp_p is not None
    assert diag_reach_p is not None
    assert diag_imp_p is not None


def test_model_conformal_metadata():
    meta = get_model_metadata()
    assert "conformal_quantile_80" in meta["evaluation"]["reach"]
    assert "conformal_quantile_90" in meta["evaluation"]["reach"]
    assert "conformal_quantile_80" in meta["evaluation"]["impressions"]
    assert "group_cv_r2_mean" in meta["evaluation"]["reach"]

    # Pre-publishing model evaluation in metadata
    assert "pre_publish_reach" in meta["evaluation"]
    assert "pre_publish_impressions" in meta["evaluation"]
    assert "tier_conformal_quantiles" in meta["evaluation"]["pre_publish_reach"]
    assert "tier_conformal_quantiles" in meta["evaluation"]["pre_publish_impressions"]


def test_pre_publishing_feature_isolation():
    """Verify pre-publishing pipelines execute strictly on pre-publish columns without post-pub metrics."""
    pre_reach_pipe = get_pre_publish_reach_pipeline()
    pre_imp_pipe = get_pre_publish_impressions_pipeline()

    # Construct input with strictly pre-publication columns
    sample_data = {
        "username": "tester",
        "country": "US",
        "total_followers": 50_000,
        "total_following": 300,
        "total_media_posts": 100,
        "account_age_years": 2.5,
        "posting_frequency_per_week": 4.0,
        "follower_growth_rate_30d": 0.03,
        "account_bio_has_link": True,
        "media_type": "Reel",
        "category": "Sports",
        "categorization": "Educational / How-To",
        "caption_length_chars": 280,
        "hashtags_count": 6,
        "mentions_count": 1,
        "has_call_to_action": True,
        "video_duration_seconds": 15.0,
        "carousel_slide_count": 1,
        "posted_day_of_week": "Friday",
        "posted_hour_of_day": 18,
        "top_country": "US",
        "secondary_country": "GB",
        "primary_age_group": "25-34",
        "gender_female_pct": 0.52,
        "gender_male_pct": 0.48,
        "audience_activity_score": 0.80,
    }

    # Derive features - notice zero post-pub metrics provided
    df_derived = compute_derived_metrics(pd.DataFrame([sample_data]))

    # Ensure strictly ALL_PRE_PUBLISH_FEATURE_COLUMNS are passed
    X_pre = df_derived[ALL_PRE_PUBLISH_FEATURE_COLUMNS]
    assert "per_media_likes" not in ALL_PRE_PUBLISH_FEATURE_COLUMNS
    assert "per_media_shares" not in ALL_PRE_PUBLISH_FEATURE_COLUMNS
    assert "per_media_saves" not in ALL_PRE_PUBLISH_FEATURE_COLUMNS

    pred_reach = pre_reach_pipe.predict(X_pre)
    pred_imp = pre_imp_pipe.predict(X_pre)

    assert len(pred_reach) == 1
    assert len(pred_imp) == 1
    assert pred_reach[0] > 0
    assert pred_imp[0] > 0
    assert np.isfinite(pred_reach[0])
    assert np.isfinite(pred_imp[0])


def test_batch_prediction_automatic_routing():
    """Verify predict_batch automatically routes based on presence of post-publication metrics."""
    df_full = load_dataset().head(5)

    # 1. With post-pub metrics: routes to diagnostic model
    preds_diag = predict_batch(df_full)
    assert "predicted_reach" in preds_diag.columns
    assert "predicted_impressions" in preds_diag.columns
    assert (preds_diag["predicted_reach"] <= preds_diag["predicted_impressions"]).all()

    # 2. Lacking post-pub metrics: routes to pre-publishing model
    post_pub_cols = ["per_media_likes", "per_media_comments", "per_media_shares", "per_media_saves",
                     "per_media_video_views", "per_media_completion_rate", "reach_from_home_pct",
                     "reach_from_explore_pct", "reach_from_hashtags_pct", "reach_from_other_pct",
                     "per_media_reach", "per_media_impressions"]
    df_no_post_pub = df_full.drop(columns=[c for c in post_pub_cols if c in df_full.columns])
    preds_pre = predict_batch(df_no_post_pub)
    assert "predicted_reach" in preds_pre.columns
    assert "predicted_impressions" in preds_pre.columns
    assert (preds_pre["predicted_reach"] <= preds_pre["predicted_impressions"]).all()


def test_diagnostic_pipelines_execute_with_engagement_metrics():
    """Verify diagnostic pipelines execute on full diagnostic feature space with verified metrics."""
    profile = ProfileInput(
        username="verified_creator",
        full_name="Verified Creator",
        country="US",
        total_followers=250_000,
        total_following=500,
        total_media_posts=300,
        is_verified=True,
        account_category=ContentCategory.FASHION_BEAUTY,
    )
    post_published = PostInput(
        media_type=MediaType.CAROUSEL,
        category=ContentCategory.FASHION_BEAUTY,
        categorization=ContentStyle.INSPIRATIONAL,
        demographics=Demographics(top_country="US", primary_age_group="25-34", gender_female_pct=0.6),
        metrics=PostMetrics(likes=12_000, comments=450, shares=600, saves=1_200, video_views=0, completion_rate=0.0),
    )

    sim = simulate_post_performance(profile, post_published)
    assert sim.projected_reach.lower <= sim.projected_reach.point_estimate <= sim.projected_reach.upper
    assert sim.projected_impressions.lower <= sim.projected_impressions.point_estimate <= sim.projected_impressions.upper
    assert sim.projected_reach.point_estimate <= sim.projected_impressions.point_estimate
    assert sim.projected_engagement_rate > 0
    assert sim.projected_save_rate > 0


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
    """ML-01: Verify pre-publish What-If simulation responds directly and sensitively to creative parameters."""
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


def test_explain_post_prediction_structure_and_contributions():
    """Phase 3: Verify TreeSHAP explain_post_prediction returns valid drivers, non-zero contributions, and matches final reach."""
    profile = ProfileInput(
        username="creative_influencer",
        full_name="Creative Influencer",
        country="US",
        total_followers=100_000,
        total_following=400,
        total_media_posts=250,
        account_category=ContentCategory.EDUCATION_CAREERS,
    )
    post = PostInput(
        media_type=MediaType.REEL,
        category=ContentCategory.EDUCATION_CAREERS,
        categorizations=[ContentStyle.EDUCATIONAL, ContentStyle.ENTERTAINING],
        categorization=ContentStyle.EDUCATIONAL,
        has_call_to_action=True,
        caption_length_chars=350,
        hashtags_count=7,
        posted_hour_of_day=18,
        posted_day_of_week="Friday",
        video_duration_seconds=20.0,
        carousel_slide_count=1,
        demographics=Demographics(top_country="US", primary_age_group="25-34", gender_female_pct=0.52),
    )

    explanation = explain_post_prediction(post=post, profile=profile)

    # 1. Verify schema structure
    assert "base_reach" in explanation
    assert "final_reach" in explanation
    assert "drivers" in explanation
    assert isinstance(explanation["base_reach"], int)
    assert isinstance(explanation["final_reach"], int)
    assert explanation["base_reach"] > 0
    assert explanation["final_reach"] > 0

    # 2. Verify all 6 creative levers are represented
    drivers = explanation["drivers"]
    assert len(drivers) == 6
    for d in drivers:
        assert "name" in d
        assert "impact" in d
        assert "pct" in d
        assert "direction" in d
        assert "description" in d
        assert isinstance(d["name"], str)
        assert isinstance(d["impact"], int)
        assert isinstance(d["pct"], float)
        assert d["direction"] in ["positive", "negative"]
        assert isinstance(d["description"], str)
        assert len(d["description"]) > 0

    # 3. Verify non-zero contributions
    assert any(d["impact"] != 0 for d in drivers)
    reel_driver = next((d for d in drivers if "Reel" in d["name"]), None)
    assert reel_driver is not None
    assert reel_driver["impact"] > 0, "Reel media format should have a positive reach impact"
    assert reel_driver["direction"] == "positive"

    # 4. Verify mathematical reach conservation: base_reach + sum(impacts) == final_reach
    total_impact = sum(d["impact"] for d in drivers)
    assert explanation["base_reach"] + total_impact == explanation["final_reach"]


def test_explain_post_prediction_argument_order_robustness():
    """Verify explain_post_prediction produces identical results regardless of argument order."""
    profile = ProfileInput(
        username="order_tester",
        country="US",
        total_followers=50_000,
        total_following=300,
        total_media_posts=150,
        account_category=ContentCategory.SPORTS,
    )
    post = PostInput(
        media_type=MediaType.CAROUSEL,
        category=ContentCategory.SPORTS,
        categorization=ContentStyle.INSPIRATIONAL,
        carousel_slide_count=5,
        has_call_to_action=True,
        hashtags_count=5,
    )

    res1 = explain_post_prediction(post=post, profile=profile)
    res2 = explain_post_prediction(profile, post)

    assert res1["base_reach"] == res2["base_reach"]
    assert res1["final_reach"] == res2["final_reach"]
    assert len(res1["drivers"]) == len(res2["drivers"])
    for d1, d2 in zip(res1["drivers"], res2["drivers"]):
        assert d1["name"] == d2["name"]
        assert d1["impact"] == d2["impact"]
        assert d1["pct"] == d2["pct"]
        assert d1["direction"] == d2["direction"]


def test_simulate_post_performance_includes_feature_explanations():
    """Verify simulate_post_performance populates feature_explanations for pre-publishing simulations."""
    profile = ProfileInput(
        username="sim_explained_user",
        country="US",
        total_followers=75_000,
        total_following=450,
        total_media_posts=200,
        account_category=ContentCategory.SCIENCE_TECHNOLOGY,
    )
    post = PostInput(
        media_type=MediaType.REEL,
        category=ContentCategory.SCIENCE_TECHNOLOGY,
        categorization=ContentStyle.EDUCATIONAL,
        has_call_to_action=True,
        hashtags_count=8,
        posted_hour_of_day=19,
        posted_day_of_week="Thursday",
    )

    sim = simulate_post_performance(profile, post)
    assert sim.feature_explanations is not None
    assert sim.feature_explanations["final_reach"] == sim.projected_reach.point_estimate
    assert (
        sim.feature_explanations["base_reach"] + sum(d["impact"] for d in sim.feature_explanations["drivers"])
        == sim.feature_explanations["final_reach"]
    )


def test_simulator_service_explain_post_simulation():
    """Verify SimulatorService provides explain_post_simulation method."""
    from instagram_predictor.services import explain_post_simulation, run_post_simulation

    profile_data = {
        "username": "service_tester",
        "country": "US",
        "total_followers": 60_000,
        "total_following": 300,
        "total_media_posts": 150,
        "account_category": "Sports",
    }
    post_data = {
        "media_type": "Reel",
        "category": "Sports",
        "categorizations": ["Entertaining / Trend"],
        "categorization": "Entertaining / Trend",
        "has_call_to_action": True,
        "hashtags_count": 6,
        "posted_hour_of_day": 18,
        "posted_day_of_week": "Friday",
    }

    # Test dedicated explain method
    success, errs, explanations = explain_post_simulation(profile_data, post_data)
    assert success is True
    assert errs == []
    assert explanations is not None
    assert "drivers" in explanations
    assert (
        explanations["base_reach"] + sum(d["impact"] for d in explanations["drivers"])
        == explanations["final_reach"]
    )

    # Test simulation result includes feature_explanations
    sim_success, sim_errs, sim_res = run_post_simulation(profile_data, post_data)
    assert sim_success is True
    assert sim_errs == []
    assert sim_res.feature_explanations is not None
    assert sim_res.feature_explanations["final_reach"] == sim_res.projected_reach.point_estimate

