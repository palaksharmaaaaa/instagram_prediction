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
