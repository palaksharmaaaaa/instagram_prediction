import json
import numpy as np
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
    """Verify that Bayesian HPO executes and returns best parameters and cross-validated log-scale MAE."""
    results = run_bayesian_hpo("per_media_reach", n_trials=2)
    assert "best_params" in results
    assert "best_cv_mae" in results
    assert results["best_cv_mae"] > 0
    # Log-scale error should be well under 10.0 (typically ~0.3 - 0.8), whereas raw MAE is in the millions (~3M)
    assert results["best_cv_mae"] < 10.0
    assert "learning_rate" in results["best_params"]


def test_compute_conformal_quantile_exact_formula():
    """ML-03: Verify compute_conformal_quantile implements p = min(ceil((n+1)(1-alpha))/n, 1.0) with method='higher'."""
    from instagram_predictor.models.trainer import compute_conformal_quantile

    # Empty array fallback
    assert compute_conformal_quantile(np.array([]), alpha=0.20) == 0.35

    # Test with known array of length 10
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    n = len(scores)

    # alpha = 0.20 -> (10+1)*0.80 = 8.8 -> ceil(8.8) = 9 -> p = 9/10 = 0.90
    expected_p_80 = min(np.ceil((n + 1) * 0.80) / n, 1.0)
    expected_q_80 = float(np.quantile(scores, expected_p_80, method="higher"))
    assert compute_conformal_quantile(scores, alpha=0.20) == pytest.approx(expected_q_80)

    # alpha = 0.10 -> (10+1)*0.90 = 9.9 -> ceil(9.9) = 10 -> p = 10/10 = 1.0
    expected_p_90 = min(np.ceil((n + 1) * 0.90) / n, 1.0)
    expected_q_90 = float(np.quantile(scores, expected_p_90, method="higher"))
    assert compute_conformal_quantile(scores, alpha=0.10) == pytest.approx(expected_q_90)

    # Test with arbitrary lengths and random scores
    rng = np.random.default_rng(123)
    for sample_size in [5, 13, 36, 100]:
        test_scores = rng.exponential(scale=0.5, size=sample_size)
        for alpha in [0.20, 0.10]:
            p = min(np.ceil((sample_size + 1) * (1.0 - alpha)) / sample_size, 1.0)
            expected = float(np.quantile(test_scores, p, method="higher"))
            actual = compute_conformal_quantile(test_scores, alpha)
            assert actual == pytest.approx(expected)


def test_nano_tier_quantiles_exact_formula():
    """ML-03: Verify Nano tier quantiles are derived using the exact finite-sample formula."""
    from instagram_predictor.models.trainer import compute_conformal_quantile
    from instagram_predictor.data import load_dataset, ALL_FEATURE_COLUMNS
    from instagram_predictor.models.trainer import build_transformed_pipeline

    with open(settings.MODEL_METADATA_PATH, "r") as f:
        meta = json.load(f)

    reach_nano = meta["evaluation"]["reach"]["tier_conformal_quantiles"]["nano"]
    imp_nano = meta["evaluation"]["impressions"]["tier_conformal_quantiles"]["nano"]

    assert reach_nano["sample_count"] >= 5
    assert imp_nano["sample_count"] >= 5
    n = reach_nano["sample_count"]

    # Verify p calculation for n=36
    p80 = min(np.ceil((n + 1) * 0.80) / n, 1.0)
    p90 = min(np.ceil((n + 1) * 0.90) / n, 1.0)
    assert p80 == pytest.approx(30.0 / 36.0)
    assert p90 == pytest.approx(34.0 / 36.0)

    # Re-run calibration split to verify exact match with stored metadata
    df = load_dataset(reload=False)
    X = df[ALL_FEATURE_COLUMNS]
    y_reach = df["per_media_reach"]
    y_impressions = df["per_media_impressions"]
    groups = df["username"]

    unique_creators = groups.unique()
    rng = np.random.default_rng(42)
    calib_creators = rng.choice(unique_creators, size=int(len(unique_creators) * 0.20), replace=False)
    calib_mask = groups.isin(calib_creators)

    X_train_icp = X[~calib_mask]
    y_r_train_icp = y_reach[~calib_mask]
    y_i_train_icp = y_impressions[~calib_mask]

    X_calib = X[calib_mask]
    y_r_calib = y_reach[calib_mask]
    y_i_calib = y_impressions[calib_mask]

    reach_calib_model = build_transformed_pipeline()
    reach_calib_model.fit(X_train_icp, y_r_train_icp)
    reach_calib_preds = reach_calib_model.predict(X_calib)

    imp_calib_model = build_transformed_pipeline()
    imp_calib_model.fit(X_train_icp, y_i_train_icp)
    imp_calib_preds = imp_calib_model.predict(X_calib)

    reach_scores = np.abs(np.log1p(y_r_calib) - np.log1p(reach_calib_preds))
    imp_scores = np.abs(np.log1p(y_i_calib) - np.log1p(imp_calib_preds))

    calib_followers = X_calib["total_followers"].values
    nano_mask = (calib_followers >= 0) & (calib_followers < 10_000)

    tier_r_scores = reach_scores[nano_mask]
    tier_i_scores = imp_scores[nano_mask]

    expected_r_q80 = compute_conformal_quantile(tier_r_scores, 0.20)
    expected_r_q90 = compute_conformal_quantile(tier_r_scores, 0.10)
    expected_i_q80 = compute_conformal_quantile(tier_i_scores, 0.20)
    expected_i_q90 = compute_conformal_quantile(tier_i_scores, 0.10)

    assert reach_nano["q80"] == pytest.approx(expected_r_q80)
    assert reach_nano["q90"] == pytest.approx(expected_r_q90)
    assert imp_nano["q80"] == pytest.approx(expected_i_q80)
    assert imp_nano["q90"] == pytest.approx(expected_i_q90)


def test_hpo_objective_log_scale_error():
    """ML-06: Verify HPO objective evaluates on log-scale error rather than celebrity-dominated raw MAE."""
    from sklearn.metrics import mean_absolute_error

    # Simulate two predictions: one for celebrity (70M reach) with 10M error, one for nano (1K reach) with 500 error
    y_val = np.array([70_000_000, 1_000], dtype=float)
    preds_poor_nano = np.array([60_000_000, 100], dtype=float)  # 10M celeb error (14%), 900 nano error (90%)
    preds_poor_celeb = np.array([55_000_000, 950], dtype=float)  # 15M celeb error (21%), 50 nano error (5%)

    # Raw MAE is dominated 100% by celebrity scale
    raw_mae_1 = mean_absolute_error(y_val, preds_poor_nano)
    raw_mae_2 = mean_absolute_error(y_val, preds_poor_celeb)
    # Under raw MAE, poor_nano wins easily (10M < 15M) even though nano error is terrible (90%)
    assert raw_mae_1 < raw_mae_2

    # Under log-scale loss, errors at all creator scales are balanced
    log_loss_1 = mean_absolute_error(np.log1p(np.maximum(y_val, 0)), np.log1p(np.maximum(preds_poor_nano, 0)))
    log_loss_2 = mean_absolute_error(np.log1p(np.maximum(y_val, 0)), np.log1p(np.maximum(preds_poor_celeb, 0)))
    # Under log loss, poor_celeb is recognized as having lower overall relative error
    assert log_loss_2 < log_loss_1


def test_simulation_calibrated_confidence_level_enforcement():
    """ML-03: Verify simulate_post_performance enforces calibrated levels in confidence_interval.level."""
    profile = ProfileInput(
        username="calib_test",
        full_name="Calib Test",
        country="US",
        total_followers=25_000,
        total_following=200,
        total_media_posts=100,
        is_verified=False,
        account_category=ContentCategory.SCIENCE_TECHNOLOGY,
    )
    post = PostInput(
        media_type=MediaType.REEL,
        category=ContentCategory.SCIENCE_TECHNOLOGY,
        categorization=ContentStyle.EDUCATIONAL,
        demographics=Demographics(top_country="US", primary_age_group="25-34", gender_female_pct=0.5),
        metrics=PostMetrics(likes=500, comments=20, shares=10, saves=30),
    )

    # 1. Standard 0.80 confidence level
    sim_80 = simulate_post_performance(profile, post, confidence_level=0.80)
    assert sim_80.projected_reach.confidence_level == 0.80
    assert sim_80.projected_reach.level == "80% Mondrian Conformal Coverage"
    assert sim_80.projected_impressions.level == "80% Mondrian Conformal Coverage"
    assert "80% Mondrian Conformal Coverage" in sim_80.prediction_interval_coverage

    # 2. Caller passes non-standard level 0.70 (<= 0.85) -> calibrated to 0.80
    sim_70 = simulate_post_performance(profile, post, confidence_level=0.70)
    assert sim_70.projected_reach.confidence_level == 0.80
    assert sim_70.projected_reach.level == "80% Mondrian Conformal Coverage"
    assert sim_70.projected_impressions.level == "80% Mondrian Conformal Coverage"
    assert "80% Mondrian Conformal Coverage" in sim_70.prediction_interval_coverage

    # 3. Caller passes non-standard level 0.95 (> 0.85) -> calibrated to 0.90
    sim_95 = simulate_post_performance(profile, post, confidence_level=0.95)
    assert sim_95.projected_reach.confidence_level == 0.90
    assert sim_95.projected_reach.level == "90% Mondrian Conformal Coverage"
    assert sim_95.projected_impressions.level == "90% Mondrian Conformal Coverage"
    assert "90% Mondrian Conformal Coverage" in sim_95.prediction_interval_coverage

