"""
Audit Hardening & Edge-Case Verification Test Suite (TEST-01)
=============================================================
Consolidates and rigorously verifies end-to-end edge cases across all 21 audit findings:
1. Negative / Malformed Inputs: Empty DataFrames, missing columns, all-NaN rows.
2. Invariant Enforcement: Invariant reach <= impressions across extreme and adversarial inputs.
3. Epistemic OOD & Boundary Handling: Extreme follower scales (1B, 0), negative value rejection.
4. Security Hardening: Tamper detection, regex metacharacters, nested HTML evasion, CSV injection.
5. Concurrency & Thread-Safety: Deadlock-free concurrent registry, loader, and simulation calls.
6. Conformal Calibration & Mondrian Tiers: Exact finite-sample quantile formula & 4-tier validity.
7. NLP Robustness: Conversational 'for' disambiguation and decimal follower multipliers.
8. Simulation Sensitivity: Tangible impact of creative controls (CTA, hashtags, media, style, timing).
"""

import concurrent.futures
import json
import os
import tempfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from instagram_predictor.config import settings
from instagram_predictor.data import (
    ALL_FEATURE_COLUMNS,
    FEATURE_COLUMNS_NUMERIC,
    FEATURE_COLUMNS_CATEGORICAL,
    compute_derived_metrics,
    load_dataset,
    clear_loader_cache,
)
from instagram_predictor.guardrails import (
    sanitize_query_input,
    sanitize_prompt,
    sanitize_csv_cell,
    sanitize_dataframe_for_csv,
)
from instagram_predictor.models import (
    predict_batch,
    simulate_post_performance,
    get_reach_pipeline,
    get_impressions_pipeline,
    get_model_metadata,
    clear_registry_cache,
    verify_artifact_integrity,
    SecurityError,
)
from instagram_predictor.models.trainer import compute_conformal_quantile
from instagram_predictor.nlp import parse_query
from instagram_predictor.schemas import (
    ProfileInput,
    PostInput,
    MediaType,
    ContentCategory,
    ContentStyle,
    Demographics,
    PostMetrics,
)
from instagram_predictor.services.analytics_service import apply_query_filters


# =============================================================================
# 1. Negative & Malformed Inputs
# =============================================================================

def test_negative_malformed_empty_dataframe():
    """Verify compute_derived_metrics and predict_batch handle empty DataFrames gracefully."""
    empty_df = pd.DataFrame()

    # compute_derived_metrics must not crash on empty DataFrame
    derived_empty = compute_derived_metrics(empty_df)
    assert isinstance(derived_empty, pd.DataFrame)
    assert derived_empty.empty
    assert len(derived_empty) == 0

    # predict_batch must return an empty DataFrame without exceptions
    preds_empty = predict_batch(empty_df)
    assert isinstance(preds_empty, pd.DataFrame)
    assert preds_empty.empty
    assert len(preds_empty) == 0

    # Empty DataFrame with predefined columns
    empty_with_cols = pd.DataFrame(columns=["total_followers", "per_media_likes"])
    preds_with_cols = predict_batch(empty_with_cols)
    assert isinstance(preds_with_cols, pd.DataFrame)
    assert preds_with_cols.empty


def test_negative_malformed_missing_columns():
    """Verify handling when required ML columns are absent from input DataFrame."""
    # Input with only arbitrary unrelated columns
    df_missing = pd.DataFrame([
        {"unrelated_col": 123, "random_text": "hello"},
        {"unrelated_col": 456, "random_text": "world"},
    ])

    derived = compute_derived_metrics(df_missing)
    assert isinstance(derived, pd.DataFrame)
    assert len(derived) == 2

    # All required numeric and categorical columns must be populated with defaults
    for col in ALL_FEATURE_COLUMNS:
        assert col in derived.columns, f"Expected column {col} to be populated in derived DataFrame"

    # Batch prediction should succeed with default imputed features
    preds = predict_batch(df_missing)
    assert "predicted_reach" in preds.columns
    assert "predicted_impressions" in preds.columns
    assert len(preds) == 2
    assert (preds["predicted_reach"] >= 100).all()
    assert (preds["predicted_impressions"] >= preds["predicted_reach"]).all()


def test_negative_malformed_all_nan_rows():
    """Verify compute_derived_metrics and predict_batch gracefully handle rows of all NaNs."""
    df_all_nan = pd.DataFrame([{col: np.nan for col in ALL_FEATURE_COLUMNS}])

    derived = compute_derived_metrics(df_all_nan)
    assert len(derived) == 1

    # Verify no unhandled NaNs or infs in numeric feature columns
    for col in FEATURE_COLUMNS_NUMERIC:
        val = derived[col].iloc[0]
        assert not np.isnan(val), f"Column {col} resulted in NaN on all-NaN input row"
        assert not np.isinf(val), f"Column {col} resulted in Inf on all-NaN input row"

    # Verify predict_batch works on all-NaN inputs
    preds = predict_batch(df_all_nan)
    assert "predicted_reach" in preds.columns
    assert "predicted_impressions" in preds.columns
    reach = preds["predicted_reach"].iloc[0]
    imp = preds["predicted_impressions"].iloc[0]
    assert isinstance(reach, (int, np.integer)) and reach >= 100
    assert isinstance(imp, (int, np.integer)) and imp >= reach


# =============================================================================
# 2. Invariant Enforcement (reach <= impressions)
# =============================================================================

def test_invariant_enforcement_predict_batch():
    """Verify predict_batch never produces reach > impressions under adversarial inputs."""
    adversarial_df = pd.DataFrame([
        # Scenario 1: Extreme likes/comments with 0 views
        {
            "total_followers": 500_000,
            "per_media_likes": 100_000,
            "per_media_comments": 10_000,
            "per_media_shares": 50_000,
            "per_media_saves": 20_000,
            "per_media_video_views": 0,
            "media_type": "Static Image",
        },
        # Scenario 2: Massive followers with minimal engagement
        {
            "total_followers": 80_000_000,
            "per_media_likes": 10,
            "per_media_comments": 1,
            "per_media_shares": 0,
            "per_media_saves": 0,
            "media_type": "Reel",
        },
        # Scenario 3: Negative numeric values in raw input
        {
            "total_followers": -10_000,
            "per_media_likes": -500,
            "per_media_comments": -50,
            "per_media_shares": -10,
            "per_media_saves": -5,
        },
        # Scenario 4: Extreme engagement rate exceeding 1000%
        {
            "total_followers": 100,
            "per_media_likes": 50_000,
            "per_media_comments": 5_000,
            "per_media_shares": 10_000,
            "per_media_saves": 5_000,
        },
    ])

    preds = predict_batch(adversarial_df)
    assert len(preds) == 4
    for idx, row in preds.iterrows():
        reach = row["predicted_reach"]
        imp = row["predicted_impressions"]
        assert imp >= reach, (
            f"Row {idx} violated invariant: reach ({reach}) > impressions ({imp})"
        )


def test_invariant_enforcement_simulate_post_performance():
    """Verify simulate_post_performance enforces reach <= impressions and valid CI bounds."""
    test_scenarios = [
        # Nano creator with high virality
        (500, ContentCategory.SPORTS, MediaType.REEL, ContentStyle.ENTERTAINING, 50, 10, 40, 20),
        # Micro creator standard post
        (25_000, ContentCategory.FASHION_BEAUTY, MediaType.CAROUSEL, ContentStyle.INSPIRATIONAL, 800, 30, 20, 150),
        # Macro creator with high shares
        (350_000, ContentCategory.EDUCATION_CAREERS, MediaType.STATIC_IMAGE, ContentStyle.EDUCATIONAL, 10_000, 500, 2_000, 4_000),
        # Mega creator unpublished post (no metrics provided)
        (15_000_000, ContentCategory.MUSIC_ENTERTAINMENT, MediaType.REEL, ContentStyle.ENTERTAINING, None, None, None, None),
    ]

    for followers, cat, m_type, style, likes, comments, shares, saves in test_scenarios:
        profile = ProfileInput(
            username="invariant_tester",
            full_name="Invariant Tester",
            country="US",
            total_followers=followers,
            total_following=300,
            total_media_posts=150,
            account_category=cat,
        )

        metrics = None
        if likes is not None:
            metrics = PostMetrics(likes=likes, comments=comments, shares=shares, saves=saves)

        post = PostInput(
            media_type=m_type,
            category=cat,
            categorization=style,
            metrics=metrics,
        )

        sim = simulate_post_performance(profile, post, confidence_level=0.80)

        # Invariant 1: Point estimate reach <= impressions
        assert sim.projected_reach.point_estimate <= sim.projected_impressions.point_estimate, (
            f"Point estimate violation for {followers} followers: "
            f"reach={sim.projected_reach.point_estimate}, imp={sim.projected_impressions.point_estimate}"
        )

        # Invariant 2: Reach lower <= point <= upper
        assert sim.projected_reach.lower <= sim.projected_reach.point_estimate <= sim.projected_reach.upper, (
            f"Reach CI ordering violation: lower={sim.projected_reach.lower}, "
            f"point={sim.projected_reach.point_estimate}, upper={sim.projected_reach.upper}"
        )

        # Invariant 3: Impressions lower <= point <= upper
        assert sim.projected_impressions.lower <= sim.projected_impressions.point_estimate <= sim.projected_impressions.upper, (
            f"Impression CI ordering violation: lower={sim.projected_impressions.lower}, "
            f"point={sim.projected_impressions.point_estimate}, upper={sim.projected_impressions.upper}"
        )

        # Invariant 4: Conformal interval reach bounds <= impression bounds
        assert sim.projected_reach.lower <= sim.projected_impressions.lower
        assert sim.projected_reach.upper <= sim.projected_impressions.upper


# =============================================================================
# 3. Epistemic OOD & Boundary Handling
# =============================================================================

def test_boundary_handling_extreme_followers():
    """Verify handling of extreme follower counts (1B, 0) with finite bounds and OOD alerts."""
    # 1. Billion follower scale
    profile_billion = ProfileInput(
        username="billionaire_account",
        full_name="Billionaire Account",
        country="US",
        total_followers=1_000_000_000,
        total_following=50,
        total_media_posts=2000,
        is_verified=True,
        account_category=ContentCategory.SPORTS,
    )
    post = PostInput(
        media_type=MediaType.REEL,
        category=ContentCategory.SPORTS,
        categorization=ContentStyle.ENTERTAINING,
    )

    sim_billion = simulate_post_performance(profile_billion, post)
    # Must trigger Extrapolation Alert
    assert "Extrapolation Alert" in sim_billion.uncertainty_rating
    # Bounds must remain finite positive integers (no overflow to inf or NaN)
    assert 0 < sim_billion.projected_reach.lower <= sim_billion.projected_reach.point_estimate <= sim_billion.projected_reach.upper
    assert sim_billion.projected_reach.upper < float("inf")
    assert sim_billion.projected_impressions.upper < float("inf")

    # 2. Zero followers scale
    profile_zero = ProfileInput(
        username="zero_account",
        full_name="Zero Account",
        country="US",
        total_followers=0,
        total_following=10,
        total_media_posts=0,
        account_category=ContentCategory.SPORTS,
    )
    sim_zero = simulate_post_performance(profile_zero, post)
    assert "Extrapolation Alert" in sim_zero.uncertainty_rating
    assert sim_zero.projected_reach.point_estimate >= 100
    assert sim_zero.projected_impressions.point_estimate >= sim_zero.projected_reach.point_estimate
    assert sim_zero.projected_reach.lower >= 50


def test_boundary_handling_negative_values():
    """Verify negative values in schemas raise ValidationError, and in DataFrames are gracefully clipped."""
    # 1. ProfileInput negative follower count
    with pytest.raises(ValidationError):
        ProfileInput(
            username="neg_user",
            total_followers=-100,
            total_following=100,
            total_media_posts=10,
        )

    # 2. ProfileInput following count exceeding platform limit (7,500)
    with pytest.raises(ValidationError):
        ProfileInput(
            username="overflow_user",
            total_followers=1000,
            total_following=7501,
            total_media_posts=10,
        )

    # 3. PostMetrics negative counts
    with pytest.raises(ValidationError):
        PostMetrics(likes=-10, comments=0, shares=0, saves=0)

    # 4. DataFrame with negative values passed to compute_derived_metrics & predict_batch
    df_negative = pd.DataFrame([{
        "total_followers": -50_000,
        "total_following": -200,
        "total_media_posts": -10,
        "per_media_likes": -500,
        "per_media_comments": -50,
        "per_media_shares": -100,
        "per_media_saves": -20,
    }])
    derived = compute_derived_metrics(df_negative)
    assert derived["total_followers"].iloc[0] >= 1.0
    assert derived["total_following"].iloc[0] >= 0.0
    assert derived["per_media_likes"].iloc[0] >= 0.0

    preds = predict_batch(df_negative)
    assert preds["predicted_reach"].iloc[0] >= 100
    assert preds["predicted_impressions"].iloc[0] >= preds["predicted_reach"].iloc[0]


# =============================================================================
# 4. Security Hardening
# =============================================================================

def test_security_tamper_detection(monkeypatch):
    """SEC-01: Verify verify_artifact_integrity catches tampered model artifacts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_model = Path(tmp_dir) / "tampered_reach_model.joblib"
        tmp_model.write_bytes(b"tampered-binary-model-data")

        tmp_meta = Path(tmp_dir) / "model_metadata.json"
        tmp_meta.write_text(
            json.dumps({
                "artifact_hashes": {
                    "tampered_reach_model.joblib": "badc0de000000000000000000000000000000000000000000000000000000000"
                }
            })
        )
        monkeypatch.setattr(settings, "MODEL_METADATA_PATH", tmp_meta)

        with pytest.raises(SecurityError, match="Artifact integrity verification failed"):
            verify_artifact_integrity(tmp_model, "reach_pipeline")


def test_security_regex_metacharacters_in_filters():
    """SEC-02: Verify regex special characters in query filters do not trigger re.error."""
    df = pd.DataFrame({
        "category": ["Sports (Football)", "Tech [Gadgets]", "Fashion & Style"],
        "account_category": ["Sports", "Tech", "Fashion"],
        "categorization": ["Educational (Tips)", "Entertaining [Fun]", "Standard"],
        "username": ["star(athlete)", "tech[guru]", "fashion_icon"],
        "full_name": ["Star Athlete (Official)", "Tech Guru [Pro]", "Fashion Icon"],
    })

    dangerous_chars = ["(", "[", "*", "+", "?", "\\", "^", "$", "(?=", "[a-z]+", ".*"]
    for char in dangerous_chars:
        # Category filter
        res_cat = apply_query_filters(df, {"category": char})
        assert isinstance(res_cat, pd.DataFrame)

        # Username filter
        res_user = apply_query_filters(df, {"username": char})
        assert isinstance(res_user, pd.DataFrame)

        # Categorization filter
        res_style = apply_query_filters(df, {"categorization": char})
        assert isinstance(res_style, pd.DataFrame)


def test_security_nested_html_evasion():
    """SEC-03: Verify nested and malformed HTML tags are completely neutralized."""
    malicious_inputs = [
        ("<<<<script>script>script>alert(1)</script>", "HTML_TAGS_STRIPPED"),
        ("<img src=x onerror=alert(1)>", "HTML_TAGS_STRIPPED"),
        ("<svg onload=alert(1)>", "HTML_TAGS_STRIPPED"),
        ("<<SCRIPT>script>alert('xss')<</SCRIPT>/script>", "HTML_TAGS_STRIPPED"),
    ]

    for raw, expected_flag in malicious_inputs:
        clean, flags = sanitize_query_input(raw)
        assert "<script>" not in clean.lower()
        assert "<img" not in clean.lower()
        assert "<svg" not in clean.lower()
        assert expected_flag in flags


def test_security_csv_formula_injection_defense():
    """SEC-04: Verify CSV formula injection triggers are escaped with prepended single quote."""
    formula_samples = [
        "=1+1",
        "=cmd|' /C calc'!A0",
        "+12345",
        "-5+2",
        "@SUM(A1:A10)",
    ]

    for formula in formula_samples:
        sanitized = sanitize_csv_cell(formula)
        assert sanitized.startswith("'"), f"Formula '{formula}' was not escaped with leading quote"

    # Verify DataFrame batch sanitization
    test_df = pd.DataFrame({
        "handle": ["=calc", "+test", "-danger", "@mention", "safe_handle"],
        "numeric_val": [-42.5, 100.0, -0.01, 5.0, 0.0],
        "integer_val": [-10, 20, -30, 40, 50],
    })
    sanitized_df = sanitize_dataframe_for_csv(test_df)

    assert sanitized_df.loc[0, "handle"] == "'=calc"
    assert sanitized_df.loc[1, "handle"] == "'+test"
    assert sanitized_df.loc[2, "handle"] == "'-danger"
    assert sanitized_df.loc[3, "handle"] == "'@mention"
    assert sanitized_df.loc[4, "handle"] == "safe_handle"

    # Numerical columns must not be modified or string-escaped
    assert sanitized_df.loc[0, "numeric_val"] == -42.5
    assert sanitized_df.loc[0, "integer_val"] == -10


# =============================================================================
# 5. Concurrency & Thread-Safety
# =============================================================================

def test_concurrency_registry_and_loader_thread_safety():
    """Verify concurrent calls to model registry and dataset loader execute without deadlocks."""
    clear_registry_cache()
    clear_loader_cache()

    num_threads = 10
    barrier = threading.Barrier(num_threads)

    def worker(worker_id: int):
        barrier.wait()
        reach_pipe = get_reach_pipeline()
        imp_pipe = get_impressions_pipeline()
        meta = get_model_metadata()
        df = load_dataset()
        return id(reach_pipe), id(imp_pipe), meta.get("version"), len(df)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        results = [f.result(timeout=20) for f in concurrent.futures.as_completed(futures)]

    reach_ids, imp_ids, versions, row_counts = zip(*results)

    # Double-checked locking must ensure all threads got the exact same cached pipeline references
    assert len(set(reach_ids)) == 1, f"Multiple reach pipeline instances: {set(reach_ids)}"
    assert len(set(imp_ids)) == 1, f"Multiple impressions pipeline instances: {set(imp_ids)}"
    assert len(set(versions)) == 1
    assert len(set(row_counts)) == 1
    assert row_counts[0] > 0


# =============================================================================
# 6. Conformal Calibration & Mondrian Tiers
# =============================================================================

def test_conformal_exact_finite_sample_formula():
    """ML-03: Verify compute_conformal_quantile implements p = min(ceil((n+1)(1-alpha))/n, 1.0)."""
    # Empty array returns default fallback
    assert compute_conformal_quantile(np.array([]), alpha=0.20) == 0.35

    # Test exact finite-sample quantile across several sample sizes
    rng = np.random.default_rng(999)
    for n in [7, 15, 36, 120]:
        scores = rng.exponential(scale=0.5, size=n)
        for alpha in [0.20, 0.10]:
            p = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
            expected = float(np.quantile(scores, p, method="higher"))
            actual = compute_conformal_quantile(scores, alpha)
            assert actual == pytest.approx(expected)


def test_conformal_mondrian_all_four_tiers():
    """ML-03: Verify that Nano, Micro, Macro, and Mega tiers all provide valid intervals."""
    with open(settings.MODEL_METADATA_PATH, "r") as f:
        meta = json.load(f)

    reach_eval = meta["evaluation"]["reach"]
    imp_eval = meta["evaluation"]["impressions"]

    tiers = [
        ("nano", 5_000, "Nano (<10k followers)"),
        ("micro", 50_000, "Micro (10k-100k followers)"),
        ("macro", 500_000, "Macro (100k-1M followers)"),
        ("mega", 5_000_000, "Mega (1M+ followers)"),
    ]

    for tier_key, followers, expected_label in tiers:
        # 1. Metadata quantiles must exist and be strictly positive
        assert tier_key in reach_eval["tier_conformal_quantiles"]
        assert tier_key in imp_eval["tier_conformal_quantiles"]
        r_tier = reach_eval["tier_conformal_quantiles"][tier_key]
        i_tier = imp_eval["tier_conformal_quantiles"][tier_key]
        assert 0.0 < r_tier["q80"] <= r_tier["q90"]
        assert 0.0 < i_tier["q80"] <= i_tier["q90"]

        # 2. Simulation assigns correct calibration tier and produces valid bounds
        profile = ProfileInput(
            username=f"tier_user_{tier_key}",
            total_followers=followers,
            total_following=300,
            total_media_posts=100,
            account_category=ContentCategory.SPORTS,
        )
        post = PostInput(
            media_type=MediaType.REEL,
            category=ContentCategory.SPORTS,
            categorization=ContentStyle.ENTERTAINING,
        )
        sim = simulate_post_performance(profile, post, confidence_level=0.80)

        assert sim.calibration_tier == expected_label
        assert sim.projected_reach.lower <= sim.projected_reach.point_estimate <= sim.projected_reach.upper
        assert sim.projected_impressions.lower <= sim.projected_impressions.point_estimate <= sim.projected_impressions.upper
        assert sim.projected_reach.point_estimate <= sim.projected_impressions.point_estimate


# =============================================================================
# 7. NLP Robustness
# =============================================================================

def test_nlp_for_preposition_conversational_queries():
    """NLP-01: Verify conversational English queries with 'for' do not extract false usernames."""
    conversational_queries = [
        "Show accounts for marketing campaigns",
        "Find creators for summer festival promotion",
        "Predict reach for new product launch",
        "Forecast impressions for upcoming holiday sale",
        "Show posts for fashion trends in Paris",
    ]

    for query in conversational_queries:
        parsed = parse_query(query)
        assert parsed.username is None, f"Extracted false username '{parsed.username}' from query: {query}"
        assert parsed.filters.get("username") is None

    # Handles with explicit @ or 'account of' must still be accurately extracted
    valid_handle_queries = [
        ("Predict reach for @cristiano", "cristiano"),
        ("Show posts for @selenagomez in US", "selenagomez"),
        ("Account of leomessi with above 10m followers", "leomessi"),
    ]

    for query, expected_handle in valid_handle_queries:
        parsed = parse_query(query)
        assert parsed.username == expected_handle
        assert parsed.filters.get("username") == expected_handle


def test_nlp_decimal_multipliers():
    """NLP-02: Verify decimal follower multipliers (k, m, b) parse accurately into float values."""
    test_cases = [
        ("Creators with over 0.5m followers", 500_000.0, ">"),
        ("Creators with above 1.5m followers", 1_500_000.0, ">"),
        ("Accounts under 2.5k followers", 2_500.0, "<"),
        ("Accounts with over 1.2b followers", 1_200_000_000.0, ">"),
        ("Creators with over 0.5M followers", 500_000.0, ">"),
    ]

    for query, expected_val, expected_op in test_cases:
        parsed = parse_query(query)
        assert "total_followers" in parsed.filters
        assert parsed.filters["total_followers"]["value"] == expected_val
        assert parsed.filters["total_followers"]["operator"] == expected_op

    # Between range with decimal multipliers
    range_query = "Creators between 1.5m and 3m followers"
    parsed_range = parse_query(range_query)
    assert parsed_range.filters["total_followers"]["operator"] == "between"
    assert parsed_range.filters["total_followers"]["min"] == 1_500_000.0
    assert parsed_range.filters["total_followers"]["max"] == 3_000_000.0


# =============================================================================
# 8. Simulation Sensitivity to Creative Controls
# =============================================================================

def test_simulation_creative_controls_sensitivity():
    """ML-01: Verify that changing creative controls tangibly affects projected metrics."""
    profile = ProfileInput(
        username="creative_influencer",
        full_name="Creative Influencer",
        country="US",
        total_followers=100_000,
        total_following=400,
        total_media_posts=250,
        account_category=ContentCategory.EDUCATION_CAREERS,
    )

    # 1. Media Type Sensitivity: Reel vs Static Image
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
        "Reel reach should tangibly exceed Static Image reach"
    )
    assert sim_reel.projected_impressions.point_estimate > sim_static.projected_impressions.point_estimate

    # 2. Call to Action Sensitivity: CTA=True vs CTA=False
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

    # 3. Hashtag Sensitivity: Optimal hashtags (8) vs No hashtags (0)
    post_no_hashtags = PostInput(
        media_type=MediaType.REEL,
        category=ContentCategory.EDUCATION_CAREERS,
        categorization=ContentStyle.EDUCATIONAL,
        hashtags_count=0,
        posted_hour_of_day=15,
        posted_day_of_week="Tuesday",
    )
    post_optimal_hashtags = PostInput(
        media_type=MediaType.REEL,
        category=ContentCategory.EDUCATION_CAREERS,
        categorization=ContentStyle.EDUCATIONAL,
        hashtags_count=8,
        posted_hour_of_day=15,
        posted_day_of_week="Tuesday",
    )
    sim_no_tags = simulate_post_performance(profile, post_no_hashtags)
    sim_opt_tags = simulate_post_performance(profile, post_optimal_hashtags)
    assert sim_opt_tags.projected_reach.point_estimate >= sim_no_tags.projected_reach.point_estimate

    # 4. Content Style Sensitivity: Educational (high saves) vs Promotional (lower engagement)
    post_educational = PostInput(
        media_type=MediaType.CAROUSEL,
        category=ContentCategory.EDUCATION_CAREERS,
        categorization=ContentStyle.EDUCATIONAL,
        carousel_slide_count=5,
    )
    post_promotional = PostInput(
        media_type=MediaType.CAROUSEL,
        category=ContentCategory.EDUCATION_CAREERS,
        categorization=ContentStyle.PROMOTIONAL,
        carousel_slide_count=5,
    )
    sim_edu = simulate_post_performance(profile, post_educational)
    sim_promo = simulate_post_performance(profile, post_promotional)
    assert sim_edu.projected_save_rate > sim_promo.projected_save_rate
    assert sim_edu.projected_reach.point_estimate >= sim_promo.projected_reach.point_estimate

    # 5. Timing Sensitivity: Peak hour (18:00) vs Off-peak hour (03:00)
    post_peak = PostInput(
        media_type=MediaType.REEL,
        category=ContentCategory.EDUCATION_CAREERS,
        categorization=ContentStyle.EDUCATIONAL,
        has_call_to_action=True,
        posted_hour_of_day=18,
        posted_day_of_week="Tuesday",
    )
    post_off_peak = PostInput(
        media_type=MediaType.REEL,
        category=ContentCategory.EDUCATION_CAREERS,
        categorization=ContentStyle.EDUCATIONAL,
        has_call_to_action=True,
        posted_hour_of_day=3,
        posted_day_of_week="Tuesday",
    )
    sim_peak = simulate_post_performance(profile, post_peak)
    sim_off_peak = simulate_post_performance(profile, post_off_peak)
    assert sim_peak.projected_reach.point_estimate > sim_off_peak.projected_reach.point_estimate
    assert sim_peak.projected_impressions.point_estimate > sim_off_peak.projected_impressions.point_estimate
