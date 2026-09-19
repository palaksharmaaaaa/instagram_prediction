import os
import sys
import pytest
import pandas as pd
import numpy as np

from instagram_predictor.nlp.parser import parse_query
from instagram_predictor.data.loader import load_dataset
from instagram_predictor.data.generator import generate_enterprise_dataset
from instagram_predictor.models.registry import get_reach_pipeline, get_impressions_pipeline
from instagram_predictor.models.engine import predict_batch
from instagram_predictor.services.analytics_service import run_analytics_pipeline


# =============================================================================
# 1. Prompt Parser Unit Tests
# =============================================================================

def test_follower_unit_scaling():
    # 500k
    res_k = parse_query("Accounts above 500k followers")
    assert res_k.filters["total_followers"]["operator"] == ">"
    assert res_k.filters["total_followers"]["value"] == 500_000.0

    # 5m / 5 million
    res_m = parse_query("Accounts over 5m followers")
    assert res_m.filters["total_followers"]["value"] == 5_000_000.0

    res_mil = parse_query("Accounts with more than 10 million followers")
    assert res_mil.filters["total_followers"]["value"] == 10_000_000.0

    # Raw integer without unit
    res_raw = parse_query("Accounts with greater than 10000000 followers")
    assert res_raw.filters["total_followers"]["value"] == 10_000_000.0


def test_follower_comparison_operators():
    # Under / less than
    res_under = parse_query("Give me accounts under 5m followers")
    assert res_under.filters["total_followers"]["operator"] == "<"
    assert res_under.filters["total_followers"]["value"] == 5_000_000.0

    res_less = parse_query("Accounts less than 20m followers")
    assert res_less.filters["total_followers"]["operator"] == "<"
    assert res_less.filters["total_followers"]["value"] == 20_000_000.0

    # Inverted order: followers above 5m
    res_inv = parse_query("Accounts with followers above 5m")
    assert res_inv.filters["total_followers"]["operator"] == ">"
    assert res_inv.filters["total_followers"]["value"] == 5_000_000.0

    # Between range
    res_between = parse_query("Accounts with between 10m and 50m followers")
    assert res_between.filters["total_followers"]["operator"] == "between"
    assert res_between.filters["total_followers"]["min"] == 10_000_000.0
    assert res_between.filters["total_followers"]["max"] == 50_000_000.0


def test_mixed_clause_operators_no_leakage():
    # Followers above, Engagement below
    res = parse_query("Accounts above 10m followers and engagement below 2%")
    assert res.filters["total_followers"]["operator"] == ">"
    assert res.filters["total_followers"]["value"] == 10_000_000.0
    assert res.filters["engagement_rate"]["operator"] == "<"
    assert abs(res.filters["engagement_rate"]["value"] - 0.02) < 1e-6

    # Followers below, Engagement above
    res2 = parse_query("Accounts below 50m followers and engagement above 3%")
    assert res2.filters["total_followers"]["operator"] == "<"
    assert res2.filters["total_followers"]["value"] == 50_000_000.0
    assert res2.filters["engagement_rate"]["operator"] == ">"
    assert abs(res2.filters["engagement_rate"]["value"] - 0.03) < 1e-6


def test_category_and_country_detection():
    # Categories
    res_sports = parse_query("Sports accounts above 5m followers")
    assert res_sports.filters["category"] == "Sports"

    res_music = parse_query("Music accounts in Spain")
    assert res_music.filters["category"] == "Music & Entertainment"
    assert res_music.filters["country"] == "ES"

    res_fashion = parse_query("Fashion accounts in US")
    assert res_fashion.filters["category"] == "Fashion & Beauty"
    assert res_fashion.filters["country"] == "US"

    res_india = parse_query("Top influencers in India")
    assert res_india.filters["country"] == "IN"


def test_username_and_top_n_extraction():
    res_handle = parse_query("Predict reach for @cristiano")
    assert res_handle.filters["username"] == "cristiano"
    assert res_handle.predict_reach is True
    assert res_handle.predict_impressions is False

    res_top = parse_query("Top 5 accounts by followers")
    assert res_top.filters["_top_n"]["limit"] == 5
    assert res_top.filters["_top_n"]["sort_by"] == "total_followers"


def test_prediction_flags():
    # Only reach
    r_reach = parse_query("Predict reach for sports accounts")
    assert r_reach.predict_reach is True
    assert r_reach.predict_impressions is False

    # Only impressions
    r_imp = parse_query("Forecast impressions for music creators")
    assert r_imp.predict_reach is False
    assert r_imp.predict_impressions is True

    # Both
    r_both = parse_query("Estimate reach and impressions for fashion accounts")
    assert r_both.predict_reach is True
    assert r_both.predict_impressions is True

    # Non-prediction query
    r_none = parse_query("Show me all accounts with over 10m followers")
    assert r_none.predict_reach is False
    assert r_none.predict_impressions is False


# =============================================================================
# 2. Data & Preprocessing Tests
# =============================================================================

def test_load_data_integrity():
    df = load_dataset()
    assert len(df) > 0
    assert "total_followers" in df.columns
    assert pd.api.types.is_numeric_dtype(df["total_followers"])
    assert pd.api.types.is_numeric_dtype(df["engagement_rate"])
    # Verify no unexpected duplicate rows
    assert df.duplicated(subset=["post_id"]).sum() == 0


def test_synthetic_data_generator_integrity():
    df = generate_enterprise_dataset(num_profiles=20, posts_per_profile=3, random_seed=42)
    assert len(df) == 60
    assert "per_media_reach" in df.columns
    assert "per_media_impressions" in df.columns
    assert (df["per_media_impressions"] >= df["per_media_reach"]).all()
    assert (df["per_media_reach"] > 0).all()


# =============================================================================
# 3. Model & Prediction Pipeline Tests
# =============================================================================

def test_model_loading_and_prediction():
    reach_pipeline = get_reach_pipeline()
    impressions_pipeline = get_impressions_pipeline()
    assert reach_pipeline is not None
    assert impressions_pipeline is not None

    df = load_dataset().head(5)

    # Reach only
    pred_reach = predict_batch(df, predict_reach=True, predict_impressions=False)
    assert "predicted_reach" in pred_reach.columns
    assert "predicted_impressions" not in pred_reach.columns
    assert (pred_reach["predicted_reach"] > 0).all()

    # Impressions only
    pred_imp = predict_batch(df, predict_reach=False, predict_impressions=True)
    assert "predicted_impressions" in pred_imp.columns
    assert "predicted_reach" not in pred_imp.columns
    assert (pred_imp["predicted_impressions"] > 0).all()

    # Both
    pred_both = predict_batch(df, predict_reach=True, predict_impressions=True)
    assert "predicted_reach" in pred_both.columns
    assert "predicted_impressions" in pred_both.columns
    assert (pred_both["predicted_impressions"] >= pred_both["predicted_reach"]).all()


def test_pipeline_end_to_end_scenarios():
    # Scenario 1: Complex filter with prediction
    req, res = run_analytics_pipeline("Predict reach for sports accounts with more than 50m followers")
    assert req.predict_reach is True
    assert not res.empty
    assert (res["total_followers"] > 50_000_000).all()
    assert "predicted_reach" in res.columns
    assert "predicted_impressions" not in res.columns

    # Scenario 2: Top 3 by followers
    req2, res2 = run_analytics_pipeline("Top 3 accounts by followers")
    assert len(res2) == 3
    assert res2.iloc[0]["total_followers"] >= res2.iloc[1]["total_followers"] >= res2.iloc[2]["total_followers"]

    # Scenario 3: Country search
    req3, res3 = run_analytics_pipeline("Accounts in US")
    assert not res3.empty
    assert (res3["country"] == "US").all()
