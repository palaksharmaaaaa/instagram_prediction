import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add src to path
SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from prompt_parser import parse_prompt
from preprocessing import load_data, NUMERIC_COLUMNS
from synthetic_targets import create_synthetic_targets
from predictor import apply_filter, predict, get_reach_model, get_impression_model
from pipeline import run_pipeline, DATA_PATH


# =============================================================================
# 1. Prompt Parser Unit Tests
# =============================================================================

def test_follower_unit_scaling():
    # 500k
    res_k = parse_prompt("Accounts above 500k followers")
    assert res_k["filters"]["Followers"]["operator"] == ">"
    assert res_k["filters"]["Followers"]["value"] == 500_000.0

    # 5m / 5 million
    res_m = parse_prompt("Accounts over 5m followers")
    assert res_m["filters"]["Followers"]["value"] == 5_000_000.0

    res_mil = parse_prompt("Accounts with more than 10 million followers")
    assert res_mil["filters"]["Followers"]["value"] == 10_000_000.0

    # Raw integer without unit
    res_raw = parse_prompt("Accounts with greater than 10000000 followers")
    assert res_raw["filters"]["Followers"]["value"] == 10_000_000.0


def test_follower_comparison_operators():
    # Under / less than
    res_under = parse_prompt("Give me accounts under 5m followers")
    assert res_under["filters"]["Followers"]["operator"] == "<"
    assert res_under["filters"]["Followers"]["value"] == 5_000_000.0

    res_less = parse_prompt("Accounts less than 20m followers")
    assert res_less["filters"]["Followers"]["operator"] == "<"
    assert res_less["filters"]["Followers"]["value"] == 20_000_000.0

    # Inverted order: followers above 5m
    res_inv = parse_prompt("Accounts with followers above 5m")
    assert res_inv["filters"]["Followers"]["operator"] == ">"
    assert res_inv["filters"]["Followers"]["value"] == 5_000_000.0

    # Between range
    res_between = parse_prompt("Accounts with between 10m and 50m followers")
    assert res_between["filters"]["Followers"]["operator"] == "between"
    assert res_between["filters"]["Followers"]["min"] == 10_000_000.0
    assert res_between["filters"]["Followers"]["max"] == 50_000_000.0


def test_mixed_clause_operators_no_leakage():
    # Followers above, Engagement below
    res = parse_prompt("Accounts above 10m followers and engagement below 2%")
    assert res["filters"]["Followers"]["operator"] == ">"
    assert res["filters"]["Followers"]["value"] == 10_000_000.0
    assert res["filters"]["Engagement Rate"]["operator"] == "<"
    assert abs(res["filters"]["Engagement Rate"]["value"] - 0.02) < 1e-6

    # Followers below, Engagement above
    res2 = parse_prompt("Accounts below 50m followers and engagement above 3%")
    assert res2["filters"]["Followers"]["operator"] == "<"
    assert res2["filters"]["Followers"]["value"] == 50_000_000.0
    assert res2["filters"]["Engagement Rate"]["operator"] == ">"
    assert abs(res2["filters"]["Engagement Rate"]["value"] - 0.03) < 1e-6


def test_category_and_country_detection():
    # Categories
    res_sports = parse_prompt("Sports accounts above 5m followers")
    assert res_sports["filters"]["category"] == "Sports"

    res_music = parse_prompt("Music accounts in Spain")
    assert res_music["filters"]["category"] == "Music"
    assert res_music["filters"]["Country"] == "ES"

    res_fashion = parse_prompt("Fashion accounts in US")
    assert res_fashion["filters"]["category"] == "Fashion & Beauty"
    assert res_fashion["filters"]["Country"] == "US"

    res_india = parse_prompt("Top influencers in India")
    assert res_india["filters"]["Country"] == "IN"


def test_username_and_top_n_extraction():
    res_handle = parse_prompt("Predict reach for @cristiano")
    assert res_handle["filters"]["Username"] == "cristiano"
    assert res_handle["predict_reach"] is True
    assert res_handle["predict_impressions"] is False

    res_top = parse_prompt("Top 5 accounts by followers")
    assert res_top["filters"]["_top_n"]["limit"] == 5
    assert res_top["filters"]["_top_n"]["sort_by"] == "Followers"


def test_prediction_flags():
    # Only reach
    r_reach = parse_prompt("Predict reach for sports accounts")
    assert r_reach["predict_reach"] is True
    assert r_reach["predict_impressions"] is False

    # Only impressions
    r_imp = parse_prompt("Forecast impressions for music creators")
    assert r_imp["predict_reach"] is False
    assert r_imp["predict_impressions"] is True

    # Both
    r_both = parse_prompt("Estimate reach and impressions for fashion accounts")
    assert r_both["predict_reach"] is True
    assert r_both["predict_impressions"] is True

    # Non-prediction query
    r_none = parse_prompt("Show me all accounts with over 10m followers")
    assert r_none["predict_reach"] is False
    assert r_none["predict_impressions"] is False


# =============================================================================
# 2. Data & Preprocessing Tests
# =============================================================================

def test_load_data_integrity():
    df = load_data(DATA_PATH)
    assert len(df) == 200
    assert "Followers" in df.columns
    assert pd.api.types.is_numeric_dtype(df["Followers"])
    assert pd.api.types.is_numeric_dtype(df["Engagement Rate"])
    # Verify no unexpected duplicate rows
    assert df.duplicated().sum() == 0


def test_synthetic_targets_copy_safety():
    df = load_data(DATA_PATH)
    orig_cols = df.columns.tolist()
    augmented = create_synthetic_targets(df)

    # Input df was not mutated with synthetic targets
    assert "synthetic_reach" not in orig_cols
    assert "synthetic_reach" in augmented.columns
    assert "synthetic_impressions" in augmented.columns
    assert (augmented["synthetic_impressions"] >= augmented["synthetic_reach"]).all()


# =============================================================================
# 3. Model & Prediction Pipeline Tests
# =============================================================================

def test_model_loading_and_prediction():
    df = load_data(DATA_PATH).head(5)
    
    # Reach only
    pred_reach = predict(df, predict_reach=True, predict_impressions=False)
    assert "Predicted Reach" in pred_reach.columns
    assert "Predicted Impressions" not in pred_reach.columns
    assert (pred_reach["Predicted Reach"] > 0).all()

    # Impressions only
    pred_imp = predict(df, predict_reach=False, predict_impressions=True)
    assert "Predicted Impressions" in pred_imp.columns
    assert "Predicted Reach" not in pred_imp.columns

    # Both
    pred_both = predict(df, predict_reach=True, predict_impressions=True)
    assert "Predicted Reach" in pred_both.columns
    assert "Predicted Impressions" in pred_both.columns


def test_pipeline_end_to_end_scenarios():
    # Scenario 1: Complex filter with prediction
    req, res = run_pipeline("Predict reach for sports accounts with more than 50m followers")
    assert req["predict_reach"] is True
    assert not res.empty
    assert (res["Followers"] > 50_000_000).all()
    assert "Predicted Reach" in res.columns
    assert "Predicted Impressions" not in res.columns

    # Scenario 2: Top 3 by followers
    req2, res2 = run_pipeline("Top 3 accounts by followers")
    assert len(res2) == 3
    assert res2.iloc[0]["Followers"] >= res2.iloc[1]["Followers"] >= res2.iloc[2]["Followers"]

    # Scenario 3: Country search
    req3, res3 = run_pipeline("Accounts in US")
    assert not res3.empty
    assert (res3["Country"] == "US").all()
