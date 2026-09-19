"""
Tests for Conversational Result / Output Analyzer & Interpreter Service.

Verifies:
- Multi-platform interpretation across Instagram (Reel, Carousel), YouTube (Short, Video), and Snapchat (Spotlight)
- Prompt handling (with and without user query prompts)
- Numerical consistency across generated text and input simulation metrics
- Correct TreeSHAP lever categorization into boosts vs opportunity areas
- Math jargon avoidance in creator-facing confidence summaries
- Seamless integration with SimulationPrediction objects
"""

import pytest
from instagram_predictor.services import interpret_simulation_result, format_simulation_for_interpretation
from instagram_predictor.schemas import ProfileInput, PostInput, PlatformType, MediaType, ContentCategory
from instagram_predictor.models import simulate_post_performance


def test_interpret_instagram_reel_without_prompt():
    """Verify Instagram Reel interpretation without a user prompt."""
    sim_data = {
        "predicted_reach": 52000,
        "predicted_impressions": 78000,
        "reach_80_ci": (42000, 65000),
        "reach_90_ci": (35000, 78000),
        "impressions_80_ci": (60000, 95000),
        "impressions_90_ci": (50000, 110000),
        "feature_explanations": {
            "base_reach": 40000,
            "final_reach": 52000,
            "drivers": [
                {
                    "name": "Reel Media Format",
                    "impact": 12000,
                    "pct": 30.0,
                    "direction": "positive",
                    "description": "Reel format vs Static Image",
                },
                {
                    "name": "Call-to-Action (CTA)",
                    "impact": 3500,
                    "pct": 8.75,
                    "direction": "positive",
                    "description": "Clear CTA prompt",
                },
                {
                    "name": "Peak Schedule & Timing",
                    "impact": -3500,
                    "pct": -8.75,
                    "direction": "negative",
                    "description": "Off-peak posting time",
                },
            ],
        },
        "engagement_rate": 4.8,
        "virality_tier": "🔥 High Virality",
        "platform": "Instagram",
        "format": "Reel",
        "total_followers": 100000,
    }

    result = interpret_simulation_result(sim_data)

    # 1. Verify required dictionary fields exist and are non-empty
    required_keys = [
        "headline",
        "executive_summary",
        "key_drivers",
        "actionable_tips",
        "confidence_summary",
        "dialogue_markdown",
        "driver_breakdown",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"
        assert result[key], f"Field {key} is empty"

    # 2. Verify numerical figures match input simulation exactly
    assert "52,000" in result["headline"] or "52000" in result["headline"]
    assert "52,000" in result["executive_summary"]
    assert "78,000" in result["executive_summary"]
    assert "100,000" in result["executive_summary"]
    assert "4.8" in result["executive_summary"]

    assert "42,000" in result["confidence_summary"]
    assert "65,000" in result["confidence_summary"]
    assert "78,000" in result["confidence_summary"]
    assert "110,000" in result["confidence_summary"]

    # 3. Verify TreeSHAP lever categorization into boosts vs opportunities
    boosts = result["driver_breakdown"]["boosts"]
    opportunities = result["driver_breakdown"]["opportunities"]

    assert len(boosts) == 2
    assert len(opportunities) == 1

    boost_names = [b["name"] for b in boosts]
    assert "Reel Media Format" in boost_names
    assert "Call-to-Action (CTA)" in boost_names
    assert opportunities[0]["name"] == "Peak Schedule & Timing"
    assert opportunities[0]["impact"] == -3500

    # 4. Verify actionable tips count (2-4 concrete tips)
    assert 2 <= len(result["actionable_tips"]) <= 4
    # Timing opportunity should be targeted in tips
    assert any("reschedule" in tip.lower() or "peak" in tip.lower() for tip in result["actionable_tips"])

    # 5. Verify dialogue markdown formatting
    assert "### 🤖 AI Content Strategist Performance Briefing" in result["dialogue_markdown"]
    assert "52,000" in result["dialogue_markdown"]
    assert "78,000" in result["dialogue_markdown"]


def test_interpret_instagram_carousel_with_prompt():
    """Verify Instagram Carousel interpretation with a user query prompt."""
    sim_data = {
        "predicted_reach": 38000,
        "predicted_impressions": 58000,
        "reach_80_ci": (30000, 48000),
        "reach_90_ci": (26000, 55000),
        "impressions_80_ci": (45000, 72000),
        "impressions_90_ci": (40000, 85000),
        "feature_explanations": {
            "base_reach": 30000,
            "final_reach": 38000,
            "drivers": [
                {
                    "name": "Carousel Media Format",
                    "impact": 6000,
                    "pct": 20.0,
                    "direction": "positive",
                    "description": "Multi-slide carousel",
                },
                {
                    "name": "Caption Depth",
                    "impact": -2000,
                    "pct": -6.67,
                    "direction": "negative",
                    "description": "Short caption length",
                },
                {
                    "name": "Call-to-Action (CTA)",
                    "impact": 4000,
                    "pct": 13.33,
                    "direction": "positive",
                    "description": "CTA present",
                },
            ],
        },
        "engagement_rate": 5.2,
        "virality_tier": "📈 Solid Distribution",
        "platform": "Instagram",
        "format": "Carousel",
        "total_followers": 60000,
    }

    user_prompt = "How can I get more saves and shares on this carousel?"
    result = interpret_simulation_result(sim_data, prompt=user_prompt)

    assert result["prompt"] == user_prompt
    assert user_prompt in result["dialogue_markdown"]
    assert "AI Strategist Assessment" in result["dialogue_markdown"]
    assert "saves" in result["dialogue_markdown"].lower() or "shares" in result["dialogue_markdown"].lower()

    # Numerical consistency
    assert "38,000" in result["executive_summary"]
    assert "58,000" in result["executive_summary"]
    assert "30,000" in result["confidence_summary"]
    assert "48,000" in result["confidence_summary"]

    # TreeSHAP categorization
    boosts = result["driver_breakdown"]["boosts"]
    opportunities = result["driver_breakdown"]["opportunities"]
    assert len(boosts) == 2
    assert len(opportunities) == 1
    assert opportunities[0]["name"] == "Caption Depth"


def test_interpret_youtube_short():
    """Verify YouTube Short interpretation with platform-specific terminology."""
    sim_data = {
        "predicted_reach": 185000,
        "predicted_impressions": 260000,
        "reach_80_ci": (140000, 230000),
        "reach_90_ci": (115000, 275000),
        "impressions_80_ci": (200000, 320000),
        "impressions_90_ci": (170000, 380000),
        "feature_explanations": {
            "base_reach": 100000,
            "final_reach": 185000,
            "drivers": [
                {
                    "name": "YouTube Shorts Media Format",
                    "impact": 75000,
                    "pct": 75.0,
                    "direction": "positive",
                    "description": "Shorts feed distribution",
                },
                {
                    "name": "Hashtag Discovery",
                    "impact": 10000,
                    "pct": 10.0,
                    "direction": "positive",
                    "description": "Targeted tags",
                },
            ],
        },
        "engagement_rate": 6.5,
        "virality_tier": "🚀 Viral Breakthrough",
        "platform": "YouTube",
        "format": "YouTube Short",
        "total_followers": 250000,
    }

    result = interpret_simulation_result(sim_data)

    assert result["platform"] == "YouTube"
    assert result["format"] == "YouTube Short"
    # YouTube terminology check (viewers / Shorts)
    assert "viewers" in result["executive_summary"].lower()
    assert "185,000" in result["headline"]
    assert "185,000" in result["executive_summary"]
    assert "260,000" in result["executive_summary"]

    # Verify boosts & tips
    assert len(result["driver_breakdown"]["boosts"]) == 2
    assert len(result["driver_breakdown"]["opportunities"]) == 0
    assert 2 <= len(result["actionable_tips"]) <= 4


def test_interpret_youtube_video_with_prompt():
    """Verify YouTube Video (long-form) interpretation answering reach drag question."""
    sim_data = {
        "predicted_reach": 45000,
        "predicted_impressions": 85000,
        "reach_80_ci": (35000, 58000),
        "reach_90_ci": (28000, 68000),
        "impressions_80_ci": (65000, 110000),
        "impressions_90_ci": (55000, 135000),
        "feature_explanations": {
            "base_reach": 55000,
            "final_reach": 45000,
            "drivers": [
                {
                    "name": "Peak Schedule & Timing",
                    "impact": -12000,
                    "pct": -21.8,
                    "direction": "negative",
                    "description": "Off-peak upload hour",
                },
                {
                    "name": "Content Styles & Theme",
                    "impact": 2000,
                    "pct": 3.6,
                    "direction": "positive",
                    "description": "Educational tech theme",
                },
            ],
        },
        "engagement_rate": 4.1,
        "virality_tier": "📈 Moderate Distribution",
        "platform": "YouTube",
        "format": "YouTube Video",
        "total_followers": 120000,
    }

    user_prompt = "Why is my reach lower than expected?"
    result = interpret_simulation_result(sim_data, prompt=user_prompt)

    assert result["prompt"] == user_prompt
    assert "Why is my reach lower than expected?" in result["dialogue_markdown"]
    # The prompt response must explicitly identify the negative driver
    assert "Peak Schedule & Timing" in result["dialogue_markdown"]
    assert "-12,000" in result["dialogue_markdown"] or "12,000" in result["dialogue_markdown"]

    opportunities = result["driver_breakdown"]["opportunities"]
    assert len(opportunities) == 1
    assert opportunities[0]["name"] == "Peak Schedule & Timing"


def test_interpret_snapchat_spotlight_with_prompt():
    """Verify Snapchat Spotlight interpretation with virality query."""
    sim_data = {
        "predicted_reach": 92000,
        "predicted_impressions": 135000,
        "reach_80_ci": (70000, 120000),
        "reach_90_ci": (58000, 145000),
        "impressions_80_ci": (100000, 175000),
        "impressions_90_ci": (85000, 210000),
        "feature_explanations": {
            "base_reach": 60000,
            "final_reach": 92000,
            "drivers": [
                {
                    "name": "Snapchat Spotlight Format",
                    "impact": 28000,
                    "pct": 46.7,
                    "direction": "positive",
                    "description": "Full-screen Spotlight delivery",
                },
                {
                    "name": "Call-to-Action (CTA)",
                    "impact": 4000,
                    "pct": 6.7,
                    "direction": "positive",
                    "description": "Interactive CTA",
                },
            ],
        },
        "engagement_rate": 7.2,
        "virality_tier": "🚀 Explosive / Viral",
        "platform": "Snapchat",
        "format": "Snapchat Spotlight",
        "total_followers": 40000,
    }

    user_prompt = "Will this Spotlight post go viral?"
    result = interpret_simulation_result(sim_data, prompt=user_prompt)

    assert result["platform"] == "Snapchat"
    assert result["format"] == "Snapchat Spotlight"
    assert "Snapchat" in result["headline"]
    assert "92,000" in result["headline"]
    assert "92,000" in result["executive_summary"]
    assert "135,000" in result["executive_summary"]
    assert "70,000" in result["confidence_summary"]
    assert "120,000" in result["confidence_summary"]

    assert user_prompt in result["dialogue_markdown"]
    assert "Explosive / Viral" in result["dialogue_markdown"]


def test_confidence_summary_avoids_mathematical_jargon():
    """Verify that confidence_summary avoids non-intuitive mathematical jargon."""
    sim_data = {
        "predicted_reach": 50000,
        "predicted_impressions": 75000,
        "reach_80_ci": (40000, 62000),
        "reach_90_ci": (32000, 72000),
        "impressions_80_ci": (60000, 92000),
        "impressions_90_ci": (50000, 110000),
        "feature_explanations": {"drivers": []},
        "engagement_rate": 4.5,
        "virality_tier": "📈 Moderate Distribution",
        "platform": "Instagram",
        "format": "Reel",
    }

    result = interpret_simulation_result(sim_data)
    summary = result["confidence_summary"].lower()

    # Jargon terms to strictly avoid in user-facing creator confidence summary
    forbidden_terms = [
        "mondrian",
        "conformal quantile",
        "finite-sample",
        "coverage guarantee",
        "epistemic",
        "ood extrapolation",
        "log1p",
    ]
    for term in forbidden_terms:
        assert term not in summary, f"Found forbidden jargon '{term}' in confidence_summary: {result['confidence_summary']}"

    # Friendly creator terms that should be present
    assert any(term in summary for term in ["expected", "window", "confidence", "bounds", "benchmark"])


def test_interpret_end_to_end_with_simulation_prediction_object():
    """Verify that interpret_simulation_result accepts a SimulationPrediction instance directly."""
    profile = ProfileInput(
        username="growth_hacker",
        total_followers=80000,
        total_following=300,
        total_media_posts=150,
        account_category=ContentCategory.FINANCE_BUSINESS,
    )
    post = PostInput(
        platform=PlatformType.INSTAGRAM,
        media_type=MediaType.REEL,
        category=ContentCategory.FINANCE_BUSINESS,
        caption_length_chars=320,
        has_call_to_action=True,
        hashtags_count=5,
        posted_hour_of_day=19,
        posted_day_of_week="Thursday",
    )

    # 1. Run actual ML simulation pipeline
    sim_prediction = simulate_post_performance(profile, post)
    assert sim_prediction.projected_reach.point_estimate > 0

    # 2. Pass SimulationPrediction object directly to interpreter
    interpretation = interpret_simulation_result(sim_prediction)

    assert interpretation["predicted_reach"] == sim_prediction.projected_reach.point_estimate
    assert interpretation["predicted_impressions"] == sim_prediction.projected_impressions.point_estimate
    assert str(interpretation["predicted_reach"]) in interpretation["headline"].replace(",", "")
    assert str(interpretation["predicted_reach"]) in interpretation["executive_summary"].replace(",", "")
    assert 2 <= len(interpretation["actionable_tips"]) <= 4
    assert len(interpretation["key_drivers"]) > 0


def test_format_simulation_for_interpretation_helper():
    """Verify that format_simulation_for_interpretation creates standard dictionaries."""
    profile = ProfileInput(
        username="beauty_guru",
        total_followers=50000,
        total_following=150,
        total_media_posts=80,
        account_category=ContentCategory.FASHION_BEAUTY,
    )
    post = PostInput(
        platform=PlatformType.INSTAGRAM,
        media_type=MediaType.CAROUSEL,
        category=ContentCategory.FASHION_BEAUTY,
        carousel_slide_count=5,
        caption_length_chars=250,
    )

    prediction = simulate_post_performance(profile, post)
    formatted = format_simulation_for_interpretation(
        prediction,
        profile_data={"total_followers": 50000},
        post_data={"platform": "Instagram", "media_type": "Carousel"},
    )

    assert formatted["platform"] == "Instagram"
    assert formatted["format"] == "Carousel"
    assert formatted["predicted_reach"] == prediction.projected_reach.point_estimate
    assert formatted["total_followers"] == 50000
    assert formatted["reach_80_ci"][0] == prediction.projected_reach.lower
    assert formatted["reach_80_ci"][1] == prediction.projected_reach.upper


def test_minimal_simulation_input_graceful_handling():
    """Verify that interpreter handles minimal dictionary inputs without crashing."""
    minimal_data = {
        "predicted_reach": 25000,
        "predicted_impressions": 40000,
    }

    result = interpret_simulation_result(minimal_data)
    assert result["predicted_reach"] == 25000
    assert result["predicted_impressions"] == 40000
    assert result["headline"]
    assert result["executive_summary"]
    assert 2 <= len(result["actionable_tips"]) <= 4
    assert result["confidence_summary"]
    assert result["dialogue_markdown"]


def test_run_post_simulation_and_interpret():
    """Verify run_post_simulation_and_interpret service pipeline with raw dictionary inputs."""
    from instagram_predictor.services import run_post_simulation_and_interpret

    profile_dict = {
        "username": "fit_coach",
        "full_name": "Fitness Coach",
        "country": "US",
        "total_followers": 75000,
        "total_following": 250,
        "total_media_posts": 120,
        "is_verified": False,
        "account_category": "Health & Fitness",
    }
    post_dict = {
        "media_type": "Reel",
        "category": "Health & Fitness",
        "categorization": "Educational / How-To",
        "caption_length_chars": 280,
        "has_call_to_action": True,
        "posted_hour_of_day": 18,
        "posted_day_of_week": "Wednesday",
    }

    success, errors, interpretation = run_post_simulation_and_interpret(
        profile_dict,
        post_dict,
        prompt="Will this Reel reach new followers?"
    )

    assert success is True
    assert len(errors) == 0
    assert interpretation is not None
    assert interpretation["predicted_reach"] > 0
    assert interpretation["predicted_impressions"] >= interpretation["predicted_reach"]
    assert interpretation["prompt"] == "Will this Reel reach new followers?"
    assert "### 🤖 AI Content Strategist Performance Briefing" in interpretation["dialogue_markdown"]

