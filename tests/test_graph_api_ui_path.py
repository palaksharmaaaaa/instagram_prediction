"""
Dedicated test suite verifying the Meta Graph API client methods and Streamlit UI ingestion flow.
Tests:
1. InstagramGraphAPIClient.fetch_full_creator_state returns validated (ProfileInput, Demographics, List[PostInput]).
2. Modern v22.0 API version configuration and URL building.
3. Category mapping and estimation transparency flags.
4. Streamlit UI path in Pro Mode executes without KeyError or TypeError when accounts are discovered and fetched.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from streamlit.testing.v1 import AppTest

from instagram_predictor.integrations.instagram_graph_api import (
    InstagramGraphAPIClient,
    MetaGraphAPIError,
)
from instagram_predictor.schemas.profile import (
    ProfileInput,
    Demographics,
    PostInput,
    ContentCategory,
)


@pytest.fixture
def mock_client():
    client = InstagramGraphAPIClient(access_token="EAABtesttoken1234567890")
    return client


def test_api_version_is_v22():
    client = InstagramGraphAPIClient(access_token="test")
    assert client.DEFAULT_API_VERSION == "v22.0"
    assert "v22.0" in client.base_url


def test_fetch_full_creator_state_success(mock_client):
    """Verify fetch_full_creator_state returns validated profile, demographics, and media."""
    mock_profile_resp = {
        "id": "17841405822304914",
        "username": "tech_creator",
        "name": "Tech Reviews Daily",
        "biography": "Reviewing latest gadgets and software",
        "followers_count": 85000,
        "follows_count": 420,
        "media_count": 310,
        "website": "https://techreviews.example.com",
        "category": "Science, technology & engineering",
        "profile_picture_url": "https://cdn.example.com/pfp.jpg",
    }

    mock_demographics_resp = {
        "data": [
            {"name": "audience_country", "values": [{"value": {"US": 45000, "GB": 12000, "CA": 8000}}]},
            {"name": "audience_gender_age", "values": [{"value": {"M.25-34": 30000, "F.25-34": 20000}}]},
        ]
    }

    mock_media_resp = {
        "data": [
            {
                "id": "180123456789",
                "caption": "Check out this new AI laptop! #tech #ai What do you think?",
                "media_type": "VIDEO",
                "media_product_type": "REELS",
                "timestamp": "2026-09-18T18:30:00Z",
                "like_count": 3200,
                "comments_count": 140,
                "permalink": "https://www.instagram.com/reel/xyz123/",
                "insights": {
                    "data": [
                        {"name": "reach", "values": [{"value": 45000}]},
                        {"name": "impressions", "values": [{"value": 62000}]},
                        {"name": "saved", "values": [{"value": 850}]},
                        {"name": "shares", "values": [{"value": 1100}]},
                    ]
                },
            }
        ]
    }

    def mock_request(endpoint, params=None, **kwargs):
        if "17841405822304914/insights" in str(endpoint) or (endpoint == "17841405822304914/insights"):
            return mock_demographics_resp
        elif "17841405822304914/media" in str(endpoint) or (endpoint == "17841405822304914/media"):
            return mock_media_resp
        else:
            return mock_profile_resp

    with patch.object(mock_client, "_request", side_effect=mock_request):
        profile, demographics, media = mock_client.fetch_full_creator_state("17841405822304914")

        assert isinstance(profile, ProfileInput)
        assert isinstance(demographics, Demographics)
        assert isinstance(media, list)
        assert len(media) == 1
        assert isinstance(media[0], PostInput)

        assert profile.username == "tech_creator"
        assert profile.total_followers == 85000
        assert profile.account_category == ContentCategory.SCIENCE_TECHNOLOGY
        assert profile.country == "US"
        assert getattr(profile, "prior_metrics_estimated", False) is True

        assert demographics.top_country == "US"
        assert demographics.primary_age_group == "25-34"

        assert media[0].metrics.reach == 45000
        assert media[0].metrics.impressions == 62000
        assert media[0].category == ContentCategory.SCIENCE_TECHNOLOGY


def test_ui_meta_graph_flow_with_discovered_accounts():
    """Verify that in Pro Mode, discovering accounts and clicking fetch creator works without KeyError or TypeError."""
    app_file = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_file, default_timeout=30)
    at.run()

    # Switch to Pro Mode
    mode_radio = at.sidebar.radio[0]
    mode_radio.set_value("🔬 Pro / Data Scientist Mode").run()
    assert len(at.exception) == 0

    # Inject discovered account into session state (simulating successful discovery)
    at.session_state["meta_user_token"] = "EAABtesttoken12345"
    at.session_state["discovered_ig_accounts"] = [
        {
            "instagram_account_id": "17841405822304914",
            "username": "verified_creator",
            "name": "Verified Creator Official",
            "profile_picture_url": "https://cdn.example.com/pfp.jpg",
            "page_id": "999888777",
            "page_name": "Verified Page",
        }
    ]

    # Re-run so UI renders the selectbox with discovered account
    at.run()
    assert len(at.exception) == 0

    # Check that selectbox renders and contains the discovered account
    ingest_select = None
    for s in at.selectbox:
        if "select instagram account" in s.label.lower():
            ingest_select = s
            break
    assert ingest_select is not None, "Selectbox for discovered Instagram account not found"
    assert "17841405822304914" in ingest_select.options[0]

    # Mock InstagramGraphAPIClient.fetch_full_creator_state
    mock_profile = ProfileInput(
        username="verified_creator",
        full_name="Verified Creator Official",
        country="US",
        total_followers=120000,
        total_following=350,
        total_media_posts=500,
        account_category=ContentCategory.MUSIC_ENTERTAINMENT,
    )
    setattr(mock_profile, "prior_metrics_estimated", True)

    with patch.object(
        InstagramGraphAPIClient,
        "fetch_full_creator_state",
        return_value=(mock_profile, Demographics(), []),
    ):
        # Click "🚀 Fetch Live Creator Data & Insights"
        fetch_btn = None
        for b in at.button:
            if "fetch live creator" in b.label.lower():
                fetch_btn = b
                break
        assert fetch_btn is not None, "Fetch Live Creator button not found"
        fetch_btn.click().run()

        # Must not raise KeyError or TypeError
        assert len(at.exception) == 0
        assert "live_creator_profile" in at.session_state
        assert at.session_state["live_creator_profile"].username == "verified_creator"
