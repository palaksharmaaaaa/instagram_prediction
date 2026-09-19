import io
import json
import urllib.request
import urllib.error
from unittest.mock import MagicMock, patch
import pytest

from instagram_predictor.integrations import (
    InstagramGraphAPIClient,
    MetaGraphAPIError,
    mask_token,
    sanitize_tokens_in_text,
)
from instagram_predictor.schemas.profile import (
    ProfileInput,
    PostInput,
    MediaType,
    Demographics,
)


def _create_mock_response(data: dict, status: int = 200) -> MagicMock:
    """Helper to mock urllib.request.urlopen context manager response."""
    body_bytes = json.dumps(data).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = body_bytes
    mock_resp.status = status
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None
    return mock_resp


def _create_mock_http_error(body_dict: dict, code: int = 400, reason: str = "Bad Request") -> urllib.error.HTTPError:
    """Helper to construct urllib.error.HTTPError with a JSON payload."""
    fp = io.BytesIO(json.dumps(body_dict).encode("utf-8"))
    return urllib.error.HTTPError(
        url="https://graph.facebook.com/v19.0/endpoint",
        code=code,
        msg=reason,
        hdrs={},
        fp=fp,
    )


# =============================================================================
# 1. Token Masking and Sanitization Tests
# =============================================================================
def test_mask_token_utility():
    assert mask_token(None) == ""
    assert mask_token("") == ""
    assert mask_token("short") == "***"
    assert mask_token("12345678") == "***"
    
    long_token = "EAABwz0123456789abcdef9876"
    masked = mask_token(long_token)
    assert masked == "EAAB...***...9876"
    assert "0123456789abcdef" not in masked


def test_sanitize_tokens_in_text():
    # Mask query parameters
    url = "https://graph.facebook.com/v19.0/me?access_token=EAAB1234567890abcdef123456&client_secret=SECRET_999"
    sanitized = sanitize_tokens_in_text(url)
    assert "access_token=***MASKED***" in sanitized
    assert "client_secret=***MASKED***" in sanitized
    assert "EAAB1234567890abcdef123456" not in sanitized
    assert "SECRET_999" not in sanitized

    # Redact standalone tokens
    msg = "Call failed for token EAAB12345678901234567890ab!"
    sanitized_msg = sanitize_tokens_in_text(msg)
    assert "EAAB12345678901234567890ab" not in sanitized_msg
    assert "***" in sanitized_msg


def test_client_repr_masks_token():
    client = InstagramGraphAPIClient(
        access_token="EAAB1234567890abcdef123456",
        app_id="123456789",
        app_secret="super_secret_key",
    )
    rep = repr(client)
    assert "EAAB1234567890abcdef123456" not in rep
    assert "super_secret_key" not in rep
    assert "EAAB...***...3456" in rep


# =============================================================================
# 2. Token Exchange Tests
# =============================================================================
def test_exchange_for_long_lived_token_success():
    client = InstagramGraphAPIClient(
        access_token="short_token_123",
        app_id="my_app_id",
        app_secret="my_app_secret",
    )
    mock_payload = {
        "access_token": "long_lived_token_60_days",
        "token_type": "bearer",
        "expires_in": 5184000,
    }

    with patch("urllib.request.urlopen", return_value=_create_mock_response(mock_payload)):
        result = client.exchange_for_long_lived_token()

    assert result["access_token"] == "long_lived_token_60_days"
    assert client.access_token == "long_lived_token_60_days"
    assert result["expires_in"] == 5184000


def test_exchange_for_long_lived_token_missing_credentials():
    client_no_token = InstagramGraphAPIClient(app_id="aid", app_secret="asec")
    with pytest.raises(MetaGraphAPIError, match="A short-lived user access token is required"):
        client_no_token.exchange_for_long_lived_token()

    client_no_secret = InstagramGraphAPIClient(access_token="tok", app_id="aid")
    with pytest.raises(MetaGraphAPIError, match="Both Meta App ID and App Secret are required"):
        client_no_secret.exchange_for_long_lived_token()


# =============================================================================
# 3. Connected Accounts Discovery Tests
# =============================================================================
def test_get_connected_instagram_accounts():
    client = InstagramGraphAPIClient(access_token="valid_token")
    mock_pages = {
        "data": [
            {
                "id": "page_101",
                "name": "Creator Facebook Page",
                "instagram_business_account": {
                    "id": "17841400000000001",
                    "username": "live_creator",
                    "name": "Live Creator Official",
                    "profile_picture_url": "https://img.meta.com/pfp1.jpg",
                },
            },
            {
                "id": "page_102",
                "name": "Page Without Instagram",
            },
            {
                "id": "page_103",
                "name": "Brand Facebook Page",
                "instagram_business_account": {
                    "id": "17841400000000002",
                    "username": "brand_official",
                    "name": "Brand Official",
                    "profile_picture_url": "https://img.meta.com/pfp2.jpg",
                },
            },
        ]
    }

    with patch("urllib.request.urlopen", return_value=_create_mock_response(mock_pages)):
        accounts = client.get_connected_instagram_accounts()

    assert len(accounts) == 2
    assert accounts[0]["instagram_account_id"] == "17841400000000001"
    assert accounts[0]["username"] == "live_creator"
    assert accounts[0]["page_name"] == "Creator Facebook Page"
    assert accounts[1]["instagram_account_id"] == "17841400000000002"
    assert accounts[1]["username"] == "brand_official"


def test_get_connected_instagram_accounts_no_token():
    client = InstagramGraphAPIClient()
    with pytest.raises(MetaGraphAPIError, match="An access token is required"):
        client.get_connected_instagram_accounts()


# =============================================================================
# 4. Profile Ingestion Tests
# =============================================================================
def test_fetch_profile_data_success():
    client = InstagramGraphAPIClient(access_token="valid_token")
    mock_profile = {
        "id": "17841405822304914",
        "username": "tech_guru",
        "name": "Tech Guru Reviews",
        "biography": "Daily tech breakdowns and AI reviews! Check out https://techguru.com",
        "followers_count": 850000,
        "follows_count": 420,
        "media_count": 560,
        "profile_picture_url": "https://img.meta.com/tech_guru.jpg",
        "website": "https://techguru.com",
    }

    with patch("urllib.request.urlopen", return_value=_create_mock_response(mock_profile)):
        profile = client.fetch_profile_data("17841405822304914")

    assert isinstance(profile, ProfileInput)
    assert profile.username == "tech_guru"
    assert profile.full_name == "Tech Guru Reviews"
    assert profile.total_followers == 850000
    assert profile.total_following == 420
    assert profile.total_media_posts == 560
    assert profile.account_bio_has_link is True
    assert getattr(profile, "profile_picture_url") == "https://img.meta.com/tech_guru.jpg"


def test_fetch_profile_data_following_limit_capped():
    """Verify following count exceeding Instagram platform 7,500 limit is capped gracefully."""
    client = InstagramGraphAPIClient(access_token="valid_token")
    mock_profile = {
        "id": "17841405822304914",
        "username": "bulk_follow_account",
        "followers_count": 50000,
        "follows_count": 8500,  # Exceeds platform 7,500 limit
        "media_count": 120,
    }

    with patch("urllib.request.urlopen", return_value=_create_mock_response(mock_profile)):
        profile = client.fetch_profile_data("17841405822304914")

    assert profile.total_following == 7500
    assert getattr(profile, "raw_following") == 8500


def test_fetch_profile_data_missing_id():
    client = InstagramGraphAPIClient(access_token="valid_token")
    with pytest.raises(MetaGraphAPIError, match="Instagram Account ID is required"):
        client.fetch_profile_data("")


# =============================================================================
# 5. Media & Post Ingestion Tests
# =============================================================================
def test_fetch_recent_media_success():
    client = InstagramGraphAPIClient(access_token="valid_token")
    mock_media_response = {
        "data": [
            {
                "id": "media_001",
                "caption": "Check out our newest AI tutorial! Comment 'AI' below for the link! #ai #machinelearning @meta",
                "media_type": "VIDEO",
                "media_product_type": "REELS",
                "timestamp": "2024-05-15T14:30:00+0000",
                "like_count": 12500,
                "comments_count": 850,
                "permalink": "https://www.instagram.com/reel/C7XYZ123/",
                "insights": {
                    "data": [
                        {"name": "reach", "values": [{"value": 145000}]},
                        {"name": "impressions", "values": [{"value": 210000}]},
                        {"name": "saved", "values": [{"value": 3400}]},
                        {"name": "shares", "values": [{"value": 1800}]},
                        {"name": "video_views", "values": [{"value": 160000}]},
                    ]
                },
            },
            {
                "id": "media_002",
                "caption": "Swipe through the top 5 productivity frameworks! Save this post for later. #productivity #tech",
                "media_type": "CAROUSEL_ALBUM",
                "media_product_type": "FEED",
                "timestamp": "2024-05-10T18:00:00+0000",
                "like_count": 8200,
                "comments_count": 310,
                "permalink": "https://www.instagram.com/p/C7ABC456/",
                "insights": {
                    "data": [
                        {"name": "reach", "values": [{"value": 95000}]},
                        {"name": "impressions", "values": [{"value": 130000}]},
                        {"name": "saved", "values": [{"value": 4200}]},
                        {"name": "shares", "values": [{"value": 950}]},
                    ]
                },
            },
            {
                "id": "media_003",
                "caption": "Behind the scenes at the studio. #photography",
                "media_type": "IMAGE",
                "media_product_type": "FEED",
                "timestamp": "2024-05-02T09:15:00+0000",
                "like_count": 4500,
                "comments_count": 90,
                "permalink": "https://www.instagram.com/p/C7DEF789/",
                "insights": {
                    "data": [
                        {"name": "reach", "total_value": {"value": 52000}},
                        {"name": "impressions", "total_value": {"value": 68000}},
                        {"name": "saved", "total_value": {"value": 310}},
                        {"name": "shares", "total_value": {"value": 120}},
                    ]
                },
            },
        ]
    }

    with patch("urllib.request.urlopen", return_value=_create_mock_response(mock_media_response)):
        posts = client.fetch_recent_media("17841405822304914", limit=3)

    assert len(posts) == 3

    # Post 1: Reel
    p1 = posts[0]
    assert isinstance(p1, PostInput)
    assert p1.media_type == MediaType.REEL
    assert p1.has_call_to_action is True
    assert p1.hashtags_count == 2
    assert p1.mentions_count == 1
    assert p1.metrics.likes == 12500
    assert p1.metrics.comments == 850
    assert p1.metrics.reach == 145000
    assert p1.metrics.impressions == 210000
    assert p1.metrics.shares == 1800
    assert p1.metrics.saves == 3400
    assert p1.metrics.video_views == 160000
    assert p1.posted_day_of_week == "Wednesday"
    assert p1.posted_hour_of_day == 14

    # Post 2: Carousel
    p2 = posts[1]
    assert p2.media_type == MediaType.CAROUSEL
    assert p2.carousel_slide_count == 3
    assert p2.has_call_to_action is True
    assert p2.metrics.likes == 8200
    assert p2.metrics.reach == 95000

    # Post 3: Image (handles total_value format)
    p3 = posts[2]
    assert p3.media_type == MediaType.STATIC_IMAGE
    assert p3.metrics.likes == 4500
    assert p3.metrics.reach == 52000
    assert p3.metrics.impressions == 68000


def test_fetch_recent_media_insights_fallback():
    """When combined insights request fails, falls back to basic media fields gracefully."""
    client = InstagramGraphAPIClient(access_token="valid_token")

    err_payload = {"error": {"message": "Invalid metric parameter for this object", "code": 100}}
    mock_http_err = _create_mock_http_error(err_payload, code=400)

    mock_basic_response = {
        "data": [
            {
                "id": "media_basic_01",
                "caption": "Simple post without insights permissions.",
                "media_type": "IMAGE",
                "timestamp": "2024-05-15T12:00:00Z",
                "like_count": 300,
                "comments_count": 25,
            }
        ]
    }

    with patch("urllib.request.urlopen", side_effect=[mock_http_err, _create_mock_response(mock_basic_response)]):
        posts = client.fetch_recent_media("17841405822304914")

    assert len(posts) == 1
    assert posts[0].metrics.likes == 300
    assert posts[0].metrics.comments == 25
    assert posts[0].metrics.reach is None


# =============================================================================
# 6. Demographics Ingestion Tests
# =============================================================================
def test_fetch_audience_demographics_success():
    client = InstagramGraphAPIClient(access_token="valid_token")
    mock_insights = {
        "data": [
            {
                "name": "audience_country",
                "period": "lifetime",
                "values": [
                    {
                        "value": {
                            "US": 45000,
                            "IN": 32000,
                            "GB": 12000,
                            "BR": 9000,
                        }
                    }
                ],
            },
            {
                "name": "audience_gender_age",
                "period": "lifetime",
                "values": [
                    {
                        "value": {
                            "F.18-24": 15000,
                            "F.25-34": 35000,
                            "M.18-24": 12000,
                            "M.25-34": 38000,
                        }
                    }
                ],
            },
        ]
    }

    with patch("urllib.request.urlopen", return_value=_create_mock_response(mock_insights)):
        demos = client.fetch_audience_demographics("17841405822304914")

    assert isinstance(demos, Demographics)
    assert demos.top_country == "US"
    assert demos.secondary_country == "IN"
    assert demos.primary_age_group == "25-34"
    # Female: 50,000, Male: 50,000 -> 0.50 / 0.50
    assert pytest.approx(demos.gender_female_pct, 0.01) == 0.50
    assert pytest.approx(demos.gender_male_pct, 0.01) == 0.50
    assert pytest.approx(demos.gender_female_pct + demos.gender_male_pct, 0.001) == 1.0


def test_fetch_audience_demographics_fallback_on_error():
    """Verify accounts with < 100 followers fallback to default Demographics without failing."""
    client = InstagramGraphAPIClient(access_token="valid_token")
    err_payload = {
        "error": {
            "message": "(#100) Insufficient data. Account must have at least 100 followers for insights.",
            "code": 100,
        }
    }
    mock_err = _create_mock_http_error(err_payload, code=400)

    with patch("urllib.request.urlopen", side_effect=mock_err):
        demos = client.fetch_audience_demographics("17841405822304914")

    assert isinstance(demos, Demographics)
    assert demos.top_country == "US"
    assert demos.secondary_country == "IN"
    assert pytest.approx(demos.gender_female_pct + demos.gender_male_pct, 0.01) == 1.0


# =============================================================================
# 7. Error Handling & Diagnostics Tests
# =============================================================================
def test_error_handling_invalid_or_expired_token():
    client = InstagramGraphAPIClient(access_token="EAAB_EXPIRED_TOKEN_1234567890")
    err_payload = {
        "error": {
            "message": "Error validating access token: Session has expired on Monday, 10-Jun-24.",
            "type": "OAuthException",
            "code": 190,
            "error_subcode": 463,
            "fbtrace_id": "ABC123Trace",
        }
    }
    mock_err = _create_mock_http_error(err_payload, code=400)

    with patch("urllib.request.urlopen", side_effect=mock_err):
        with pytest.raises(MetaGraphAPIError) as exc_info:
            client.fetch_profile_data("17841405822304914")

    err_str = str(exc_info.value)
    assert "[Code: 190]" in err_str
    assert "[Subcode: 463]" in err_str
    assert "Troubleshooting:" in err_str
    assert "User Access Token is invalid or has expired" in err_str
    # Token must be sanitized
    assert "EAAB_EXPIRED_TOKEN_1234567890" not in err_str


def test_error_handling_rate_limit():
    client = InstagramGraphAPIClient(access_token="valid_token")
    err_payload = {
        "error": {
            "message": "(#613) Calls to this api have exceeded the rate limit.",
            "type": "OAuthException",
            "code": 613,
        }
    }
    mock_err = _create_mock_http_error(err_payload, code=429)

    with patch("urllib.request.urlopen", side_effect=mock_err):
        with pytest.raises(MetaGraphAPIError) as exc_info:
            client.get_connected_instagram_accounts()

    err_str = str(exc_info.value)
    assert "[Code: 613]" in err_str
    assert "rate limit reached" in err_str.lower()


def test_error_handling_missing_permissions():
    client = InstagramGraphAPIClient(access_token="valid_token")
    err_payload = {
        "error": {
            "message": "(#200) Requires pages_read_engagement and instagram_basic permissions.",
            "type": "OAuthException",
            "code": 200,
        }
    }
    mock_err = _create_mock_http_error(err_payload, code=403)

    with patch("urllib.request.urlopen", side_effect=mock_err):
        with pytest.raises(MetaGraphAPIError) as exc_info:
            client.get_connected_instagram_accounts()

    err_str = str(exc_info.value)
    assert "[Code: 200]" in err_str
    assert "Missing required Meta permissions" in err_str


def test_error_handling_network_url_error():
    client = InstagramGraphAPIClient(access_token="valid_token")
    mock_url_err = urllib.error.URLError(reason="getaddrinfo failed - temporary failure in name resolution")

    with patch("urllib.request.urlopen", side_effect=mock_url_err):
        with pytest.raises(MetaGraphAPIError) as exc_info:
            client.fetch_profile_data("17841405822304914")

    err_str = str(exc_info.value)
    assert "Network connection to Meta Graph API failed" in err_str
    assert "Verify network connectivity" in err_str
