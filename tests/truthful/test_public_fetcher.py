"""
Tests for InstagramPublicFetcher and upsert_creator_profile.
"""

import json
import urllib.error
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from instagram_predictor.data.loader import upsert_creator_profile
from instagram_predictor.integrations.instagram_public_fetcher import InstagramPublicFetcher


def test_public_fetcher_via_meta_business_discovery():
    mock_resp = {
        "business_discovery": {
            "id": "17841404467516249",
            "ig_id": "4369996917",
            "username": "techburner",
            "name": "TechBurner",
            "biography": "Tech creator #tech",
            "website": "https://techburner.in",
            "followers_count": 4240000,
            "follows_count": 170,
            "media_count": 850,
            "profile_picture_url": "https://instagram.com/pic.jpg",
            "media": {
                "data": [
                    {
                        "id": "18121843840929035",
                        "caption": "Cool new tech laptop #tech #ai",
                        "like_count": 50000,
                        "comments_count": 1200,
                        "timestamp": "2026-09-23T16:15:01+0000",
                        "permalink": "https://www.instagram.com/reel/Ddotql0v2_s/",
                        "media_type": "VIDEO",
                        "media_product_type": "REELS",
                        "media_url": "https://instagram.com/video.mp4",
                    },
                    {
                        "id": "18121843840929036",
                        "caption": "Carousel slides testing",
                        "like_count": 30000,
                        "comments_count": 500,
                        "timestamp": "2026-09-20T10:00:00+0000",
                        "permalink": "https://www.instagram.com/p/slide/",
                        "media_type": "CAROUSEL_ALBUM",
                        "media_product_type": "FEED",
                        "children": {
                            "data": [
                                {"id": "c1", "media_type": "IMAGE", "media_url": "https://img1"},
                                {"id": "c2", "media_type": "IMAGE", "media_url": "https://img2"},
                            ]
                        },
                    },
                ]
            },
        }
    }

    mock_urlopen = MagicMock()
    mock_urlopen.read.return_value = json.dumps(mock_resp).encode("utf-8")
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = mock_urlopen

    with patch("urllib.request.urlopen", return_value=mock_ctx):
        fetcher = InstagramPublicFetcher(meta_access_token="test_token", meta_business_account_id="178414000")
        snapshot = fetcher.fetch_creator("@techburner")

    assert snapshot.profile.username == "techburner"
    assert snapshot.profile.total_followers == 4240000
    assert snapshot.metadata.meta_id == "17841404467516249"
    assert snapshot.metadata.instagram_id == "4369996917"
    assert snapshot.metadata.website == "https://techburner.in"
    assert snapshot.metadata.has_website_link is True
    assert snapshot.metadata.account_category == "Technology"
    assert len(snapshot.posts) == 2

    # Post 1 (Reel)
    p1 = snapshot.posts[0]
    assert p1.media_type == "Reel"
    assert p1.media_product_type == "REELS"
    assert p1.likes == 50000
    assert p1.comments == 1200
    assert "#tech" in p1.hashtags

    # Post 2 (Carousel)
    p2 = snapshot.posts[1]
    assert p2.media_type == "Carousel"
    assert p2.carousel_slide_count == 2
    assert len(p2.carousel_children) == 2


def test_public_fetcher_missing_credentials_raises():
    fetcher = InstagramPublicFetcher(meta_access_token="", meta_business_account_id="")
    with pytest.raises(ValueError, match="credentials are required"):
        fetcher.fetch_creator("techburner")


def test_public_fetcher_private_or_not_found_raises():
    err_body = json.dumps({"error": {"message": "Not found", "code": 100}}).encode("utf-8")
    mock_http_err = urllib.error.HTTPError("url", 400, "Bad Request", {}, None)
    mock_http_err.read = MagicMock(return_value=err_body)

    with patch("urllib.request.urlopen", side_effect=mock_http_err):
        fetcher = InstagramPublicFetcher(meta_access_token="tok", meta_business_account_id="123")
        with pytest.raises(RuntimeError, match="does not exist, is private"):
            fetcher.fetch_creator("some_private_or_fake_user")


def test_upsert_creator_profile(tmp_path):
    csv_file = tmp_path / "creator_profiles.csv"

    # 1. Insert new creator
    prof1 = {
        "username": "newcreator",
        "full_name": "New Creator",
        "platform": "Instagram",
        "total_followers": "50000",
        "total_media_posts": "120",
        "account_category": "Sports",
        "engagement_rate": "0.035",
        "avg_likes": "1700.0",
        "avg_comments": "50.0",
        "profile_url": "https://www.instagram.com/newcreator/",
    }
    upsert_creator_profile(prof1, path=csv_file)

    df1 = pd.read_csv(csv_file, dtype=str)
    assert len(df1) == 1
    assert df1.iloc[0]["username"] == "newcreator"
    assert df1.iloc[0]["total_followers"] == "50000"

    # 2. Update existing creator (same username, different followers and ER)
    prof1_updated = {
        "username": "NEWCREATOR",
        "total_followers": "65000",
        "engagement_rate": "0.042",
    }
    upsert_creator_profile(prof1_updated, path=csv_file)

    df2 = pd.read_csv(csv_file, dtype=str)
    assert len(df2) == 1  # No duplicate
    assert df2.iloc[0]["username"] == "newcreator"
    assert df2.iloc[0]["total_followers"] == "65000"
    assert df2.iloc[0]["engagement_rate"] == "0.042"
    assert df2.iloc[0]["full_name"] == "New Creator"
    assert df2.iloc[0]["account_category"] == "Sports"
