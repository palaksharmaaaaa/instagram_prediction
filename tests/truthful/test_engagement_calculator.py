"""
Tests for EngagementCalculatorService and metric calculations.
"""

from instagram_predictor.integrations.instagram_public_fetcher import (
    PublicCreatorSnapshot,
    PublicCreatorMetadata,
    PublicPostItem,
)
from instagram_predictor.schemas.profile import ProfileInput, PlatformType
from instagram_predictor.services.engagement_calculator import (
    EngagementCalculatorService,
)


def _make_snapshot(followers=100000, n_posts=10, likes_per_post=2000, comments_per_post=100):
    posts = [
        PublicPostItem(
            post_id=f"p_{i}",
            media_type="Reel" if i % 2 == 0 else "Static Image",
            caption="test post #vibes",
            likes=likes_per_post,
            comments=comments_per_post,
            timestamp_utc="2026-09-20T12:00:00+00:00",
            posted_day_of_week="Sunday",
            posted_hour_of_day=12,
            permalink="https://instagram.com/p/test",
            views=likes_per_post * 4 if i % 2 == 0 else None,
        )
        for i in range(n_posts)
    ]
    prof = ProfileInput(
        username="testcreator",
        platform=PlatformType.INSTAGRAM,
        total_followers=followers,
        full_name="Test Creator",
        account_category="Technology",
    )
    meta = PublicCreatorMetadata(
        meta_id="123",
        instagram_id="456",
        username="testcreator",
        full_name="Test Creator",
        biography="Bio text",
        website="https://test.com",
        followers_count=followers,
        follows_count=200,
        media_count=n_posts,
        profile_picture_url=None,
        follower_to_following_ratio=followers / 200,
        bio_hashtags=[],
        bio_mentions=[],
        has_website_link=True,
        account_category="Technology",
    )
    return PublicCreatorSnapshot(
        profile=prof,
        metadata=meta,
        posts=posts,
        raw_response={},
        source="TEST",
    )


def test_engagement_rate_computation():
    # 2100 interactions per post, 100,000 followers -> ER = 2.1%
    snap = _make_snapshot(followers=100000, n_posts=5, likes_per_post=2000, comments_per_post=100)
    res = EngagementCalculatorService.calculate(snap)

    assert res.username == "testcreator"
    assert res.total_followers == 100000
    assert res.avg_likes == 2000.0
    assert res.avg_comments == 100.0
    assert abs(res.engagement_rate - 0.021) < 1e-5
    assert res.engagement_rate_pct == 2.1
    assert res.like_rate == 2.0
    assert res.comment_rate == 0.1


def test_tier_benchmark_classification():
    # Nano tier (< 10,000)
    res_nano = EngagementCalculatorService.calculate(_make_snapshot(followers=5000, likes_per_post=250, comments_per_post=10))
    assert res_nano.benchmark.tier == "Nano"
    assert res_nano.benchmark.tier_average == 4.8

    # Micro tier (10k - 100k)
    res_micro = EngagementCalculatorService.calculate(_make_snapshot(followers=50000, likes_per_post=1000, comments_per_post=50))
    assert res_micro.benchmark.tier == "Micro"
    assert res_micro.benchmark.tier_average == 3.2

    # Mega tier (> 1M)
    res_mega = EngagementCalculatorService.calculate(_make_snapshot(followers=2000000, likes_per_post=40000, comments_per_post=2000))
    assert res_mega.benchmark.tier == "Mega"
    assert res_mega.benchmark.tier_average == 1.2


def test_reach_and_commercial_estimations():
    snap = _make_snapshot(followers=50000, likes_per_post=1500, comments_per_post=50)
    res = EngagementCalculatorService.calculate(snap)

    assert res.estimated_reach_min > 0
    assert res.estimated_reach_max > res.estimated_reach_min
    assert res.estimated_impressions >= res.estimated_reach_min
    assert res.post_rate_min > 0
    assert res.post_rate_max >= res.post_rate_min
    assert res.reel_rate_min >= res.post_rate_min
    assert res.reel_rate_max >= res.reel_rate_min
    assert res.avg_reach_per_post > 0
    assert res.overall_account_avg_reach > 0
    assert res.avg_impressions_per_post >= res.avg_reach_per_post
    assert 0.0 <= res.reels_ratio <= 1.0


def test_schema_instagram_post_record():
    from instagram_predictor.schemas.profile import (
        InstagramPostRecord,
        calculate_per_post_reach,
        calculate_per_post_impressions,
        classify_post_category,
    )

    # Test mandatory fields creation
    post = InstagramPostRecord(
        post_id="1234567890",
        username="techtester",
        media_type="Reel",
        media_product_type="REELS",
        like_count=5000,
        comments_count=200,
        caption="Reviewing the newest phone #tech #ai",
        permalink="https://instagram.com/reel/xyz",
        timestamp_utc="2026-09-24T10:00:00+0000",
        posted_day_of_week="Thursday",
        posted_hour_of_day=10,
        total_followers=100_000,
    )

    # Verify auto-calculated fields
    assert post.per_media_engagement == 5200
    assert abs(post.per_media_engagement_rate - 5.2) < 1e-3
    assert post.per_media_reach > 5200
    assert post.per_media_impressions == round(post.per_media_reach * 1.25, 1)
    assert post.post_category == "Technology & AI"
    assert post.hashtags_count == 2
    # Verify optional fields default to None / safe defaults
    assert post.comment_text is None
    assert post.replies_text is None
    assert post.shares_count is None
    assert post.saved_count is None


def test_schema_instagram_profile_record():
    from instagram_predictor.schemas.profile import InstagramProfileRecord

    profile = InstagramProfileRecord(
        id="17841400000000000",
        ig_id="12345678",
        username="creator_x",
        name="Creator X",
        biography="Creating tech content",
        total_followers=250_000,
        total_following=500,
        total_media_posts=450,
        profile_picture_url="https://cdn.example.com/avatar.jpg",
        website="https://linktr.ee/creator_x",
        avg_likes=10000.0,
        avg_comments=400.0,
        reels_ratio=0.6,
    )

    assert profile.username == "creator_x"
    assert profile.follower_to_following_ratio == 500.0
    assert profile.has_website_link is True
    assert profile.avg_engagement == 10400.0
    assert abs(profile.overall_engagement_rate - 4.16) < 1e-2
    assert profile.overall_account_avg_reach > 250000 * 0.14
    assert profile.is_verified is None


def test_calculative_formulas_pure():
    from instagram_predictor.schemas.profile import (
        calculate_per_post_reach,
        calculate_per_post_impressions,
        calculate_avg_reach_per_post,
        calculate_overall_account_avg_reach,
        classify_post_category,
    )

    # Micro tier (50k followers, 1000 likes, 100 comments, Reel)
    # Tier rate = 18% -> 9000. Reel mult = 4.2 -> 1100 * 4.2 = 4620 -> 13620.0
    reach = calculate_per_post_reach(50000, 1000, 100, media_type="Reel", media_product_type="REELS")
    assert reach == 13620.0

    # Impressions: reach * 1.25 -> 17025.0
    imp = calculate_per_post_impressions(reach)
    assert imp == 17025.0

    # Avg reach
    avg_r = calculate_avg_reach_per_post([10000.0, 20000.0, 30000.0])
    assert avg_r == 20000.0

    # Category classification
    assert classify_post_category("Check out my gym workout #fitness") == "Fitness & Health"
    assert classify_post_category("Loving this new outfit #fashion #glam") == "Fashion & Beauty"
    assert classify_post_category("Paid partnership with brand #ad") == "Sponsored / Brand Collab"
    assert classify_post_category("Just a Sunday morning stroll") == "General / Lifestyle"
