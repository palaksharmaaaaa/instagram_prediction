"""
Engagement calculation and commercial metrics service matching project-influencenexus.
"""

from dataclasses import dataclass
from typing import List, Optional
from ..integrations.instagram_public_fetcher import PublicCreatorSnapshot, PublicPostItem


@dataclass
class BenchmarkComparison:
    tier: str
    tier_average: float
    status: str  # 'above', 'average', 'below'
    diff_percentage: float


@dataclass
class CalculatedEngagementResult:
    username: str
    full_name: str
    total_followers: int
    engagement_rate: float          # Decimal (e.g. 0.024 for 2.4%)
    engagement_rate_pct: float      # Percentage (e.g. 2.4)
    avg_likes: float
    avg_comments: float
    avg_video_views: float
    like_rate: float
    comment_rate: float
    rating: str                     # 'Excellent', 'Good', 'Average', 'Below Average'
    benchmark: BenchmarkComparison
    estimated_reach_min: int
    estimated_reach_max: int
    estimated_impressions: int
    post_rate_min: int
    post_rate_max: int
    reel_rate_min: int
    reel_rate_max: int
    total_posts_sampled: int
    recent_posts: List[PublicPostItem]
    # Calculative fields with documented formulas:
    avg_reach_per_post: float = 0.0
    overall_account_avg_reach: float = 0.0
    avg_impressions_per_post: float = 0.0
    reels_ratio: float = 0.0
    carousel_ratio: float = 0.0
    static_image_ratio: float = 0.0
    follower_to_following_ratio: float = 0.0


class EngagementCalculatorService:
    @staticmethod
    def calculate(snapshot: PublicCreatorSnapshot) -> CalculatedEngagementResult:
        followers = max(snapshot.profile.total_followers, 1)
        posts = snapshot.posts

        total_likes = sum(p.likes for p in posts)
        total_comments = sum(p.comments for p in posts)
        video_views = [p.views for p in posts if p.views is not None and p.views > 0]

        n_posts = max(len(posts), 1)
        avg_likes = total_likes / n_posts
        avg_comments = total_comments / n_posts
        avg_views = (sum(video_views) / len(video_views)) if video_views else (avg_likes * 4.2)

        er_decimal = (avg_likes + avg_comments) / followers
        er_pct = round(er_decimal * 100, 2)
        like_rate = round((avg_likes / followers) * 100, 2)
        comment_rate = round((avg_comments / followers) * 100, 3)

        # Tier benchmarks
        if followers < 10000:
            tier, tier_avg = "Nano", 4.8
        elif followers < 100000:
            tier, tier_avg = "Micro", 3.2
        elif followers < 500000:
            tier, tier_avg = "Mid", 2.1
        elif followers < 1000000:
            tier, tier_avg = "Macro", 1.6
        else:
            tier, tier_avg = "Mega", 1.2

        diff_pct = round(((er_pct - tier_avg) / tier_avg) * 100, 1)
        status = "above" if diff_pct > 15 else ("below" if diff_pct < -15 else "average")

        # Rating
        if er_pct >= tier_avg * 1.5 or er_pct >= 6.0:
            rating = "Excellent"
        elif er_pct >= tier_avg * 1.1 or er_pct >= 3.5:
            rating = "Good"
        elif er_pct >= tier_avg * 0.75 or er_pct >= 1.5:
            rating = "Average"
        else:
            rating = "Below Average"

        # Estimated Reach
        er_multiplier = max(0.5, min(3.0, er_pct / 3.0))
        min_reach = round(followers * 0.12 * er_multiplier)
        max_reach = round(followers * 0.38 * er_multiplier)
        impressions = max(round((min_reach + max_reach) * 0.65), round(avg_views * 1.2))

        # Commercial Value (Estimated Post & Reel Rates in INR)
        base_rate = (followers / 1000) * 150
        val_multiplier = max(0.7, er_pct / 2.5)
        post_min = max(1500, round(base_rate * 0.8 * val_multiplier))
        post_max = max(post_min * 1.5, round(base_rate * 1.4 * val_multiplier))
        reel_min = max(2500, round(base_rate * 1.25 * val_multiplier))
        reel_max = max(reel_min * 1.6, round(base_rate * 2.1 * val_multiplier))

        # Content format distribution
        reels_count = sum(1 for p in posts if p.media_type == "Reel" or p.media_product_type == "REELS")
        carousel_count = sum(1 for p in posts if p.media_type == "Carousel")
        image_count = sum(1 for p in posts if p.media_type == "Static Image")

        reels_ratio = round(reels_count / n_posts, 2)
        carousel_ratio = round(carousel_count / n_posts, 2)
        image_ratio = round(image_count / n_posts, 2)

        # Calculative Per-Post Reach & Impressions Averages
        posts_reach = [p.per_media_reach for p in posts if p.per_media_reach and p.per_media_reach > 0]
        avg_reach = round(sum(posts_reach) / len(posts_reach), 1) if posts_reach else float(min_reach)

        posts_imp = [p.per_media_impressions for p in posts if p.per_media_impressions and p.per_media_impressions > 0]
        avg_imp = round(sum(posts_imp) / len(posts_imp), 1) if posts_imp else float(impressions)

        tier_rate = 0.25 if followers < 10000 else (0.18 if followers < 100000 else (0.14 if followers < 500000 else (0.10 if followers < 1000000 else 0.06)))
        blended_multiplier = (reels_ratio * 4.2) + ((1.0 - reels_ratio) * 2.5)
        overall_reach = round(min(followers * 5.0, max(avg_likes + avg_comments, (followers * tier_rate) + ((avg_likes + avg_comments) * blended_multiplier))), 1)

        f_ratio = getattr(snapshot.metadata, "follower_to_following_ratio", 0.0) if hasattr(snapshot, "metadata") else 0.0

        return CalculatedEngagementResult(
            username=snapshot.profile.username,
            full_name=snapshot.profile.full_name or snapshot.profile.username,
            total_followers=followers,
            engagement_rate=er_decimal,
            engagement_rate_pct=er_pct,
            avg_likes=avg_likes,
            avg_comments=avg_comments,
            avg_video_views=avg_views,
            like_rate=like_rate,
            comment_rate=comment_rate,
            rating=rating,
            benchmark=BenchmarkComparison(
                tier=tier,
                tier_average=tier_avg,
                status=status,
                diff_percentage=diff_pct,
            ),
            estimated_reach_min=min_reach,
            estimated_reach_max=max_reach,
            estimated_impressions=impressions,
            post_rate_min=round(post_min),
            post_rate_max=round(post_max),
            reel_rate_min=round(reel_min),
            reel_rate_max=round(reel_max),
            total_posts_sampled=len(posts),
            recent_posts=posts,
            avg_reach_per_post=avg_reach,
            overall_account_avg_reach=overall_reach,
            avg_impressions_per_post=avg_imp,
            reels_ratio=reels_ratio,
            carousel_ratio=carousel_ratio,
            static_image_ratio=image_ratio,
            follower_to_following_ratio=f_ratio,
        )

