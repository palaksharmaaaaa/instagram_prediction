from typing import Dict, List, Any
import numpy as np
from ..config import settings


def detect_profile_anomalies(
    followers: int,
    avg_likes: float,
    avg_comments: float,
    engagement_rate: float
) -> List[str]:
    """
    Detects potential fraud, bot followers, or artificial engagement inflation.
    """
    anomalies = []

    # 1. Bot follower risk: Huge follower count with virtually zero engagement
    if followers > 100_000 and engagement_rate < settings.MIN_ENGAGEMENT_RATE_WARN:
        anomalies.append(
            "HIGH_BOT_RISK: Extremely low engagement rate (<0.05%) for a profile with >100K followers."
        )

    # 2. Suspicious like-to-comment ratio: Normal ratio is 10:1 to 100:1
    if avg_comments == 0 and avg_likes > 100:
        anomalies.append(
            f"SUSPICIOUS_ENGAGEMENT: Zero comments detected alongside high like volume ({int(avg_likes):,}). High likelihood of purchased or bot likes."
        )
    elif avg_comments > 0 and avg_likes > 0:
        ratio = avg_likes / avg_comments
        if ratio > 500:
            anomalies.append(
                f"UNUSUAL_LIKE_SPIKE: Likes to comments ratio exceeds 500:1 ({avg_likes/avg_comments:.1f}:1)."
            )
        elif ratio < 1.0:
            anomalies.append(
                "COMMENT_SPIKE: Comments exceed likes, indicating a giveaway, controversial topic, or comment pod."
            )

    return anomalies


def detect_media_anomalies(
    likes: int,
    shares: int,
    saves: int,
    reach: int,
    followers: int
) -> List[str]:
    """
    Detects virality anomalies or organic outperformance on individual posts.
    """
    anomalies = []

    # Virality breakout
    if reach > (followers * 2.5) and followers > 1000:
        anomalies.append(
            "EXPLORE_VIRAL_BREAKOUT: Post reach exceeded 250% of total follower base via Explore/Reels distribution."
        )

    # High share velocity
    if likes > 0 and (shares / likes) > 0.5:
        anomalies.append(
            "HIGH_SHARE_VIRALITY: Post shares exceed 50% of total likes."
        )

    return anomalies
