"""
Public Instagram creator fetcher using official Meta Graph API Business Discovery.
Fetches only observed, real data directly reported by Meta.
No simulation, no synthetic generation, no defaulted values.
"""

import json
import logging
import os
import re
import urllib.request
import urllib.parse
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..config import settings
from ..schemas.profile import (
    ProfileInput,
    PlatformType,
    calculate_per_post_reach,
    calculate_per_post_impressions,
    classify_post_category,
)

logger = logging.getLogger(__name__)


@dataclass
class CarouselChildItem:
    id: str
    media_type: str
    media_url: Optional[str] = None
    permalink: Optional[str] = None


@dataclass
class PublicPostItem:
    post_id: str
    media_type: str                      # 'Reel', 'Carousel', 'Static Image'
    caption: str
    likes: int
    comments: int
    timestamp_utc: str
    posted_day_of_week: str
    posted_hour_of_day: int
    permalink: str
    media_product_type: str = "FEED"
    caption_length_chars: int = 0
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    views: Optional[int] = None
    media_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    carousel_slide_count: Optional[int] = None
    carousel_children: List[CarouselChildItem] = field(default_factory=list)
    per_media_reach: float = 0.0
    per_media_impressions: float = 0.0
    engagement_rate: float = 0.0
    post_category: str = "General / Lifestyle"


@dataclass
class PublicCreatorMetadata:
    meta_id: str
    instagram_id: Optional[str]
    username: str
    full_name: str
    biography: str
    website: Optional[str]
    followers_count: int
    follows_count: int
    media_count: int
    profile_picture_url: Optional[str]
    follower_to_following_ratio: float
    bio_hashtags: List[str]
    bio_mentions: List[str]
    has_website_link: bool
    account_category: str


@dataclass
class PublicCreatorSnapshot:
    profile: ProfileInput
    metadata: PublicCreatorMetadata
    posts: List[PublicPostItem]
    raw_response: Dict[str, Any]
    source: str = "META_BUSINESS_DISCOVERY"


class InstagramPublicFetcher:
    """
    Official Meta Graph API Business Discovery client for querying public Instagram
    creator and business accounts. Returns only observed data from Meta.
    """

    DEFAULT_API_VERSION = "v20.0"

    def __init__(
        self,
        meta_access_token: Optional[str] = None,
        meta_business_account_id: Optional[str] = None,
        api_version: Optional[str] = None,
        timeout: int = 15,
    ):
        self.access_token = (
            meta_access_token.strip() if meta_access_token is not None
            else os.getenv("META_ACCESS_TOKEN", "").strip()
        )
        self.business_account_id = (
            meta_business_account_id.strip() if meta_business_account_id is not None
            else os.getenv("META_IG_BUSINESS_ACCOUNT_ID", "").strip()
        )
        self.api_version = api_version or self.DEFAULT_API_VERSION
        self.timeout = timeout

    def fetch_creator(self, username: str, media_limit: int = 25) -> PublicCreatorSnapshot:
        clean = username.strip().lstrip("@").lower()
        if not clean:
            raise ValueError("Username cannot be empty")

        if not self.access_token or not self.business_account_id:
            raise ValueError(
                "Meta Graph API credentials are required. Please configure META_ACCESS_TOKEN "
                "and META_IG_BUSINESS_ACCOUNT_ID in your .env file or enter them in the app settings."
            )

        limit = min(max(1, int(media_limit)), 100)

        profile_fields = (
            "id,ig_id,username,name,biography,website,"
            "followers_count,follows_count,media_count,profile_picture_url"
        )
        media_fields = (
            "id,caption,comments_count,like_count,media_product_type,media_type,"
            "media_url,permalink,thumbnail_url,timestamp,username,"
            "children{id,media_type,media_url,permalink}"
        )
        query = (
            f"business_discovery.username({clean}){{"
            f"{profile_fields},"
            f"media.limit({limit}){{{media_fields}}}"
            f"}}"
        )

        url = (
            f"https://graph.facebook.com/{self.api_version}/{self.business_account_id}"
            f"?fields={urllib.parse.quote(query)}&access_token={self.access_token}"
        )

        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "InstagramPredictorEngine/3.0",
            },
            method="GET",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as he:
            err_msg = he.reason
            try:
                err_json = json.loads(he.read().decode("utf-8"))
                err_dict = err_json.get("error", {})
                err_msg = err_dict.get("message", err_msg)
                code = err_dict.get("code")
                if code == 100:
                    raise RuntimeError(
                        f"Instagram account @{clean} does not exist, is private, "
                        "or is not a professional/business account accessible via Business Discovery."
                    ) from he
                if code == 190:
                    raise RuntimeError(
                        "Your Meta User Access Token is invalid or has expired. Please refresh your token."
                    ) from he
            except json.JSONDecodeError:
                pass
            raise RuntimeError(f"Meta Graph API error (HTTP {he.code}): {err_msg}") from he
        except urllib.error.URLError as ue:
            raise RuntimeError(f"Network connection to Meta Graph API failed: {ue.reason}") from ue

        disc = data.get("business_discovery")
        if not disc or "followers_count" not in disc:
            raise RuntimeError(f"Meta Graph API did not return profile metrics for @{clean}.")

        return self._parse_snapshot(clean, disc, data)

    def _parse_snapshot(
        self,
        username: str,
        disc: Dict[str, Any],
        raw_response: Dict[str, Any],
    ) -> PublicCreatorSnapshot:
        followers = int(disc.get("followers_count") or 0)
        following = int(disc.get("follows_count") or 0)
        media_count = int(disc.get("media_count") or 0)
        bio = str(disc.get("biography") or "")
        name = str(disc.get("name") or username)
        website = disc.get("website")

        bio_hashtags = re.findall(r"#\w+", bio)
        bio_mentions = re.findall(r"@\w+", bio)
        has_website_link = bool(website) or any(t in bio.lower() for t in ("http://", "https://"))
        ratio = round(followers / max(1, following), 2)
        category = self._detect_niche(f"{bio} {name}")

        metadata = PublicCreatorMetadata(
            meta_id=str(disc.get("id") or ""),
            instagram_id=str(disc.get("ig_id")) if disc.get("ig_id") else None,
            username=disc.get("username", username),
            full_name=name,
            biography=bio,
            website=website,
            followers_count=followers,
            follows_count=following,
            media_count=media_count,
            profile_picture_url=disc.get("profile_picture_url"),
            follower_to_following_ratio=ratio,
            bio_hashtags=bio_hashtags,
            bio_mentions=bio_mentions,
            has_website_link=has_website_link,
            account_category=category,
        )

        posts: List[PublicPostItem] = []
        for m in (disc.get("media", {}).get("data") or []):
            ts = m.get("timestamp")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    dt = datetime.now(timezone.utc)
            else:
                dt = datetime.now(timezone.utc)

            raw_type = str(m.get("media_type") or "").upper()
            prod_type = str(m.get("media_product_type") or "").upper()
            if prod_type == "REELS" or raw_type in ("VIDEO", "REEL"):
                media_type = "Reel"
            elif raw_type == "CAROUSEL_ALBUM":
                media_type = "Carousel"
            else:
                media_type = "Static Image"

            caption = m.get("caption") or ""
            hashtags = re.findall(r"#\w+", caption)
            mentions = re.findall(r"@\w+", caption)

            # Parse carousel children slides if present
            children: List[CarouselChildItem] = []
            for child in (m.get("children", {}).get("data") or []):
                children.append(CarouselChildItem(
                    id=str(child.get("id") or ""),
                    media_type=str(child.get("media_type") or ""),
                    media_url=child.get("media_url"),
                    permalink=child.get("permalink"),
                ))

            post_likes = int(m.get("like_count") or 0)
            post_comments = int(m.get("comments_count") or 0)
            post_reach = calculate_per_post_reach(
                total_followers=followers,
                like_count=post_likes,
                comments_count=post_comments,
                media_type=media_type,
                media_product_type=prod_type or "FEED",
            )
            post_impressions = calculate_per_post_impressions(post_reach)
            post_er = round(((post_likes + post_comments) / max(followers, 1)) * 100, 3)
            post_cat = classify_post_category(caption, hashtags)

            posts.append(PublicPostItem(
                post_id=str(m.get("id") or ""),
                media_type=media_type,
                media_product_type=prod_type or "FEED",
                caption=caption,
                caption_length_chars=len(caption),
                hashtags=hashtags,
                mentions=mentions,
                likes=post_likes,
                comments=post_comments,
                timestamp_utc=ts or dt.isoformat(),
                posted_day_of_week=dt.strftime("%A"),
                posted_hour_of_day=dt.hour,
                permalink=m.get("permalink") or "",
                media_url=m.get("media_url"),
                thumbnail_url=m.get("thumbnail_url"),
                carousel_slide_count=len(children) if children else None,
                carousel_children=children,
                per_media_reach=post_reach,
                per_media_impressions=post_impressions,
                engagement_rate=post_er,
                post_category=post_cat,
            ))


        profile = ProfileInput(
            username=metadata.username,
            platform=PlatformType.INSTAGRAM,
            total_followers=followers,
            total_following=following,
            total_media_posts=media_count,
            full_name=name,
            account_category=category,
            biography=bio,
            has_bio_link=has_website_link,
            profile_picture_url=metadata.profile_picture_url,
        )

        return PublicCreatorSnapshot(
            profile=profile,
            metadata=metadata,
            posts=posts,
            raw_response=raw_response,
            source="META_BUSINESS_DISCOVERY",
        )

    @staticmethod
    def _detect_niche(text: str) -> str:
        low = text.lower()
        keywords = {
            "Technology": ["tech", "software", "code", "developer", "ai", "gadgets", "engineering", "robotics", "burner", "hardware"],
            "Sports": ["football", "soccer", "cricket", "fitness", "athlete", "nba", "gym", "workout", "sports", "coach"],
            "Fashion": ["fashion", "style", "model", "outfit", "clothing", "streetwear", "apparel", "designer"],
            "Beauty": ["beauty", "makeup", "skincare", "hair", "cosmetics", "glow", "glam"],
            "Food": ["food", "chef", "recipe", "cooking", "baking", "foodie", "restaurant", "kitchen"],
            "Travel": ["travel", "wanderlust", "explore", "adventure", "tourism", "nomad", "destinations"],
            "Finance": ["finance", "investing", "crypto", "stocks", "money", "trading", "business", "founder"],
            "Gaming": ["gaming", "gamer", "esports", "streamer", "playstation", "xbox", "pcbuild"],
        }
        for cat, words in keywords.items():
            if any(w in low for w in words):
                return cat
        return "Lifestyle"
