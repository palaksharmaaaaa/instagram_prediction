"""
Input schemas.

Design rule: a field is either supplied by the caller or it is None. There are
no silent defaults for facts about a creator or a post; the model layer reports
exactly which optional fields were missing.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PlatformType(str, Enum):
    INSTAGRAM = "Instagram"
    YOUTUBE = "YouTube"
    SNAPCHAT = "Snapchat"


class MediaType(str, Enum):
    # Instagram
    REEL = "Reel"
    CAROUSEL = "Carousel"
    STATIC_IMAGE = "Static Image"
    STORY = "Story"
    VIDEO = "Video"
    # YouTube
    YOUTUBE_SHORT = "YouTube Short"
    YOUTUBE_VIDEO = "YouTube Video"
    COMMUNITY_POST = "Community Post"
    # Snapchat
    SNAPCHAT_SPOTLIGHT = "Snapchat Spotlight"
    SNAPCHAT_STORY = "Snapchat Story"
    SNAPCHAT_POST = "Snapchat Post"


FORMAT_TO_PLATFORM = {
    MediaType.REEL: PlatformType.INSTAGRAM,
    MediaType.CAROUSEL: PlatformType.INSTAGRAM,
    MediaType.STATIC_IMAGE: PlatformType.INSTAGRAM,
    MediaType.STORY: PlatformType.INSTAGRAM,
    MediaType.VIDEO: PlatformType.INSTAGRAM,
    MediaType.YOUTUBE_SHORT: PlatformType.YOUTUBE,
    MediaType.YOUTUBE_VIDEO: PlatformType.YOUTUBE,
    MediaType.COMMUNITY_POST: PlatformType.YOUTUBE,
    MediaType.SNAPCHAT_SPOTLIGHT: PlatformType.SNAPCHAT,
    MediaType.SNAPCHAT_STORY: PlatformType.SNAPCHAT,
    MediaType.SNAPCHAT_POST: PlatformType.SNAPCHAT,
}

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def platform_of(media_type: "MediaType | str") -> PlatformType:
    return FORMAT_TO_PLATFORM[MediaType(media_type)]


def normalize_day(value: str) -> str:
    """Returns the canonical weekday name or raises ValueError. Never guesses."""
    v = str(value).strip().title()
    if v not in DAYS_OF_WEEK:
        raise ValueError(f"'{value}' is not a weekday; expected one of {DAYS_OF_WEEK}")
    return v


class ProfileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    username: str = Field(min_length=1, max_length=50)
    platform: PlatformType
    total_followers: int = Field(ge=1, description="Follower count; must be a positive integer")
    total_following: Optional[int] = Field(default=None, ge=0)
    total_media_posts: Optional[int] = Field(default=None, ge=0)
    full_name: Optional[str] = Field(default=None, max_length=100)
    account_category: Optional[str] = Field(default=None, description="Free text exactly as reported by the source")
    biography: Optional[str] = None
    has_bio_link: Optional[bool] = None
    profile_picture_url: Optional[str] = None

    @field_validator("username")
    @classmethod
    def clean_username(cls, v: str) -> str:
        return v.strip().lstrip("@").lower()


class PostInput(BaseModel):
    """
    A post to forecast. Required: format and publication time. Everything else
    is optional and, when omitted, is reported as missing by the forecaster.
    """
    model_config = ConfigDict(extra="forbid")

    media_type: MediaType
    posted_day_of_week: str
    posted_hour_of_day: int = Field(ge=0, le=23)
    caption_length_chars: Optional[int] = Field(default=None, ge=0)
    hashtags_count: Optional[int] = Field(default=None, ge=0)
    mentions_count: Optional[int] = Field(default=None, ge=0)
    has_call_to_action: Optional[bool] = None
    video_duration_seconds: Optional[float] = Field(default=None, ge=0.0)
    carousel_slide_count: Optional[int] = Field(default=None, ge=1)

    @field_validator("posted_day_of_week")
    @classmethod
    def validate_day(cls, v: str) -> str:
        return normalize_day(v)

    @model_validator(mode="after")
    def carousel_only_slides(self) -> "PostInput":
        if self.carousel_slide_count is not None and self.carousel_slide_count > 1 \
                and self.media_type != MediaType.CAROUSEL:
            raise ValueError("carousel_slide_count > 1 is only valid for the Carousel format")
        return self

    @property
    def platform(self) -> PlatformType:
        return platform_of(self.media_type)


# ============================================================================
# FORMULAS FOR CALCULATIVE FIELDS & METRICS
# ============================================================================

def calculate_per_post_reach(
    total_followers: int,
    like_count: int,
    comments_count: int,
    media_type: str = "Reel",
    media_product_type: str = "REELS",
) -> float:
    """
    Calculative Formula for Per-Post Reach.
    
    Formula:
      per_media_reach = max(
          like_count + comments_count,
          round((total_followers * tier_rate) + ((like_count + comments_count) * format_multiplier), 1)
      )
      
    Where:
      1. Audience Tier Baseline Penetration (tier_rate):
         - Nano   (< 10k followers):      25.0% (0.25)
         - Micro  (10k - 100k followers):  18.0% (0.18)
         - Mid    (100k - 500k followers): 14.0% (0.14)
         - Macro  (500k - 1M followers):   10.0% (0.10)
         - Mega   (> 1M followers):        6.0% (0.06)
      2. Algorithmic Viral Multiplier (format_multiplier):
         - Reels / Video:  4.2x  (High non-follower Explore & Reels distribution)
         - Carousels:      2.8x  (Repeated feed impressions from multi-slides)
         - Static Images:  2.2x  (Core feed distribution)
      3. Bounds & Guardrails:
         - Floor: Cannot be lower than total verified interactions (likes + comments)
         - Cap: Limited to min(reach, total_followers * 5.0) for viral reels
    """
    followers = max(int(total_followers), 1)
    likes = max(int(like_count), 0)
    comments = max(int(comments_count), 0)
    total_engagement = likes + comments

    if followers < 10_000:
        tier_rate = 0.25
    elif followers < 100_000:
        tier_rate = 0.18
    elif followers < 500_000:
        tier_rate = 0.14
    elif followers < 1_000_000:
        tier_rate = 0.10
    else:
        tier_rate = 0.06

    is_reel = (
        str(media_product_type).upper() == "REELS"
        or str(media_type).lower() in ("reel", "video")
    )
    if is_reel:
        format_multiplier = 4.2
    elif str(media_type).lower() == "carousel":
        format_multiplier = 2.8
    else:
        format_multiplier = 2.2

    reach = (followers * tier_rate) + (total_engagement * format_multiplier)
    reach = max(float(total_engagement), reach)
    reach = min(reach, float(followers * 5.0))
    return float(round(reach, 1))


def calculate_per_post_impressions(per_media_reach: float, frequency_multiplier: float = 1.25) -> float:
    """
    Calculative Formula for Per-Post Impressions.
    
    Formula:
      per_media_impressions = round(per_media_reach * frequency_multiplier, 1)
      
    Where:
      frequency_multiplier = 1.25 (typical repeat exposure frequency per unique viewer on Instagram)
    """
    return float(round(max(per_media_reach, 0.0) * frequency_multiplier, 1))


def calculate_avg_reach_per_post(posts_reach: list[float]) -> float:
    """
    Calculative Formula for Average Reach Per Post.
    
    Formula:
      avg_reach_per_post = sum(posts_reach) / count(posts_reach)
    """
    if not posts_reach:
        return 0.0
    return float(round(sum(posts_reach) / len(posts_reach), 1))


def calculate_overall_account_avg_reach(
    total_followers: int,
    avg_likes: float,
    avg_comments: float,
    reels_ratio: float = 0.5,
) -> float:
    """
    Calculative Formula for Overall Account Average Reach.
    
    Formula:
      overall_account_avg_reach = (total_followers * tier_rate) + (avg_engagement * blended_multiplier)
      
    Where:
      avg_engagement = avg_likes + avg_comments
      blended_multiplier = (reels_ratio * 4.2) + ((1 - reels_ratio) * 2.5)
    """
    followers = max(int(total_followers), 1)
    avg_eng = max(avg_likes, 0.0) + max(avg_comments, 0.0)

    if followers < 10_000:
        tier_rate = 0.25
    elif followers < 100_000:
        tier_rate = 0.18
    elif followers < 500_000:
        tier_rate = 0.14
    elif followers < 1_000_000:
        tier_rate = 0.10
    else:
        tier_rate = 0.06

    r_ratio = max(0.0, min(1.0, float(reels_ratio)))
    blended_multiplier = (r_ratio * 4.2) + ((1.0 - r_ratio) * 2.5)
    reach = (followers * tier_rate) + (avg_eng * blended_multiplier)
    reach = max(float(avg_eng), reach)
    reach = min(reach, float(followers * 5.0))
    return float(round(reach, 1))


def classify_post_category(caption: str, hashtags: Optional[list[str]] = None) -> str:
    """
    Rule-based post category classifier based on caption content and hashtags.
    """
    text = (caption or "").lower()
    tags = [t.lower().lstrip("#") for t in (hashtags or [])]
    combined = f"{text} {' '.join(tags)}"

    if any(k in combined for k in ["#ad", "#sponsored", "#collab", "#partner", "paid partnership", "sponsored"]):
        return "Sponsored / Brand Collab"
    elif any(k in combined for k in ["tech", "ai", "gadget", "phone", "iphone", "samsung", "laptop", "code", "software", "unboxing", "review"]):
        return "Technology & AI"
    elif any(k in combined for k in ["workout", "fitness", "gym", "exercise", "training", "diet", "protein", "muscle", "health"]):
        return "Fitness & Health"
    elif any(k in combined for k in ["fashion", "outfit", "style", "beauty", "makeup", "look", "dress", "glam", "hair", "skincare"]):
        return "Fashion & Beauty"
    elif any(k in combined for k in ["funny", "comedy", "meme", "joke", "lol", "entertainment", "reelscomedy"]):
        return "Comedy & Entertainment"
    elif any(k in combined for k in ["recipe", "food", "cook", "chef", "delicious", "restaurant", "yummy", "meal"]):
        return "Food & Cooking"
    elif any(k in combined for k in ["travel", "wanderlust", "trip", "vacation", "explore", "nature", "destination"]):
        return "Travel & Adventure"
    elif any(k in combined for k in ["business", "money", "finance", "crypto", "invest", "startup", "entrepreneur", "marketing"]):
        return "Business & Finance"
    return "General / Lifestyle"


# ============================================================================
# COMPREHENSIVE SCHEMAS AS PER LIVE GRAPH API RESEARCH
# ============================================================================

class InstagramPostRecord(BaseModel):
    """
    Comprehensive Per-Post Schema for Instagram content.
    
    Fields are partitioned strictly according to the Meta Graph API research:
      - Mandatory: Everything verified & accessible directly via username (likes, comments, caption, permalink, timestamps, format, followers).
      - Calculative: Calculated deterministically with documented formulas (reach, impressions, ER, categorization).
      - Optional: Metrics that Meta restricts (#100 comment text, #10 private shares/saves) or format-specific.
    """
    model_config = ConfigDict(extra="ignore")

    # --- MANDATORY FIELDS (Directly Observable from Meta Graph API) ---
    post_id: str = Field(description="Unique Instagram Media ID (e.g. '18003160360600852')")
    username: str = Field(description="Instagram username/handle of the author")
    media_type: str = Field(description="Format classification: 'Reel', 'Carousel', 'Static Image', 'Video', 'IMAGE', etc.")
    media_product_type: str = Field(description="Product surface: 'REELS', 'FEED', or 'AD'")
    like_count: int = Field(ge=0, description="Verified real-time likes count")
    comments_count: int = Field(ge=0, description="Verified real-time comments count")
    caption: str = Field(default="", description="Full caption text")
    permalink: str = Field(description="Official direct URL (https://www.instagram.com/reel/... or /p/...)")
    timestamp_utc: str = Field(description="Publication timestamp in UTC ISO 8601")
    posted_day_of_week: str = Field(description="Weekday name: Monday through Sunday")
    posted_hour_of_day: int = Field(ge=0, le=23, description="UTC hour of publication 0-23")
    total_followers: int = Field(ge=1, description="Follower count at post/snapshot time")

    # --- CALCULATIVE FIELDS (Formula-Driven) ---
    per_media_engagement: int = Field(
        default=0,
        description="Formula: like_count + comments_count"
    )
    per_media_engagement_rate: float = Field(
        default=0.0,
        description="Formula: ((like_count + comments_count) / total_followers) * 100"
    )
    per_media_reach: float = Field(
        default=0.0,
        description="Calculative benchmark reach formula: followers * tier_rate + engagement * format_multiplier"
    )
    per_media_impressions: float = Field(
        default=0.0,
        description="Calculative benchmark impressions: per_media_reach * 1.25"
    )
    engagement_rate_by_reach: float = Field(
        default=0.0,
        description="Formula: (engagement / per_media_reach) * 100"
    )
    caption_length_chars: int = Field(
        default=0,
        description="Formula: len(caption)"
    )
    hashtags_count: int = Field(
        default=0,
        description="Count of #tags in caption"
    )
    mentions_count: int = Field(
        default=0,
        description="Count of @mentions in caption"
    )
    post_category: str = Field(
        default="General / Lifestyle",
        description="Niche classification derived from caption and hashtag NLP rules"
    )

    # --- OPTIONAL FIELDS (Meta-Restricted or Format-Specific) ---
    comment_text: Optional[str] = Field(
        default=None,
        description="Restricted by Meta (#100) on 3rd-party accounts. Available only for owned accounts."
    )
    replies_text: Optional[list[str]] = Field(
        default=None,
        description="Restricted by Meta (#100) on 3rd-party accounts."
    )
    shares_count: Optional[int] = Field(
        default=None,
        description="Restricted by Meta (#10) on 3rd-party accounts; private creator insight."
    )
    saved_count: Optional[int] = Field(
        default=None,
        description="Restricted by Meta (#10) on 3rd-party accounts; private creator insight."
    )
    media_url: Optional[str] = Field(
        default=None,
        description="Direct CDN media download stream URL"
    )
    thumbnail_url: Optional[str] = Field(
        default=None,
        description="Direct CDN poster thumbnail URL"
    )
    video_duration_seconds: Optional[float] = Field(
        default=None,
        description="Video runtime duration in seconds"
    )
    carousel_slide_count: Optional[int] = Field(
        default=None,
        description="Number of slides in carousel album"
    )
    carousel_children: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Sub-media items for multi-slide carousels"
    )
    has_call_to_action: Optional[bool] = Field(
        default=None,
        description="Whether caption contains an explicit CTA"
    )
    insights_available: bool = Field(
        default=False,
        description="True if native Meta insights were returned for this media"
    )
    insights_unavailable_reason: Optional[str] = Field(
        default=None,
        description="Reason explanation when Meta withholds private per-post insights"
    )

    @model_validator(mode="after")
    def compute_calculative_fields(self) -> "InstagramPostRecord":
        """Automatically synchronizes all calculative fields if left at default."""
        eng = self.like_count + self.comments_count
        self.per_media_engagement = eng
        if self.total_followers >= 1:
            self.per_media_engagement_rate = float(round((eng / self.total_followers) * 100, 3))
        
        if self.per_media_reach <= 0.0:
            self.per_media_reach = calculate_per_post_reach(
                self.total_followers,
                self.like_count,
                self.comments_count,
                self.media_type,
                self.media_product_type,
            )
        
        if self.per_media_impressions <= 0.0:
            self.per_media_impressions = calculate_per_post_impressions(self.per_media_reach)
            
        if self.per_media_reach > 0.0:
            self.engagement_rate_by_reach = float(round((eng / self.per_media_reach) * 100, 2))
            
        if self.caption:
            self.caption_length_chars = len(self.caption)
            tags = [w for w in self.caption.split() if w.startswith("#")]
            mentions = [w for w in self.caption.split() if w.startswith("@")]
            self.hashtags_count = len(tags)
            self.mentions_count = len(mentions)
            if self.post_category == "General / Lifestyle":
                self.post_category = classify_post_category(self.caption, tags)

        return self


class InstagramProfileRecord(BaseModel):
    """
    Comprehensive User Profile Schema for Instagram creator & business accounts.
    
    Fields:
      - Mandatory: All verified attributes retrieved directly from Meta Graph API.
      - Calculative: Account-level benchmarks, reach, format distribution, engagement rates.
      - Optional: Meta-restricted attributes (#100 is_verified) or optional contact links.
    """
    model_config = ConfigDict(extra="ignore")

    # --- MANDATORY FIELDS (Directly Observable from Meta Graph API) ---
    id: str = Field(description="Meta Graph API Scoped Account ID (e.g. '17841404467516249')")
    ig_id: str = Field(description="Traditional Instagram Numeric ID (e.g. '4369996917')")
    username: str = Field(description="Clean account handle (e.g. 'techburner')")
    name: str = Field(description="Full Display Name")
    biography: str = Field(default="", description="Complete bio string")
    total_followers: int = Field(ge=1, description="Verified real-time follower count")
    total_following: int = Field(ge=0, description="Verified real-time following count")
    total_media_posts: int = Field(ge=0, description="Total lifetime media posts count")
    profile_picture_url: str = Field(description="Direct CDN high-resolution avatar URL")
    website: Optional[str] = Field(default=None, description="External bio link (Linktree/site)")
    platform: str = Field(default="Instagram", description="Platform identifier")

    # --- CALCULATIVE FIELDS (Formula-Driven Benchmarks) ---
    avg_likes: float = Field(
        default=0.0,
        description="Formula: sum(post_likes) / sample_posts_count"
    )
    avg_comments: float = Field(
        default=0.0,
        description="Formula: sum(post_comments) / sample_posts_count"
    )
    avg_engagement: float = Field(
        default=0.0,
        description="Formula: avg_likes + avg_comments"
    )
    overall_engagement_rate: float = Field(
        default=0.0,
        description="Formula: ((avg_likes + avg_comments) / total_followers) * 100"
    )
    avg_reach_per_post: float = Field(
        default=0.0,
        description="Formula: sum(per_media_reach) / sample_posts_count"
    )
    overall_account_avg_reach: float = Field(
        default=0.0,
        description="Formula: followers * tier_rate + avg_engagement * format_multiplier"
    )
    avg_impressions_per_post: float = Field(
        default=0.0,
        description="Formula: sum(per_media_impressions) / sample_posts_count"
    )
    follower_to_following_ratio: float = Field(
        default=0.0,
        description="Formula: total_followers / max(total_following, 1)"
    )
    reels_ratio: float = Field(
        default=0.0,
        description="Formula: count(reels) / sample_posts_count"
    )
    carousel_ratio: float = Field(
        default=0.0,
        description="Formula: count(carousels) / sample_posts_count"
    )
    static_image_ratio: float = Field(
        default=0.0,
        description="Formula: count(static_images) / sample_posts_count"
    )

    # --- OPTIONAL FIELDS (Restricted by Meta #100 or Inferred) ---
    is_verified: Optional[bool] = Field(
        default=None,
        description="Restricted by Meta (#100) on business discovery. Manual or heuristic."
    )
    account_category: Optional[str] = Field(
        default=None,
        description="Inferred or reported creator niche / category"
    )
    bio_hashtags: list[str] = Field(
        default_factory=list,
        description="Extracted #tags from biography"
    )
    bio_mentions: list[str] = Field(
        default_factory=list,
        description="Extracted @handles from biography"
    )
    has_website_link: bool = Field(
        default=False,
        description="True if website URL is present"
    )

    @field_validator("username")
    @classmethod
    def clean_username(cls, v: str) -> str:
        return v.strip().lstrip("@").lower()

    @model_validator(mode="after")
    def compute_calculative_profile_fields(self) -> "InstagramProfileRecord":
        if self.total_following >= 0 and self.total_followers >= 1:
            self.follower_to_following_ratio = float(
                round(self.total_followers / max(self.total_following, 1), 2)
            )
        self.has_website_link = bool(self.website and self.website.strip())
        self.avg_engagement = float(round(self.avg_likes + self.avg_comments, 1))
        if self.total_followers >= 1:
            self.overall_engagement_rate = float(
                round((self.avg_engagement / self.total_followers) * 100, 3)
            )
        if self.overall_account_avg_reach <= 0.0 and self.total_followers >= 1:
            self.overall_account_avg_reach = calculate_overall_account_avg_reach(
                self.total_followers,
                self.avg_likes,
                self.avg_comments,
                self.reels_ratio or 0.5,
            )
        return self
