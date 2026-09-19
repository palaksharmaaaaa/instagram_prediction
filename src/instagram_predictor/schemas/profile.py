"""
Input schemas.

Design rule: a field is either supplied by the caller or it is None. There are
no silent defaults for facts about a creator or a post; the model layer reports
exactly which optional fields were missing.
"""

from enum import Enum
from typing import Optional
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
