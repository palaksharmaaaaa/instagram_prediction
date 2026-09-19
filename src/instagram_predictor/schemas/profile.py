from enum import Enum
from typing import Optional, List, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class PlatformType(str, Enum):
    INSTAGRAM = "Instagram"
    YOUTUBE = "YouTube"
    SNAPCHAT = "Snapchat"


class MediaType(str, Enum):
    # Instagram formats
    REEL = "Reel"
    CAROUSEL = "Carousel"
    STATIC_IMAGE = "Static Image"
    STORY = "Story"
    VIDEO = "Video"
    # YouTube formats
    YOUTUBE_SHORT = "YouTube Short"
    YOUTUBE_VIDEO = "YouTube Video"
    COMMUNITY_POST = "Community Post"
    # Snapchat formats
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


class ContentCategory(str, Enum):
    SPORTS = "Sports"
    HEALTH_FITNESS = "Health & Fitness"
    FINANCE_BUSINESS = "Finance & Business"
    FASHION_BEAUTY = "Fashion & Beauty"
    SCIENCE_TECHNOLOGY = "Science & Technology"
    TRAVEL_EVENTS = "Travel & Events"
    FOOD_DINING = "Food & Dining"
    MUSIC_ENTERTAINMENT = "Music & Entertainment"
    EDUCATION_CAREERS = "Education & Careers"


class ContentStyle(str, Enum):
    EDUCATIONAL = "Educational / How-To"
    ENTERTAINING = "Entertaining / Trend"
    PROMOTIONAL = "Promotional / Sponsored"
    BEHIND_THE_SCENES = "Behind The Scenes"
    INSPIRATIONAL = "Inspirational / Storytelling"


class Demographics(BaseModel):
    model_config = ConfigDict(extra="allow")

    top_country: str = Field(default="US", description="Top audience country ISO code")
    secondary_country: str = Field(default="IN", description="Secondary audience country ISO code")
    primary_age_group: str = Field(default="25-34", description="Dominant audience age group")
    gender_female_pct: float = Field(default=0.50, ge=0.0, le=1.0, description="Female audience percentage (0.0 to 1.0)")
    gender_male_pct: float = Field(default=0.50, ge=0.0, le=1.0, description="Male audience percentage (0.0 to 1.0)")
    audience_activity_score: float = Field(default=0.75, ge=0.0, le=1.0, description="Audience daily active index (0.0 to 1.0)")
    is_estimated: bool = Field(default=False, description="Whether demographics data is an estimated baseline prior")

    @field_validator("top_country", "secondary_country")
    def clean_country(cls, v: str) -> str:
        return v.strip().upper()

    @model_validator(mode="before")
    @classmethod
    def sync_gender_pct(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "gender_female_pct" in data and "gender_male_pct" not in data:
                data["gender_male_pct"] = round(1.0 - float(data["gender_female_pct"]), 4)
            elif "gender_male_pct" in data and "gender_female_pct" not in data:
                data["gender_female_pct"] = round(1.0 - float(data["gender_male_pct"]), 4)
        return data

    @model_validator(mode="after")
    def validate_gender_sum(self) -> "Demographics":
        if abs((self.gender_female_pct + self.gender_male_pct) - 1.0) > 0.05:
            raise ValueError(
                f"Demographics female and male percentages must sum to 1.0 (got {self.gender_female_pct + self.gender_male_pct:.2f})"
            )
        return self


class PostMetrics(BaseModel):
    model_config = ConfigDict(extra="allow")

    likes: int = Field(default=0, ge=0)
    comments: int = Field(default=0, ge=0)
    shares: int = Field(default=0, ge=0)
    saves: int = Field(default=0, ge=0)
    video_views: Optional[int] = Field(default=None, ge=0)
    completion_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reach_from_home_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reach_from_explore_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reach_from_hashtags_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reach: Optional[int] = Field(default=None, ge=0)
    impressions: Optional[int] = Field(default=None, ge=0)
    views: Optional[int] = Field(default=None, ge=0, description="Modern Meta v22.0 video/media views")
    total_interactions: Optional[int] = Field(default=None, ge=0, description="Meta v22.0 aggregate total interactions")


class ProfileInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    platform: PlatformType = Field(default=PlatformType.INSTAGRAM, description="Social media publishing platform")
    username: str = Field(min_length=1, max_length=50)
    full_name: str = Field(default="", max_length=100)
    country: str = Field(default="US")
    total_followers: int = Field(ge=0)
    total_following: int = Field(ge=0, le=7500, description="Instagram platform limit is 7,500 following")
    total_media_posts: int = Field(ge=0)
    account_age_years: float = Field(default=4.0, ge=0.0, description="Account maturity in years")
    posting_frequency_per_week: float = Field(default=3.5, ge=0.0, description="Average posts published per week")
    follower_growth_rate_30d: float = Field(default=0.02, description="30-day net follower growth rate")
    account_bio_has_link: bool = Field(default=False, description="Whether bio contains external URL")
    is_verified: bool = Field(default=False)
    account_category: ContentCategory = Field(default=ContentCategory.MUSIC_ENTERTAINMENT, description="Primary profile topic category")
    account_categories: List[ContentCategory] = Field(default_factory=list, description="Array of distinct categories represented across creator posts")
    profile_picture_url: Optional[str] = Field(default=None, description="Direct URL to creator profile picture")
    biography: Optional[str] = Field(default=None, description="Creator profile bio text")
    raw_following: Optional[int] = Field(default=None, description="Uncapped following count reported by platform")
    country_inferred: bool = Field(default=False, description="Whether country was inferred/defaulted rather than from live insights")
    estimated_metrics: bool = Field(default=False, description="Whether account_age_years, posting_frequency, and follower_growth are estimated baseline priors")
    raw_category: Optional[str] = Field(default=None, description="Raw category string from Meta Graph API")
    meta_category_raw: Optional[str] = Field(default=None, description="Raw Meta API category string")
    prior_metrics_estimated: bool = Field(default=False, description="Whether historical reach/impressions metrics are estimated baseline priors")


    @field_validator("username")
    def clean_username(cls, v: str) -> str:
        return v.strip().lstrip("@").lower()

    @model_validator(mode="before")
    @classmethod
    def sync_categories(cls, data: Any) -> Any:
        if isinstance(data, dict):
            cats = data.get("account_categories")
            primary = data.get("account_category")
            if cats and isinstance(cats, list) and len(cats) > 0:
                if not primary:
                    data["account_category"] = cats[0]
            elif primary:
                data["account_categories"] = [primary]
        return data


class PostInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: PlatformType = Field(default=PlatformType.INSTAGRAM, description="Social media publishing platform")
    media_type: MediaType = Field(default=MediaType.REEL)
    category: ContentCategory = Field(default=ContentCategory.SPORTS, description="Single primary category for this media post")
    categorizations: List[ContentStyle] = Field(default_factory=list, description="Array of content style categorizations for this media post")
    categorization: Optional[ContentStyle] = Field(default=None, description="Primary content style (backward-compatible)")
    caption_length_chars: int = Field(default=220, ge=0, le=2200, description="Character count of the post caption")
    hashtags_count: int = Field(default=5, ge=0, le=30, description="Number of hashtags used (0 to 30)")
    mentions_count: int = Field(default=1, ge=0, le=20, description="Number of tagged user handles")
    has_call_to_action: bool = Field(default=True, description="Whether post explicitly prompts save/share/comment")
    video_duration_seconds: float = Field(default=0.0, ge=0.0, description="Duration in seconds (for Reels/Videos)")
    carousel_slide_count: int = Field(default=1, ge=1, le=10, description="Number of slides (for Carousels)")
    video_title_length: int = Field(default=60, ge=0, le=100, description="Title length for YouTube content")
    thumbnail_has_face: bool = Field(default=True, description="Whether thumbnail features a human face for YouTube content")
    screenshot_count: int = Field(default=0, ge=0, description="Audience screenshot count for Snapchat content")
    posted_day_of_week: str = Field(default="Wednesday", description="Day of the week published")
    posted_hour_of_day: int = Field(default=18, ge=0, le=23, description="Hour of the day published (0 to 23)")
    demographics: Demographics = Field(default_factory=Demographics)
    metrics: Optional[PostMetrics] = None
    id: Optional[str] = Field(default=None, description="Platform media identifier")
    permalink: Optional[str] = Field(default=None, description="Direct web link to the media post")
    caption: Optional[str] = Field(default=None, description="Raw text caption of the media post")
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp of media publishing")


    @model_validator(mode="before")
    @classmethod
    def sync_platform_and_categorizations(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # 1. Platform auto-inference & validation
            media_type_raw = data.get("media_type")
            platform_raw = data.get("platform")

            if media_type_raw is not None:
                try:
                    resolved_mt = MediaType(media_type_raw) if not isinstance(media_type_raw, MediaType) else media_type_raw
                    expected_platform = FORMAT_TO_PLATFORM.get(resolved_mt)
                    if expected_platform:
                        if platform_raw is None:
                            data["platform"] = expected_platform
                        else:
                            try:
                                resolved_platform = PlatformType(platform_raw) if not isinstance(platform_raw, PlatformType) else platform_raw
                                if resolved_platform != expected_platform:
                                    raise ValueError(
                                        f"Media format '{resolved_mt.value}' does not match specified platform '{resolved_platform.value}'"
                                    )
                            except ValueError as ve:
                                if "does not match" in str(ve):
                                    raise
                                raise
                except ValueError as ve:
                    if "does not match" in str(ve):
                        raise

            # 2. Synchronize categorizations
            styles = data.get("categorizations")
            single = data.get("categorization")
            if styles and isinstance(styles, list) and len(styles) > 0:
                if not single:
                    data["categorization"] = styles[0]
            elif single:
                data["categorizations"] = [single]
            else:
                data["categorization"] = ContentStyle.ENTERTAINING
                data["categorizations"] = [ContentStyle.ENTERTAINING]
        return data

    @model_validator(mode="after")
    def validate_platform_media_compatibility(self) -> "PostInput":
        expected_platform = FORMAT_TO_PLATFORM.get(self.media_type)
        if expected_platform and self.platform != expected_platform:
            raise ValueError(
                f"Media format '{self.media_type.value}' is not supported on platform '{self.platform.value}'"
            )
        return self

    @property
    def primary_categorization(self) -> ContentStyle:
        if self.categorizations:
            return self.categorizations[0]
        return self.categorization or ContentStyle.ENTERTAINING
