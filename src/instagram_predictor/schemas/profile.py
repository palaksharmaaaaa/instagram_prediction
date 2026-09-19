from enum import Enum
from typing import Optional, List, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class MediaType(str, Enum):
    REEL = "Reel"
    CAROUSEL = "Carousel"
    STATIC_IMAGE = "Static Image"
    STORY = "Story"


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
    top_country: str = Field(default="US", description="Top audience country ISO code")
    secondary_country: str = Field(default="IN", description="Secondary audience country ISO code")
    primary_age_group: str = Field(default="25-34", description="Dominant audience age group")
    gender_female_pct: float = Field(default=0.50, ge=0.0, le=1.0, description="Female audience percentage (0.0 to 1.0)")
    gender_male_pct: float = Field(default=0.50, ge=0.0, le=1.0, description="Male audience percentage (0.0 to 1.0)")
    audience_activity_score: float = Field(default=0.75, ge=0.0, le=1.0, description="Audience daily active index (0.0 to 1.0)")

    @field_validator("top_country", "secondary_country")
    def clean_country(cls, v: str) -> str:
        return v.strip().upper()


class PostMetrics(BaseModel):
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


class ProfileInput(BaseModel):
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
    media_type: MediaType = Field(default=MediaType.REEL)
    category: ContentCategory = Field(default=ContentCategory.SPORTS, description="Single primary category for this media post")
    categorizations: List[ContentStyle] = Field(default_factory=list, description="Array of content style categorizations for this media post")
    categorization: Optional[ContentStyle] = Field(default=None, description="Primary content style (backward-compatible)")
    caption_length_chars: int = Field(default=220, ge=0, description="Character count of the post caption")
    hashtags_count: int = Field(default=5, ge=0, le=30, description="Number of hashtags used (0 to 30)")
    mentions_count: int = Field(default=1, ge=0, le=20, description="Number of tagged user handles")
    has_call_to_action: bool = Field(default=True, description="Whether post explicitly prompts save/share/comment")
    video_duration_seconds: float = Field(default=0.0, ge=0.0, description="Duration in seconds (for Reels/Videos)")
    carousel_slide_count: int = Field(default=1, ge=1, le=10, description="Number of slides (for Carousels)")
    posted_day_of_week: str = Field(default="Wednesday", description="Day of the week published")
    posted_hour_of_day: int = Field(default=18, ge=0, le=23, description="Hour of the day published (0 to 23)")
    demographics: Demographics = Field(default_factory=Demographics)
    metrics: Optional[PostMetrics] = None

    @model_validator(mode="before")
    @classmethod
    def sync_categorizations(cls, data: Any) -> Any:
        if isinstance(data, dict):
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

    @property
    def primary_categorization(self) -> ContentStyle:
        if self.categorizations:
            return self.categorizations[0]
        return self.categorization or ContentStyle.ENTERTAINING
