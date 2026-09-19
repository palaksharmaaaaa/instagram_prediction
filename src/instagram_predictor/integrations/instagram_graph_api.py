import re
import json
import logging
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Tuple

from instagram_predictor.schemas.profile import (
    ProfileInput,
    PostInput,
    MediaType,
    ContentCategory,
    ContentStyle,
    Demographics,
    PostMetrics,
)

logger = logging.getLogger(__name__)


def mask_token(token: Optional[str]) -> str:
    """
    Masks an access token for safe logging and UI rendering.
    Example: 'EAAB12345678abcdefgh' -> 'EAAB...***...efgh'
    """
    if not token:
        return ""
    token_str = str(token).strip()
    if len(token_str) <= 8:
        return "***"
    return f"{token_str[:4]}...***...{token_str[-4:]}"


def sanitize_tokens_in_text(text: str) -> str:
    """
    Scans a string or URL and redacts any query parameter tokens or standalone tokens.
    """
    if not text:
        return ""
    # Redact query parameters
    sanitized = re.sub(
        r"(access_token|client_secret|fb_exchange_token)=([^& \n\r\t]+)",
        r"\1=***MASKED***",
        text,
        flags=re.IGNORECASE,
    )
    # Redact standard Meta / Instagram Graph tokens
    sanitized = re.sub(
        r"\b(EAA[A-Za-z0-9]{20,}|IGQ[A-Za-z0-9]{20,})\b",
        lambda m: mask_token(m.group(1)),
        sanitized,
    )
    return sanitized


class MetaGraphAPIError(Exception):
    """
    Custom exception for Meta Instagram Graph API communication and data validation errors.
    Automatically masks tokens in error messages and attaches actionable troubleshooting advice.
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_code: Optional[int] = None,
        error_subcode: Optional[int] = None,
        fbtrace_id: Optional[str] = None,
        troubleshooting_tip: Optional[str] = None,
    ):
        self.raw_message = message
        self.status_code = status_code
        self.error_code = error_code
        self.error_subcode = error_subcode
        self.fbtrace_id = fbtrace_id
        self.troubleshooting_tip = troubleshooting_tip

        sanitized_msg = sanitize_tokens_in_text(message)
        details = [sanitized_msg]
        if error_code is not None:
            details.append(f"[Code: {error_code}]")
        if error_subcode is not None:
            details.append(f"[Subcode: {error_subcode}]")
        if status_code is not None:
            details.append(f"[HTTP {status_code}]")
        if fbtrace_id is not None:
            details.append(f"[Trace: {fbtrace_id}]")
        if troubleshooting_tip:
            details.append(f"\n💡 Troubleshooting: {troubleshooting_tip}")

        super().__init__(" ".join(details))


class InstagramGraphAPIClient:
    """
    Client for the Meta Instagram Graph API (v22.0).
    Provides methods to:
      1. Exchange short-lived tokens for 60-day long-lived tokens.
      2. Auto-discover linked Instagram Creator/Business accounts from Facebook Pages.
      3. Fetch creator profile metrics into a validated ProfileInput.
      4. Fetch recent media posts with performance metrics into validated PostInput objects.
      5. Fetch audience country, gender, and age distribution into a validated Demographics object.
      6. Fetch full creator state (Profile, Demographics, Media) in a single unified method.
    """

    DEFAULT_API_VERSION = "v22.0"
    DEFAULT_BASE_URL = f"https://graph.facebook.com/{DEFAULT_API_VERSION}"

    def __init__(
        self,
        access_token: Optional[str] = None,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
        timeout: int = 15,
        base_url: Optional[str] = None,
    ):
        self.access_token = access_token.strip() if access_token else None
        self.app_id = app_id.strip() if app_id else None
        self.app_secret = app_secret.strip() if app_secret else None
        self.timeout = timeout
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")

    def __repr__(self) -> str:
        return (
            f"InstagramGraphAPIClient(access_token='{mask_token(self.access_token)}', "
            f"app_id={repr(self.app_id)}, timeout={self.timeout})"
        )

    def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        method: str = "GET",
        access_token: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Executes an HTTP request to the Meta Graph API using standard library urllib.request.
        Automatically injects the access token, parses JSON, sanitizes error bodies, and handles timeouts.
        """
        token = access_token or self.access_token
        req_params = dict(params or {})

        if token and "access_token" not in req_params:
            req_params["access_token"] = token

        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            url = endpoint
        else:
            clean_endpoint = endpoint.lstrip("/")
            url = f"{self.base_url}/{clean_endpoint}"

        query_string = urllib.parse.urlencode(req_params)
        if query_string:
            sep = "&" if "?" in url else "?"
            full_url = f"{url}{sep}{query_string}"
        else:
            full_url = url

        req_headers = {
            "Accept": "application/json",
            "User-Agent": "InstagramPredictorEngine/2.0",
        }
        if headers:
            req_headers.update(headers)

        req = urllib.request.Request(full_url, headers=req_headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                content = response.read().decode("utf-8")
                try:
                    return json.loads(content)
                except json.JSONDecodeError as jde:
                    raise MetaGraphAPIError(
                        f"Malformed JSON returned by Meta Graph API: {jde}",
                        status_code=response.status,
                    ) from jde

        except urllib.error.HTTPError as he:
            body_text = ""
            err_code = None
            err_subcode = None
            fbtrace_id = None
            msg = he.reason

            try:
                body_text = he.read().decode("utf-8")
                parsed_err = json.loads(body_text)
                err_dict = parsed_err.get("error", {})
                msg = err_dict.get("message", msg)
                err_code = err_dict.get("code")
                err_subcode = err_dict.get("error_subcode")
                fbtrace_id = err_dict.get("fbtrace_id")
            except Exception:
                if body_text:
                    msg = body_text

            tip = self._diagnose_error(err_code, err_subcode, he.code, msg)
            raise MetaGraphAPIError(
                message=msg,
                status_code=he.code,
                error_code=err_code,
                error_subcode=err_subcode,
                fbtrace_id=fbtrace_id,
                troubleshooting_tip=tip,
            ) from he

        except urllib.error.URLError as ue:
            sanitized_reason = sanitize_tokens_in_text(str(ue.reason))
            raise MetaGraphAPIError(
                f"Network connection to Meta Graph API failed: {sanitized_reason}",
                troubleshooting_tip="Verify network connectivity, proxy settings, or Meta API availability.",
            ) from ue

    @staticmethod
    def _diagnose_error(
        error_code: Optional[int],
        error_subcode: Optional[int],
        http_code: int,
        message: str,
    ) -> str:
        """
        Translates Meta Graph API error codes into descriptive troubleshooting recommendations.
        """
        msg_lower = (message or "").lower()

        # Token expiration / invalid session
        if (
            error_code == 190
            or "session has expired" in msg_lower
            or "invalid oauth access token" in msg_lower
            or "error validating access token" in msg_lower
        ):
            return (
                "Your Meta User Access Token is invalid or has expired. "
                "Generate a fresh token via Meta Graph API Explorer or your Facebook Developer App."
            )

        # Rate limits
        if (
            error_code in (4, 17, 32, 613)
            or http_code == 429
            or "rate limit" in msg_lower
            or "too many calls" in msg_lower
        ):
            return (
                "Meta Graph API rate limit reached. "
                "Pause requests and wait 15-60 minutes for your quota to replenish."
            )

        # Missing permissions
        if (
            (error_code is not None and 200 <= error_code <= 299)
            or error_code == 10
            or "permission" in msg_lower
            or "scope" in msg_lower
            or "not authorized" in msg_lower
        ):
            return (
                "Missing required Meta permissions. Ensure your token is granted: "
                "'instagram_basic', 'instagram_manage_insights', 'pages_show_list', and 'pages_read_engagement'."
            )

        # Missing or unlinked business account
        if (
            error_code == 100
            or "unsupported get request" in msg_lower
            or "does not exist" in msg_lower
        ):
            return (
                "Target account not found or unsupported. Verify the Instagram account is converted to a "
                "Professional (Creator or Business) account and linked to a Facebook Page."
            )

        return "Check your Meta Developer App settings, access token validity, and parameter values."

    def exchange_for_long_lived_token(
        self,
        short_lived_token: Optional[str] = None,
        app_id: Optional[str] = None,
        app_secret: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Exchanges a short-lived (2-hour) user access token for a long-lived (60-day) token.
        Queries: GET /oauth/access_token?grant_type=fb_exchange_token...
        """
        token = short_lived_token or self.access_token
        aid = app_id or self.app_id
        asec = app_secret or self.app_secret

        if not token:
            raise MetaGraphAPIError(
                "A short-lived user access token is required to perform token exchange."
            )
        if not aid or not asec:
            raise MetaGraphAPIError(
                "Both Meta App ID and App Secret are required to exchange for a long-lived token."
            )

        params = {
            "grant_type": "fb_exchange_token",
            "client_id": aid,
            "client_secret": asec,
            "fb_exchange_token": token,
        }

        result = self._request("oauth/access_token", params=params)
        if "access_token" in result:
            self.access_token = result["access_token"]
        return result

    def get_connected_instagram_accounts(
        self, access_token: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries GET /me/accounts to discover linked Instagram Creator/Business accounts
        from Facebook Pages managed by the token holder.
        """
        token = access_token or self.access_token
        if not token:
            raise MetaGraphAPIError(
                "An access token is required to query connected Instagram accounts."
            )

        params = {
            "fields": "id,name,instagram_business_account{id,username,name,profile_picture_url}"
        }
        response = self._request("me/accounts", params=params, access_token=token)
        pages = response.get("data", [])

        connected: List[Dict[str, Any]] = []
        for page in pages:
            ig_data = page.get("instagram_business_account")
            if ig_data and isinstance(ig_data, dict) and "id" in ig_data:
                connected.append({
                    "instagram_account_id": str(ig_data["id"]),
                    "username": ig_data.get("username", ""),
                    "name": ig_data.get("name") or page.get("name", ""),
                    "profile_picture_url": ig_data.get("profile_picture_url"),
                    "page_id": str(page.get("id", "")),
                    "page_name": page.get("name", ""),
                })

        return connected

    def fetch_full_creator_state(
        self,
        instagram_account_id: str,
        access_token: Optional[str] = None,
        media_limit: int = 6,
    ) -> Tuple[ProfileInput, Demographics, List[PostInput]]:
        """
        Fetches the complete creator state: validated profile, audience demographics, and recent media.
        Synchronizes demographic country and categorizations cleanly across objects.
        """
        token = access_token or self.access_token
        profile = self.fetch_profile_data(instagram_account_id, access_token=token)
        demographics = self.fetch_audience_demographics(instagram_account_id, access_token=token)
        media = self.fetch_recent_media(instagram_account_id, limit=media_limit, access_token=token)

        # Synchronize country from demographics if available
        if demographics and demographics.top_country and demographics.top_country != "US":
            profile.country = demographics.top_country
            profile.top_country = demographics.top_country

        return profile, demographics, media

    def fetch_profile_data(
        self,
        instagram_account_id: str,
        access_token: Optional[str] = None,
    ) -> ProfileInput:
        """
        Queries GET /{id}?fields=id,username,name,biography,followers_count,follows_count,media_count,profile_picture_url,website,category
        and constructs a validated ProfileInput instance with transparent estimation flags.
        """
        if not instagram_account_id or not str(instagram_account_id).strip():
            raise MetaGraphAPIError("An Instagram Account ID is required to fetch profile data.")

        account_id = str(instagram_account_id).strip()
        fields = "id,username,name,biography,followers_count,follows_count,media_count,profile_picture_url,website,category"
        data = self._request(account_id, params={"fields": fields}, access_token=access_token)

        raw_following = int(data.get("follows_count", 0))
        # Platform ceiling guardrail: Instagram platform enforces a maximum of 7,500 following
        sanitized_following = min(max(0, raw_following), 7500)
        followers = max(0, int(data.get("followers_count", 0)))
        media_count = max(0, int(data.get("media_count", 0)))

        biography = data.get("biography") or ""
        website = data.get("website") or ""
        has_bio_link = bool(website) or ("http://" in biography.lower()) or ("https://" in biography.lower())

        username = data.get("username") or f"ig_{account_id}"

        # Resolve category from Meta API or map to best ContentCategory
        meta_category = data.get("category") or ""
        cat_lower = meta_category.lower()
        if any(w in cat_lower for w in ["fitness", "gym", "health", "workout"]):
            resolved_category = ContentCategory.HEALTH_FITNESS
        elif any(w in cat_lower for w in ["sport", "athlete", "team", "football", "soccer", "basketball"]):
            resolved_category = ContentCategory.SPORTS
        elif any(w in cat_lower for w in ["tech", "software", "science", "engineer", "computer", "ai"]):
            resolved_category = ContentCategory.SCIENCE_TECHNOLOGY
        elif any(w in cat_lower for w in ["beauty", "fashion", "model", "cosmetic", "clothing", "apparel"]):
            resolved_category = ContentCategory.FASHION_BEAUTY
        elif any(w in cat_lower for w in ["food", "restaurant", "chef", "dining", "bakery", "cooking"]):
            resolved_category = ContentCategory.FOOD_DINING
        elif any(w in cat_lower for w in ["travel", "hotel", "destination", "tour", "flight"]):
            resolved_category = ContentCategory.TRAVEL_EVENTS
        elif any(w in cat_lower for w in ["business", "finance", "invest", "entrepreneur", "company", "consult"]):
            resolved_category = ContentCategory.FINANCE_BUSINESS
        elif any(w in cat_lower for w in ["education", "teacher", "school", "career", "university", "tutor"]):
            resolved_category = ContentCategory.EDUCATION_CAREERS
        else:
            resolved_category = ContentCategory.MUSIC_ENTERTAINMENT

        profile = ProfileInput(
            username=username,
            full_name=data.get("name") or username,
            country="US",
            total_followers=followers,
            total_following=sanitized_following,
            total_media_posts=media_count,
            account_age_years=4.0,
            posting_frequency_per_week=3.5,
            follower_growth_rate_30d=0.02,
            account_bio_has_link=has_bio_link,
            is_verified=False,
            account_category=resolved_category,
        )

        # Attach raw attributes and transparent estimation flags
        setattr(profile, "profile_picture_url", data.get("profile_picture_url"))
        setattr(profile, "biography", biography)
        setattr(profile, "raw_following", raw_following)
        setattr(profile, "meta_category_raw", meta_category)
        setattr(profile, "prior_metrics_estimated", True)
        return profile

    def fetch_recent_media(
        self,
        instagram_account_id: str,
        limit: int = 25,
        access_token: Optional[str] = None,
    ) -> List[PostInput]:
        """
        Queries GET /{id}/media with insights metrics to construct validated PostInput objects.
        Extracts reach, impressions, likes, comments, saved, shares, and video views.
        """
        if not instagram_account_id or not str(instagram_account_id).strip():
            raise MetaGraphAPIError("An Instagram Account ID is required to fetch media.")

        account_id = str(instagram_account_id).strip()
        clamped_limit = min(max(1, limit), 100)

        # Nested field query for media fields & post insights (supporting modern v21/v22 metrics)
        combined_fields = (
            "id,caption,media_type,media_product_type,timestamp,like_count,comments_count,permalink,"
            "insights.metric(reach,impressions,views,saved,shares,video_views)"
        )

        try:
            response = self._request(
                f"{account_id}/media",
                params={"fields": combined_fields, "limit": clamped_limit},
                access_token=access_token,
            )
        except MetaGraphAPIError as me:
            # Defensive fallback: If insights.metric is rejected (e.g. permission or older post restriction),
            # fall back to basic media fields
            logger.warning("Combined media insights request failed (%s). Falling back to basic media fields.", me)
            basic_fields = "id,caption,media_type,media_product_type,timestamp,like_count,comments_count,permalink"
            response = self._request(
                f"{account_id}/media",
                params={"fields": basic_fields, "limit": clamped_limit},
                access_token=access_token,
            )

        items = response.get("data", [])
        posts: List[PostInput] = []

        for item in items:
            caption = item.get("caption") or ""
            raw_media_type = str(item.get("media_type") or "").upper()
            raw_product = str(item.get("media_product_type") or "").upper()

            # Map to MediaType enum
            if "REEL" in raw_product or "VIDEO" in raw_media_type:
                media_type = MediaType.REEL
                video_sec = 30.0
                slide_count = 1
            elif "CAROUSEL" in raw_media_type or "ALBUM" in raw_media_type:
                media_type = MediaType.CAROUSEL
                video_sec = 0.0
                slide_count = 3
            else:
                media_type = MediaType.STATIC_IMAGE
                video_sec = 0.0
                slide_count = 1

            # Parse timestamp for Day of Week and Hour of Day
            timestamp_str = item.get("timestamp")
            day_of_week = "Wednesday"
            hour_of_day = 18
            if timestamp_str:
                try:
                    dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    day_of_week = dt.strftime("%A")
                    hour_of_day = dt.hour
                except Exception:
                    pass

            # Extract metrics from insights
            insights_map: Dict[str, int] = {}
            for metric_entry in item.get("insights", {}).get("data", []):
                name = metric_entry.get("name")
                val = None
                if "values" in metric_entry and metric_entry["values"]:
                    val = metric_entry["values"][0].get("value")
                elif "total_value" in metric_entry and isinstance(metric_entry["total_value"], dict):
                    val = metric_entry["total_value"].get("value")
                elif "value" in metric_entry:
                    val = metric_entry.get("value")

                if name and val is not None:
                    try:
                        insights_map[name] = int(val)
                    except (ValueError, TypeError):
                        pass

            likes = max(0, int(item.get("like_count", 0)))
            comments = max(0, int(item.get("comments_count", 0)))
            shares = max(0, insights_map.get("shares", 0))
            saves = max(0, insights_map.get("saved", 0))
            reach = insights_map.get("reach")
            views_val = insights_map.get("views")
            impressions = insights_map.get("impressions") or views_val
            video_views = insights_map.get("video_views") or views_val

            metrics = PostMetrics(
                likes=likes,
                comments=comments,
                shares=shares,
                saves=saves,
                reach=reach,
                impressions=impressions,
                video_views=video_views,
            )

            # Caption analysis
            hashtags = re.findall(r"#\w+", caption)
            mentions = re.findall(r"@\w+", caption)
            cta_pattern = r"\b(comment|share|save|tap|link in bio|follow|dm|tag)\b"
            has_cta = bool(re.search(cta_pattern, caption, re.IGNORECASE))

            # Infer category and style from caption keywords instead of hardcoding
            c_low = caption.lower()
            if any(w in c_low for w in ["fitness", "workout", "gym", "health", "diet"]):
                post_cat = ContentCategory.HEALTH_FITNESS
            elif any(w in c_low for w in ["tech", "software", "ai", "coding", "crypto"]):
                post_cat = ContentCategory.SCIENCE_TECHNOLOGY
            elif any(w in c_low for w in ["travel", "vacation", "trip", "explore"]):
                post_cat = ContentCategory.TRAVEL_EVENTS
            elif any(w in c_low for w in ["fashion", "style", "outfit", "beauty"]):
                post_cat = ContentCategory.FASHION_BEAUTY
            elif any(w in c_low for w in ["food", "recipe", "cook", "dining"]):
                post_cat = ContentCategory.FOOD_DINING
            elif any(w in c_low for w in ["business", "finance", "money", "career"]):
                post_cat = ContentCategory.FINANCE_BUSINESS
            elif any(w in c_low for w in ["sport", "football", "soccer", "match"]):
                post_cat = ContentCategory.SPORTS
            else:
                post_cat = ContentCategory.MUSIC_ENTERTAINMENT

            if any(w in c_low for w in ["how to", "tips", "tutorial", "guide", "lesson"]):
                post_style = ContentStyle.EDUCATIONAL
            elif any(w in c_low for w in ["promo", "discount", "sale", "shop", "link in bio"]):
                post_style = ContentStyle.PROMOTIONAL
            elif any(w in c_low for w in ["story", "journey", "lesson", "mindset"]):
                post_style = ContentStyle.INSPIRATIONAL
            elif any(w in c_low for w in ["behind the scenes", "bts", "process"]):
                post_style = ContentStyle.BEHIND_THE_SCENES
            else:
                post_style = ContentStyle.ENTERTAINING

            post = PostInput(
                media_type=media_type,
                category=post_cat,
                categorizations=[post_style],
                categorization=post_style,
                caption_length_chars=min(len(caption), 2200),
                hashtags_count=min(len(hashtags), 30),
                mentions_count=min(len(mentions), 20),
                has_call_to_action=has_cta,
                video_duration_seconds=video_sec,
                carousel_slide_count=slide_count,
                posted_day_of_week=day_of_week,
                posted_hour_of_day=hour_of_day,
                demographics=Demographics(),
                metrics=metrics,
            )

            # Attach metadata for UI display
            setattr(post, "id", item.get("id"))
            setattr(post, "permalink", item.get("permalink"))
            setattr(post, "caption", caption)
            setattr(post, "timestamp", timestamp_str)
            posts.append(post)

        return posts

    def fetch_audience_demographics(
        self,
        instagram_account_id: str,
        access_token: Optional[str] = None,
    ) -> Demographics:
        """
        Queries GET /{id}/insights?metric=audience_country,audience_gender_age&period=lifetime
        to extract top country, secondary country, and gender male/female split into Demographics.
        """
        if not instagram_account_id or not str(instagram_account_id).strip():
            raise MetaGraphAPIError("An Instagram Account ID is required to fetch demographics.")

        account_id = str(instagram_account_id).strip()
        params = {
            "metric": "audience_country,audience_gender_age",
            "period": "lifetime",
        }

        try:
            response = self._request(f"{account_id}/insights", params=params, access_token=access_token)
            insights = response.get("data", [])
        except MetaGraphAPIError as me:
            logger.warning("Audience demographics unavailable (%s). Returning safe baseline defaults.", me)
            return Demographics()

        country_data: Dict[str, int] = {}
        gender_age_data: Dict[str, int] = {}

        for entry in insights:
            name = entry.get("name")
            values = entry.get("values", [])
            val = values[0].get("value", {}) if values else {}
            if name == "audience_country" and isinstance(val, dict):
                country_data = val
            elif name == "audience_gender_age" and isinstance(val, dict):
                gender_age_data = val

        # Country distribution
        top_country = "US"
        secondary_country = "IN"
        if country_data:
            sorted_countries = sorted(country_data.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_countries) >= 1:
                top_country = sorted_countries[0][0].strip().upper()
            if len(sorted_countries) >= 2:
                secondary_country = sorted_countries[1][0].strip().upper()
            else:
                secondary_country = "IN" if top_country != "IN" else "US"

        # Gender & age breakdown
        female_sum = 0
        male_sum = 0
        age_bucket_counts: Dict[str, int] = {}

        for key, count in gender_age_data.items():
            try:
                cnt = int(count)
            except (ValueError, TypeError):
                continue

            parts = key.split(".", 1)
            gender_code = parts[0].upper()
            age_bucket = parts[1] if len(parts) > 1 else "25-34"

            if gender_code == "F":
                female_sum += cnt
            elif gender_code == "M":
                male_sum += cnt

            age_bucket_counts[age_bucket] = age_bucket_counts.get(age_bucket, 0) + cnt

        total_gender = female_sum + male_sum
        if total_gender > 0:
            female_pct = round(female_sum / total_gender, 4)
            male_pct = round(1.0 - female_pct, 4)
        else:
            female_pct = 0.50
            male_pct = 0.50

        # Dominant age bucket
        primary_age_group = "25-34"
        if age_bucket_counts:
            sorted_ages = sorted(age_bucket_counts.items(), key=lambda x: x[1], reverse=True)
            primary_age_group = sorted_ages[0][0]

        return Demographics(
            top_country=top_country,
            secondary_country=secondary_country,
            primary_age_group=primary_age_group,
            gender_female_pct=female_pct,
            gender_male_pct=male_pct,
            audience_activity_score=0.75,
        )
