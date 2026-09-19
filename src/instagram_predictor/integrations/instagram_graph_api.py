import re
import json
import logging
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from datetime import timezone
from typing import Optional, List, Dict, Any, Tuple

from instagram_predictor.schemas.profile import ProfileInput, PlatformType

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
    Client for the Meta Instagram Graph API.

    It returns ONLY values Meta reports. A field Meta does not return is left None / absent; it is never
    defaulted, guessed from keywords, or estimated. Post rows are shaped like data/posts.csv so real
    insights can be appended to the training data.
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

    def fetch_profile_data(
        self,
        instagram_account_id: str,
        access_token: Optional[str] = None,
    ) -> ProfileInput:
        """GET /{id}?fields=... -> ProfileInput containing only what Meta returned."""
        if not instagram_account_id or not str(instagram_account_id).strip():
            raise MetaGraphAPIError("An Instagram Account ID is required to fetch profile data.")
        account_id = str(instagram_account_id).strip()
        fields = "id,username,name,biography,followers_count,follows_count,media_count,profile_picture_url,website"
        data = self._request(account_id, params={"fields": fields}, access_token=access_token)
        if "followers_count" not in data:
            raise MetaGraphAPIError("Meta did not return followers_count for this account; cannot build a profile.")

        biography = data.get("biography")
        website = data.get("website")
        has_link = None
        if website is not None or biography is not None:
            has_link = bool(website) or any(t in (biography or "").lower() for t in ("http://", "https://"))

        return ProfileInput(
            username=data.get("username") or f"ig_{account_id}",
            platform=PlatformType.INSTAGRAM,
            total_followers=int(data["followers_count"]),
            total_following=int(data["follows_count"]) if data.get("follows_count") is not None else None,
            total_media_posts=int(data["media_count"]) if data.get("media_count") is not None else None,
            full_name=data.get("name"),
            biography=biography,
            has_bio_link=has_link,
            profile_picture_url=data.get("profile_picture_url"),
        )

    def _fetch_media_insights(self, media_id: str, token: Optional[str]) -> Dict[str, int]:
        """
        Insights for one media item. Meta rejects the whole call if any metric is unsupported for that media type,
        so we retry with narrower metric sets. Returns only metrics Meta actually returned (possibly empty).
        """
        for metrics in ("reach,saved,shares,views,total_interactions", "reach,saved,shares", "reach"):
            try:
                resp = self._request(f"{media_id}/insights", params={"metric": metrics}, access_token=token)
            except MetaGraphAPIError:
                continue
            out: Dict[str, int] = {}
            for entry in resp.get("data", []):
                name = entry.get("name")
                val = None
                if entry.get("values"):
                    val = entry["values"][0].get("value")
                elif isinstance(entry.get("total_value"), dict):
                    val = entry["total_value"].get("value")
                if name and isinstance(val, (int, float)):
                    out[name] = int(val)
            return out
        return {}

    @staticmethod
    def _map_media_type(media_type: Optional[str], product_type: Optional[str]) -> Optional[str]:
        mt, pt = (media_type or "").upper(), (product_type or "").upper()
        if pt in ("REELS", "REEL"):
            return "Reel"
        if mt == "CAROUSEL_ALBUM":
            return "Carousel"
        if mt == "IMAGE":
            return "Static Image"
        if mt == "VIDEO":
            return "Video"
        return None   # unknown type: the post is skipped rather than mislabelled

    def fetch_media_records(
        self,
        instagram_account_id: str,
        limit: int = 25,
        access_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recent media as rows in the data/posts.csv layout. Only observed values are included:
          * per_media_reach / saves / shares / views / total_interactions exist only if Meta returned them
          * per_media_impressions is NOT available in current API versions and is left absent
          * posted_day_of_week / posted_hour_of_day are derived from Meta's timestamp and are in UTC
          * has_call_to_action and video_duration_seconds are not reported by the API and are left absent
        """
        if not instagram_account_id or not str(instagram_account_id).strip():
            raise MetaGraphAPIError("An Instagram Account ID is required to fetch media.")
        account_id = str(instagram_account_id).strip()
        token = access_token or self.access_token
        response = self._request(
            f"{account_id}/media",
            params={"fields": "id,caption,media_type,media_product_type,timestamp,like_count,comments_count,permalink,children{id}",
                    "limit": min(max(1, limit), 100)},
            access_token=token,
        )
        rows: List[Dict[str, Any]] = []
        for item in response.get("data", []):
            mt = self._map_media_type(item.get("media_type"), item.get("media_product_type"))
            ts = item.get("timestamp")
            if mt is None or not ts:
                continue
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
            row: Dict[str, Any] = {
                "post_id": item.get("id"),
                "media_type": mt,
                "posted_day_of_week": dt.strftime("%A"),
                "posted_hour_of_day": dt.hour,
                "timestamp_utc": ts,
                "permalink": item.get("permalink"),
            }
            caption = item.get("caption")
            if caption is not None:
                row["caption_length_chars"] = len(caption)
                row["hashtags_count"] = len(re.findall(r"#\w+", caption))
                row["mentions_count"] = len(re.findall(r"@\w+", caption))
            children = (item.get("children") or {}).get("data")
            if mt == "Carousel" and children:
                row["carousel_slide_count"] = len(children)
            for src, dst in (("like_count", "per_media_likes"), ("comments_count", "per_media_comments")):
                if item.get(src) is not None:
                    row[dst] = int(item[src])
            ins = self._fetch_media_insights(item.get("id"), token)
            for src, dst in (("reach", "per_media_reach"), ("saved", "per_media_saves"), ("shares", "per_media_shares"),
                             ("views", "per_media_views"), ("total_interactions", "per_media_total_interactions")):
                if src in ins:
                    row[dst] = ins[src]
            rows.append(row)
        return rows

    def fetch_creator_snapshot(
        self,
        instagram_account_id: str,
        media_limit: int = 25,
        access_token: Optional[str] = None,
    ) -> Tuple[ProfileInput, List[Dict[str, Any]]]:
        """
        Profile plus post rows. Each row carries the CURRENT follower count (`total_followers`) and the fetch time,
        because Meta does not report the follower count at post time. Rows without observed reach are kept
        but cannot be used for training (the loader rejects them).
        """
        profile = self.fetch_profile_data(instagram_account_id, access_token=access_token)
        rows = self.fetch_media_records(instagram_account_id, limit=media_limit, access_token=access_token)
        now = datetime.now(timezone.utc).isoformat()
        for r in rows:
            r["username"] = profile.username
            r["total_followers"] = profile.total_followers
            r["followers_snapshot_utc"] = now
        return profile, rows
