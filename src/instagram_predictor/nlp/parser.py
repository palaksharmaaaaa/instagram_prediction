import html
import re
from typing import Any, Dict, List, Optional
from ..schemas import ParsedQuery, ContentCategory, MediaType, ContentStyle
from ..guardrails.safety import sanitize_prompt
from .intent_analyzer import analyze_query_intent
from .auditor import audit_query

COUNTRY_LOOKUP = {
    "united states": "US", "usa": "US", "us": "US", "america": "US",
    "spain": "ES", "es": "ES",
    "india": "IN", "in": "IN",
    "brazil": "BR", "brasil": "BR", "br": "BR",
    "united kingdom": "GB", "uk": "GB", "great britain": "GB", "england": "GB", "gb": "GB",
    "canada": "CA", "ca": "CA",
    "france": "FR", "fr": "FR",
    "australia": "AU", "au": "AU",
    "germany": "DE", "de": "DE",
    "italy": "IT", "it": "IT",
    "mexico": "MX", "mx": "MX",
    "united arab emirates": "AE", "uae": "AE", "dubai": "AE", "ae": "AE"
}

CATEGORY_SYNONYMS = {
    "sports": ContentCategory.SPORTS.value,
    "football": ContentCategory.SPORTS.value,
    "soccer": ContentCategory.SPORTS.value,
    "cricket": ContentCategory.SPORTS.value,
    "fitness": ContentCategory.HEALTH_FITNESS.value,
    "health": ContentCategory.HEALTH_FITNESS.value,
    "workout": ContentCategory.HEALTH_FITNESS.value,
    "finance": ContentCategory.FINANCE_BUSINESS.value,
    "business": ContentCategory.FINANCE_BUSINESS.value,
    "crypto": ContentCategory.FINANCE_BUSINESS.value,
    "money": ContentCategory.FINANCE_BUSINESS.value,
    "fashion": ContentCategory.FASHION_BEAUTY.value,
    "beauty": ContentCategory.FASHION_BEAUTY.value,
    "makeup": ContentCategory.FASHION_BEAUTY.value,
    "tech": ContentCategory.SCIENCE_TECHNOLOGY.value,
    "technology": ContentCategory.SCIENCE_TECHNOLOGY.value,
    "science": ContentCategory.SCIENCE_TECHNOLOGY.value,
    "gadgets": ContentCategory.SCIENCE_TECHNOLOGY.value,
    "travel": ContentCategory.TRAVEL_EVENTS.value,
    "events": ContentCategory.TRAVEL_EVENTS.value,
    "tourism": ContentCategory.TRAVEL_EVENTS.value,
    "food": ContentCategory.FOOD_DINING.value,
    "dining": ContentCategory.FOOD_DINING.value,
    "cooking": ContentCategory.FOOD_DINING.value,
    "recipe": ContentCategory.FOOD_DINING.value,
    "music": ContentCategory.MUSIC_ENTERTAINMENT.value,
    "entertainment": ContentCategory.MUSIC_ENTERTAINMENT.value,
    "comedy": ContentCategory.MUSIC_ENTERTAINMENT.value,
    "education": ContentCategory.EDUCATION_CAREERS.value,
    "career": ContentCategory.EDUCATION_CAREERS.value,
    "learning": ContentCategory.EDUCATION_CAREERS.value
}

MEDIA_TYPE_SYNONYMS = {
    "reels": MediaType.REEL.value,
    "reel": MediaType.REEL.value,
    "carousels": MediaType.CAROUSEL.value,
    "carousel": MediaType.CAROUSEL.value,
    "static images": MediaType.STATIC_IMAGE.value,
    "static image": MediaType.STATIC_IMAGE.value,
    "photos": MediaType.STATIC_IMAGE.value,
    "photo": MediaType.STATIC_IMAGE.value,
    "stories": MediaType.STORY.value,
    "story": MediaType.STORY.value
}


def _parse_multiplier(unit: Optional[str]) -> float:
    if not unit:
        return 1.0
    u = unit.lower().strip()
    if u in ["k", "thousand"]:
        return 1_000.0
    if u in ["m", "mil", "million"]:
        return 1_000_000.0
    if u in ["b", "bil", "billion"]:
        return 1_000_000_000.0
    return 1.0


def _parse_op(word: str) -> str:
    w = html.unescape(word.lower().strip())
    if w in ["above", "over", "more than", "greater than", "higher than", ">"]:
        return ">"
    if w in ["at least", "min", "minimum", ">="]:
        return ">="
    if w in ["below", "under", "less than", "fewer than", "lower than", "<"]:
        return "<"
    if w in ["at most", "max", "maximum", "up to", "<="]:
        return "<="
    return ">="


def _parse_num(num_str: str) -> float:
    return float(num_str.replace(",", "").strip())


_OP_PATTERN = r"(above|over|more than|greater than|higher than|below|under|less than|fewer than|lower than|at least|at most|min|max|up to|>=|<=|>|<|&gt;=|&lt;=|&gt;|&lt;)"
_NUM_PATTERN = r"((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
_FOLLOWER_UNITS = r"(k|m|mil|million|thousand|b|bil|billion)?"


def parse_query(prompt: str) -> ParsedQuery:
    """
    Parses a user query with localized clause evaluation, guardrail checks,
    and returns a validated ParsedQuery schema.
    """
    sanitized, safety_flags = sanitize_prompt(prompt)
    text_lower = sanitized.lower()
    filters: Dict[str, Any] = {}

    # ---------------------------------------------------------
    # 1. FOLLOWERS FILTER
    # ---------------------------------------------------------
    # Pattern A: between X and Y followers OR followers between X and Y
    between_followers = re.search(
        r"(?:(?:between|from)\s*"
        rf"{_NUM_PATTERN}\s*{_FOLLOWER_UNITS}\s*(?:and|to)\s*{_NUM_PATTERN}\s*{_FOLLOWER_UNITS}\s*followers?|"
        r"followers?\s*(?:between|from)\s*"
        rf"{_NUM_PATTERN}\s*{_FOLLOWER_UNITS}\s*(?:and|to)\s*{_NUM_PATTERN}\s*{_FOLLOWER_UNITS})",
        text_lower
    )
    if between_followers:
        if between_followers.group(1):
            n1, u1, n2, u2 = between_followers.group(1), between_followers.group(2), between_followers.group(3), between_followers.group(4)
        else:
            n1, u1, n2, u2 = between_followers.group(5), between_followers.group(6), between_followers.group(7), between_followers.group(8)
        v1 = _parse_num(n1) * _parse_multiplier(u1 or u2)
        v2 = _parse_num(n2) * _parse_multiplier(u2)
        filters["total_followers"] = {"operator": "between", "min": min(v1, v2), "max": max(v1, v2)}
    else:
        # Pattern B: (op) (num) (unit) followers
        f_match = re.search(
            rf"{_OP_PATTERN}\s*"
            rf"{_NUM_PATTERN}\s*{_FOLLOWER_UNITS}\s*followers?",
            text_lower
        )
        # Pattern C: followers (op) (num) (unit)
        f_inv = re.search(
            rf"followers?\s*{_OP_PATTERN}\s*"
            rf"{_NUM_PATTERN}\s*{_FOLLOWER_UNITS}",
            text_lower
        )
        # Pattern D: (num)(unit)+ followers
        f_plus = re.search(
            rf"{_NUM_PATTERN}\s*(k|m|mil|million|thousand|b|bil|billion)\s*(\+)?\s*followers?",
            text_lower
        )

        if f_match:
            val = _parse_num(f_match.group(2)) * _parse_multiplier(f_match.group(3))
            filters["total_followers"] = {"operator": _parse_op(f_match.group(1)), "value": val}
        elif f_inv:
            val = _parse_num(f_inv.group(2)) * _parse_multiplier(f_inv.group(3))
            filters["total_followers"] = {"operator": _parse_op(f_inv.group(1)), "value": val}
        elif f_plus:
            val = _parse_num(f_plus.group(1)) * _parse_multiplier(f_plus.group(2))
            filters["total_followers"] = {"operator": ">=", "value": val}

    # ---------------------------------------------------------
    # 2. ENGAGEMENT RATE FILTER
    # ---------------------------------------------------------
    eng_between = re.search(
        r"(?:engagement|engagement rate|\ber\b)\s*(?:between|from)\s*(\d+(?:\.\d+)?)\s*%?\s*(?:and|to)\s*(\d+(?:\.\d+)?)\s*%?",
        text_lower
    )
    if eng_between:
        v1 = float(eng_between.group(1))
        v2 = float(eng_between.group(2))
        if v1 > 1.0 or v2 > 1.0:
            v1 /= 100.0
            v2 /= 100.0
        filters["engagement_rate"] = {"operator": "between", "min": min(v1, v2), "max": max(v1, v2)}
    else:
        eng_match = re.search(
            rf"(?:engagement|engagement rate|\ber\b)\s*{_OP_PATTERN}\s*(\d+(?:\.\d+)?)\s*%",
            text_lower
        )
        eng_inv = re.search(
            rf"{_OP_PATTERN}\s*(\d+(?:\.\d+)?)\s*%\s*(?:engagement|engagement rate|\ber\b)",
            text_lower
        )
        eng_dec = re.search(
            rf"(?:engagement|engagement rate|\ber\b)\s*{_OP_PATTERN}\s*(0\.\d+)",
            text_lower
        )

        if eng_match:
            filters["engagement_rate"] = {"operator": _parse_op(eng_match.group(1)), "value": float(eng_match.group(2)) / 100.0}
        elif eng_inv:
            filters["engagement_rate"] = {"operator": _parse_op(eng_inv.group(1)), "value": float(eng_inv.group(2)) / 100.0}
        elif eng_dec:
            filters["engagement_rate"] = {"operator": _parse_op(eng_dec.group(1)), "value": float(eng_dec.group(2))}

    # ---------------------------------------------------------
    # 3. PER-MEDIA REACH & METRICS NUMERIC FILTER
    # ---------------------------------------------------------
    reach_match = re.search(
        rf"\breach\s*{_OP_PATTERN}\s*{_NUM_PATTERN}\s*{_FOLLOWER_UNITS}",
        text_lower
    )
    reach_inv = re.search(
        rf"{_OP_PATTERN}\s*{_NUM_PATTERN}\s*{_FOLLOWER_UNITS}\s*reach\b",
        text_lower
    )
    if reach_match:
        val = _parse_num(reach_match.group(2)) * _parse_multiplier(reach_match.group(3))
        filters["per_media_reach"] = {"operator": _parse_op(reach_match.group(1)), "value": val}
    elif reach_inv:
        val = _parse_num(reach_inv.group(2)) * _parse_multiplier(reach_inv.group(3))
        filters["per_media_reach"] = {"operator": _parse_op(reach_inv.group(1)), "value": val}

    # ---------------------------------------------------------
    # 4. MEDIA TYPE
    # ---------------------------------------------------------
    for kw, m_type in MEDIA_TYPE_SYNONYMS.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            filters["media_type"] = m_type
            break

    # ---------------------------------------------------------
    # 5. CATEGORY
    # ---------------------------------------------------------
    for kw, cat_name in sorted(CATEGORY_SYNONYMS.items(), key=lambda x: -len(x[0])):
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            filters["category"] = cat_name
            break

    # ---------------------------------------------------------
    # 6. COUNTRY
    # ---------------------------------------------------------
    for place, code in sorted(COUNTRY_LOOKUP.items(), key=lambda x: -len(x[0])):
        pat = r"\b(?:in|from|country)\s+" + re.escape(place) + r"\b|\b" + re.escape(place) + r"\s+(?:accounts?|influencers?|creators?)\b"
        if re.search(pat, text_lower):
            filters["country"] = code
            break
        elif len(place) == 2 and re.search(r"\b(?:in|from)\s+" + re.escape(place.upper()) + r"\b", sanitized):
            filters["country"] = code
            break

    # ---------------------------------------------------------
    # 7. SPECIFIC HANDLE / USERNAME (NLP-01: Robust handle extraction)
    # ---------------------------------------------------------
    explicit_at_match = re.search(r"@([a-zA-Z0-9_\.]{3,30})\b", text_lower)

    prefix_pattern = (
        r"\b(?:"
        r"(?:account|handle|profile)\s+of|"
        r"(?:creator|user|influencer|profile|account)\s+named|"
        r"handle\s*[:\s]|"
        r"for\s+(?:account|handle|profile|creator|user|influencer)\s+(?:named\s+|of\s+)?"
        r")\s*@?([a-zA-Z0-9_\.]{3,30})\b"
    )
    prefix_match = re.search(prefix_pattern, text_lower)

    candidate: Optional[str] = None
    is_explicit = False

    if explicit_at_match:
        candidate = explicit_at_match.group(1)
        is_explicit = True
    elif prefix_match:
        candidate = prefix_match.group(1)
        is_explicit = False

    if candidate:
        candidate = candidate.rstrip(".!?,;:")
        reserved = [
            "accounts", "followers", "reach", "impressions", "reels", "carousels",
            "sports", "music", "fashion", "fitness", "finance", "spain", "india",
            "brazil", "america", "stories", "photos", "posts", "influencers",
            "creators"
        ]
        common_nouns = {
            "marketing", "summer", "new", "business", "tech", "technology", "product",
            "campaign", "campaigns", "festival", "festivals", "brand", "brands",
            "content", "post", "posts", "account", "accounts", "creator", "creators",
            "user", "users", "influencer", "influencers", "profile", "profiles",
            "video", "videos", "image", "images", "photo", "photos", "reel", "reels",
            "story", "stories", "carousel", "carousels", "followers", "reach",
            "impression", "impressions", "engagement", "virality", "analytics",
            "prediction", "forecast", "data", "stats", "statistics", "report"
        }

        if is_explicit:
            if candidate not in reserved and len(candidate) >= 3:
                filters["username"] = candidate
        else:
            if (
                candidate not in reserved
                and candidate not in common_nouns
                and candidate not in CATEGORY_SYNONYMS
                and candidate not in COUNTRY_LOOKUP
                and candidate not in MEDIA_TYPE_SYNONYMS
                and len(candidate) >= 3
            ):
                filters["username"] = candidate

    # ---------------------------------------------------------
    # 8. TOP-N & RANKINGS (Word-boundary safe)
    # ---------------------------------------------------------
    top_match = re.search(r"\btop\s*(\d+)\b", text_lower)
    most_followers = re.search(r"\b(?:most|highest|biggest)\s+(?:followers?|accounts?)\b", text_lower)
    most_engaging = re.search(r"\b(?:most|highest)\s+(?:engaging|engagement)\b", text_lower)
    most_viral = re.search(r"\b(?:most|highest)\s+(?:viral|virality)\b", text_lower)

    if top_match:
        lim = int(top_match.group(1))
        sort_by = "total_followers"
        if "engagement" in text_lower or re.search(r"\ber\b", text_lower):
            sort_by = "engagement_rate"
        elif "reach" in text_lower:
            sort_by = "per_media_reach"
        elif "virality" in text_lower or re.search(r"\bshares?\b", text_lower):
            sort_by = "virality_score"

        filters["_top_n"] = {"limit": lim, "sort_by": sort_by, "ascending": False}
    elif most_followers:
        filters["_top_n"] = {"limit": 10, "sort_by": "total_followers", "ascending": False}
    elif most_engaging:
        filters["_top_n"] = {"limit": 10, "sort_by": "engagement_rate", "ascending": False}
    elif most_viral:
        filters["_top_n"] = {"limit": 10, "sort_by": "virality_score", "ascending": False}

    # ---------------------------------------------------------
    # 9. PREDICTION INTENT
    # ---------------------------------------------------------
    predict_kw = ["predict", "prediction", "predicting", "estimate", "estimating", "forecast", "forecasting", "calculate"]
    has_pred = any(k in text_lower for k in predict_kw)

    predict_reach = ("reach" in text_lower and has_pred) or ("reach" in text_lower and "what is" in text_lower)
    predict_impressions = ("impression" in text_lower and has_pred) or ("impressions" in text_lower and has_pred)

    if has_pred and not predict_reach and not predict_impressions:
        predict_reach = True
        predict_impressions = True

    # ---------------------------------------------------------
    # 10. INTENT ANALYSIS & AUDITING
    # ---------------------------------------------------------
    intent, implicit_filters = analyze_query_intent(sanitized)

    # Augment filters with implicit filters if not already explicitly specified
    for k, v in implicit_filters.items():
        if k not in filters:
            filters[k] = v

    audit_report = audit_query(
        prompt=prompt,
        filters=filters,
        intent=intent,
        predict_reach=bool(predict_reach),
        predict_impressions=bool(predict_impressions)
    )

    return ParsedQuery(
        filters=filters,
        predict_reach=bool(predict_reach),
        predict_impressions=bool(predict_impressions),
        original_prompt=prompt,
        sanitized_prompt=sanitized,
        safety_flags=safety_flags,
        intent=intent,
        audit_report=audit_report
    )
