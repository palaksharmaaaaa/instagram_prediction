import re
from typing import Dict, Any, Tuple
from ..schemas.query import QueryIntent

VIRALITY_CUES = [
    r"\bviral\b", r"\bblow\s*up\b", r"\bexplosive\b", r"\bexplore\s*page\b",
    r"\btrending\b", r"\bhigh\s*shares?\b", r"\bvirality\b", r"\bshareable\b",
    r"\bmaximum\s*reach\b", r"\balgorithm\s*boost\b"
]

COMMUNITY_CUES = [
    r"\bloyal\b", r"\bactive\s*community\b", r"\bhigh\s*engagement\b",
    r"\bcomments?\b", r"\bconversations?\b", r"\bengaged\s*followers\b",
    r"\bdiscussion\b", r"\bcommunity\b"
]

EDUCATIONAL_CUES = [
    r"\beducational\b", r"\bhow\s*to\b", r"\btips\b", r"\bguide\b",
    r"\bcheatsheet\b", r"\bframework\b", r"\bcarousel\b", r"\bbookmark\b",
    r"\bsaves?\b", r"\blearn\b", r"\bknowledge\b", r"\btutorial\b"
]

COMMERCIAL_CUES = [
    r"\bsponsored\b", r"\bbrand\s*deal\b", r"\broi\b", r"\bpaid\s*partnership\b",
    r"\bproduct\s*placement\b", r"\bcommercial\b", r"\binfluencer\s*marketing\b",
    r"\badvertis\w*\b", r"\bmonetiz\w*\b"
]

TIER_CUES = [
    (r"\bnano\b", "Nano (<10k)", 1_000, 10_000),
    (r"\bmicro\b", "Micro (10k-100k)", 10_000, 100_000),
    (r"\bmid\s*tier\b|\bmid-tier\b", "Mid-Tier (100k-500k)", 100_000, 500_000),
    (r"\bmacro\b", "Macro (500k-1M)", 500_000, 1_000_000),
    (r"\bmega\b|\bcelebrity\b|\bcelebrities\b|\ba-list\b", "Mega / Celebrity (>1M)", 1_000_000, 1_000_000_000)
]

DEMO_CUES = [
    (r"\bgen\s*z\b|\bteen\w*\b|\byouth\b", "Gen Z (18-24)", "18-24", None),
    (r"\bmillennial\w*\b|\bprofessionals?\b", "Millennials (25-34)", "25-34", None),
    (r"\bfemale\b|\bwomen\b|\bgirls?\b", "Female Dominant (60%+)", None, 0.60),
    (r"\bmale\b|\bmen\b|\bguys?\b", "Male Dominant (60%+)", None, 0.40)
]


def analyze_query_intent(prompt: str) -> Tuple[QueryIntent, Dict[str, Any]]:
    """
    Analyzes the 'feel', strategic goals, creator tiers, and demographic needs of the prompt.
    Returns structured QueryIntent and implicit filter augmentations.
    """
    text_lower = prompt.lower()
    implicit_filters: Dict[str, Any] = {}

    # 1. Primary Goal Detection
    primary_goal = "Exploratory / Discovery"
    if any(re.search(pat, text_lower) for pat in VIRALITY_CUES):
        primary_goal = "Virality & Explore Reach"
        implicit_filters["_top_n"] = {"limit": 10, "sort_by": "virality_score", "ascending": False}
    elif any(re.search(pat, text_lower) for pat in COMMUNITY_CUES):
        primary_goal = "Community & High Engagement"
        implicit_filters["_top_n"] = {"limit": 10, "sort_by": "engagement_rate", "ascending": False}
    elif any(re.search(pat, text_lower) for pat in EDUCATIONAL_CUES):
        primary_goal = "Saveable / Educational Value"
        implicit_filters["categorization"] = "Educational / How-To"
    elif any(re.search(pat, text_lower) for pat in COMMERCIAL_CUES):
        primary_goal = "Commercial / Brand Sponsorship"
        implicit_filters["categorization"] = "Promotional / Sponsored"

    # 2. Creator Scale / Tier
    creator_tier = "Any"
    for pat, label, min_f, max_f in TIER_CUES:
        if re.search(pat, text_lower):
            creator_tier = label
            implicit_filters["total_followers"] = {"operator": "between", "min": float(min_f), "max": float(max_f)}
            break

    # 3. Audience Demographic Focus
    audience_focus = "Broad Audience"
    for pat, label, age_grp, female_pct in DEMO_CUES:
        if re.search(pat, text_lower):
            audience_focus = label
            if age_grp:
                implicit_filters["primary_age_group"] = age_grp
            if female_pct is not None:
                op = ">=" if female_pct >= 0.5 else "<="
                implicit_filters["gender_female_pct"] = {"operator": op, "value": female_pct}
            break

    # 4. Tone
    tone = "Informational"
    if re.search(r"\bpredict\b|\bforecast\b|\bestimate\b|\bcalculate\b|\brate\b", text_lower):
        tone = "Analytical"
    elif re.search(r"\bsponsor\w*\b|\bcampaign\b|\bbudget\b|\broi\b|\bbrand\b", text_lower):
        tone = "Commercial Strategy"
    elif re.search(r"\bshow\b|\bgive\b|\bwho\b|\bwhat\b|\bfind\b", text_lower):
        tone = "Exploratory"

    intent = QueryIntent(
        primary_goal=primary_goal,
        creator_tier=creator_tier,
        audience_focus=audience_focus,
        tone=tone
    )

    return intent, implicit_filters
