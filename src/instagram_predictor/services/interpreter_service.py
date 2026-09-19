"""
Conversational Result / Output Analyzer & Interpreter Service.

Translates raw ML predictions and TreeSHAP attributions into a conversational,
actionable dialogue tailored for creators, explaining projected performance,
top positive and negative algorithmic drivers, safe confidence windows,
and concrete pre-publishing recommendations.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from ..schemas import SimulationPrediction


def format_simulation_for_interpretation(
    prediction: Union[SimulationPrediction, Dict[str, Any]],
    profile_data: Optional[Dict[str, Any]] = None,
    post_data: Optional[Dict[str, Any]] = None,
    platform: Optional[str] = None,
    media_format: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Converts a SimulationPrediction object or raw simulation dictionary into the
    standardized dictionary format expected by interpret_simulation_result.
    """
    profile_data = profile_data or {}
    post_data = post_data or {}

    # Extract platform and format
    plat = (
        platform
        or post_data.get("platform")
        or profile_data.get("platform")
        or getattr(post_data, "platform", None)
        or "Instagram"
    )
    if hasattr(plat, "value"):
        plat = plat.value

    fmt = (
        media_format
        or post_data.get("media_type")
        or post_data.get("format")
        or "Reel"
    )
    if hasattr(fmt, "value"):
        fmt = fmt.value

    followers = (
        profile_data.get("total_followers")
        or profile_data.get("followers")
        or profile_data.get("follower_count")
    )

    if isinstance(prediction, SimulationPrediction):
        reach_point = prediction.projected_reach.point_estimate
        reach_lower_80 = prediction.projected_reach.lower
        reach_upper_80 = prediction.projected_reach.upper

        # Estimate 90% CI from 80% CI if only 80% is in object
        lower_delta = reach_point - reach_lower_80
        upper_delta = reach_upper_80 - reach_point
        reach_lower_90 = max(50, int(reach_point - lower_delta * 1.35))
        reach_upper_90 = max(reach_point, int(reach_point + upper_delta * 1.35))

        imp_point = prediction.projected_impressions.point_estimate
        imp_lower_80 = prediction.projected_impressions.lower
        imp_upper_80 = prediction.projected_impressions.upper

        imp_lower_delta = imp_point - imp_lower_80
        imp_upper_delta = imp_upper_80 - imp_point
        imp_lower_90 = max(reach_lower_90, int(imp_point - imp_lower_delta * 1.35))
        imp_upper_90 = max(imp_point, int(imp_point + imp_upper_delta * 1.35))

        return {
            "predicted_reach": reach_point,
            "predicted_impressions": imp_point,
            "reach_80_ci": (reach_lower_80, reach_upper_80),
            "reach_90_ci": (reach_lower_90, reach_upper_90),
            "impressions_80_ci": (imp_lower_80, imp_upper_80),
            "impressions_90_ci": (imp_lower_90, imp_upper_90),
            "feature_explanations": prediction.feature_explanations,
            "engagement_rate": prediction.projected_engagement_rate,
            "virality_tier": prediction.virality_tier,
            "platform": plat,
            "format": fmt,
            "total_followers": followers,
        }

    # If it's already a dict, normalize it
    return _normalize_simulation_dict(prediction, plat, fmt, followers)


def _normalize_simulation_dict(
    data: Dict[str, Any],
    fallback_platform: str = "Instagram",
    fallback_format: str = "Reel",
    fallback_followers: Optional[int] = None
) -> Dict[str, Any]:
    """Helper to extract and normalize simulation fields from heterogeneous dictionary inputs."""
    # Predicted reach
    reach = data.get("predicted_reach")
    if reach is None and "projected_reach" in data:
        proj_reach = data["projected_reach"]
        reach = (
            proj_reach.get("point_estimate")
            if isinstance(proj_reach, dict)
            else getattr(proj_reach, "point_estimate", 1000)
        )
    reach = int(reach or 1000)

    # Predicted impressions
    impressions = data.get("predicted_impressions")
    if impressions is None and "projected_impressions" in data:
        proj_imp = data["projected_impressions"]
        impressions = (
            proj_imp.get("point_estimate")
            if isinstance(proj_imp, dict)
            else getattr(proj_imp, "point_estimate", reach)
        )
    impressions = max(int(impressions or reach), reach)

    # Helper for CI extraction
    def extract_ci(val: Any, default_lower: int, default_upper: int) -> Tuple[int, int]:
        if isinstance(val, (tuple, list)) and len(val) >= 2:
            return (int(val[0]), int(val[1]))
        if isinstance(val, dict) and "lower" in val and "upper" in val:
            return (int(val["lower"]), int(val["upper"]))
        if hasattr(val, "lower") and hasattr(val, "upper"):
            return (int(val.lower), int(val.upper))
        return (default_lower, default_upper)

    default_r80_lower = max(50, int(reach * 0.75))
    default_r80_upper = max(reach, int(reach * 1.30))
    reach_80 = extract_ci(data.get("reach_80_ci") or data.get("projected_reach"), default_r80_lower, default_r80_upper)

    lower_delta = reach - reach_80[0]
    upper_delta = reach_80[1] - reach
    default_r90_lower = max(50, int(reach - lower_delta * 1.35))
    default_r90_upper = max(reach, int(reach + upper_delta * 1.35))
    reach_90 = extract_ci(data.get("reach_90_ci"), default_r90_lower, default_r90_upper)

    default_i80_lower = max(reach_80[0], int(impressions * 0.80))
    default_i80_upper = max(impressions, int(impressions * 1.30))
    imp_80 = extract_ci(data.get("impressions_80_ci") or data.get("projected_impressions"), default_i80_lower, default_i80_upper)

    imp_lower_delta = impressions - imp_80[0]
    imp_upper_delta = imp_80[1] - impressions
    default_i90_lower = max(reach_90[0], int(impressions - imp_lower_delta * 1.35))
    default_i90_upper = max(impressions, int(impressions + imp_upper_delta * 1.35))
    imp_90 = extract_ci(data.get("impressions_90_ci"), default_i90_lower, default_i90_upper)

    # Engagement rate
    er = data.get("engagement_rate")
    if er is None:
        er = data.get("projected_engagement_rate", 4.0)
    er = float(er or 4.0)
    if 0.0 < er <= 1.0:
        er = round(er * 100, 2)
    else:
        er = round(er, 2)

    # Virality tier
    virality_tier = str(data.get("virality_tier") or "📈 Moderate Distribution")

    # Platform & Format
    platform = str(data.get("platform") or fallback_platform)
    fmt = str(data.get("format") or data.get("media_type") or fallback_format)

    # Followers
    followers = (
        data.get("total_followers")
        or data.get("followers")
        or data.get("follower_count")
        or fallback_followers
    )
    if followers is not None:
        followers = int(followers)

    # Feature explanations
    feat_exp = data.get("feature_explanations")

    return {
        "predicted_reach": reach,
        "predicted_impressions": impressions,
        "reach_80_ci": reach_80,
        "reach_90_ci": reach_90,
        "impressions_80_ci": imp_80,
        "impressions_90_ci": imp_90,
        "feature_explanations": feat_exp,
        "engagement_rate": er,
        "virality_tier": virality_tier,
        "platform": platform,
        "format": fmt,
        "total_followers": followers,
    }


def _get_platform_terms(platform: str) -> Dict[str, str]:
    """Returns platform-specific terminology."""
    plat_lower = platform.lower()
    if "youtube" in plat_lower:
        return {
            "audience": "viewers",
            "explore_feed": "YouTube Shorts & Browse Feeds",
            "connection": "subscribers",
            "feed_action": "watching and subscribing",
        }
    elif "snapchat" in plat_lower:
        return {
            "audience": "viewers",
            "explore_feed": "Snapchat Spotlight Feed",
            "connection": "friends & subscribers",
            "feed_action": "viewing and sharing Snaps",
        }
    else:  # Instagram / Default
        return {
            "audience": "users",
            "explore_feed": "Instagram Explore & Reels feeds",
            "connection": "followers",
            "feed_action": "scrolling, saving, and sharing",
        }


def _generate_headline(platform: str, media_format: str, predicted_reach: int, virality_tier: str) -> str:
    """Generates a punchy one-line summary reflecting virality tier, reach, and platform."""
    terms = _get_platform_terms(platform)
    aud = terms["audience"]

    tier_lower = virality_tier.lower()
    if "explosive" in tier_lower or "viral" in tier_lower:
        prefix = "🚀 Viral Breakthrough"
        desc = f"Projected to reach ~{predicted_reach:,} {aud} on {platform}!"
    elif "high" in tier_lower:
        prefix = "🔥 Strong Momentum"
        desc = f"Projected to reach ~{predicted_reach:,} {aud}!"
    elif "moderate" in tier_lower or "growth" in tier_lower:
        prefix = "📈 Solid Distribution"
        desc = f"Projected to reach ~{predicted_reach:,} {aud}."
    else:
        prefix = "🎯 Targeted Core Reach"
        desc = f"Projected to reach ~{predicted_reach:,} {aud}."

    return f"{prefix}: {desc}"


def _generate_executive_summary(
    platform: str,
    media_format: str,
    predicted_reach: int,
    predicted_impressions: int,
    engagement_rate: float,
    virality_tier: str,
    followers: Optional[int]
) -> str:
    """Conversational paragraph putting the reach and impressions in perspective relative to follower scale."""
    terms = _get_platform_terms(platform)
    aud = terms["audience"]
    conn = terms["connection"]

    if followers and followers > 0:
        ratio = (predicted_reach / followers) * 100
        if ratio >= 100.0:
            scale_desc = (
                f"Relative to your follower base of **{followers:,}** {conn}, this represents an extraordinary "
                f"**{ratio:.1f}% audience reach**—projected to break well outside your current audience and surge into algorithmic discovery."
            )
        elif ratio >= 30.0:
            scale_desc = (
                f"Relative to your audience of **{followers:,}** {conn}, this represents a robust "
                f"**{ratio:.1f}% reach penetration**, signaling strong engagement among loyal followers alongside healthy feed exploration."
            )
        else:
            scale_desc = (
                f"Relative to your audience of **{followers:,}** {conn}, this represents a focused "
                f"**{ratio:.1f}% reach**, indicating performance concentrated primarily within your most active core community."
            )
    else:
        view_ratio = predicted_impressions / max(predicted_reach, 1)
        scale_desc = (
            f"This translates to an average of **{view_ratio:.2f}** impressions per reached account, "
            f"indicating strong repeat viewing interest and sustained viewer attention."
        )

    return (
        f"Your upcoming {platform} {media_format} is forecasted to reach approximately **{predicted_reach:,}** unique {aud} "
        f"with **{predicted_impressions:,}** total impressions. {scale_desc} "
        f"With an anticipated engagement rate of **{engagement_rate:.1f}%**, this post is tracking in the **{virality_tier}** tier."
    )


def _categorize_and_translate_drivers(
    drivers: List[Dict[str, Any]],
    platform: str,
    media_format: str
) -> Tuple[List[str], Dict[str, List[Dict[str, Any]]]]:
    """
    Translates raw TreeSHAP drivers into plain-English explanations and categorizes them
    into positive boosts vs negative opportunity areas.
    """
    terms = _get_platform_terms(platform)
    boosts: List[Dict[str, Any]] = []
    opportunities: List[Dict[str, Any]] = []
    plain_strings: List[str] = []

    for driver in drivers:
        name = str(driver.get("name", "Creative Choice"))
        impact = int(driver.get("impact", 0))
        pct = float(driver.get("pct", 0.0))
        direction = driver.get("direction")
        if direction is None:
            direction = "positive" if impact >= 0 else "negative"

        is_positive = (direction == "positive" or impact >= 0)
        name_lower = name.lower()

        # Format factor
        if any(w in name_lower for w in ["format", "media_type", "reel", "carousel", "short", "video", "spotlight"]):
            if is_positive:
                if "reel" in media_format.lower():
                    desc = f"Format Boost: Reels format adds +{impact:,} reach (+{pct:.1f}%) through active recommendations in the Reels & Explore feeds."
                elif "carousel" in media_format.lower():
                    desc = f"Format Boost: Carousel format adds +{impact:,} reach (+{pct:.1f}%) by driving multi-slide swiping and save actions."
                elif "short" in media_format.lower():
                    desc = f"Format Boost: YouTube Shorts format adds +{impact:,} reach (+{pct:.1f}%) through rapid swipe discovery on the Shorts Feed."
                elif "video" in media_format.lower():
                    desc = f"Format Boost: Long-form Video adds +{impact:,} reach (+{pct:.1f}%) by capturing long watch sessions and search ranking."
                elif "spotlight" in media_format.lower():
                    desc = f"Format Boost: Snapchat Spotlight format adds +{impact:,} reach (+{pct:.1f}%) with algorithmic distribution across non-friends."
                else:
                    desc = f"Format Boost: {media_format} format adds +{impact:,} reach (+{pct:.1f}%) compared to static baselines."
            else:
                desc = f"Opportunity Area - Format Drag: {media_format} reduces reach by -{abs(impact):,} ({pct:.1f}%) compared to dynamic video formats."

        # Timing factor
        elif any(w in name_lower for w in ["timing", "schedule", "peak", "hour", "day"]):
            if is_positive:
                desc = f"Peak Timing Boost: Posting during peak active hours adds +{impact:,} reach (+{pct:.1f}%) when audience activity is highest."
            else:
                desc = f"Opportunity Area - Off-Peak Timing: Publishing during off-peak hours reduces potential reach by -{abs(impact):,} ({pct:.1f}%) due to low initial scroll traffic."

        # Call-to-action (CTA) factor
        elif any(w in name_lower for w in ["cta", "call-to-action", "call to action"]):
            if is_positive:
                desc = f"Call-to-Action Boost: Explicit CTA prompt adds +{impact:,} reach (+{pct:.1f}%) by compelling viewers to save, share, and comment."
            else:
                desc = f"Opportunity Area - Missing CTA: Without a clear call-to-action, potential engagement drops by -{abs(impact):,} reach ({pct:.1f}%)."

        # Caption factor
        elif any(w in name_lower for w in ["caption", "length"]):
            if is_positive:
                desc = f"Caption Depth Boost: Detailed caption adds +{impact:,} reach (+{pct:.1f}%) by extending dwell time while reading."
            else:
                desc = f"Opportunity Area - Caption Depth: Brief caption reduces reach by -{abs(impact):,} ({pct:.1f}%) due to limited reading dwell time."

        # Hashtags factor
        elif any(w in name_lower for w in ["hashtag", "tag"]):
            if is_positive:
                desc = f"Hashtag Discovery Boost: Targeted hashtags add +{impact:,} reach (+{pct:.1f}%) by indexing content in relevant topic feeds."
            else:
                desc = f"Opportunity Area - Hashtag Discovery: Zero or limited hashtags reduce reach by -{abs(impact):,} ({pct:.1f}%) through lost topic discovery."

        # Style / Categorization factor
        elif any(w in name_lower for w in ["style", "theme", "categorization", "content styles"]):
            if is_positive:
                desc = f"Theme Alignment Boost: Thematic style adds +{impact:,} reach (+{pct:.1f}%) by matching high-interest viewer tastes."
            else:
                desc = f"Opportunity Area - Theme Positioning: Theme positioning reduces reach by -{abs(impact):,} ({pct:.1f}%) relative to viral trends."

        # Generic fallback
        else:
            if is_positive:
                desc = f"Boost - {name}: Contributes +{impact:,} reach (+{pct:.1f}%) to projected distribution."
            else:
                desc = f"Opportunity Area - {name}: Decreases potential reach by -{abs(impact):,} ({pct:.1f}%)."

        plain_strings.append(desc)
        item_data = {
            "name": name,
            "impact": impact,
            "pct": pct,
            "category": "boost" if is_positive else "opportunity",
            "summary": desc,
            "description": driver.get("description", desc),
        }
        if is_positive:
            boosts.append(item_data)
        else:
            opportunities.append(item_data)

    return plain_strings, {"boosts": boosts, "opportunities": opportunities}


def _generate_actionable_tips(
    opportunities: List[Dict[str, Any]],
    platform: str,
    media_format: str,
    engagement_rate: float
) -> List[str]:
    """Generates 2 to 4 concrete, highly actionable recommendations for boosting performance before publishing."""
    tips: List[str] = []
    opp_names = " ".join([o["name"].lower() for o in opportunities])

    # Address negative TreeSHAP levers first
    if any(w in opp_names for w in ["timing", "schedule", "peak", "hour"]):
        tips.append("🕒 **Reschedule to Peak Hours:** Shift your publishing window to 18:00–21:00 or 12:00–13:00 local audience time to hit peak scrolling activity.")

    if any(w in opp_names for w in ["cta", "call-to-action", "call to action"]):
        tips.append("📣 **Add a Friction-Free Call-to-Action:** Include an explicit prompt in the caption and closing screen (e.g. 'Save this guide for later' or 'Share with a friend').")

    if any(w in opp_names for w in ["caption", "length"]):
        tips.append("📝 **Deepen Your Caption:** Expand your caption to 150–300 characters with an arresting opening hook and clear formatting to maximize viewer dwell time.")

    if any(w in opp_names for w in ["hashtag", "tag"]):
        tips.append("🏷️ **Optimize Topic Hashtags:** Add 3–5 niche, highly relevant hashtags to help the platform's recommendation engine properly index your post.")

    if any(w in opp_names for w in ["format", "static"]):
        tips.append("🔄 **Upgrade Format:** Repackage static content into a multi-slide Carousel (+45% average saves) or a 15–30s Reel to trigger algorithmic explore distribution.")

    # Platform/Format specific additions if we still have room or no opportunity drivers
    plat_lower = platform.lower()
    fmt_lower = media_format.lower()

    if len(tips) < 4:
        if "youtube" in plat_lower:
            if "short" in fmt_lower:
                tips.append("⚡ **Hook in 2 Seconds:** Deliver your core punchline or visual question within the first 1.5–2 seconds to keep the Viewed vs Swiped-away ratio above 75%.")
            else:
                tips.append("🎬 **Thumbnail & Title Synergy:** Use a high-contrast thumbnail with an expressive human focal point and front-load keywords in your video title.")
        elif "snapchat" in plat_lower:
            tips.append("⚡ **Fast Pacing:** Keep the video tightly edited under 15 seconds with animated text overlays to retain quick-scrolling Snapchat viewers.")
        else:  # Instagram
            if "reel" in fmt_lower:
                tips.append("🎵 **Pair with Trending Audio:** Choose a rising audio track with less than 10k uses to gain early distribution via the Reels audio page.")
            elif "carousel" in fmt_lower:
                tips.append("📌 **Slide 2 Curiosity Hook:** Ensure slide 2 delivers immediate value and tease the best takeaway on slide 3+ to drive full swipe completions.")
            else:
                tips.append("💬 **Seed Early Engagement:** Pin a thoughtful conversation-starter in the comments and respond to every early commenter within the first 60 minutes.")

    # Guarantee 2 to 4 actionable tips
    if len(tips) < 2:
        tips.append("🚀 **Accelerate Early Signals:** Share the post directly to your Stories / Feed within 5 minutes of publishing with an interactive poll or sticker.")
    if len(tips) < 2:
        tips.append("💬 **Community Velocity:** Engage with similar creator accounts in your niche immediately before and after publishing to invite reciprocal visits.")

    return tips[:4]


def _generate_confidence_summary(
    reach_80_ci: Tuple[int, int],
    reach_90_ci: Tuple[int, int],
    impressions_80_ci: Tuple[int, int],
    impressions_90_ci: Tuple[int, int],
    predicted_reach: int,
    predicted_impressions: int
) -> str:
    """
    Plain-English explanation of the confidence range, avoiding mathematical jargon
    like Mondrian conformal quantiles, finite-sample coverage guarantees, or epistemic uncertainty.
    """
    r80_low, r80_high = reach_80_ci
    r90_low, r90_high = reach_90_ci
    i90_high = impressions_90_ci[1]

    return (
        f"Based on historical creator patterns, your reach is most reliably expected to land between "
        f"**{r80_low:,}** and **{r80_high:,}** users in standard publishing conditions (an 80% certainty window). "
        f"In a high-velocity breakout scenario where early saves and shares surge, outer reach bounds project up to "
        f"**{r90_high:,}** accounts, with impressions reaching up to **{i90_high:,}**. "
        f"This safe operating range gives you a dependable benchmark to measure actual post performance."
    )


def _generate_prompt_answer(
    prompt: str,
    platform: str,
    media_format: str,
    predicted_reach: int,
    predicted_impressions: int,
    engagement_rate: float,
    virality_tier: str,
    boosts: List[Dict[str, Any]],
    opportunities: List[Dict[str, Any]],
    tips: List[str]
) -> str:
    """Generates an AI strategist conversational answer tailored to the creator's explicit prompt."""
    p_lower = prompt.lower()

    if any(w in p_lower for w in ["why is my reach", "why lower", "lower than", "low reach", "decrease"]):
        if opportunities:
            top_opp = opportunities[0]
            ans = (
                f"The primary factor tempering your forecasted reach of **{predicted_reach:,}** is "
                f"**{top_opp['name']}**, which creates an estimated drag of **{top_opp['impact']:,} reach** ({top_opp['pct']:.1f}%). "
                f"By addressing this lever—specifically: {tips[0]}—you can immediately reclaim that lost distribution."
            )
        else:
            ans = (
                f"Your reach is actually pacing well at **{predicted_reach:,}** accounts with a **{virality_tier}** rating! "
                f"To push distribution even higher, focus on boosting viewer saves and shares before publishing."
            )
    elif any(w in p_lower for w in ["timing", "when to post", "time", "hour", "day", "schedule"]):
        timing_driver = next((d for d in (boosts + opportunities) if any(w in d["name"].lower() for w in ["timing", "schedule", "peak"])), None)
        if timing_driver and timing_driver["category"] == "boost":
            ans = (
                f"Your chosen schedule is working strongly in your favor (+{timing_driver['impact']:,} reach)! "
                f"It places your content right in front of your audience during an active browsing window."
            )
        elif timing_driver and timing_driver["category"] == "opportunity":
            ans = (
                f"Your scheduled time is currently an off-peak slot, reducing potential reach by -{abs(timing_driver['impact']):,} accounts. "
                f"We strongly recommend shifting your post time to a peak window (typically 18:00–21:00) to capture immediate scroll velocity."
            )
        else:
            ans = (
                f"Timing is a key multiplier for {platform} {media_format}s. Publishing between 18:00 and 21:00 local audience time "
                f"consistently drives the fastest initial view velocity."
            )
    elif any(w in p_lower for w in ["viral", "virality", "explore", "blow up", "trend"]):
        ans = (
            f"This post is currently tracking in the **{virality_tier}** tier with **{predicted_reach:,}** projected reach. "
            f"To trigger viral feed recommendations, maximize early direct message shares and comments in the first 30 minutes: {tips[0]}"
        )
    elif any(w in p_lower for w in ["save", "saves", "share", "shares"]):
        ans = (
            f"Saves and shares are the strongest ranking signals on {platform}. With an anticipated engagement rate of "
            f"**{engagement_rate:.1f}%**, incorporating an explicit bookmarking prompt and bookmarkable insights will directly drive repeat feed circulation."
        )
    else:
        # General question response
        lead_lever = boosts[0]["name"] if boosts else (opportunities[0]["name"] if opportunities else "Creative Setup")
        ans = (
            f"Taking into account your overall creative setup, this {platform} {media_format} is set to deliver "
            f"**{predicted_reach:,} reach** and **{predicted_impressions:,} impressions**. Your standout driver is **{lead_lever}**. "
            f"Review the strategic tips below to optimize your delivery before hitting publish."
        )

    return ans


def _generate_dialogue_markdown(
    prompt: Optional[str],
    headline: str,
    executive_summary: str,
    key_drivers: List[str],
    actionable_tips: List[str],
    confidence_summary: str,
    predicted_reach: int,
    predicted_impressions: int,
    reach_80_ci: Tuple[int, int],
    reach_90_ci: Tuple[int, int],
    impressions_80_ci: Tuple[int, int],
    impressions_90_ci: Tuple[int, int],
    engagement_rate: float,
    virality_tier: str,
    prompt_answer: Optional[str]
) -> str:
    """Formats the entire interpretation into clean, ready-to-render GitHub-flavored Markdown."""
    lines: List[str] = [
        "### 🤖 AI Content Strategist Performance Briefing",
        "",
    ]

    if prompt and prompt.strip():
        lines.extend([
            f"> 💬 **Creator Question:** *\"{prompt.strip()}\"*",
            "",
            f"**AI Strategist Assessment:** {prompt_answer}",
            "",
            "---",
            "",
        ])

    lines.extend([
        f"#### {headline}",
        "",
        executive_summary,
        "",
        "---",
        "",
        "#### 📊 Forecast Overview & Confidence Range",
        f"- **🎯 Projected Reach:** **{predicted_reach:,}** accounts *(safe window: {reach_80_ci[0]:,} – {reach_80_ci[1]:,})*",
        f"- **👁️ Projected Impressions:** **{predicted_impressions:,}** total views *(upper bound: up to {impressions_90_ci[1]:,})*",
        f"- **💬 Anticipated Engagement Rate:** **{engagement_rate:.1f}%**",
        f"- **⚡ Virality Potential:** **{virality_tier}**",
        "",
        f"> 🛡️ **Confidence Range:** {confidence_summary}",
        "",
        "---",
        "",
        "#### 🌳 Algorithmic Drivers & Levers (TreeSHAP Attribution)",
    ])

    if key_drivers:
        for driver_text in key_drivers:
            icon = "🟢" if ("boost" in driver_text.lower() and "opportunity" not in driver_text.lower()) else "⚠️"
            lines.append(f"- {icon} {driver_text}")
    else:
        lines.append("- ℹ️ Baseline model distribution calibrated with standard feature weights.")

    lines.extend([
        "",
        "---",
        "",
        "#### 💡 Strategic Steps Before You Publish",
    ])

    for i, tip in enumerate(actionable_tips, 1):
        lines.append(f"{i}. {tip}")

    lines.append("")
    return "\n".join(lines)


def interpret_simulation_result(
    simulation_result: Union[Dict[str, Any], SimulationPrediction],
    prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Translates raw ML simulation results and TreeSHAP attributions into a conversational,
    prompt-response dialogue that feels like a real-time AI strategist explaining
    the post to a creator.

    Args:
        simulation_result: Dict containing:
            - `predicted_reach`: int
            - `predicted_impressions`: int
            - `reach_80_ci`: tuple/dict (lower, upper)
            - `reach_90_ci`: tuple/dict (lower, upper)
            - `impressions_80_ci`: tuple/dict (lower, upper)
            - `impressions_90_ci`: tuple/dict (lower, upper)
            - `feature_explanations`: dict of TreeSHAP attribution drivers
            - `engagement_rate`: float
            - `virality_tier`: str
            - `platform`: str
            - `format`: str
            (Or an instance of SimulationPrediction, which is automatically converted)
        prompt: Optional creator query string (e.g. "Why is my reach lower than expected?").

    Returns:
        Dict containing:
            - `headline`: Punchy, one-line summary
            - `executive_summary`: Conversational paragraph putting reach & impressions in perspective
            - `key_drivers`: Structured list of positive and negative factors in plain English
            - `actionable_tips`: 2-4 concrete, highly actionable recommendations
            - `confidence_summary`: Plain-English explanation of confidence range without math jargon
            - `dialogue_markdown`: Ready-to-render GitHub-flavored Markdown text formatted as AI response
            - `driver_breakdown`: Dict with categorized 'boosts' and 'opportunities'
            - Plus normalized numeric simulation fields
    """
    # 1. Normalize input payload
    if isinstance(simulation_result, SimulationPrediction):
        payload = format_simulation_for_interpretation(simulation_result)
    elif isinstance(simulation_result, dict):
        payload = _normalize_simulation_dict(simulation_result)
    else:
        raise ValueError("simulation_result must be a dictionary or SimulationPrediction instance")

    predicted_reach = payload["predicted_reach"]
    predicted_impressions = payload["predicted_impressions"]
    reach_80_ci = payload["reach_80_ci"]
    reach_90_ci = payload["reach_90_ci"]
    impressions_80_ci = payload["impressions_80_ci"]
    impressions_90_ci = payload["impressions_90_ci"]
    engagement_rate = payload["engagement_rate"]
    virality_tier = payload["virality_tier"]
    platform = payload["platform"]
    media_format = payload["format"]
    followers = payload.get("total_followers")
    feat_exp = payload.get("feature_explanations") or {}

    # 2. Extract and translate TreeSHAP drivers
    raw_drivers = feat_exp.get("drivers", []) if isinstance(feat_exp, dict) else []
    key_drivers, driver_breakdown = _categorize_and_translate_drivers(raw_drivers, platform, media_format)

    # 3. Generate headline
    headline = _generate_headline(platform, media_format, predicted_reach, virality_tier)

    # 4. Generate executive summary
    executive_summary = _generate_executive_summary(
        platform=platform,
        media_format=media_format,
        predicted_reach=predicted_reach,
        predicted_impressions=predicted_impressions,
        engagement_rate=engagement_rate,
        virality_tier=virality_tier,
        followers=followers
    )

    # 5. Generate actionable tips (2-4 concrete tips)
    actionable_tips = _generate_actionable_tips(
        opportunities=driver_breakdown["opportunities"],
        platform=platform,
        media_format=media_format,
        engagement_rate=engagement_rate
    )

    # 6. Generate confidence summary without mathematical jargon
    confidence_summary = _generate_confidence_summary(
        reach_80_ci=reach_80_ci,
        reach_90_ci=reach_90_ci,
        impressions_80_ci=impressions_80_ci,
        impressions_90_ci=impressions_90_ci,
        predicted_reach=predicted_reach,
        predicted_impressions=predicted_impressions
    )

    # 7. Answer prompt if provided
    prompt_answer = None
    if prompt and prompt.strip():
        prompt_answer = _generate_prompt_answer(
            prompt=prompt,
            platform=platform,
            media_format=media_format,
            predicted_reach=predicted_reach,
            predicted_impressions=predicted_impressions,
            engagement_rate=engagement_rate,
            virality_tier=virality_tier,
            boosts=driver_breakdown["boosts"],
            opportunities=driver_breakdown["opportunities"],
            tips=actionable_tips
        )

    # 8. Generate dialogue markdown
    dialogue_markdown = _generate_dialogue_markdown(
        prompt=prompt,
        headline=headline,
        executive_summary=executive_summary,
        key_drivers=key_drivers,
        actionable_tips=actionable_tips,
        confidence_summary=confidence_summary,
        predicted_reach=predicted_reach,
        predicted_impressions=predicted_impressions,
        reach_80_ci=reach_80_ci,
        reach_90_ci=reach_90_ci,
        impressions_80_ci=impressions_80_ci,
        impressions_90_ci=impressions_90_ci,
        engagement_rate=engagement_rate,
        virality_tier=virality_tier,
        prompt_answer=prompt_answer
    )

    return {
        "headline": headline,
        "executive_summary": executive_summary,
        "key_drivers": key_drivers,
        "actionable_tips": actionable_tips,
        "confidence_summary": confidence_summary,
        "dialogue_markdown": dialogue_markdown,
        "driver_breakdown": driver_breakdown,
        "predicted_reach": predicted_reach,
        "predicted_impressions": predicted_impressions,
        "reach_80_ci": reach_80_ci,
        "reach_90_ci": reach_90_ci,
        "impressions_80_ci": impressions_80_ci,
        "impressions_90_ci": impressions_90_ci,
        "engagement_rate": engagement_rate,
        "virality_tier": virality_tier,
        "platform": platform,
        "format": media_format,
        "prompt": prompt,
    }
