from typing import Dict, Any, List
from ..schemas.query import AuditReport, QueryIntent


def audit_query(
    prompt: str,
    filters: Dict[str, Any],
    intent: QueryIntent,
    predict_reach: bool,
    predict_impressions: bool
) -> AuditReport:
    """
    Audits the user prompt, scores clarity, highlights detected entities,
    and provides intelligent refinement recommendations.
    """
    clarity_score = 100
    suggestions: List[str] = []
    intents_list: List[str] = [intent.primary_goal]

    if intent.creator_tier != "Any":
        intents_list.append(f"Tier: {intent.creator_tier}")
    if intent.audience_focus != "Broad Audience":
        intents_list.append(f"Demographics: {intent.audience_focus}")

    # Entity extraction summary
    extracted_entities = {}
    if "category" in filters:
        extracted_entities["Category"] = filters["category"]
    if "media_type" in filters:
        extracted_entities["Media Format"] = filters["media_type"]
    if "country" in filters:
        extracted_entities["Country"] = filters["country"]
    if "username" in filters:
        extracted_entities["Target Handle"] = f"@{filters['username']}"
    if "total_followers" in filters:
        extracted_entities["Follower Constraint"] = str(filters["total_followers"])
    if "engagement_rate" in filters:
        extracted_entities["Engagement Constraint"] = str(filters["engagement_rate"])

    # Clarity scoring and suggestions
    num_entities = len(extracted_entities)

    if not prompt.strip():
        clarity_score = 0
        suggestions.append("Enter specific creator niches (e.g. Sports, Fashion), media formats (Reels), or follower tiers.")
        feedback = "Empty prompt received. Showing global top creators."
    elif num_entities == 0:
        clarity_score = 45
        suggestions.append("Specify a topic category (e.g. 'Fitness', 'Tech', 'Music') to isolate relevant creators.")
        suggestions.append("Add follower or engagement thresholds (e.g. 'over 1M followers', 'engagement > 2%').")
        feedback = "Broad query without explicit categorical or metric boundaries. Consider adding a niche or format."
    elif num_entities == 1:
        clarity_score = 70
        if "category" in extracted_entities:
            suggestions.append("Try specifying a media format (e.g. 'Reels' or 'Carousels') to see format-specific reach.")
        feedback = "Good baseline query. You can add geographic or performance constraints for finer targeting."
    elif num_entities >= 2:
        clarity_score = 95
        if not predict_reach and not predict_impressions:
            suggestions.append("Add 'predict reach' or 'forecast impressions' to activate ML performance projections.")
        feedback = "High-precision query with well-defined dimensional filters."

    # Contradiction check
    if "total_followers" in filters and intent.creator_tier != "Any":
        f_cond = filters["total_followers"]
        if isinstance(f_cond, dict) and f_cond.get("operator") == ">" and f_cond.get("value", 0) > 1_000_000 and "Micro" in intent.creator_tier:
            suggestions.append("⚠️ Potential contradiction: Query mentions 'Micro' creators but specifies followers > 1M.")
            clarity_score -= 20

    return AuditReport(
        clarity_score=max(clarity_score, 10),
        detected_intents=intents_list,
        extracted_entities=extracted_entities,
        refinement_suggestions=suggestions,
        audit_feedback=feedback
    )
