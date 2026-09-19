from typing import Dict, List, Tuple
from ..config import settings


class SanityViolationError(ValueError):
    """Raised when an Instagram business logic constraint is strictly violated."""
    pass


def validate_profile_sanity(
    followers: int,
    following: int,
    posts: int,
    strict: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validates Instagram platform constraints for profile-level metrics.
    """
    violations = []

    if followers < 0:
        violations.append("Followers cannot be negative.")
    if following < 0:
        violations.append("Following cannot be negative.")
    if posts < 0:
        violations.append("Posts cannot be negative.")

    if following > settings.MAX_INSTAGRAM_FOLLOWING:
        violations.append(
            f"Instagram platform hard limit allows a maximum of {settings.MAX_INSTAGRAM_FOLLOWING} following (received: {following})."
        )

    if violations and strict:
        raise SanityViolationError("; ".join(violations))

    return len(violations) == 0, violations


def validate_media_sanity(
    reach: int,
    impressions: int,
    likes: int,
    comments: int,
    shares: int,
    saves: int,
    strict: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validates mathematical and platform invariants for per-media metrics:
    - Reach cannot exceed Impressions (impressions = reach * frequency, where frequency >= 1)
    - Likes/saves/shares cannot exceed impressions
    """
    violations = []

    if reach < 0 or impressions < 0 or likes < 0 or comments < 0 or shares < 0 or saves < 0:
        violations.append("Per-media counts must be non-negative.")

    if reach > impressions:
        violations.append(
            f"Invalid metric relation: Reach ({reach:,}) cannot exceed Impressions ({impressions:,})."
        )

    if likes > impressions and impressions > 0:
        violations.append(
            f"Invalid metric relation: Likes ({likes:,}) cannot exceed Impressions ({impressions:,})."
        )

    if saves > impressions and impressions > 0:
        violations.append(
            f"Invalid metric relation: Saves ({saves:,}) cannot exceed Impressions ({impressions:,})."
        )

    if violations and strict:
        raise SanityViolationError("; ".join(violations))

    return len(violations) == 0, violations
