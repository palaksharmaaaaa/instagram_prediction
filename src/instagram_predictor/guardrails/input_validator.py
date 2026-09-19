from typing import Any, Dict, List, Tuple

from ..schemas import PostInput, ProfileInput
from .sanity_rules import validate_profile_sanity


def validate_profile_dict(data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """Validates a profile dict against the schema and platform limits. Values are never altered."""
    try:
        profile = ProfileInput(**data)
    except Exception as e:  # pydantic ValidationError
        return False, [str(e)], data
    errors: List[str] = []
    if profile.total_following is not None and profile.platform.value == "Instagram":
        _, sanity = validate_profile_sanity(
            followers=profile.total_followers,
            following=profile.total_following,
            posts=profile.total_media_posts or 0,
        )
        errors.extend(sanity)
    return len(errors) == 0, errors, profile.model_dump()


def validate_post_dict(data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    try:
        post = PostInput(**data)
    except Exception as e:
        return False, [str(e)], data
    return True, [], post.model_dump()
