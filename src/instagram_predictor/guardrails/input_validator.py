from typing import Any, Dict, List, Tuple
import pandas as pd
from ..schemas import ProfileInput, PostInput
from .sanity_rules import validate_profile_sanity, validate_media_sanity


def validate_profile_dict(data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validates and standardizes a raw profile dictionary against Pydantic schema and platform limits.
    """
    errors = []
    try:
        profile = ProfileInput(**data)
        _, sanity_errors = validate_profile_sanity(
            followers=profile.total_followers,
            following=profile.total_following,
            posts=profile.total_media_posts
        )
        errors.extend(sanity_errors)
        return len(errors) == 0, errors, profile.model_dump()
    except Exception as e:
        errors.append(str(e))
        return False, errors, data


def validate_post_dict(data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validates and standardizes a raw post dictionary against Pydantic schema and invariant rules.
    """
    errors = []
    try:
        post = PostInput(**data)
        if post.metrics:
            if post.metrics.reach is not None and post.metrics.impressions is not None:
                _, sanity_errors = validate_media_sanity(
                    reach=post.metrics.reach,
                    impressions=post.metrics.impressions,
                    likes=post.metrics.likes,
                    comments=post.metrics.comments,
                    shares=post.metrics.shares,
                    saves=post.metrics.saves
                )
                errors.extend(sanity_errors)
        return len(errors) == 0, errors, post.model_dump()
    except Exception as e:
        errors.append(str(e))
        return False, errors, data
