from typing import Tuple, List, Dict, Any, Optional
from ..schemas import ProfileInput, PostInput, SimulationPrediction
from ..guardrails import validate_profile_dict, validate_post_dict
from ..models import simulate_post_performance


def run_post_simulation(
    profile_data: Dict[str, Any],
    post_data: Dict[str, Any]
) -> Tuple[bool, List[str], Optional[SimulationPrediction]]:
    """
    Simulates post performance with full schema validation and guardrail checks.
    """
    valid_prof, prof_errors, clean_prof = validate_profile_dict(profile_data)
    valid_post, post_errors, clean_post = validate_post_dict(post_data)

    errors = prof_errors + post_errors
    if not valid_prof or not valid_post:
        return False, errors, None

    profile = ProfileInput(**clean_prof)
    post = PostInput(**clean_post)

    prediction = simulate_post_performance(profile, post)
    return True, [], prediction
