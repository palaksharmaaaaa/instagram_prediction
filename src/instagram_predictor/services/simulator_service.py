from typing import Tuple, List, Dict, Any, Optional
from ..schemas import ProfileInput, PostInput, SimulationPrediction
from ..guardrails import validate_profile_dict, validate_post_dict
from ..models import simulate_post_performance, explain_post_prediction


def run_post_simulation(
    profile_data: Dict[str, Any],
    post_data: Dict[str, Any]
) -> Tuple[bool, List[str], Optional[SimulationPrediction]]:
    """
    Simulates post performance with full schema validation and guardrail checks.
    Includes feature_explanations in the resulting SimulationPrediction.
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


def explain_post_simulation(
    profile_data: Dict[str, Any],
    post_data: Dict[str, Any]
) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """
    Computes TreeSHAP / creative feature importance explainability for a simulation scenario.
    """
    valid_prof, prof_errors, clean_prof = validate_profile_dict(profile_data)
    valid_post, post_errors, clean_post = validate_post_dict(post_data)

    errors = prof_errors + post_errors
    if not valid_prof or not valid_post:
        return False, errors, None

    profile = ProfileInput(**clean_prof)
    post = PostInput(**clean_post)

    explanations = explain_post_prediction(post=post, profile=profile)
    return True, [], explanations


def run_post_simulation_and_interpret(
    profile_data: Dict[str, Any],
    post_data: Dict[str, Any],
    prompt: Optional[str] = None
) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """
    Simulates post performance and translates the result into a conversational,
    creator-facing AI interpretation dialogue.
    """
    success, errors, prediction = run_post_simulation(profile_data, post_data)
    if not success or prediction is None:
        return False, errors, None

    from .interpreter_service import interpret_simulation_result, format_simulation_for_interpretation
    sim_dict = format_simulation_for_interpretation(
        prediction,
        profile_data=profile_data,
        post_data=post_data
    )
    interpretation = interpret_simulation_result(sim_dict, prompt=prompt)
    return True, [], interpretation

