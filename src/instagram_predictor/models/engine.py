"""
Forecasting. Returns only what the trained model and its held-out evaluation support.

The engine REFUSES (raises OutOfSupportError) instead of guessing when an input lies outside what the
model was trained on: unseen platform/format, or a follower count outside the observed range.
Optional inputs left blank are reported as `imputed_fields`; extreme optional values are reported as
`extrapolated_fields`. No default value is ever substituted silently, and no outcome is post-processed
into something the model did not predict.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..data import build_features
from ..schemas import ConfidenceInterval, Forecast, PostInput, ProfileInput
from .registry import load_model
from .trainer import follower_tier


class OutOfSupportError(ValueError):
    """The requested forecast is outside the data the model was trained on."""


def _interval(point_log: float, q: float, level: float, calibration: str) -> ConfidenceInterval:
    lower = max(int(np.floor(np.expm1(point_log - q))), 0)
    upper = max(int(np.ceil(np.expm1(point_log + q))), 0)
    point = max(int(round(float(np.expm1(point_log)))), 0)
    return ConfidenceInterval(lower=min(lower, point), point_estimate=point, upper=max(upper, point),
                              confidence_level=level, calibration=calibration)


def _quantiles(cal: Dict[str, Any], followers: float) -> Tuple[float, float, str]:
    tier = follower_tier(followers)
    t = cal["tiers"].get(tier, {})
    if t.get("used"):
        return t["q80"], t["q90"], f"follower tier '{tier}' ({t['n_residuals']} residuals from {t['n_creators']} creators)"
    return cal["global"]["q80"], cal["global"]["q90"], (
        f"all creators ({cal['n_residuals']} residuals); tier '{tier}' had too little data for its own calibration")


def _check_support(profile: ProfileInput, post: PostInput, support: Dict[str, Any]) -> None:
    if profile.platform != post.platform:
        raise ValueError(f"Profile platform {profile.platform.value} does not match the platform of format '{post.media_type.value}'")
    if post.platform.value not in support["platforms"]:
        raise OutOfSupportError(f"No training data for platform '{post.platform.value}' (trained on: {support['platforms']})")
    if post.media_type.value not in support["media_types"]:
        raise OutOfSupportError(f"No training data for format '{post.media_type.value}' (trained on: {support['media_types']})")
    lo, hi = support["followers_min"], support["followers_max"]
    if not (lo <= profile.total_followers <= hi):
        raise OutOfSupportError(
            f"{profile.total_followers:,} followers is outside the range the model was trained on "
            f"({lo:,} to {hi:,}); no forecast is given rather than extrapolating."
        )


def _row(profile: ProfileInput, post: PostInput) -> pd.DataFrame:
    return pd.DataFrame([{
        "total_followers": profile.total_followers,
        "platform": post.platform.value,
        "media_type": post.media_type.value,
        "posted_day_of_week": post.posted_day_of_week,
        "posted_hour_of_day": post.posted_hour_of_day,
        "caption_length_chars": post.caption_length_chars,
        "hashtags_count": post.hashtags_count,
        "mentions_count": post.mentions_count,
        "has_call_to_action": np.nan if post.has_call_to_action is None else float(post.has_call_to_action),
        "video_duration_seconds": post.video_duration_seconds,
        "carousel_slide_count": post.carousel_slide_count,
    }]).astype({"caption_length_chars": "float64", "hashtags_count": "float64", "mentions_count": "float64",
                "video_duration_seconds": "float64", "carousel_slide_count": "float64"})


def forecast_post(profile: ProfileInput, post: PostInput) -> Forecast:
    bundle, meta = load_model()
    support, spec = meta["support"], bundle["spec"]
    _check_support(profile, post, support)

    row = _row(profile, post)
    X = build_features(row, spec)

    imputed = [c for c in spec["optional_used"] if pd.isna(row[c].iloc[0])]
    extrapolated: List[str] = []
    for c, (lo, hi) in support["optional_ranges"].items():
        v = row[c].iloc[0]
        if pd.notna(v) and not (lo <= float(v) <= hi):
            extrapolated.append(f"{c}={v:g} (trained on {lo:g} to {hi:g})")

    reach_log = float(bundle["reach_model"].predict(X)[0])
    reach_eval = meta["reach"]
    q80, q90, cal_txt = _quantiles(reach_eval["calibration"], profile.total_followers)
    reach_80 = _interval(reach_log, q80, 0.80, cal_txt)
    reach_90 = _interval(reach_log, q90, 0.90, cal_txt)

    imp_80 = imp_90 = None
    imp_eval = meta["impressions"]
    if bundle.get("ratio_model") is not None and imp_eval.get("modelled"):
        ratio = max(float(bundle["ratio_model"].predict(X)[0]), 0.0)   # impressions >= reach by definition
        imp_log = reach_log + ratio
        iq80, iq90, icat = _quantiles(imp_eval["calibration"], profile.total_followers)
        imp_80 = _interval(imp_log, iq80, 0.80, icat)
        imp_90 = _interval(imp_log, iq90, 0.90, icat)

    chosen = reach_eval["selected"]
    is_baseline = bool(reach_eval["selected_is_baseline"])
    val = reach_eval.get("interval_validation", {})
    notes: List[str] = []
    if is_baseline:
        notes.append("Selected model is the follower-proportional baseline: no richer model beat it on held-out creators, "
                     "so post details (format, time, caption...) do not change this forecast.")
    if val.get("available") and not val.get("intervals_reliable", True):
        notes.append("Measured coverage of these intervals on held-out creators was below nominal; treat them as optimistic.")
    if not val.get("available"):
        notes.append("Interval coverage could not be validated on held-out creators with this amount of data.")
    if imp_80 is None:
        notes.append(imp_eval.get("reason", "Impressions are not modelled."))
    notes.extend(meta.get("limits", []))

    return Forecast(
        reach_80=reach_80, reach_90=reach_90, impressions_80=imp_80, impressions_90=imp_90,
        model_name=chosen, model_is_baseline=is_baseline,
        n_train_posts=meta["data"]["n_posts"], n_train_creators=meta["data"]["n_creators"],
        heldout_median_abs_pct_error=reach_eval["candidates"][chosen]["median_ratio_error"],
        empirical_coverage_80=val.get("measured_80_mean") if val.get("available") else None,
        empirical_coverage_90=val.get("measured_90_mean") if val.get("available") else None,
        imputed_fields=imputed, extrapolated_fields=extrapolated, notes=notes,
    )


def forecast_formats(profile: ProfileInput, post: PostInput) -> Dict[str, Forecast]:
    """
    Model forecasts for the same post under every format the model was trained on for this platform.
    These are model comparisons (association in the data), not a promise that switching format changes reach.
    """
    _, meta = load_model()
    out: Dict[str, Forecast] = {}
    for mt in meta["support"]["media_types"]:
        try:
            alt = post.model_copy(update={"media_type": type(post.media_type)(mt)})
            if alt.platform != profile.platform:
                continue
            if alt.media_type.value != "Carousel":
                alt = alt.model_copy(update={"carousel_slide_count": None})
            out[mt] = forecast_post(profile, alt)
        except (OutOfSupportError, ValueError):
            continue
    return out
