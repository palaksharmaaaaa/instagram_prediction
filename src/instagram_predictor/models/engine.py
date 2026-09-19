import itertools
import math
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from ..config import settings
from ..data import (
    ALL_FEATURE_COLUMNS,
    ALL_PRE_PUBLISH_FEATURE_COLUMNS,
    ALL_DIAGNOSTIC_FEATURE_COLUMNS,
    compute_derived_metrics,
)
from ..schemas import ConfidenceInterval, SimulationPrediction, PostInput, ProfileInput
from .registry import (
    get_reach_pipeline,
    get_impressions_pipeline,
    get_pre_publish_reach_pipeline,
    get_pre_publish_impressions_pipeline,
    get_diagnostic_reach_pipeline,
    get_diagnostic_impressions_pipeline,
    get_model_metadata,
)


def predict_batch(
    df: pd.DataFrame,
    predict_reach: bool = True,
    predict_impressions: bool = True,
    use_pre_publish: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Vectorized batch inference on a DataFrame with feature derivation and invariant enforcement.
    Automatically routes to pre-publishing pipelines if post-publication metrics are missing.
    """
    if df.empty:
        return df.copy()

    # Determine whether DataFrame contains post-publication metrics
    post_pub_cols = ["per_media_likes", "per_media_comments", "per_media_shares", "per_media_saves"]
    has_post_pub = (
        all(col in df.columns for col in post_pub_cols)
        and not df[post_pub_cols].isna().all().all()
    )

    if use_pre_publish is None:
        use_pre_publish = not has_post_pub

    # Ensure all derived features exist
    df = compute_derived_metrics(df)

    if use_pre_publish:
        X = df[ALL_PRE_PUBLISH_FEATURE_COLUMNS]
        reach_model = get_pre_publish_reach_pipeline() if predict_reach else None
        imp_model = get_pre_publish_impressions_pipeline() if predict_impressions else None
    else:
        X = df[ALL_DIAGNOSTIC_FEATURE_COLUMNS]
        reach_model = get_diagnostic_reach_pipeline() if predict_reach else None
        imp_model = get_diagnostic_impressions_pipeline() if predict_impressions else None

    if predict_reach and reach_model is not None:
        raw_reach = reach_model.predict(X)
        clean_reach = np.nan_to_num(raw_reach, nan=100.0, posinf=1e10, neginf=100.0)
        df["predicted_reach"] = np.maximum(np.round(clean_reach).astype(int), 100)

    if predict_impressions and imp_model is not None:
        raw_imp = imp_model.predict(X)
        clean_imp = np.nan_to_num(raw_imp, nan=100.0, posinf=1e10, neginf=100.0)
        df["predicted_impressions"] = np.maximum(np.round(clean_imp).astype(int), 100)

    # Invariant guardrail: reach cannot exceed impressions
    if predict_reach and predict_impressions:
        df["predicted_impressions"] = np.maximum(df["predicted_impressions"], df["predicted_reach"])

    return df


def explain_post_prediction(
    post: Union[PostInput, ProfileInput],
    profile: Optional[Union[ProfileInput, PostInput]] = None,
) -> Dict[str, Any]:
    """
    Evaluates the pre-publishing model on the current post vs baseline/reference variations
    using exact Combinatorial Shapley Attribution (evaluating 64 coalitions over 6 creative levers) to compute the exact impact of each creative choice:
      - Format Impact (e.g. Reel or Carousel vs Static Image)
      - Content Styles Impact (e.g. Educational / Entertaining vs neutral)
      - Call-to-Action Impact (CTA enabled vs disabled)
      - Caption Depth Impact (caption length vs short caption)
      - Hashtag Discovery Impact (hashtag count vs zero hashtags)
      - Timing & Schedule Impact (chosen hour/day vs off-peak)
    Computes both raw reach impact (+15,400, -3,200) and percentage impact (+18.5%, -4.2%).
    Returns a structured dictionary with base_reach, final_reach, and drivers.
    """
    # Robust argument order handling
    if isinstance(post, ProfileInput) and isinstance(profile, PostInput):
        post, profile = profile, post

    if not isinstance(post, PostInput) or not isinstance(profile, ProfileInput):
        raise ValueError("explain_post_prediction requires valid PostInput and ProfileInput instances")

    reach_pipeline = get_pre_publish_reach_pipeline()

    # Demographic defaults
    demo = post.demographics
    top_country = demo.top_country if demo else profile.country
    sec_country = demo.secondary_country if demo else "US"
    primary_age = demo.primary_age_group if demo else "25-34"
    female_pct = demo.gender_female_pct if demo else 0.50
    male_pct = demo.gender_male_pct if demo else 0.50
    activity_score = demo.audience_activity_score if demo else 0.75

    media_type_val = post.media_type.value if hasattr(post.media_type, "value") else str(post.media_type)
    platform_val = post.platform.value if hasattr(post.platform, "value") else str(getattr(post, "platform", "Instagram"))
    category_val = post.category.value if hasattr(post.category, "value") else str(post.category)
    acc_cat_str = profile.account_category.value if hasattr(profile.account_category, "value") else str(profile.account_category)

    style_list = post.categorizations if post.categorizations else ([post.categorization] if post.categorization else [])
    cat_str = ", ".join([c.value if hasattr(c, "value") else str(c) for c in style_list]) if style_list else "General"

    factors = ["format", "style", "cta", "caption", "hashtags", "timing"]

    if platform_val == "YouTube":
        base_format = ("Community Post", 0.0, 1)
    elif platform_val == "Snapchat":
        base_format = ("Snapchat Post", 0.0, 1)
    else:
        base_format = ("Static Image", 0.0, 1)

    base_vals = {
        "format": base_format,
        "style": "General",
        "cta": False,
        "caption": 50,
        "hashtags": 0,
        "timing": (3, "Tuesday"),
    }
    post_vals = {
        "format": (media_type_val, post.video_duration_seconds, post.carousel_slide_count),
        "style": cat_str,
        "cta": bool(post.has_call_to_action),
        "caption": int(post.caption_length_chars),
        "hashtags": int(post.hashtags_count),
        "timing": (int(post.posted_hour_of_day), str(post.posted_day_of_week)),
    }

    combinations = list(itertools.product([0, 1], repeat=len(factors)))
    rows = []
    for combo in combinations:
        vals = {}
        for factor, is_active in zip(factors, combo):
            vals[factor] = post_vals[factor] if is_active else base_vals[factor]

        fmt, vid_sec, slides = vals["format"]
        hour, day = vals["timing"]
        row = {
            "platform": platform_val,
            "username": profile.username,
            "country": profile.country,
            "total_followers": profile.total_followers,
            "total_following": profile.total_following,
            "total_media_posts": profile.total_media_posts,
            "account_age_years": profile.account_age_years,
            "posting_frequency_per_week": profile.posting_frequency_per_week,
            "follower_growth_rate_30d": profile.follower_growth_rate_30d,
            "account_bio_has_link": profile.account_bio_has_link,
            "media_type": fmt,
            "category": category_val,
            "categorization": vals["style"],
            "account_category": acc_cat_str,
            "caption_length_chars": vals["caption"],
            "hashtags_count": vals["hashtags"],
            "mentions_count": post.mentions_count,
            "has_call_to_action": vals["cta"],
            "video_duration_seconds": vid_sec,
            "carousel_slide_count": slides,
            "posted_day_of_week": day,
            "posted_hour_of_day": hour,
            "is_weekend": 1.0 if day in ["Saturday", "Sunday"] else 0.0,
            "is_peak_posting_hour": 1.0 if hour in [11, 12, 13, 18, 19, 20, 21] else 0.0,
            "top_country": top_country,
            "secondary_country": sec_country,
            "primary_age_group": primary_age,
            "gender_female_pct": female_pct,
            "gender_male_pct": male_pct,
            "audience_activity_score": activity_score,
        }
        rows.append(row)

    df_batch = compute_derived_metrics(pd.DataFrame(rows))
    raw_preds = reach_pipeline.predict(df_batch[ALL_PRE_PUBLISH_FEATURE_COLUMNS])
    clean_preds = np.nan_to_num(raw_preds, nan=100.0, posinf=1e10, neginf=100.0)
    preds = np.maximum(np.round(clean_preds).astype(int), 100)

    base_reach = int(preds[0])
    final_reach = int(preds[-1])

    # Exact Shapley values calculation across all factor subsets
    n = len(factors)
    weights = [math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n) for r in range(n)]
    shap_values = {f: 0.0 for f in factors}
    combo_to_idx = {combo: idx for idx, combo in enumerate(combinations)}

    for i, f in enumerate(factors):
        for combo in combinations:
            if combo[i] == 1:
                continue
            combo_with = list(combo)
            combo_with[i] = 1
            combo_with = tuple(combo_with)

            idx_without = combo_to_idx[combo]
            idx_with = combo_to_idx[combo_with]

            subset_size = sum(combo)
            weight = weights[subset_size]
            diff = float(preds[idx_with] - preds[idx_without])
            shap_values[f] += weight * diff

    # Integer conservation enforcement: base_reach + sum(impacts) == final_reach
    raw_impacts = {f: int(np.round(shap_values[f])) for f in factors}
    residual = (final_reach - base_reach) - sum(raw_impacts.values())
    if residual != 0:
        dominant_factor = max(raw_impacts.keys(), key=lambda k: abs(raw_impacts[k]))
        raw_impacts[dominant_factor] += residual

    driver_metadata = {
        "format": {
            "name": f"{media_type_val} Media Format",
            "description": f"{media_type_val} format vs baseline {base_format[0]}",
        },
        "style": {
            "name": "Content Styles & Theme",
            "description": f"Thematic positioning ({cat_str}) vs neutral baseline",
        },
        "cta": {
            "name": "Call-to-Action (CTA)",
            "description": "Explicit call-to-action driving comments & saves" if post.has_call_to_action else "No explicit call-to-action prompt",
        },
        "caption": {
            "name": "Caption Depth",
            "description": f"Caption length ({post.caption_length_chars} chars) vs short baseline (50 chars)",
        },
        "hashtags": {
            "name": "Hashtag Discovery",
            "description": f"Algorithmic search & explore reach via {post.hashtags_count} hashtags vs zero hashtags",
        },
        "timing": {
            "name": "Peak Schedule & Timing",
            "description": f"Scheduled for {post.posted_day_of_week} at {post.posted_hour_of_day:02d}:00 vs off-peak baseline",
        },
    }

    drivers = []
    for f in factors:
        impact = raw_impacts[f]
        pct = round((impact / base_reach) * 100, 2) if base_reach > 0 else 0.0
        direction = "positive" if impact >= 0 else "negative"
        drivers.append({
            "name": driver_metadata[f]["name"],
            "impact": impact,
            "pct": pct,
            "direction": direction,
            "description": driver_metadata[f]["description"],
        })

    return {
        "base_reach": base_reach,
        "final_reach": final_reach,
        "drivers": drivers,
    }


def simulate_post_performance(
    profile: ProfileInput,
    post: PostInput,
    confidence_level: float = 0.80
) -> SimulationPrediction:
    """
    Simulates projected post performance for a given profile and media scenario:
    - When post.metrics is None: evaluates directly with the Pre-Publishing Forecasting Pipelines
      on ALL_PRE_PUBLISH_FEATURE_COLUMNS using pre-publish Mondrian conformal quantiles.
    - When post.metrics is provided: evaluates with the Post-Publishing Diagnostic Pipelines
      on ALL_DIAGNOSTIC_FEATURE_COLUMNS using diagnostic conformal quantiles.
    """
    meta = get_model_metadata()

    # Determine creator follower scale tier
    followers = profile.total_followers
    if followers < 10_000:
        follower_tier = "nano"
        tier_label = "Nano (<10k followers)"
    elif followers < 100_000:
        follower_tier = "micro"
        tier_label = "Micro (10k-100k followers)"
    elif followers < 1_000_000:
        follower_tier = "macro"
        tier_label = "Macro (100k-1M followers)"
    else:
        follower_tier = "mega"
        tier_label = "Mega (1M+ followers)"

    calibrated_level = 0.80 if confidence_level <= 0.85 else 0.90
    q_key = "q80" if calibrated_level == 0.80 else "q90"

    media_type_val = post.media_type.value if hasattr(post.media_type, "value") else str(post.media_type)
    platform_val = post.platform.value if hasattr(post.platform, "value") else str(getattr(post, "platform", "Instagram"))
    category_val = post.category.value if hasattr(post.category, "value") else str(post.category)
    style_list = post.categorizations if post.categorizations else ([post.categorization] if post.categorization else [])
    style_tokens = " ".join([s.value if hasattr(s, "value") else str(s) for s in style_list]).lower()
    cat_str = ", ".join([c.value if hasattr(c, "value") else str(c) for c in post.categorizations]) if post.categorizations else (
        post.categorization.value if hasattr(post.categorization, "value") else str(post.categorization or "Entertaining / Trend")
    )
    acc_cat_str = profile.account_category.value if hasattr(profile.account_category, "value") else str(profile.account_category)

    is_unpublished = (post.metrics is None)

    if is_unpublished:
        # Pre-Publishing Forecasting Model (Direct algorithmic prediction without mock metrics)
        reach_pipeline = get_pre_publish_reach_pipeline()
        imp_pipeline = get_pre_publish_impressions_pipeline()

        reach_eval = meta.get("evaluation", {}).get("pre_publish_reach", meta.get("evaluation", {}).get("reach", {}))
        imp_eval = meta.get("evaluation", {}).get("pre_publish_impressions", meta.get("evaluation", {}).get("impressions", {}))

        global_q_reach = reach_eval.get("conformal_quantile_80" if calibrated_level == 0.80 else "conformal_quantile_90", 0.35)
        global_q_imp = imp_eval.get("conformal_quantile_80" if calibrated_level == 0.80 else "conformal_quantile_90", 0.35)

        tier_reach_meta = reach_eval.get("tier_conformal_quantiles", {}).get(follower_tier, {})
        tier_imp_meta = imp_eval.get("tier_conformal_quantiles", {}).get(follower_tier, {})

        q_reach = tier_reach_meta.get(q_key, global_q_reach)
        q_imp = tier_imp_meta.get(q_key, global_q_imp)

        row = {
            "platform": platform_val,
            "username": profile.username,
            "country": profile.country,
            "total_followers": profile.total_followers,
            "total_following": profile.total_following,
            "total_media_posts": profile.total_media_posts,
            "account_age_years": profile.account_age_years,
            "posting_frequency_per_week": profile.posting_frequency_per_week,
            "follower_growth_rate_30d": profile.follower_growth_rate_30d,
            "account_bio_has_link": profile.account_bio_has_link,
            "media_type": media_type_val,
            "category": category_val,
            "categorization": cat_str,
            "account_category": acc_cat_str,
            "caption_length_chars": post.caption_length_chars,
            "hashtags_count": post.hashtags_count,
            "mentions_count": post.mentions_count,
            "has_call_to_action": post.has_call_to_action,
            "video_duration_seconds": post.video_duration_seconds,
            "carousel_slide_count": post.carousel_slide_count,
            "posted_day_of_week": post.posted_day_of_week,
            "posted_hour_of_day": post.posted_hour_of_day,
            "is_weekend": 1.0 if post.posted_day_of_week in ["Saturday", "Sunday"] else 0.0,
            "is_peak_posting_hour": 1.0 if post.posted_hour_of_day in [11, 12, 13, 18, 19, 20, 21] else 0.0,
            "top_country": post.demographics.top_country,
            "secondary_country": post.demographics.secondary_country,
            "primary_age_group": post.demographics.primary_age_group,
            "gender_female_pct": post.demographics.gender_female_pct,
            "gender_male_pct": post.demographics.gender_male_pct,
            "audience_activity_score": post.demographics.audience_activity_score,
        }

        df_single = compute_derived_metrics(pd.DataFrame([row]))
        X_single = df_single[ALL_PRE_PUBLISH_FEATURE_COLUMNS]

        # Points estimates directly from pre-publish pipelines
        raw_point_reach = reach_pipeline.predict(X_single)[0]
        clean_point_reach = float(np.nan_to_num(raw_point_reach, nan=100.0, posinf=1e10, neginf=100.0))
        point_reach = max(int(np.round(clean_point_reach)), 100)

        raw_point_imp = imp_pipeline.predict(X_single)[0]
        clean_point_imp = float(np.nan_to_num(raw_point_imp, nan=100.0, posinf=1e10, neginf=100.0))
        point_imp = max(int(np.round(clean_point_imp)), point_reach)

        # Baseline expected engagement rates for downstream creator metrics
        base_er = float(np.clip(0.045 - 0.003 * np.log10(max(followers, 100)), 0.015, 0.08))
        cta_boost = 1.15 if post.has_call_to_action else 1.0
        peak_boost = 1.10 if post.posted_hour_of_day in [11, 12, 13, 18, 19, 20, 21] else 1.0
        proj_er = float(base_er * cta_boost * peak_boost)

        if media_type_val == "Carousel":
            proj_save = 0.025 * (1.5 if "educational" in style_tokens else 1.0) * (1.2 if post.has_call_to_action else 1.0)
            proj_share = 0.015
        elif media_type_val in ["Reel", "YouTube Short", "Snapchat Spotlight"]:
            proj_save = 0.018 * (1.4 if "educational" in style_tokens else 1.0) * (1.2 if post.has_call_to_action else 1.0)
            proj_share = 0.035 * (1.3 if "entertaining" in style_tokens else 1.0)
        elif media_type_val == "YouTube Video":
            proj_save = 0.022 * (1.4 if "educational" in style_tokens else 1.0)
            proj_share = 0.020
        elif media_type_val in ["Story", "Snapchat Story"]:
            proj_save = 0.005
            proj_share = 0.020
        else:
            proj_save = 0.012 * (1.2 if post.has_call_to_action else 1.0)
            proj_share = 0.012

        virality = float(proj_share / max(proj_er, 0.01))
        virality_momentum = virality * (2.0 if media_type_val in ["Reel", "YouTube Short", "Snapchat Spotlight"] else 1.0)
        interaction_density = 0.40

        feature_explanations = explain_post_prediction(post=post, profile=profile)

    else:
        feature_explanations = None
        # Post-Publishing Diagnostic Model (Requires verified engagement metrics)
        reach_pipeline = get_diagnostic_reach_pipeline()
        imp_pipeline = get_diagnostic_impressions_pipeline()

        reach_eval = meta.get("evaluation", {}).get("diagnostic_reach", meta.get("evaluation", {}).get("reach", {}))
        imp_eval = meta.get("evaluation", {}).get("diagnostic_impressions", meta.get("evaluation", {}).get("impressions", {}))

        global_q_reach = reach_eval.get("conformal_quantile_80" if calibrated_level == 0.80 else "conformal_quantile_90", 0.35)
        global_q_imp = imp_eval.get("conformal_quantile_80" if calibrated_level == 0.80 else "conformal_quantile_90", 0.35)

        tier_reach_meta = reach_eval.get("tier_conformal_quantiles", {}).get(follower_tier, {})
        tier_imp_meta = imp_eval.get("tier_conformal_quantiles", {}).get(follower_tier, {})

        q_reach = tier_reach_meta.get(q_key, global_q_reach)
        q_imp = tier_imp_meta.get(q_key, global_q_imp)

        likes = post.metrics.likes
        comments = post.metrics.comments
        shares = post.metrics.shares
        saves = post.metrics.saves
        video_views = post.metrics.video_views if post.metrics.video_views is not None else (
            int(likes * 5.0) if media_type_val in ["Reel", "YouTube Short", "Snapchat Spotlight", "YouTube Video"] else 0
        )
        completion_rate = post.metrics.completion_rate if post.metrics.completion_rate is not None else (
            0.45 if media_type_val in ["Reel", "YouTube Short", "Snapchat Spotlight"] else (
                0.35 if media_type_val in ["YouTube Video"] else 0.0
            )
        )
        reach_home_pct = post.metrics.reach_from_home_pct if post.metrics.reach_from_home_pct is not None else 0.55
        reach_explore_pct = post.metrics.reach_from_explore_pct if post.metrics.reach_from_explore_pct is not None else (
            0.35 if media_type_val in ["Reel", "YouTube Short", "Snapchat Spotlight"] else 0.15
        )
        reach_hashtags_pct = post.metrics.reach_from_hashtags_pct if post.metrics.reach_from_hashtags_pct is not None else 0.05

        row = {
            "platform": platform_val,
            "username": profile.username,
            "country": profile.country,
            "total_followers": profile.total_followers,
            "total_following": profile.total_following,
            "total_media_posts": profile.total_media_posts,
            "account_age_years": profile.account_age_years,
            "posting_frequency_per_week": profile.posting_frequency_per_week,
            "follower_growth_rate_30d": profile.follower_growth_rate_30d,
            "account_bio_has_link": profile.account_bio_has_link,
            "media_type": media_type_val,
            "category": category_val,
            "categorization": cat_str,
            "account_category": acc_cat_str,
            "caption_length_chars": post.caption_length_chars,
            "hashtags_count": post.hashtags_count,
            "mentions_count": post.mentions_count,
            "has_call_to_action": post.has_call_to_action,
            "video_duration_seconds": post.video_duration_seconds,
            "carousel_slide_count": post.carousel_slide_count,
            "posted_day_of_week": post.posted_day_of_week,
            "posted_hour_of_day": post.posted_hour_of_day,
            "is_weekend": 1.0 if post.posted_day_of_week in ["Saturday", "Sunday"] else 0.0,
            "is_peak_posting_hour": 1.0 if post.posted_hour_of_day in [11, 12, 13, 18, 19, 20, 21] else 0.0,
            "top_country": post.demographics.top_country,
            "secondary_country": post.demographics.secondary_country,
            "primary_age_group": post.demographics.primary_age_group,
            "gender_female_pct": post.demographics.gender_female_pct,
            "gender_male_pct": post.demographics.gender_male_pct,
            "audience_activity_score": post.demographics.audience_activity_score,
            "per_media_likes": likes,
            "per_media_comments": comments,
            "per_media_shares": shares,
            "per_media_saves": saves,
            "per_media_video_views": video_views,
            "per_media_completion_rate": completion_rate,
            "reach_from_home_pct": reach_home_pct,
            "reach_from_explore_pct": reach_explore_pct,
            "reach_from_hashtags_pct": reach_hashtags_pct,
            "reach_from_other_pct": 0.05,
        }

        df_single = compute_derived_metrics(pd.DataFrame([row]))
        X_single = df_single[ALL_DIAGNOSTIC_FEATURE_COLUMNS]

        raw_point_reach = reach_pipeline.predict(X_single)[0]
        clean_point_reach = float(np.nan_to_num(raw_point_reach, nan=100.0, posinf=1e10, neginf=100.0))
        point_reach = max(int(np.round(clean_point_reach)), 100)

        raw_point_imp = imp_pipeline.predict(X_single)[0]
        clean_point_imp = float(np.nan_to_num(raw_point_imp, nan=100.0, posinf=1e10, neginf=100.0))
        point_imp = max(int(np.round(clean_point_imp)), point_reach)

        total_eng = likes + comments + shares + saves
        proj_er = float(total_eng / point_reach) if point_reach > 0 else 0.0
        proj_save = float(saves / point_imp) if point_imp > 0 else 0.0
        proj_share = float(shares / point_reach) if point_reach > 0 else 0.0
        virality = float(shares / (likes + 1.0))
        virality_momentum = (shares / (likes + 1.0)) * (2.0 if media_type_val == "Reel" else 1.0)
        interaction_density = (comments + shares + saves) / (likes + 1.0)

    # Invariant check
    point_imp = max(point_imp, point_reach)

    # Epistemic Out-Of-Distribution (OOD) Extrapolation Check
    is_ood = False
    ood_reasons = []
    if followers > 25_000_000 or followers < 100:
        is_ood = True
        ood_reasons.append("Follower scale is outside typical training bounds")
    if virality_momentum > 15.0:
        is_ood = True
        ood_reasons.append("Virality momentum exceeds 99th percentile of observed posts")
    if interaction_density > 4.0:
        is_ood = True
        ood_reasons.append("Unusual comment/share/save to like ratio")

    if is_ood:
        uncertainty_rating = f"⚠️ Extrapolation Alert: {', '.join(ood_reasons)}"
    else:
        uncertainty_rating = "✅ Calibrated (In-Distribution, High Confidence)"

    coverage_label = f"{int(calibrated_level * 100)}% Mondrian Conformal Coverage"

    # Conformal calibrated prediction intervals:
    # [exp(log(1 + y) - q) - 1,  exp(log(1 + y) + q) - 1]
    log_reach = np.log1p(point_reach)
    log_reach_lower = np.clip(log_reach - q_reach, 0.0, 30.0)
    log_reach_upper = np.clip(log_reach + q_reach, 0.0, 30.0)
    reach_lower = max(int(np.expm1(log_reach_lower)), 50)
    reach_upper = max(int(np.expm1(log_reach_upper)), reach_lower)

    log_imp = np.log1p(point_imp)
    log_imp_lower = np.clip(log_imp - q_imp, 0.0, 30.0)
    log_imp_upper = np.clip(log_imp + q_imp, 0.0, 30.0)
    imp_lower = max(int(np.expm1(log_imp_lower)), reach_lower)
    imp_upper = max(int(np.expm1(log_imp_upper)), max(reach_upper, imp_lower))

    reach_ci = ConfidenceInterval(
        lower=reach_lower,
        point_estimate=point_reach,
        upper=reach_upper,
        confidence_level=calibrated_level,
        level=coverage_label
    )

    imp_ci = ConfidenceInterval(
        lower=imp_lower,
        point_estimate=point_imp,
        upper=imp_upper,
        confidence_level=calibrated_level,
        level=coverage_label
    )

    # Also compute 90% Mondrian conformal interval from calibrated q90 quantiles
    q_reach_90 = tier_reach_meta.get("q90", reach_eval.get("conformal_quantile_90", 0.45))
    q_imp_90 = tier_imp_meta.get("q90", imp_eval.get("conformal_quantile_90", 0.45))

    reach_lower_90 = max(int(np.expm1(np.clip(log_reach - q_reach_90, 0.0, 30.0))), 50)
    reach_upper_90 = max(int(np.expm1(np.clip(log_reach + q_reach_90, 0.0, 30.0))), reach_lower_90)
    imp_lower_90 = max(int(np.expm1(np.clip(log_imp - q_imp_90, 0.0, 30.0))), reach_lower_90)
    imp_upper_90 = max(int(np.expm1(np.clip(log_imp + q_imp_90, 0.0, 30.0))), max(reach_upper_90, imp_lower_90))

    reach_ci_90 = ConfidenceInterval(
        lower=reach_lower_90,
        point_estimate=point_reach,
        upper=reach_upper_90,
        confidence_level=0.90,
        level="90% Mondrian Conformal Coverage"
    )
    imp_ci_90 = ConfidenceInterval(
        lower=imp_lower_90,
        point_estimate=point_imp,
        upper=imp_upper_90,
        confidence_level=0.90,
        level="90% Mondrian Conformal Coverage"
    )

    # Virality tier
    if virality >= 0.20 or (point_reach > profile.total_followers * 2.0):
        v_tier = "🚀 Explosive / Viral"
    elif virality >= 0.10:
        v_tier = "🔥 High Virality"
    elif virality >= 0.04:
        v_tier = "📈 Moderate Distribution"
    else:
        v_tier = "⚖️ Standard Follower Reach"

    # Optimization tips
    tips = []
    if post.media_type.value == "Static Image":
        tips.append("💡 Consider converting this post into a Carousel (+45% saves) or short Reel (+120% explore reach).")
    if post.categorization.value != "Educational / How-To" and proj_save < 0.02:
        tips.append("💡 Add a bookmarkable framework or summary slide to boost your Save Rate above the 2.0% benchmark.")
    if proj_share < 0.03:
        tips.append("💡 Incorporate a shareable punchline, surprising stat, or relatable meme to trigger direct message shares.")
    if post.media_type.value == "Reel":
        tips.append("✅ Reels are currently receiving top algorithmic distribution across Explore & Reels tabs.")

    return SimulationPrediction(
        projected_reach=reach_ci,
        projected_impressions=imp_ci,
        projected_reach_90=reach_ci_90,
        projected_impressions_90=imp_ci_90,
        reach_90_ci=reach_ci_90,
        impressions_90_ci=imp_ci_90,
        projected_engagement_rate=round(proj_er * 100, 2),
        projected_save_rate=round(proj_save * 100, 2),
        projected_share_rate=round(proj_share * 100, 2),
        virality_score=round(virality, 4),
        virality_tier=v_tier,
        calibration_tier=tier_label,
        uncertainty_rating=uncertainty_rating,
        prediction_interval_coverage=coverage_label,
        optimization_tips=tips,
        feature_explanations=feature_explanations
    )
