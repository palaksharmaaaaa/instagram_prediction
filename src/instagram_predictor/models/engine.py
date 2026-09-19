from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from ..config import settings
from ..data import ALL_FEATURE_COLUMNS, compute_derived_metrics
from ..schemas import ConfidenceInterval, SimulationPrediction, PostInput, ProfileInput
from .registry import get_reach_pipeline, get_impressions_pipeline, get_model_metadata


def predict_batch(
    df: pd.DataFrame,
    predict_reach: bool = True,
    predict_impressions: bool = True
) -> pd.DataFrame:
    """
    Vectorized batch inference on a DataFrame with feature derivation and invariant enforcement.
    """
    if df.empty:
        return df.copy()

    # Ensure all derived features exist
    df = compute_derived_metrics(df)
    X = df[ALL_FEATURE_COLUMNS]

    if predict_reach:
        reach_model = get_reach_pipeline()
        raw_reach = reach_model.predict(X)
        clean_reach = np.nan_to_num(raw_reach, nan=100.0, posinf=1e10, neginf=100.0)
        df["predicted_reach"] = np.maximum(np.round(clean_reach).astype(int), 100)

    if predict_impressions:
        imp_model = get_impressions_pipeline()
        raw_imp = imp_model.predict(X)
        clean_imp = np.nan_to_num(raw_imp, nan=100.0, posinf=1e10, neginf=100.0)
        df["predicted_impressions"] = np.maximum(np.round(clean_imp).astype(int), 100)

    # Invariant guardrail: reach cannot exceed impressions
    if predict_reach and predict_impressions:
        df["predicted_impressions"] = np.maximum(df["predicted_impressions"], df["predicted_reach"])

    return df


def simulate_post_performance(
    profile: ProfileInput,
    post: PostInput,
    confidence_level: float = 0.80
) -> SimulationPrediction:
    """
    Simulates projected post performance for a given profile and media scenario,
    generating calibrated Conformal Prediction intervals and actionable creator recommendations.
    """
    reach_pipeline = get_reach_pipeline()
    imp_pipeline = get_impressions_pipeline()
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

    # Retrieve calibrated Mondrian Conformal Prediction quantiles
    reach_eval = meta.get("evaluation", {}).get("reach", {})
    imp_eval = meta.get("evaluation", {}).get("impressions", {})

    calibrated_level = 0.80 if confidence_level <= 0.85 else 0.90
    q_key = "q80" if calibrated_level == 0.80 else "q90"
    global_q_reach = reach_eval.get("conformal_quantile_80" if calibrated_level == 0.80 else "conformal_quantile_90", 0.35)
    global_q_imp = imp_eval.get("conformal_quantile_80" if calibrated_level == 0.80 else "conformal_quantile_90", 0.35)

    tier_reach_meta = reach_eval.get("tier_conformal_quantiles", {}).get(follower_tier, {})
    tier_imp_meta = imp_eval.get("tier_conformal_quantiles", {}).get(follower_tier, {})

    q_reach = tier_reach_meta.get(q_key, global_q_reach)
    q_imp = tier_imp_meta.get(q_key, global_q_imp)

    # Baseline metrics based on creator scale
    if post.metrics:
        likes = post.metrics.likes
        comments = post.metrics.comments
        shares = post.metrics.shares
        saves = post.metrics.saves
        video_views = post.metrics.video_views if post.metrics.video_views is not None else (int(likes * 5.0) if post.media_type.value == "Reel" else 0)
        completion_rate = post.metrics.completion_rate if post.metrics.completion_rate is not None else (0.45 if post.media_type.value == "Reel" else 0.0)
        reach_home_pct = post.metrics.reach_from_home_pct if post.metrics.reach_from_home_pct is not None else 0.55
        reach_explore_pct = post.metrics.reach_from_explore_pct if post.metrics.reach_from_explore_pct is not None else (0.35 if post.media_type.value == "Reel" else 0.15)
        reach_hashtags_pct = post.metrics.reach_from_hashtags_pct if post.metrics.reach_from_hashtags_pct is not None else 0.05
    else:
        # Dynamic baseline modulation for unpublished post (ML-01)
        base_er = float(np.clip(0.045 - 0.003 * np.log10(max(followers, 100)), 0.015, 0.08))

        media_type_val = post.media_type.value if hasattr(post.media_type, "value") else str(post.media_type)
        if media_type_val == "Reel":
            like_mult = 1.15
            share_mult = 2.20
            save_mult = 1.05
            comment_mult = 1.10
        elif media_type_val == "Carousel":
            slide_bonus = 1.0 + 0.03 * min(max(post.carousel_slide_count, 1), 10)
            like_mult = 1.05
            share_mult = 1.15
            save_mult = 2.40 * slide_bonus
            comment_mult = 1.25
        elif media_type_val == "Story":
            like_mult = 0.70
            share_mult = 0.50
            save_mult = 0.40
            comment_mult = 0.60
        else:  # Static Image
            like_mult = 0.95
            share_mult = 0.80
            save_mult = 0.90
            comment_mult = 0.90

        style_list = post.categorizations if post.categorizations else ([post.categorization] if post.categorization else [])
        style_tokens = " ".join([s.value if hasattr(s, "value") else str(s) for s in style_list]).lower()

        if "educational" in style_tokens:
            save_mult *= 1.80
            share_mult *= 1.25
            comment_mult *= 1.10
        if "entertaining" in style_tokens:
            share_mult *= 1.45
            like_mult *= 1.10
        if "promotional" in style_tokens:
            like_mult *= 0.85
            share_mult *= 0.75
            save_mult *= 0.80
            comment_mult *= 0.85
        if "behind the scenes" in style_tokens:
            comment_mult *= 1.30
            like_mult *= 1.05
        if "inspirational" in style_tokens:
            save_mult *= 1.25
            share_mult *= 1.25

        if post.has_call_to_action:
            comment_mult *= 1.30
            save_mult *= 1.25
            like_mult *= 1.05

        if 150 <= post.caption_length_chars <= 900:
            caption_mult = 1.08
        elif post.caption_length_chars > 900:
            caption_mult = 1.12 if ("educational" in style_tokens or "inspirational" in style_tokens) else 1.03
        else:
            caption_mult = 0.95

        if 3 <= post.hashtags_count <= 15:
            hashtag_mult = 1.10
        elif post.hashtags_count == 0:
            hashtag_mult = 0.90
        elif post.hashtags_count > 20:
            hashtag_mult = 0.95
        else:
            hashtag_mult = 1.02

        is_peak = post.posted_hour_of_day in [11, 12, 13, 18, 19, 20, 21]
        is_wknd = post.posted_day_of_week in ["Saturday", "Sunday"]

        if is_peak:
            like_mult *= 1.15
            comment_mult *= 1.10
        if is_wknd:
            share_mult *= 1.15
            like_mult *= 1.05

        likes = max(int(followers * base_er * like_mult * caption_mult * hashtag_mult), 10)
        comments = max(int(likes * 0.04 * comment_mult), 1)
        shares = max(int(likes * 0.05 * share_mult), 1)
        saves = max(int(likes * 0.03 * save_mult), 1)

        if media_type_val == "Reel":
            video_views = max(int(likes * 5.5), 10)
            completion_rate = float(np.clip(0.65 - 0.003 * post.video_duration_seconds, 0.25, 0.85))
            reach_explore_pct = float(np.clip(0.35 + 0.15 * (shares / (likes + 1.0)), 0.20, 0.70))
        else:
            video_views = 0
            completion_rate = 0.0
            reach_explore_pct = float(np.clip(0.15 + 0.10 * (shares / (likes + 1.0)), 0.05, 0.40))

        reach_hashtags_pct = float(np.clip(0.02 + 0.005 * min(post.hashtags_count, 20), 0.01, 0.18))
        reach_home_pct = float(max(0.10, 1.0 - (reach_explore_pct + reach_hashtags_pct + 0.05)))

    is_reel = float(post.media_type.value == "Reel")
    is_carousel = float(post.media_type.value == "Carousel")
    total_eng = likes + comments + shares + saves
    virality_momentum = (shares / (likes + 1.0)) * (1.0 + is_reel)
    interaction_density = (comments + shares + saves) / (likes + 1.0)

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

    row = {
        "username": profile.username,
        "country": profile.country,
        "total_followers": profile.total_followers,
        "total_following": profile.total_following,
        "total_media_posts": profile.total_media_posts,
        "account_age_years": profile.account_age_years,
        "posting_frequency_per_week": profile.posting_frequency_per_week,
        "follower_growth_rate_30d": profile.follower_growth_rate_30d,
        "account_bio_has_link": profile.account_bio_has_link,
        "media_type": post.media_type.value,
        "category": post.category.value,
        "categorization": ", ".join([c.value if hasattr(c, "value") else str(c) for c in post.categorizations]) if post.categorizations else (post.categorization.value if post.categorization else "Entertaining / Trend"),
        "account_category": profile.account_category.value if hasattr(profile.account_category, "value") else str(profile.account_category),
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
    }

    df_single = compute_derived_metrics(pd.DataFrame([row]))
    X_single = df_single[ALL_FEATURE_COLUMNS]

    # Model point estimates (transformed pipeline outputs in original scale)
    raw_point_reach = reach_pipeline.predict(X_single)[0]
    clean_point_reach = float(np.nan_to_num(raw_point_reach, nan=100.0, posinf=1e10, neginf=100.0))
    point_reach = max(int(np.round(clean_point_reach)), 100)

    raw_point_imp = imp_pipeline.predict(X_single)[0]
    clean_point_imp = float(np.nan_to_num(raw_point_imp, nan=100.0, posinf=1e10, neginf=100.0))
    point_imp = max(int(np.round(clean_point_imp)), point_reach)

    # Invariant check
    point_imp = max(point_imp, point_reach)

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

    # Derived rates
    proj_er = float(total_eng / point_reach) if point_reach > 0 else 0.0
    proj_save = float(saves / point_imp) if point_imp > 0 else 0.0
    proj_share = float(shares / point_reach) if point_reach > 0 else 0.0
    virality = float(shares / (likes + 1.0))

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
        projected_engagement_rate=round(proj_er * 100, 2),
        projected_save_rate=round(proj_save * 100, 2),
        projected_share_rate=round(proj_share * 100, 2),
        virality_score=round(virality, 4),
        virality_tier=v_tier,
        calibration_tier=tier_label,
        uncertainty_rating=uncertainty_rating,
        prediction_interval_coverage=coverage_label,
        optimization_tips=tips
    )
