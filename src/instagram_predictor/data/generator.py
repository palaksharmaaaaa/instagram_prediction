import os
from pathlib import Path
import numpy as np
import pandas as pd
from ..config import settings


CATEGORIES = [
    "Sports",
    "Health & Fitness",
    "Finance & Business",
    "Fashion & Beauty",
    "Science & Technology",
    "Travel & Events",
    "Food & Dining",
    "Music & Entertainment",
    "Education & Careers"
]

MEDIA_TYPES = ["Reel", "Carousel", "Static Image", "Story"]

CONTENT_STYLES = [
    "Educational / How-To",
    "Entertaining / Trend",
    "Promotional / Sponsored",
    "Behind The Scenes",
    "Inspirational / Storytelling"
]

COUNTRIES = ["US", "IN", "BR", "GB", "ES", "CA", "FR", "AU", "DE", "IT", "MX", "AE"]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45+"]


def generate_enterprise_dataset(
    num_profiles: int = 250,
    posts_per_profile: int = 3,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic, multi-dimensional Instagram dataset spanning account-level
    and per-media metrics (reach, impressions, likes, comments, shares, saves, media type,
    categories, demographics, and ratios).
    """
    rng = np.random.default_rng(random_seed)

    usernames_seed = [
        ("cristiano", "Cristiano Ronaldo", "Sports", "ES", 465000000, 520, 3328, True),
        ("kyliejenner", "Kylie Jenner", "Fashion & Beauty", "US", 356000000, 98, 6921, True),
        ("leomessi", "Leo Messi", "Sports", "ES", 347000000, 290, 875, True),
        ("selenagomez", "Selena Gomez", "Music & Entertainment", "US", 334000000, 210, 1835, True),
        ("therock", "Dwayne Johnson", "Health & Fitness", "US", 327000000, 580, 6660, True),
        ("virat.kohli", "Virat Kohli", "Sports", "IN", 206000000, 240, 1390, True),
        ("nike", "Nike", "Sports", "US", 228000000, 120, 932, True),
        ("natgeo", "National Geographic", "Travel & Events", "US", 232000000, 140, 10002, True),
        ("mrbeast", "MrBeast", "Entertainment / Trend", "US", 58000000, 340, 420, True),
        ("hubermanlab", "Andrew Huberman", "Health & Fitness", "US", 5200000, 410, 890, True),
        ("aliabdaal", "Ali Abdaal", "Education & Careers", "GB", 3100000, 320, 640, True),
        ("garyvee", "Gary Vaynerchuk", "Finance & Business", "US", 10200000, 4500, 12400, True),
        ("mkbhd", "Marques Brownlee", "Science & Technology", "US", 4800000, 380, 1200, True),
        ("gordonramsay", "Gordon Ramsay", "Food & Dining", "GB", 14500000, 480, 3890, True)
    ]

    records = []

    for i in range(num_profiles):
        if i < len(usernames_seed):
            u, fn, cat, country, followers, following, posts, verified = usernames_seed[i]
        else:
            cat = rng.choice(CATEGORIES)
            country = rng.choice(COUNTRIES)
            u = f"creator_{cat[:4].lower()}_{i}"
            fn = f"Creator {i} ({cat})"
            # Log-normal follower distribution (micro, mid, macro, celebrity)
            followers = int(np.exp(rng.uniform(np.log(10_000), np.log(80_000_000))))
            following = int(rng.integers(50, 4000))
            posts = int(rng.integers(100, 5000))
            verified = bool(followers > 1_000_000 and rng.random() > 0.3)

        profile_categories = set([cat])
        profile_records = []

        for p_idx in range(posts_per_profile):
            media_type = rng.choice(MEDIA_TYPES, p=[0.45, 0.30, 0.20, 0.05])
            # 1. Single primary category for this media post
            post_cat = cat if rng.random() > 0.30 else rng.choice(CATEGORIES)
            profile_categories.add(post_cat)

            # 2. Multi-label content styles / categorizations for this media post (1 to 3 styles)
            num_styles = int(rng.choice([1, 2, 3], p=[0.45, 0.40, 0.15]))
            post_styles = list(rng.choice(CONTENT_STYLES, size=num_styles, replace=False))
            style_str = ", ".join(post_styles)

            age_group = rng.choice(AGE_GROUPS, p=[0.35, 0.40, 0.18, 0.07])
            female_pct = float(np.clip(rng.normal(0.52, 0.15), 0.10, 0.90))

            # Base engagement rate modulated by follower tier (smaller accounts have higher % ER)
            base_er = float(np.clip(0.045 - 0.003 * np.log10(max(followers, 1000)), 0.008, 0.10))

            # Media type multiplier on Reach and Virality
            media_reach_mult = {
                "Reel": rng.uniform(1.2, 2.5),
                "Carousel": rng.uniform(0.7, 1.3),
                "Static Image": rng.uniform(0.5, 0.9),
                "Story": rng.uniform(0.15, 0.35)
            }[media_type]

            # Likes, comments, shares, saves
            expected_likes = max(int(followers * base_er * rng.uniform(0.7, 1.4)), 10)
            expected_comments = max(int(expected_likes * rng.uniform(0.015, 0.08)), 1)
            
            # Shares are boosted heavily on Reels and Educational Carousels
            is_edu = "Educational / How-To" in post_styles
            share_boost = 2.5 if media_type == "Reel" or is_edu else 1.0
            expected_shares = max(int(expected_likes * rng.uniform(0.02, 0.12) * share_boost), 1)

            # Saves are boosted heavily on Carousels and Educational content
            save_boost = 3.0 if media_type == "Carousel" or is_edu else 1.0
            expected_saves = max(int(expected_likes * rng.uniform(0.01, 0.10) * save_boost), 1)

            # Reach: driven by followers, media type, and engagement velocity
            reach = int(
                (0.35 * followers + 0.65 * (expected_likes * 14 + expected_shares * 28 + expected_saves * 15))
                * media_reach_mult
                * rng.uniform(0.90, 1.12)
            )
            reach = max(reach, expected_likes + expected_comments + expected_shares + expected_saves + 50)

            # Impressions: reach * frequency (typically 1.15 to 1.85)
            frequency = rng.uniform(1.18, 1.80) if media_type != "Story" else rng.uniform(1.02, 1.15)
            impressions = int(reach * frequency)

            # Account-level granular attributes
            acc_age = round(float(rng.uniform(1.0, 12.0)), 1)
            posting_freq = round(float(rng.uniform(1.0, 14.0)), 1)
            growth_rate = round(float(rng.uniform(-0.02, 0.15)), 3)
            has_bio_link = bool(rng.random() > 0.35)

            # Post content semantics & structure
            caption_len = int(rng.integers(50, 2200))
            hashtags = int(rng.integers(0, 30))
            mentions = int(rng.integers(0, 8))
            has_cta = bool(rng.random() > 0.40)
            video_sec = round(float(rng.uniform(5.0, 90.0)), 1) if media_type == "Reel" else 0.0
            slide_count = int(rng.integers(2, 11)) if media_type == "Carousel" else 1

            # Temporal & scheduling
            day_of_week = rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            hour_of_day = int(rng.integers(0, 24))
            is_wknd = int(day_of_week in ["Saturday", "Sunday"])
            is_peak = int(hour_of_day in [11, 12, 13, 18, 19, 20, 21])

            # Audience demographics
            sec_country = rng.choice([c for c in COUNTRIES if c != country])
            audience_activity = round(float(rng.uniform(0.40, 0.98)), 3)

            # Algorithmic discovery breakdown
            explore_pct = round(float(np.clip(0.10 + 0.35 * (media_type == "Reel") + 0.20 * (expected_shares / (expected_likes + 1)), 0.05, 0.75)), 3)
            hashtag_pct = round(float(np.clip(0.02 + 0.004 * min(hashtags, 20), 0.01, 0.18)), 3)
            other_pct = round(float(rng.uniform(0.02, 0.10)), 3)
            home_pct = round(max(0.10, 1.0 - (explore_pct + hashtag_pct + other_pct)), 3)

            video_views = int(reach * rng.uniform(1.1, 1.8)) if media_type == "Reel" else 0
            completion_rate = round(float(rng.uniform(0.20, 0.85)), 3) if media_type == "Reel" else 0.0

            profile_records.append({
                "username": u,
                "full_name": fn,
                "country": country,
                "total_followers": followers,
                "total_following": following,
                "total_media_posts": posts,
                "is_verified": verified,
                "account_category": cat,
                "account_age_years": acc_age,
                "posting_frequency_per_week": posting_freq,
                "follower_growth_rate_30d": growth_rate,
                "account_bio_has_link": has_bio_link,
                "post_id": f"{u}_p{p_idx + 1}",
                "media_type": media_type,
                "category": post_cat,
                "categorization": style_str,
                "categorizations": post_styles,
                "caption_length_chars": caption_len,
                "hashtags_count": hashtags,
                "mentions_count": mentions,
                "has_call_to_action": has_cta,
                "video_duration_seconds": video_sec,
                "carousel_slide_count": slide_count,
                "posted_day_of_week": day_of_week,
                "posted_hour_of_day": hour_of_day,
                "is_weekend": is_wknd,
                "is_peak_posting_hour": is_peak,
                "top_country": country,
                "secondary_country": sec_country,
                "primary_age_group": age_group,
                "gender_female_pct": round(female_pct, 3),
                "gender_male_pct": round(1.0 - female_pct, 3),
                "audience_activity_score": audience_activity,
                "per_media_likes": expected_likes,
                "per_media_comments": expected_comments,
                "per_media_shares": expected_shares,
                "per_media_saves": expected_saves,
                "per_media_video_views": video_views,
                "per_media_completion_rate": completion_rate,
                "reach_from_home_pct": home_pct,
                "reach_from_explore_pct": explore_pct,
                "reach_from_hashtags_pct": hashtag_pct,
                "reach_from_other_pct": other_pct,
                "per_media_reach": reach,
                "per_media_impressions": impressions
            })

        # Set profile-wide distinct categories on all records for this creator
        distinct_cats = sorted(list(profile_categories))
        distinct_cats_str = ", ".join(distinct_cats)
        for rec in profile_records:
            rec["account_categories"] = distinct_cats
            rec["account_categories_str"] = distinct_cats_str
            records.append(rec)

    df = pd.DataFrame(records)
    return df


def ensure_dataset_exists(force_recreate: bool = False) -> pd.DataFrame:
    """
    Ensures the raw dataset exists at settings.RAW_DATA_PATH.
    If not present or forced, creates and saves it.
    """
    csv_path = settings.RAW_DATA_PATH
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists() or force_recreate:
        df = generate_enterprise_dataset()
        df.to_csv(csv_path, index=False)
        return df

    return pd.read_csv(csv_path)
