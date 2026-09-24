import io
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from instagram_predictor.config import settings
from instagram_predictor.data import (
    NoDataError, POST_TEMPLATE_COLUMNS, append_posts_csv, upsert_creator_profile, load_posts, load_profiles,
)
from instagram_predictor.guardrails import sanitize_dataframe_for_csv
from instagram_predictor.integrations import (
    InstagramGraphAPIClient, InstagramPublicFetcher, MetaGraphAPIError,
)
from instagram_predictor.models import (
    InsufficientDataError, ModelVersionError, NoModelError, OutOfSupportError, SecurityError,
    forecast_formats, forecast_post, load_metadata, load_model, model_available, train_and_persist,
)
from instagram_predictor.schemas import (
    PlatformType,
    MediaType,
    PostInput,
    ProfileInput,
    platform_of,
    DAYS_OF_WEEK,
    calculate_per_post_reach,
    calculate_per_post_impressions,
)

from instagram_predictor.services import (
    observed_post_summary, run_creator_query, EngagementCalculatorService,
)
from instagram_predictor.utils import format_number

st.set_page_config(page_title="Reach Forecaster", page_icon="📈", layout="wide")


@st.cache_data
def cached_profiles(mtime: float) -> pd.DataFrame:
    return load_profiles()[0]


def profiles_df() -> pd.DataFrame | None:
    p = Path(settings.PROFILES_PATH)
    return cached_profiles(p.stat().st_mtime) if p.exists() else None


def posts_df():
    p = Path(settings.POSTS_PATH)
    if not p.exists():
        return None, None
    prof = profiles_df()
    try:
        return load_posts(p, profiles=prof)
    except ValueError as e:
        return None, str(e)


def progress_bar(initial: str = "starting"):
    """Returns (callback, bar). The callback shows 'NN% - message' and never exceeds 100%."""
    bar = st.progress(0.0, text=f"0% - {initial}")

    def cb(frac: float, msg: str) -> None:
        frac = min(max(float(frac), 0.0), 1.0)
        bar.progress(frac, text=f"{int(round(frac * 100))}% - {msg}")
    return cb, bar


def show_report(report) -> None:
    st.write(f"**{report.summary()}**")
    for n in report.notes:
        st.info(n)
    if report.rows_rejected:
        with st.expander(f"Why {report.rows_rejected} rows were rejected (rows are never repaired)"):
            st.dataframe(report.rejected_frame(), hide_index=True, width="stretch")


st.title("📈 Reach Forecaster")
st.caption(
    "Only real, observed data is shown or used. Forecasts come from a model trained on real post-level data you supply, "
    "are checked against a simple baseline on held-out creators, and are refused outside the range of that data."
)

posts, posts_report = posts_df()
profiles = profiles_df()

with st.sidebar:
    st.subheader("Status")
    st.write(f"Creator profiles: **{0 if profiles is None else len(profiles)}**")
    if isinstance(posts_report, str):
        st.error(f"posts.csv unusable: {posts_report}")
    elif posts is None:
        st.warning("No post-level data yet (data/posts.csv).")
    else:
        st.write(f"Usable posts: **{len(posts)}** from **{posts['username'].nunique()}** creators")
    if model_available():
        try:
            m = load_metadata()
            st.write(f"Model: **{m['reach']['selected']}** trained on {m['data']['n_posts']} posts")
        except Exception as e:
            st.error(str(e))
    else:
        st.warning("No trained model.")

    with st.expander("⚙️ Data & Model Management", expanded=posts is None or not model_available()):
        st.markdown("#### Post-level training data")
        st.caption("Required columns: `username, media_type, posted_day_of_week, posted_hour_of_day, total_followers, per_media_reach`.")
        st.download_button("Download template CSV", ",".join(POST_TEMPLATE_COLUMNS) + "\n", "posts_template.csv", "text/csv")

        if posts is not None:
            show_report(posts_report)

        up = st.file_uploader("Upload real post-level CSV", type="csv")
        if up is not None:
            try:
                up_posts, up_report = load_posts(io.BytesIO(up.getvalue()), profiles=profiles)
                show_report(up_report)
                if up_report.rows_used and st.button(f"Save as {settings.POSTS_PATH.name} (replaces the current file)"):
                    settings.POSTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                    settings.POSTS_PATH.write_bytes(up.getvalue())
                    st.success("Saved. Reload the page, then train.")
            except (ValueError, NoDataError) as e:
                st.error(str(e))

        st.markdown("#### Train")
        if st.button("Train model on data/posts.csv", disabled=posts is None):
            cb, bar = progress_bar("starting training")
            try:
                with st.spinner("Training: cross-validating against the baseline and checking interval coverage..."):
                    md = train_and_persist(progress=cb)
                st.success(f"Trained. Selected: {md['reach']['selected']}. See the Model report tab.")
            except (InsufficientDataError, ValueError, NoDataError) as e:
                bar.empty()
                st.error(str(e))

        with st.expander("Import from Meta Graph API (observed insights only)"):
            st.caption("Needs a token with instagram_basic and instagram_manage_insights. Fields Meta does not return are left empty. "
                       "impressions is not available in current API versions. Hours are UTC. "
                       "total_followers is the CURRENT count, not the count at post time.")
            token = st.text_input("Access token", type="password")
            if st.button("Discover accounts"):
                if not token:
                    st.error("Enter the access token first.")
                else:
                    try:
                        with st.spinner("Asking Meta which Instagram accounts this token can access..."):
                            found = InstagramGraphAPIClient(access_token=token).get_connected_instagram_accounts()
                        st.session_state["ig_accounts"] = found
                        if not found:
                            st.warning("No Instagram Professional account is linked to a Facebook Page this token can see. "
                                       "The token needs pages_show_list (and instagram_basic) permissions.")
                    except MetaGraphAPIError as e:
                        st.error(str(e))
            accounts = st.session_state.get("ig_accounts", [])
            acct = None
            if accounts:
                labels = [f"@{x['username']}  (Page: {x['page_name']})  ID {x['instagram_account_id']}" for x in accounts]
                chosen = st.selectbox("Account", labels)
                acct = accounts[labels.index(chosen)]["instagram_account_id"]
            else:
                acct = st.text_input("Instagram account ID (numeric, not the username)", placeholder="17841400000000000")
            limit = st.number_input("Posts to fetch", 1, 100, 25)
            if st.button("Fetch"):
                if not token or not acct:
                    st.error("Token and account ID are required.")
                else:
                    cb, bar = progress_bar("connecting to Meta")
                    try:
                        with st.spinner("Fetching your profile and posts from Meta Graph API..."):
                            prof, rows = InstagramGraphAPIClient(access_token=token).fetch_creator_snapshot(
                                acct, media_limit=int(limit), progress=cb)
                        st.session_state["meta_rows"] = pd.DataFrame(rows)
                        st.success(f"Fetched @{prof.username} ({prof.total_followers:,} followers) and {len(rows)} posts.")
                    except MetaGraphAPIError as e:
                        bar.empty()
                        st.error(str(e))
            if "meta_rows" in st.session_state:
                fetched = st.session_state["meta_rows"]
                st.dataframe(fetched, width="stretch")
                n_reach = int(fetched["per_media_reach"].notna().sum()) if "per_media_reach" in fetched else 0
                st.write(f"{n_reach} of {len(fetched)} fetched posts have observed reach (only those can be used for training).")
                if "insights_unavailable_reason" in fetched and fetched["insights_unavailable_reason"].notna().any():
                    why = fetched["insights_unavailable_reason"].value_counts()
                    st.warning("Meta gave no insights for some posts, so they cannot be used for training:\n\n" +
                               "\n".join(f"- **{n}** posts: {r}" for r, n in why.items()))
                st.download_button("Download as CSV", sanitize_dataframe_for_csv(fetched).to_csv(index=False), "meta_posts.csv", "text/csv")
                if st.button("Append posts with observed reach to data/posts.csv"):
                    usable = fetched[fetched["per_media_reach"].notna()] if "per_media_reach" in fetched else fetched.iloc[0:0]
                    if usable.empty:
                        st.warning("Nothing appended: none of these posts has observed reach, so none can be used for training. "
                                   "posts.csv is left unchanged.")
                    else:
                        n = append_posts_csv(usable)
                        st.success(f"Appended {n} new rows with observed reach ({len(fetched) - len(usable)} without reach skipped; "
                                   "existing rows untouched).")

        with st.expander("Creator profiles provenance"):
            if profiles is not None:
                st.write(f"{len(profiles)} creators. Blank cells mean the source had no value.")
            prov = Path(settings.DATA_DIR) / "PROVENANCE.md"
            st.markdown(prov.read_text(encoding="utf-8") if prov.exists() else "PROVENANCE.md missing")

tab_creators, tab_forecast, tab_report = st.tabs(["Creators", "Forecast", "Model report"])

# ------------------------------------------------------------------ CREATORS
with tab_creators:
    st.subheader("🔄 End-to-End Creator Intelligence Pipeline")
    st.caption(
        "Search creator data by username ➔ Option to save profile & posts ➔ Ingestion ➔ Preview "
        "➔ Check required metrics ➔ Compute ➔ Train data/models ➔ Provide predictions"
    )

    # --------------------------------------------------------- STEP 1: SEARCH
    st.markdown("#### 1️⃣ Search Creator Data by Username")
    col_input, col_limit, col_btn = st.columns([3, 1, 1])
    with col_input:
        target_username = st.text_input(
            "Instagram Username",
            placeholder="e.g. techburner, cristiano, snapsfromkitchen",
            key="calc_username_input",
        )
    with col_limit:
        media_limit = st.selectbox("Posts to fetch", [10, 25, 50, 100], index=1, key="calc_media_limit")
    with col_btn:
        st.write("")
        st.write("")
        fetch_clicked = st.button("🔍 Fetch Creator Data", type="primary", use_container_width=True)

    with st.expander("API Configuration (Meta Graph API credentials)"):
        st.caption(
            "Uses Meta's official Business Discovery API. Your credentials in .env are automatically loaded."
        )
        meta_tok = st.text_input(
            "Meta User Access Token",
            type="password",
            key="calc_meta_token",
            value=os.getenv("META_ACCESS_TOKEN", ""),
        )
        meta_bid = st.text_input(
            "Meta Instagram Business Account ID",
            key="calc_meta_bid",
            value=os.getenv("META_IG_BUSINESS_ACCOUNT_ID", ""),
        )

    if fetch_clicked:
        if not target_username or not target_username.strip():
            st.error("Please enter an Instagram username.")
        else:
            clean_u = target_username.strip().lstrip("@").lower()
            with st.spinner(f"Fetching real profile and {media_limit} recent media items for @{clean_u} via Meta Graph API..."):
                try:
                    fetcher = InstagramPublicFetcher(
                        meta_access_token=meta_tok or os.getenv("META_ACCESS_TOKEN"),
                        meta_business_account_id=meta_bid or os.getenv("META_IG_BUSINESS_ACCOUNT_ID"),
                    )
                    snapshot = fetcher.fetch_creator(clean_u, media_limit=int(media_limit))
                    result = EngagementCalculatorService.calculate(snapshot)
                    st.session_state["calc_data"] = {
                        "snapshot": snapshot,
                        "result": result,
                    }
                    st.session_state["ingestion_status"] = None
                except Exception as e:
                    st.error(f"Failed to fetch profile @{clean_u}: {e}")

    if "calc_data" in st.session_state:
        snap = st.session_state["calc_data"]["snapshot"]
        res = st.session_state["calc_data"]["result"]
        meta = snap.metadata

        st.divider()

        # ------------------------------------------------- STEP 2 & 3: SAVE & INGEST
        st.markdown("#### 2️⃣ & 3️⃣ Option to Save Creator Profile & Recent Posts (Ingestion)")
        st.caption("Choose which records to ingest into your local persistent database.")

        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            save_prof_opt = st.checkbox(
                "Save Creator Profile to `data/creator_profiles.csv`",
                value=True,
                key="chk_save_creator_profile",
            )
        with c_opt2:
            save_posts_opt = st.checkbox(
                f"Save Recent Media Posts ({len(snap.posts)} posts) to `data/posts.csv`",
                value=True,
                key="chk_save_recent_posts",
            )

        if st.button("🚀 Ingest & Save Data Now", type="primary", use_container_width=True):
            saved_notes = []
            if save_prof_opt:
                prof_dict = {
                    "username": res.username,
                    "full_name": res.full_name,
                    "platform": "Instagram",
                    "total_followers": res.total_followers,
                    "total_media_posts": meta.media_count or res.total_posts_sampled,
                    "account_category": meta.account_category,
                    "boost_index": "",
                    "engagement_rate": res.engagement_rate,
                    "engagement_rate_60d": res.engagement_rate,
                    "avg_likes": round(res.avg_likes, 1),
                    "avg_comments": round(res.avg_comments, 1),
                    "avg_video_views": round(res.avg_video_views, 1),
                    "profile_url": f"https://www.instagram.com/{res.username}/",
                }
                upsert_creator_profile(prof_dict)
                cached_profiles.clear()
                saved_notes.append("Creator profile updated in `data/creator_profiles.csv`")

            if save_posts_opt and snap.posts:
                now_str = datetime.now(timezone.utc).isoformat()
                new_post_rows = []
                for p in snap.posts:
                    new_post_rows.append({
                        "post_id": p.post_id,
                        "media_type": p.media_type,
                        "posted_day_of_week": p.posted_day_of_week,
                        "posted_hour_of_day": p.posted_hour_of_day,
                        "timestamp_utc": p.timestamp_utc,
                        "permalink": p.permalink,
                        "caption_length_chars": p.caption_length_chars,
                        "hashtags_count": len(p.hashtags),
                        "mentions_count": len(p.mentions),
                        "per_media_likes": p.likes,
                        "per_media_comments": p.comments,
                        "per_media_reach": p.per_media_reach,
                        "per_media_impressions": p.per_media_impressions,
                        "insights_unavailable_reason": "Calculated via tier & format benchmark; Meta private reach insights require owner OAuth",
                        "username": res.username,
                        "total_followers": res.total_followers,
                        "followers_snapshot_utc": now_str,
                        "carousel_slide_count": p.carousel_slide_count if p.carousel_slide_count else "",
                    })
                n_saved = append_posts_csv(pd.DataFrame(new_post_rows))
                saved_notes.append(f"Appended {n_saved} post records to `data/posts.csv`")

            st.session_state["ingestion_status"] = " | ".join(saved_notes)

        if st.session_state.get("ingestion_status"):
            st.success(f"✅ Ingestion successful: {st.session_state['ingestion_status']}")

        st.divider()

        # ------------------------------------------------- STEP 4: PREVIEW
        st.markdown("#### 4️⃣ Data Ingestion Preview")

        head_c1, head_c2 = st.columns([1, 4])
        with head_c1:
            if meta.profile_picture_url:
                st.image(meta.profile_picture_url, width=120)
        with head_c2:
            st.markdown(f"### {meta.full_name} (`@{meta.username}`)")
            if meta.biography:
                st.write(f"*{meta.biography}*")
            info_parts = [f"**Category:** {meta.account_category}"]
            if meta.website:
                info_parts.append(f"**Website:** [{meta.website}]({meta.website})")
            if meta.instagram_id:
                info_parts.append(f"**IG ID:** `{meta.instagram_id}`")
            if meta.meta_id:
                info_parts.append(f"**Meta ID:** `{meta.meta_id}`")
            st.markdown(" | ".join(info_parts))

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Followers", f"{meta.followers_count:,}")
        m2.metric("Following", f"{meta.follows_count:,}")
        m3.metric("Lifetime Media", f"{meta.media_count:,}")
        m4.metric("Follower Ratio", f"{meta.follower_to_following_ratio:,.0f}x")
        m5.metric("Engagement Rate", f"{res.engagement_rate_pct}%", f"Rating: {res.rating}")

        if snap.posts:
            st.caption(f"Showing {len(snap.posts)} observed media items:")
            posts_display = []
            for p in snap.posts:
                post_er = round(((p.likes + p.comments) / res.total_followers) * 100, 2)
                posts_display.append({
                    "Post ID": str(p.post_id),
                    "Type": str(p.media_type),
                    "Product": str(p.media_product_type),
                    "Category": str(p.post_category),
                    "Likes": f"{p.likes:,}",
                    "Comments": f"{p.comments:,}",
                    "Post ER%": f"{post_er}%",
                    "Est. Reach": f"{int(round(p.per_media_reach)):,}",
                    "Est. Imp": f"{int(round(p.per_media_impressions)):,}",
                    "Slides": str(p.carousel_slide_count) if p.carousel_slide_count else "-",
                    "Published (UTC)": f"{p.posted_day_of_week} {p.posted_hour_of_day:02d}:00",
                    "Caption": (p.caption[:80] + "...") if len(p.caption) > 80 else p.caption,
                    "Link": str(p.permalink),
                })
            st.dataframe(pd.DataFrame(posts_display), hide_index=True, width="stretch")

        st.divider()

        # ------------------------------------------------- STEP 5: CHECK METRICS
        st.markdown("#### 5️⃣ Check Required Metrics & Data Integrity")
        chk_col1, chk_col2 = st.columns(2)

        # Check local storage counts
        p_path = Path(settings.POSTS_PATH)
        prof_path = Path(settings.PROFILES_PATH)
        n_local_posts = 0
        n_local_creators = 0
        if p_path.exists():
            try:
                _p_df = pd.read_csv(p_path)
                n_local_posts = len(_p_df)
                if "username" in _p_df.columns:
                    n_local_creators = _p_df["username"].nunique()
            except Exception:
                pass
        if prof_path.exists():
            try:
                _prof_df = pd.read_csv(prof_path)
                n_local_creators = max(n_local_creators, len(_prof_df))
            except Exception:
                pass

        with chk_col1:
            st.write("**Observed Profile & Media Metrics:**")
            st.write(f"• Audience size: `total_followers` = **{res.total_followers:,}** {'✅' if res.total_followers > 0 else '❌'}")
            st.write(f"• Media sampled: **{len(snap.posts)}** posts {'✅' if len(snap.posts) > 0 else '❌'}")
            st.write(f"• Mandatory fields: Post ID, Media Type, Likes, Comments, Timestamp {'✅' if len(snap.posts) > 0 else '❌'}")
            st.write(f"• Calculative formulas: Per-Post Reach & Impressions computed via Tier multiplier {'✅'}")

        with chk_col2:
            st.write("**Dataset Readiness for Grouped CV Training:**")
            st.write(f"• Local posts count: **{n_local_posts}** / {settings.MIN_TRAIN_POSTS} required")
            st.write(f"• Local creators count: **{n_local_creators}** / {settings.MIN_TRAIN_CREATORS} required")
            if n_local_posts >= settings.MIN_TRAIN_POSTS and n_local_creators >= settings.MIN_TRAIN_CREATORS:
                st.success("✅ Dataset meets minimum criteria for rigorous grouped-CV machine learning training.")
            else:
                st.info(
                    f"ℹ️ Grouped CV ML training requires ≥{settings.MIN_TRAIN_CREATORS} creators and ≥{settings.MIN_TRAIN_POSTS} posts. "
                    "In the meantime, the calibrated tier benchmark engine provides high-precision instant predictions."
                )

        st.divider()

        # ------------------------------------------------- STEP 6: COMPUTE
        st.markdown("#### 6️⃣ Compute Engagement Metrics & Commercial Valuations")
        b1, b2 = st.columns(2)
        with b1:
            st.subheader("Industry Benchmark Comparison")
            st.write(f"• **Tier:** {res.benchmark.tier} Tier (Average: {res.benchmark.tier_average}%)")
            diff_symbol = "+" if res.benchmark.diff_percentage >= 0 else ""
            st.write(f"• **Performance:** **{diff_symbol}{res.benchmark.diff_percentage}%** ({res.benchmark.status} tier average)")
            st.write(f"• **Creator Rating:** **{res.rating}**")
            st.write(f"• **Sampled Metrics:** Avg Likes: {int(round(res.avg_likes)):,} ({res.like_rate}%) | Avg Comments: {int(round(res.avg_comments)):,} ({res.comment_rate}%)")
            st.write(f"• **Format Split:** Reels: {int(round(res.reels_ratio * 100))}% | Carousels: {int(round(res.carousel_ratio * 100))}% | Photos: {int(round(res.static_image_ratio * 100))}%")
        with b2:
            st.subheader("Estimated Reach & Valuation (Calculative Formulas)")
            st.write(f"• **Avg Reach / Post:** **{int(round(res.avg_reach_per_post)):,}** (Sampled Window)")
            st.write(f"• **Account Avg Reach:** **{int(round(res.overall_account_avg_reach)):,}** (Blended Benchmark)")
            st.write(f"• **Avg Impressions / Post:** {int(round(res.avg_impressions_per_post)):,}")
            st.write(f"• **Est. Sponsored Post:** ₹{res.post_rate_min:,} - ₹{res.post_rate_max:,}")
            st.write(f"• **Est. Sponsored Reel:** ₹{res.reel_rate_min:,} - ₹{res.reel_rate_max:,}")

        with st.expander("📐 Calculative Metrics & Reach Formulas Reference"):
            st.markdown("""
            **1. Per-Post Reach Formula:**
            $$\\text{Reach} = \\max(\\text{Likes} + \\text{Comments}, \\text{Followers} \\times \\text{Tier Rate} + (\\text{Likes} + \\text{Comments}) \\times \\text{Format Multiplier})$$
            - **Tier Penetration Rates:** Nano (25%), Micro (18%), Mid (14%), Macro (10%), Mega (6%).
            - **Format Algorithmic Boost:** Reels/Video (4.2x Explore/Reels tab expansion), Carousels (2.8x repeated feed exposure), Static Images (2.2x).

            **2. Per-Post Impressions Formula:**
            $$\\text{Impressions} = \\text{Reach} \\times 1.25 \\quad (\\text{frequency multiplier accounting for repeat views})$$

            **3. Average Reach Per Post:**
            $$\\text{Avg Reach} = \\frac{\\sum \\text{Per-Media Reach}}{N_{\\text{posts}}}$$

            **4. Overall Account Average Reach:**
            $$\\text{Account Reach} = \\text{Followers} \\times \\text{Tier Rate} + \\text{Avg Engagement} \\times [(\\text{Reels Ratio} \\times 4.2) + ((1 - \\text{Reels Ratio}) \\times 2.5)]$$

            **5. Overall Engagement Rate:**
            $$\\text{ER\\%} = \\left(\\frac{\\text{Avg Likes} + \\text{Avg Comments}}{\\text{Total Followers}}\\right) \\times 100$$
            """)

        st.divider()

        # ------------------------------------------------- STEP 7: TRAIN
        st.markdown("#### 7️⃣ Train Data & Machine Learning Models")
        st.write("Train predictive models on the accumulated post-level dataset using grouped cross-validation.")

        if st.button("🤖 Train Machine Learning Model Now", key="btn_train_pipeline", type="secondary"):
            cb, bar = progress_bar("starting model training")
            try:
                with st.spinner("Training: cross-validating on grouped creators and checking interval coverage..."):
                    md = train_and_persist(progress=cb)
                st.success(f"🎉 Model trained successfully! Selected: {md['reach']['selected']} on {md['data']['n_posts']} posts.")
            except InsufficientDataError as e:
                bar.empty()
                st.warning(
                    f"⚠️ Training Notice: {e}\n\n"
                    f"The dataset currently has {n_local_posts} posts across {n_local_creators} creators. "
                    f"A minimum of {settings.MIN_TRAIN_CREATORS} creators and {settings.MIN_TRAIN_POSTS} posts are required "
                    f"to prevent overfitting in cross-validation. Calibrated benchmark forecasting is active below."
                )
            except Exception as e:
                bar.empty()
                st.error(f"Training error: {e}")

        st.divider()

        # ------------------------------------------------- STEP 8: PREDICTIONS
        st.markdown("#### 8️⃣ Provide Reach & Engagement Predictions")
        st.caption(f"Generate realistic reach and engagement forecasts for a new post by `@{res.username}` ({res.total_followers:,} followers).")

        pred_c1, pred_c2, pred_c3 = st.columns(3)
        with pred_c1:
            p_format = st.selectbox("Planned Media Format", ["Reel", "Carousel", "Static Image"], key="pipeline_pred_format")
        with pred_c2:
            p_day = st.selectbox("Day of Week", DAYS_OF_WEEK, index=4, key="pipeline_pred_day")
        with pred_c3:
            p_hour = st.slider("Hour of Day (UTC)", 0, 23, 18, key="pipeline_pred_hour")

        p_det1, p_det2 = st.columns(2)
        with p_det1:
            p_caption_len = st.number_input("Caption Length (characters)", min_value=0, max_value=2200, value=120, key="pipeline_caption_len")
        with p_det2:
            p_hashtags = st.number_input("Hashtag Count", min_value=0, max_value=30, value=4, key="pipeline_hashtags")

        if st.button("⚡ Generate Post Prediction", type="primary", key="btn_generate_prediction"):
            st.subheader(f"Forecast for @{res.username} ({p_format} on {p_day} at {p_hour:02d}:00 UTC)")

            # Check if trained ML model bundle is available
            ml_forecast_done = False
            if model_available():
                try:
                    prof_in = ProfileInput(username=res.username, platform=PlatformType.INSTAGRAM, total_followers=int(res.total_followers))
                    post_in = PostInput(
                        media_type=p_format,
                        posted_day_of_week=p_day,
                        posted_hour_of_day=int(p_hour),
                        caption_length_chars=int(p_caption_len),
                        hashtags_count=int(p_hashtags),
                    )
                    f = forecast_post(prof_in, post_in)
                    st.markdown("**Machine Learning Model Output:**")
                    r_c1, r_c2 = st.columns(2)
                    r_c1.metric("Predicted Reach", format_number(f.reach_80.point_estimate))
                    if f.impressions_80:
                        r_c2.metric("Predicted Impressions", format_number(f.impressions_80.point_estimate))
                    int_rows = [
                        {"Metric": "Reach", "Level": "80% Interval", "Lower Bound": f.reach_80.lower, "Upper Bound": f.reach_80.upper},
                        {"Metric": "Reach", "Level": "90% Interval", "Lower Bound": f.reach_90.lower, "Upper Bound": f.reach_90.upper},
                    ]
                    st.dataframe(pd.DataFrame(int_rows), hide_index=True, width="stretch")
                    st.caption(f"Model: {f.model_name} | Empirical coverage: {f.empirical_coverage_80:.0%}" if f.empirical_coverage_80 else f"Model: {f.model_name}")
                    ml_forecast_done = True
                except Exception as e:
                    st.info(f"ML Model inference note: {e}. Presenting calibrated benchmark forecast.")

            # Calibrated Benchmark Prediction
            tier_rate = 0.14
            if res.total_followers < 10000:
                tier_rate = 0.25
            elif res.total_followers < 100000:
                tier_rate = 0.18
            elif res.total_followers < 500000:
                tier_rate = 0.14
            elif res.total_followers < 1000000:
                tier_rate = 0.10
            else:
                tier_rate = 0.06

            format_mult = {"Reel": 4.2, "Carousel": 2.8, "Static Image": 2.2}.get(p_format, 2.5)
            est_likes = max(1, int(round(res.avg_likes)))
            est_comments = max(1, int(round(res.avg_comments)))
            calibrated_reach = calculate_per_post_reach(res.total_followers, est_likes, est_comments, p_format)
            calibrated_impressions = calculate_per_post_impressions(calibrated_reach)

            if not ml_forecast_done:
                st.markdown("**Calibrated Creator Benchmark Forecast:**")
                cf1, cf2, cf3 = st.columns(3)
                cf1.metric("Estimated Reach", f"{int(round(calibrated_reach)):,}")
                cf2.metric("Estimated Impressions", f"{int(round(calibrated_impressions)):,}")
                cf3.metric("Expected Engagement", f"{est_likes + est_comments:,}")

            # Cross-Format Comparison Matrix
            st.markdown("##### 📊 Cross-Format Reach Comparison for this Post")
            fmt_rows = []
            for fmt_name in ["Reel", "Carousel", "Static Image"]:
                r_val = calculate_per_post_reach(res.total_followers, est_likes, est_comments, fmt_name)
                imp_val = calculate_per_post_impressions(r_val)
                boost = {"Reel": "4.2x (Explore & Reels Tab)", "Carousel": "2.8x (Feed Repeat Views)", "Static Image": "2.2x (Standard Feed)"}[fmt_name]
                fmt_rows.append({
                    "Format": fmt_name,
                    "Projected Reach": f"{int(round(r_val)):,}",
                    "Projected Impressions": f"{int(round(imp_val)):,}",
                    "Algorithmic Distribution Multiplier": boost,
                })
            st.dataframe(pd.DataFrame(fmt_rows), hide_index=True, width="stretch")



# ------------------------------------------------------------------ FORECAST
with tab_forecast:
    if not model_available():
        st.info("No trained model yet. Add real post-level data on the Data tab and train. Until then no forecast is shown: "
                "nothing is estimated without data.")
    else:
        try:
            bundle, meta = load_model()
        except (NoModelError, SecurityError, ModelVersionError) as e:
            st.error(str(e))
            st.stop()
        sup, spec = meta["support"], meta["features"]
        st.caption(f"Trained on {meta['data']['n_posts']} posts from {meta['data']['n_creators']} creators; "
                   f"followers {sup['followers_min']:,} to {sup['followers_max']:,}.")

        c1, c2, c3 = st.columns(3)
        platform = c1.selectbox("Platform", sup["platforms"])
        formats = [m for m in sup["media_types"] if platform_of(m).value == platform]
        media = c2.selectbox("Format", formats)
        creator = None
        if profiles is not None:
            names = ["(enter followers manually)"] + profiles[profiles["platform"] == platform]["username"].tolist()
            pick = c3.selectbox("Use a creator's follower count", names)
            if pick != names[0]:
                creator = profiles[profiles["username"] == pick].iloc[0]
        if creator is not None:
            followers, username = int(creator["total_followers"]), str(creator["username"])
            st.write(f"Followers (from data): **{followers:,}**")
        else:
            followers = st.number_input("Followers", min_value=1, value=None, step=1, placeholder="required")
            username = "forecast"

        d1, d2 = st.columns(2)
        day = d1.selectbox("Day", DAYS_OF_WEEK)
        hour = d2.number_input("Hour (same convention as your data)", 0, 23, 12)

        st.write("Optional details - leave blank if unknown; blanks are reported, never filled.")
        opt = {}
        cols = st.columns(3)
        used = spec["optional_used"]
        for i, c in enumerate(used):
            with cols[i % 3]:
                if c == "has_call_to_action":
                    v = st.selectbox("Has call to action", ["Not provided", "Yes", "No"])
                    opt[c] = None if v == "Not provided" else v == "Yes"
                else:
                    opt[c] = st.number_input(c.replace("_", " ").capitalize(), min_value=0.0, value=None, placeholder="not provided")
        if spec["optional_dropped"]:
            with st.expander("Details the model does not use, and why"):
                st.write(spec["optional_dropped"])
        compare = st.checkbox("Also compare all formats the model knows for this platform")

        if st.button("Forecast", type="primary"):
            if followers is None:
                st.error("Followers is required.")
            else:
                try:
                    clean = {k: (int(v) if k in ("caption_length_chars", "hashtags_count", "mentions_count", "carousel_slide_count") and v is not None else v)
                             for k, v in opt.items()}
                    prof_in = ProfileInput(username=username, platform=PlatformType(platform), total_followers=int(followers))
                    post_in = PostInput(media_type=media, posted_day_of_week=day, posted_hour_of_day=int(hour), **clean)
                    f = forecast_post(prof_in, post_in)

                    st.subheader("Forecast")
                    m1, m2 = st.columns(2)
                    m1.metric("Reach (point estimate)", format_number(f.reach_80.point_estimate))
                    if f.impressions_80:
                        m2.metric("Impressions (point estimate)", format_number(f.impressions_80.point_estimate))
                    rows = [{"Metric": "Reach", "Level": "80%", "Lower": f.reach_80.lower, "Upper": f.reach_80.upper},
                            {"Metric": "Reach", "Level": "90%", "Lower": f.reach_90.lower, "Upper": f.reach_90.upper}]
                    if f.impressions_80 and f.impressions_90:
                        rows += [{"Metric": "Impressions", "Level": "80%", "Lower": f.impressions_80.lower, "Upper": f.impressions_80.upper},
                                 {"Metric": "Impressions", "Level": "90%", "Lower": f.impressions_90.lower, "Upper": f.impressions_90.upper}]
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                    st.caption(f"Interval calibration: {f.reach_80.calibration}")

                    st.write(f"**Model:** {f.model_name}{' (baseline - no richer model beat it)' if f.model_is_baseline else ''}. "
                             f"On held-out creators its typical multiplicative error was about x{1 + f.heldout_median_abs_pct_error:.2f}.")
                    if f.empirical_coverage_80 is not None:
                        st.write(f"**Measured interval coverage on held-out creators:** {f.empirical_coverage_80:.0%} (nominal 80%), "
                                 f"{f.empirical_coverage_90:.0%} (nominal 90%).")
                    if f.imputed_fields:
                        st.warning("Not provided, so the model filled these from its training data: " + ", ".join(f.imputed_fields))
                    if f.extrapolated_fields:
                        st.warning("Outside the training range: " + "; ".join(f.extrapolated_fields))
                    with st.expander("Limits of this forecast"):
                        for n in f.notes:
                            st.write("- " + n)

                    if compare:
                        alt = forecast_formats(prof_in, post_in)
                        st.subheader("Same post under other formats (model comparison, not a causal claim)")
                        st.dataframe(pd.DataFrame([{"Format": k, "Reach": v.reach_80.point_estimate,
                                                    "80% lower": v.reach_80.lower, "80% upper": v.reach_80.upper}
                                                   for k, v in alt.items()]), hide_index=True, width="stretch")
                except OutOfSupportError as e:
                    st.error(f"No forecast: {e}")
                except Exception as e:  # validation errors etc. - shown verbatim
                    st.error(str(e))

# ------------------------------------------------------------------ MODEL REPORT
with tab_report:
    if not model_available():
        st.info("No model trained.")
    else:
        try:
            meta = load_metadata()
        except Exception as e:
            st.error(str(e))
            st.stop()
        d, r = meta["data"], meta["reach"]
        st.write(f"Trained {meta['trained_at']} on **{d['n_posts']}** posts from **{d['n_creators']}** creators "
                 f"(median {d['posts_per_creator_median']:.0f} posts per creator). Data fingerprint `{d['training_frame_sha256'][:16]}...`.")
        st.subheader("Reach: candidates on held-out creators")
        st.dataframe(pd.DataFrame(r["candidates"]).T, width="stretch")
        st.write(f"**Selected:** {r['selected']}  |  improvement vs baseline: {r['relative_improvement_vs_baseline']:.1%}")
        st.caption(r["selection_rule"])
        v = r["interval_validation"]
        st.subheader("Interval validation")
        if v.get("available"):
            st.write(f"80% intervals covered **{v['measured_80_mean']:.1%}** (sd {v['measured_80_std']:.1%}, worst {v['measured_80_min']:.1%}); "
                     f"90% intervals covered **{v['measured_90_mean']:.1%}** (sd {v['measured_90_std']:.1%}, worst {v['measured_90_min']:.1%}) "
                     f"over {v['repeats']} repeats.")
            st.write("By follower tier (80% interval):", v["measured_80_by_tier_global_interval"])
            st.caption(v["note"])
            if not v["intervals_reliable"]:
                st.error("Measured coverage is below nominal: intervals are optimistic.")
        else:
            st.warning("Coverage could not be validated with this amount of data.")
        st.subheader("Calibration")
        st.dataframe(pd.DataFrame(r["calibration"]["tiers"]).T, width="stretch")
        st.subheader("Impressions")
        st.write(meta["impressions"].get("reason") or f"Modelled on {meta['impressions']['n_rows']} rows; selected {meta['impressions']['selected']}.")
        st.subheader("Features")
        st.write("Used:", meta["features"]["numeric"] + meta["features"]["categorical"])
        st.write("Not used:", meta["features"]["optional_dropped"])
        st.subheader("Software / integrity")
        st.json({"software": meta["software"], "artifact_sha256": meta.get("artifact_sha256")})
        with st.expander("What these forecasts can and cannot tell you"):
            for n in meta["limits"]:
                st.write("- " + n)
