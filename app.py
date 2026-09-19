import io
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from instagram_predictor.config import settings
from instagram_predictor.data import (
    NoDataError, POST_TEMPLATE_COLUMNS, append_posts_csv, load_posts, load_profiles,
)
from instagram_predictor.guardrails import sanitize_dataframe_for_csv
from instagram_predictor.integrations import InstagramGraphAPIClient, MetaGraphAPIError
from instagram_predictor.models import (
    InsufficientDataError, ModelVersionError, NoModelError, OutOfSupportError, SecurityError,
    forecast_formats, forecast_post, load_metadata, load_model, model_available, train_and_persist,
)
from instagram_predictor.schemas import PlatformType, PostInput, ProfileInput, platform_of, DAYS_OF_WEEK
from instagram_predictor.services import observed_post_summary, run_creator_query
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

tab_data, tab_creators, tab_forecast, tab_report = st.tabs(["Data", "Creators", "Forecast", "Model report"])

# ------------------------------------------------------------------ DATA
with tab_data:
    st.subheader("Creator profiles (observed, as supplied)")
    if profiles is None:
        st.warning(f"{settings.PROFILES_PATH.name} not found.")
    else:
        st.write(f"{len(profiles)} creators. Blank cells mean the source had no value; nothing is filled in.")
        with st.expander("Provenance and what was removed"):
            prov = Path(settings.DATA_DIR) / "PROVENANCE.md"
            st.markdown(prov.read_text(encoding="utf-8") if prov.exists() else "PROVENANCE.md missing")

    st.subheader("Post-level training data (yours)")
    st.write("Required columns: `username, media_type, posted_day_of_week, posted_hour_of_day, total_followers, per_media_reach`. "
             "Use the **same hour convention** here as when you forecast (the Meta connector records UTC).")
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

    st.subheader("Train")
    if st.button("Train model on data/posts.csv", disabled=posts is None):
        try:
            with st.spinner("Cross-validating against the baseline and checking interval coverage..."):
                md = train_and_persist()
            st.success(f"Trained. Selected: {md['reach']['selected']}. See the Model report tab.")
        except (InsufficientDataError, ValueError, NoDataError) as e:
            st.error(str(e))

    with st.expander("Import from Meta Graph API (observed insights only)"):
        st.caption("Needs a token with instagram_basic and instagram_manage_insights. Fields Meta does not return are left empty. "
                   "impressions is not available in current API versions. Hours are UTC. "
                   "total_followers is the CURRENT count, not the count at post time.")
        token = st.text_input("Access token", type="password")
        acct = st.text_input("Instagram account ID")
        limit = st.number_input("Posts to fetch", 1, 100, 25)
        if st.button("Fetch"):
            if not token or not acct:
                st.error("Token and account ID are required.")
            else:
                try:
                    prof, rows = InstagramGraphAPIClient(access_token=token).fetch_creator_snapshot(acct, media_limit=int(limit))
                    st.session_state["meta_rows"] = pd.DataFrame(rows)
                    st.success(f"Fetched @{prof.username} ({prof.total_followers:,} followers) and {len(rows)} posts.")
                except MetaGraphAPIError as e:
                    st.error(str(e))
        if "meta_rows" in st.session_state:
            fetched = st.session_state["meta_rows"]
            st.dataframe(fetched, width="stretch")
            n_reach = int(fetched["per_media_reach"].notna().sum()) if "per_media_reach" in fetched else 0
            st.write(f"{n_reach} of {len(fetched)} fetched posts have observed reach (only those can be used for training).")
            st.download_button("Download as CSV", sanitize_dataframe_for_csv(fetched).to_csv(index=False), "meta_posts.csv", "text/csv")
            if st.button("Append to data/posts.csv"):
                st.success(f"Appended {append_posts_csv(fetched)} new rows (existing rows untouched).")

# ------------------------------------------------------------------ CREATORS
with tab_creators:
    st.subheader("Search observed creator data")
    st.caption("Examples: `over 10m followers`, `followers between 1m and 5m`, `engagement above 2%`, `top 10 by engagement`, `@cristiano`, `category sports`.")
    if profiles is None:
        st.warning("No creator profile data.")
    else:
        q = st.text_input("Query", "")
        parsed, rows, notes = run_creator_query(q, profiles)
        if parsed.understood:
            st.caption("Applied: " + "; ".join(parsed.understood))
        for w in parsed.warnings + notes:
            st.warning(w)
        if parsed.safety_flags:
            st.info("Input contained unusual content: " + ", ".join(parsed.safety_flags))
        summary = observed_post_summary(posts)
        if summary is not None:
            rows = rows.merge(summary, on="username", how="left")
        st.write(f"{len(rows)} of {len(profiles)} creators match.")
        st.dataframe(rows, hide_index=True, width="stretch")

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
