"""
Strict loaders for real, observed data.

Rules enforced here:
  * Values are never imputed, defaulted, clipped, or rewritten. A row is either used exactly as
    supplied or rejected with a stated reason.
  * Rejections are returned in a ValidationReport so nothing disappears silently.
  * There is no data generator anywhere in this project. If the data file is missing,
    NoDataError is raised.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..config import settings
from ..schemas import DAYS_OF_WEEK, MediaType, FORMAT_TO_PLATFORM

PathOrBuffer = Union[str, Path, IO]

REQUIRED_POST_COLUMNS = [
    "username",
    "media_type",
    "posted_day_of_week",
    "posted_hour_of_day",
    "total_followers",
    "per_media_reach",
]
OPTIONAL_NUMERIC_COLUMNS = [
    "caption_length_chars",
    "hashtags_count",
    "mentions_count",
    "video_duration_seconds",
    "carousel_slide_count",
]
OPTIONAL_BOOL_COLUMNS = ["has_call_to_action"]
IMPRESSIONS_COLUMN = "per_media_impressions"
POST_TEMPLATE_COLUMNS = (
    ["post_id"] + REQUIRED_POST_COLUMNS + OPTIONAL_NUMERIC_COLUMNS + OPTIONAL_BOOL_COLUMNS + [IMPRESSIONS_COLUMN]
)

PROFILE_REQUIRED_COLUMNS = ["username", "platform", "total_followers"]


class NoDataError(FileNotFoundError):
    """Raised when a required data file does not exist. No data is ever generated."""


@dataclass
class ValidationReport:
    rows_read: int = 0
    rows_used: int = 0
    rejected: List[Tuple[int, str]] = field(default_factory=list)   # (1-based data row number, reason)
    notes: List[str] = field(default_factory=list)

    @property
    def rows_rejected(self) -> int:
        return len(self.rejected)

    def rejected_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rejected, columns=["row", "reason"])

    def summary(self) -> str:
        return f"{self.rows_used} of {self.rows_read} rows usable; {self.rows_rejected} rejected"


def _read_csv(source: PathOrBuffer, what: str) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise NoDataError(f"{what} not found at {p}")
    df = pd.read_csv(source, dtype=str, keep_default_na=False, na_values=[""])
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def load_profiles(source: Optional[PathOrBuffer] = None) -> Tuple[pd.DataFrame, ValidationReport]:
    """
    Loads observed creator-level facts. Numeric columns are parsed exactly; unparseable or
    non-positive follower counts reject the row. Blank cells stay blank (NaN).
    """
    df = _read_csv(source if source is not None else settings.PROFILES_PATH, "Creator profiles file")
    report = ValidationReport(rows_read=len(df))
    missing = [c for c in PROFILE_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Profiles file is missing required columns: {missing}")

    numeric_cols = [c for c in df.columns if c not in
                    ("username", "full_name", "platform", "account_category", "profile_url")]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    bad = pd.Series("", index=df.index, dtype=object)
    bad[df["username"].isna()] = "username is blank"
    bad[(bad == "") & (~(df["total_followers"] >= 1))] = "total_followers is missing or not a positive number"
    dup = (bad == "") & df["username"].str.lower().duplicated(keep="first")
    bad[dup] = "duplicate username (first occurrence kept)"

    for idx in df.index[bad != ""]:
        report.rejected.append((int(idx) + 1, bad[idx]))
    out = df[bad == ""].copy()
    out["username"] = out["username"].str.strip().str.lower()
    report.rows_used = len(out)
    return out.reset_index(drop=True), report


def load_posts(
    source: Optional[PathOrBuffer] = None,
    profiles: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, ValidationReport]:
    """
    Loads real per-post observations for model training.

    `total_followers` is taken from the file when present. Otherwise it is joined from the creator
    profile snapshot by username (and the report says so); rows with no follower count are rejected.
    """
    df = _read_csv(source if source is not None else settings.POSTS_PATH, "Post-level data file")
    report = ValidationReport(rows_read=len(df))

    if "total_followers" not in df.columns:
        if profiles is None:
            raise ValueError(
                "Posts file has no 'total_followers' column and no creator profiles were supplied to join it from."
            )
        snap = profiles.set_index("username")["total_followers"]
        df["total_followers"] = df["username"].str.strip().str.lower().map(snap).astype("float64").astype(str)
        df.loc[df["total_followers"] == "nan", "total_followers"] = np.nan
        report.notes.append("total_followers joined from the creator profile snapshot (not follower count at post time).")

    missing = [c for c in REQUIRED_POST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Posts file is missing required columns: {missing}")

    num = ["posted_hour_of_day", "total_followers", "per_media_reach"] + OPTIONAL_NUMERIC_COLUMNS
    if IMPRESSIONS_COLUMN in df.columns:
        num.append(IMPRESSIONS_COLUMN)
    for c in num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    valid_media = {m.value for m in MediaType}
    days_norm = df["posted_day_of_week"].str.strip().str.title()

    reasons = pd.Series("", index=df.index, dtype=object)

    def flag(mask: pd.Series, reason: str) -> None:
        m = mask & (reasons == "")
        reasons[m] = reason

    flag(df["username"].isna(), "username is blank")
    flag(~df["media_type"].isin(valid_media), "media_type is not a supported format")
    flag(~days_norm.isin(DAYS_OF_WEEK), "posted_day_of_week is not a weekday name")
    hour = df["posted_hour_of_day"]
    flag(~((hour >= 0) & (hour <= 23) & (hour == np.floor(hour))), "posted_hour_of_day is not an integer 0-23")
    flag(~(df["total_followers"] >= 1), "total_followers is missing or not positive")
    reach = df["per_media_reach"]
    flag(~(reach >= 0), "per_media_reach is missing or negative")
    for c in OPTIONAL_NUMERIC_COLUMNS:
        if c in df.columns:
            flag(df[c].notna() & (df[c] < 0), f"{c} is negative")
    if "carousel_slide_count" in df.columns:
        flag(df["carousel_slide_count"].notna() & (df["carousel_slide_count"] < 1), "carousel_slide_count < 1")
    if IMPRESSIONS_COLUMN in df.columns:
        flag(df[IMPRESSIONS_COLUMN].notna() & (df[IMPRESSIONS_COLUMN] < reach),
             "per_media_impressions is smaller than per_media_reach (impossible; row not altered, rejected)")
    if "post_id" in df.columns:
        flag(df["post_id"].notna() & df["post_id"].duplicated(keep="first"), "duplicate post_id (first occurrence kept)")
    if "platform" in df.columns:
        expected = df["media_type"].map(lambda m: FORMAT_TO_PLATFORM[MediaType(m)].value if m in valid_media else None)
        flag(df["platform"].notna() & expected.notna() & (df["platform"].str.strip() != expected),
             "platform does not match the platform of media_type")
    if "has_call_to_action" in df.columns:
        lowered = df["has_call_to_action"].str.strip().str.lower()
        allowed = {"true", "false", "1", "0", "yes", "no"}
        flag(lowered.notna() & ~lowered.isin(allowed), "has_call_to_action is not a boolean")

    for idx in df.index[reasons != ""]:
        report.rejected.append((int(idx) + 1, reasons[idx]))

    out = df[reasons == ""].copy()
    out["username"] = out["username"].str.strip().str.lower()
    out["posted_day_of_week"] = days_norm[out.index]
    out["posted_hour_of_day"] = out["posted_hour_of_day"].astype(int)
    out["platform"] = out["media_type"].map(lambda m: FORMAT_TO_PLATFORM[MediaType(m)].value)
    if "has_call_to_action" in out.columns:
        low = out["has_call_to_action"].str.strip().str.lower()
        out["has_call_to_action"] = low.map({"true": 1.0, "1": 1.0, "yes": 1.0, "false": 0.0, "0": 0.0, "no": 0.0})
    report.rows_used = len(out)
    return out.reset_index(drop=True), report


def append_posts_csv(new_rows: pd.DataFrame, path: Optional[Path] = None) -> int:
    """
    Appends observed rows to the posts CSV, skipping post_ids that already exist.
    Returns the number of rows written. Existing rows are never modified.
    """
    path = Path(path) if path else settings.POSTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
        if "post_id" in existing.columns and "post_id" in new_rows.columns:
            new_rows = new_rows[~new_rows["post_id"].astype(str).isin(set(existing["post_id"].dropna()))]
        combined_cols = list(existing.columns) + [c for c in new_rows.columns if c not in existing.columns]
        out = pd.concat([existing, new_rows.astype(str).replace("nan", "")], ignore_index=True)[combined_cols]
    else:
        out = new_rows
    out.to_csv(path, index=False)
    return len(new_rows)


def upsert_creator_profile(profile_data: Dict[str, Any], path: Optional[Path] = None) -> bool:
    """
    Safely upserts or updates a creator row in creator_profiles.csv by username.
    Preserves all existing columns and records without duplicating username.
    """
    path = Path(path) if path else settings.PROFILES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    username = str(profile_data.get("username", "")).strip().lower()
    if not username:
        raise ValueError("Profile data must contain a valid non-empty username")

    canonical_cols = [
        "username", "full_name", "platform", "total_followers", "total_media_posts",
        "account_category", "boost_index", "engagement_rate", "engagement_rate_60d",
        "avg_likes", "avg_comments", "avg_video_views", "avg_1d", "avg_3d", "avg_7d",
        "avg_14d", "avg_30d", "profile_url"
    ]

    if path.exists():
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
        cols = list(df.columns)
    else:
        cols = canonical_cols
        df = pd.DataFrame(columns=cols)

    row: Dict[str, str] = {c: "" for c in cols}
    for k, v in profile_data.items():
        k_norm = str(k).strip().lower()
        if k_norm in cols and v is not None:
            row[k_norm] = str(v)
    row["username"] = username

    # Check for match (case-insensitive)
    matches = df.index[df["username"].astype(str).str.strip().str.lower() == username]
    if len(matches) > 0:
        idx = matches[0]
        for k, v in row.items():
            if v != "":
                df.at[idx, k] = v
    else:
        new_row_df = pd.DataFrame([row], columns=cols)
        df = pd.concat([df, new_row_df], ignore_index=True)

    df.to_csv(path, index=False)
    return True
