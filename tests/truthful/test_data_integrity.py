"""Nothing fabricated may exist in the runtime data path, and the loaders may never repair or invent values."""

import io
from pathlib import Path

import pandas as pd
import pytest

import instagram_predictor.data as data_pkg
from instagram_predictor.config import settings
from instagram_predictor.data import NoDataError, load_posts, load_profiles

ROOT = Path(settings.BASE_DIR)
FABRICATED_COLUMNS = {
    "total_following", "is_verified", "country", "account_age_years", "posting_frequency_per_week",
    "follower_growth_rate_30d", "top_country", "secondary_country", "primary_age_group",
    "gender_female_pct", "gender_male_pct", "audience_activity_score", "per_media_reach", "per_media_impressions",
    "per_media_likes", "nested_posts",
}
DERIVED_FROM_SYNTHETIC = {"gordonramsay", "garyvee", "mrbeast", "hubermanlab", "mkbhd", "aliabdaal"}


def test_no_generator_in_runtime_package():
    assert not hasattr(data_pkg, "generate_enterprise_dataset")
    assert not hasattr(data_pkg, "ensure_dataset_exists")


def test_repository_contains_no_synthetic_artifacts():
    """Tripwire: files produced from fabricated data must not remain in the repository's live paths."""
    stale = [
        "data/raw", "data/consolidated_profiles_posts.csv", "data/consolidated_profiles_posts_flat.csv",
        "src/instagram_predictor/data/generator.py", "scripts/consolidate_datasets.py",
        "models/reach_pipeline.joblib", "models/impressions_pipeline.joblib",
        "models/pre_publish_reach_pipeline.joblib", "models/pre_publish_impressions_pipeline.joblib",
        "models/model_metadata.json",
    ]
    present = [p for p in stale if (ROOT / p).exists()]
    assert not present, f"Synthetic-data artifacts still present (delete them): {present}"


def test_creator_profiles_contain_only_source_columns():
    df, report = load_profiles()
    assert not (set(df.columns) & FABRICATED_COLUMNS), set(df.columns) & FABRICATED_COLUMNS
    assert report.rows_rejected == 0
    assert len(df) == 200
    assert not (set(df["username"]) & DERIVED_FROM_SYNTHETIC)
    assert set(df["platform"]) == {"Instagram"}
    assert (df["total_followers"] >= 1).all()


def test_missing_files_raise_instead_of_generating(tmp_path):
    with pytest.raises(NoDataError):
        load_posts(tmp_path / "nope.csv")
    with pytest.raises(NoDataError):
        load_profiles(tmp_path / "nope.csv")
    assert not (tmp_path / "nope.csv").exists()


BASE = ("post_id,username,media_type,posted_day_of_week,posted_hour_of_day,total_followers,per_media_reach,per_media_impressions\n")


def test_valid_rows_are_used_exactly_as_supplied():
    csv = BASE + "a1,Alice,Reel,monday,9,1000,400,500\n"
    df, rep = load_posts(io.StringIO(csv))
    assert rep.rows_used == 1 and rep.rows_rejected == 0
    r = df.iloc[0]
    assert (r.per_media_reach, r.per_media_impressions, r.total_followers, r.posted_hour_of_day) == (400, 500, 1000, 9)
    assert r.username == "alice" and r.posted_day_of_week == "Monday"     # canonical spelling only, values untouched


@pytest.mark.parametrize("row,reason", [
    ("b1,x,Reel,Monday,9,1000,600,500", "impressions is smaller than"),
    ("b2,x,Hologram,Monday,9,1000,100,200", "media_type"),
    ("b3,x,Reel,Funday,9,1000,100,200", "weekday"),
    ("b4,x,Reel,Monday,25,1000,100,200", "0-23"),
    ("b5,x,Reel,Monday,9.5,1000,100,200", "0-23"),
    ("b6,x,Reel,Monday,9,0,100,200", "total_followers"),
    ("b7,x,Reel,Monday,9,1000,-5,200", "per_media_reach"),
    ("b8,x,Reel,Monday,9,1000,,200", "per_media_reach"),
])
def test_invalid_rows_are_rejected_with_reason_not_repaired(row, reason):
    df, rep = load_posts(io.StringIO(BASE + row + "\n"))
    assert len(df) == 0
    assert rep.rows_rejected == 1 and reason in rep.rejected[0][1]


def test_duplicate_post_id_first_kept():
    csv = BASE + "d1,x,Reel,Monday,9,1000,100,200\nd1,x,Reel,Monday,9,1000,999,999\n"
    df, rep = load_posts(io.StringIO(csv))
    assert len(df) == 1 and df.iloc[0].per_media_reach == 100 and rep.rows_rejected == 1


def test_missing_required_column_is_an_error():
    with pytest.raises(ValueError, match="missing required columns"):
        load_posts(io.StringIO("username,media_type\nx,Reel\n"), profiles=pd.DataFrame({"username": ["x"], "total_followers": [5]}))


def test_followers_join_is_reported_and_unmatched_rows_rejected():
    profiles = pd.DataFrame({"username": ["known"], "total_followers": [5000.0]})
    csv = ("username,media_type,posted_day_of_week,posted_hour_of_day,per_media_reach\n"
           "known,Reel,Monday,9,300\nunknown,Reel,Monday,9,300\n")
    df, rep = load_posts(io.StringIO(csv), profiles=profiles)
    assert len(df) == 1 and df.iloc[0].total_followers == 5000
    assert any("snapshot" in n for n in rep.notes)
    assert any("total_followers" in r for _, r in rep.rejected)


def test_optional_blanks_stay_blank():
    csv = BASE.replace("\n", ",hashtags_count\n") + "e1,x,Reel,Monday,9,1000,100,200,\n"
    df, _ = load_posts(io.StringIO(csv))
    assert pd.isna(df.iloc[0].hashtags_count)
