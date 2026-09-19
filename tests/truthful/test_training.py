"""Training must refuse thin data, prefer the baseline unless a model truly wins, and measure its own coverage."""

import numpy as np
import pytest

from instagram_predictor.models import BASELINE, InsufficientDataError, conformal_quantile, train_forecaster


def test_refuses_to_train_on_too_little_data(posts_effect):
    with pytest.raises(InsufficientDataError, match="need at least"):
        train_forecaster(posts_effect.head(50))


def test_refuses_when_too_few_creators(posts_effect):
    few = posts_effect[posts_effect["username"].isin(posts_effect["username"].unique()[:10])]
    with pytest.raises(InsufficientDataError):
        train_forecaster(few, min_posts=10)


def test_baseline_is_shipped_when_nothing_beats_it(posts_noise):
    _, meta = train_forecaster(posts_noise, coverage_repeats=5)
    r = meta["reach"]
    assert r["selected"] == BASELINE and r["selected_is_baseline"] is True
    assert r["relative_improvement_vs_baseline"] == 0.0


def test_model_is_shipped_when_a_real_effect_exists(posts_effect):
    _, meta = train_forecaster(posts_effect, coverage_repeats=5)
    r = meta["reach"]
    assert r["selected"] != BASELINE
    assert r["relative_improvement_vs_baseline"] >= 0.03
    win = r["candidates"][r["selected"]]["fold_win_fraction_vs_baseline"]
    assert win >= 0.6


def test_metadata_records_only_measured_facts(posts_effect):
    _, meta = train_forecaster(posts_effect, coverage_repeats=5)
    assert meta["data"]["n_posts"] == len(posts_effect)
    assert meta["data"]["n_creators"] == posts_effect["username"].nunique()
    assert len(meta["data"]["training_frame_sha256"]) == 64
    assert {"python", "scikit_learn", "numpy", "pandas"} <= set(meta["software"])
    for c in meta["reach"]["candidates"].values():
        assert c["log_mae"] > 0 and c["n_folds_scored"] > 0
    assert meta["support"]["followers_min"] == int(posts_effect["total_followers"].min())
    assert meta["support"]["followers_max"] == int(posts_effect["total_followers"].max())


def test_interval_coverage_is_measured_and_close_to_nominal(posts_effect):
    _, meta = train_forecaster(posts_effect, coverage_repeats=15)
    v = meta["reach"]["interval_validation"]
    assert v["available"]
    assert 0.65 <= v["measured_80_mean"] <= 0.95
    assert 0.78 <= v["measured_90_mean"] <= 0.99
    assert v["measured_90_mean"] > v["measured_80_mean"]


def test_training_is_deterministic(posts_effect):
    _, a = train_forecaster(posts_effect, seed=3, coverage_repeats=3)
    _, b = train_forecaster(posts_effect, seed=3, coverage_repeats=3)
    assert a["reach"]["candidates"] == b["reach"]["candidates"]
    assert a["data"]["training_frame_sha256"] == b["data"]["training_frame_sha256"]


def test_impressions_only_modelled_when_observed(posts_effect):
    no_imp = posts_effect.drop(columns=["per_media_impressions"])
    _, meta = train_forecaster(no_imp, coverage_repeats=3)
    assert meta["impressions"]["modelled"] is False and "not modelled" in meta["impressions"]["reason"]
    _, meta2 = train_forecaster(posts_effect, coverage_repeats=3)
    assert meta2["impressions"]["modelled"] is True


def test_sparse_optional_feature_is_dropped_with_reason(posts_effect):
    df = posts_effect.copy()
    df.loc[df.index[: int(0.5 * len(df))], "hashtags_count"] = np.nan
    _, meta = train_forecaster(df, coverage_repeats=3)
    assert "hashtags_count" in meta["features"]["optional_dropped"]
    assert "hashtags_count" not in meta["features"]["optional_used"]


class TestConformalQuantile:
    def test_matches_definition(self):
        s = np.arange(1, 101, dtype=float)               # n=100
        assert conformal_quantile(s, 0.2) == 81.0        # ceil(101*0.8)=81
        assert conformal_quantile(s, 0.1) == 91.0        # ceil(101*0.9)=91

    def test_too_few_residuals_raises_instead_of_inventing(self):
        with pytest.raises(InsufficientDataError):
            conformal_quantile(np.array([1.0, 2.0, 3.0]), 0.1)
        with pytest.raises(InsufficientDataError):
            conformal_quantile(np.array([]), 0.2)


def test_end_to_end_from_csv_file_to_forecast(tmp_path, monkeypatch):
    """The real workflow: a posts.csv on disk -> validate -> train -> save -> load -> forecast."""
    from conftest import make_posts_csv
    from instagram_predictor.config import settings
    from instagram_predictor.models import forecast_post, train_and_persist
    from instagram_predictor.schemas import PlatformType, PostInput, ProfileInput

    csv_path = tmp_path / "posts.csv"
    csv_path.write_text(make_posts_csv(seed=5, reel_effect=0.6))
    monkeypatch.setattr(settings, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(settings, "MODEL_METADATA_PATH", tmp_path / "m.json")

    meta = train_and_persist(posts_source=csv_path, coverage_repeats=3)
    assert meta["data"]["validation"]["rows_rejected"] == 0
    assert (tmp_path / "m.json").exists() and meta["data"]["n_posts"] == 240

    f = forecast_post(ProfileInput(username="u", platform=PlatformType.INSTAGRAM, total_followers=200_000),
                      PostInput(media_type="Reel", posted_day_of_week="Friday", posted_hour_of_day=10))
    assert f.reach_80.lower <= f.reach_80.point_estimate <= f.reach_80.upper


def test_training_from_disk_refuses_when_file_missing(tmp_path, monkeypatch):
    from instagram_predictor.data import NoDataError
    from instagram_predictor.models import train_and_persist
    with pytest.raises(NoDataError):
        train_and_persist(posts_source=tmp_path / "absent.csv")


def test_append_posts_skips_existing_ids_and_never_edits_old_rows(tmp_path):
    import pandas as pd
    from instagram_predictor.data import append_posts_csv
    path = tmp_path / "posts.csv"
    first = pd.DataFrame({"post_id": ["a", "b"], "username": ["u", "u"], "per_media_reach": [10, 20]})
    assert append_posts_csv(first, path) == 2
    second = pd.DataFrame({"post_id": ["b", "c"], "username": ["u", "u"], "per_media_reach": [999, 30]})
    assert append_posts_csv(second, path) == 1
    out = pd.read_csv(path)
    assert list(out["post_id"]) == ["a", "b", "c"] and list(out["per_media_reach"]) == [10, 20, 30]


def test_training_reports_monotonic_progress_ending_at_100_percent(tmp_path, monkeypatch):
    from conftest import make_posts_csv
    from instagram_predictor.config import settings
    from instagram_predictor.models import train_and_persist
    csv_path = tmp_path / "posts.csv"
    csv_path.write_text(make_posts_csv(seed=6, reel_effect=0.6))
    monkeypatch.setattr(settings, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(settings, "MODEL_METADATA_PATH", tmp_path / "m.json")
    seen = []
    train_and_persist(posts_source=csv_path, coverage_repeats=3, progress=lambda f, m: seen.append((f, m)))
    fracs = [f for f, _ in seen]
    assert fracs == sorted(fracs), "progress went backwards"
    assert fracs[0] <= 0.05 and fracs[-1] == 1.0 and all(0.0 <= f <= 1.0 for f in fracs)
    assert len(seen) > 10, "progress should update throughout, not just at the ends"
