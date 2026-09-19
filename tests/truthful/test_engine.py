"""Forecasts must refuse outside the training support, report missing/extrapolated inputs, and fail closed on bad artifacts."""

import json

import pytest

from instagram_predictor.config import settings
from instagram_predictor.models import (
    ModelVersionError, NoModelError, OutOfSupportError, SecurityError, clear_registry_cache,
    forecast_formats, forecast_post, load_model,
)
from instagram_predictor.schemas import PlatformType, PostInput, ProfileInput


def prof(f=100_000, platform=PlatformType.INSTAGRAM):
    return ProfileInput(username="u", platform=platform, total_followers=f)


def post(**kw):
    base = dict(media_type="Reel", posted_day_of_week="Friday", posted_hour_of_day=12)
    base.update(kw)
    return PostInput(**base)


def test_no_model_means_no_forecast(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MODEL_PATH", tmp_path / "x.joblib")
    monkeypatch.setattr(settings, "MODEL_METADATA_PATH", tmp_path / "x.json")
    clear_registry_cache()
    with pytest.raises(NoModelError):
        forecast_post(prof(), post())


def test_intervals_are_ordered_and_nested(trained_model):
    f = forecast_post(prof(), post(hashtags_count=5, has_call_to_action=True))
    for ci in (f.reach_80, f.reach_90, f.impressions_80, f.impressions_90):
        assert ci.lower <= ci.point_estimate <= ci.upper
    assert f.reach_90.lower <= f.reach_80.lower and f.reach_90.upper >= f.reach_80.upper
    assert f.n_train_posts > 0 and f.heldout_median_abs_pct_error > 0


def test_impressions_never_below_reach_point(trained_model):
    f = forecast_post(prof(), post())
    assert f.impressions_80.point_estimate >= f.reach_80.point_estimate


@pytest.mark.parametrize("followers", [1, 10, 10**9])
def test_followers_outside_training_range_are_refused_not_extrapolated(trained_model, followers):
    with pytest.raises(OutOfSupportError, match="outside the range"):
        forecast_post(prof(followers), post())


def test_unseen_format_and_platform_are_refused(trained_model):
    with pytest.raises(OutOfSupportError, match="format"):
        forecast_post(prof(), post(media_type="Story"))
    with pytest.raises(OutOfSupportError, match="platform"):
        forecast_post(prof(platform=PlatformType.YOUTUBE), post(media_type="YouTube Short"))


def test_platform_format_mismatch_is_an_error(trained_model):
    with pytest.raises(ValueError, match="does not match"):
        forecast_post(prof(platform=PlatformType.YOUTUBE), post(media_type="Reel"))


def test_missing_optional_inputs_are_reported_not_hidden(trained_model):
    f = forecast_post(prof(), post())
    assert set(f.imputed_fields) == {"has_call_to_action", "hashtags_count"}
    g = forecast_post(prof(), post(hashtags_count=5, has_call_to_action=False))
    assert g.imputed_fields == []


def test_extrapolated_optional_values_are_flagged(trained_model):
    f = forecast_post(prof(), post(hashtags_count=500, has_call_to_action=True))
    assert any("hashtags_count" in x for x in f.extrapolated_fields)


def test_invalid_day_and_hour_rejected_by_schema():
    with pytest.raises(ValueError):
        post(posted_day_of_week="Funday")
    with pytest.raises(ValueError):
        post(posted_hour_of_day=24)


def test_reel_effect_learned_in_the_data_is_reflected(trained_model):
    reel = forecast_post(prof(), post(media_type="Reel", hashtags_count=5, has_call_to_action=True))
    static = forecast_post(prof(), post(media_type="Static Image", hashtags_count=5, has_call_to_action=True))
    assert reel.reach_80.point_estimate > 1.3 * static.reach_80.point_estimate


def test_forecast_scales_with_followers_inside_range(trained_model):
    lo = forecast_post(prof(50_000), post(hashtags_count=5, has_call_to_action=True)).reach_80.point_estimate
    hi = forecast_post(prof(500_000), post(hashtags_count=5, has_call_to_action=True)).reach_80.point_estimate
    assert 5 < hi / lo < 20        # roughly proportional: never a flat plateau


def test_format_comparison_only_uses_trained_formats(trained_model):
    out = forecast_formats(prof(), post(hashtags_count=5, has_call_to_action=True))
    assert set(out) == {"Reel", "Carousel", "Static Image"}


def test_tampered_artifact_fails_closed(trained_model):
    settings.MODEL_PATH.write_bytes(settings.MODEL_PATH.read_bytes() + b"x")
    clear_registry_cache()
    with pytest.raises(SecurityError):
        load_model()


def test_missing_hash_fails_closed(trained_model):
    meta = json.loads(settings.MODEL_METADATA_PATH.read_text())
    meta.pop("artifact_sha256")
    settings.MODEL_METADATA_PATH.write_text(json.dumps(meta))
    clear_registry_cache()
    with pytest.raises(SecurityError):
        load_model()


def test_version_mismatch_requires_retrain(trained_model):
    meta = json.loads(settings.MODEL_METADATA_PATH.read_text())
    meta["software"]["scikit_learn"] = "0.0.1"
    settings.MODEL_METADATA_PATH.write_text(json.dumps(meta))
    clear_registry_cache()
    with pytest.raises(ModelVersionError):
        load_model()
