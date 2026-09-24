"""The UI must never show a forecast without a trained model, and must load cleanly in that state."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[2] / "app.py")


def test_app_loads_with_three_tabs_and_no_forecast_without_model(tmp_path, monkeypatch):
    from instagram_predictor.config import settings
    monkeypatch.setattr(settings, "MODEL_PATH", tmp_path / "none.joblib")
    monkeypatch.setattr(settings, "MODEL_METADATA_PATH", tmp_path / "none.json")
    monkeypatch.setattr(settings, "POSTS_PATH", tmp_path / "posts.csv")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 3
    assert not [b for b in at.button if b.label == "Forecast"], "a Forecast button must not exist without a model"
    text = " ".join(str(x.value) for x in list(at.info) + list(at.warning))
    assert "No trained model" in text
    assert not [r for r in at.sidebar.radio], "the old mode selector must be gone"


def test_creator_search_reports_unapplied_constraints():
    from instagram_predictor.services import run_creator_query
    from instagram_predictor.data import load_profiles
    profiles, _ = load_profiles()
    parsed, rows, notes = run_creator_query("creators in india over 10m followers", profiles)
    warns = " ".join(parsed.warnings + notes)
    assert "cannot be filtered" in warns or "Could not interpret" in warns


def test_public_calculator_rendered_on_creators_tab():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    inputs = [t for t in at.text_input if t.label == "Instagram Username"]
    assert len(inputs) == 1



def test_fetch_shows_progress_and_explains_missing_reach(tmp_path, monkeypatch):
    """Clicking Fetch drives a progress bar to 100% and warns why posts have no reach (stubbed Meta client)."""
    from unittest.mock import patch
    from instagram_predictor.config import settings
    from instagram_predictor.integrations import InstagramGraphAPIClient
    from instagram_predictor.schemas import PlatformType, ProfileInput
    monkeypatch.setattr(settings, "POSTS_PATH", tmp_path / "posts.csv")
    steps = []

    def fake(self, acct, media_limit=25, access_token=None, progress=None):
        for f, m in [(0.02, "reading the profile"), (0.5, "reading insights for post 1 of 2"), (1.0, "done")]:
            steps.append(f)
            progress(f, m)
        prof = ProfileInput(username="me", platform=PlatformType.INSTAGRAM, total_followers=141)
        why = "Posted before the account was converted to a Professional account; Meta provides no insights for it"
        return prof, [{"post_id": "1", "media_type": "Reel", "insights_unavailable_reason": why},
                      {"post_id": "2", "media_type": "Reel", "insights_unavailable_reason": why}]

    with patch.object(InstagramGraphAPIClient, "fetch_creator_snapshot", fake):
        at = AppTest.from_file(APP, default_timeout=60).run()
        [t for t in at.text_input if t.label == "Access token"][0].set_value("tok").run()
        [t for t in at.text_input if t.label.startswith("Instagram account ID")][0].set_value("17841417366975260").run()
        [b for b in at.button if b.label == "Fetch"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert steps == [0.02, 0.5, 1.0]
    assert any("Fetched @me" in s.value for s in at.success)
    warn = " ".join(w.value for w in at.warning)
    assert "converted to a Professional account" in warn and "2** posts" in warn


def _fetch_stub(rows):
    from instagram_predictor.schemas import PlatformType, ProfileInput

    def fake(self, acct, media_limit=25, access_token=None, progress=None):
        return ProfileInput(username="me", platform=PlatformType.INSTAGRAM, total_followers=141), rows
    return fake


def _click_fetch(at):
    [t for t in at.text_input if t.label == "Access token"][0].set_value("tok").run()
    [t for t in at.text_input if t.label.startswith("Instagram account ID")][0].set_value("17841417366975260").run()
    return [b for b in at.button if b.label == "Fetch"][0].click().run()


def test_append_skips_posts_without_reach_and_leaves_file_untouched(tmp_path, monkeypatch):
    from unittest.mock import patch
    from instagram_predictor.config import settings
    from instagram_predictor.integrations import InstagramGraphAPIClient
    monkeypatch.setattr(settings, "POSTS_PATH", tmp_path / "posts.csv")
    rows = [{"post_id": "1", "media_type": "Reel", "insights_unavailable_reason": "x"}]
    with patch.object(InstagramGraphAPIClient, "fetch_creator_snapshot", _fetch_stub(rows)):
        at = _click_fetch(AppTest.from_file(APP, default_timeout=60).run())
        [b for b in at.button if b.label.startswith("Append")][0].click().run()
    assert not (tmp_path / "posts.csv").exists(), "unusable rows must never be written into the training file"
    assert "Nothing appended" in " ".join(w.value for w in at.warning)


def test_append_writes_only_rows_with_reach(tmp_path, monkeypatch):
    import pandas as pd
    from unittest.mock import patch
    from instagram_predictor.config import settings
    from instagram_predictor.integrations import InstagramGraphAPIClient
    monkeypatch.setattr(settings, "POSTS_PATH", tmp_path / "posts.csv")
    rows = [{"post_id": "1", "media_type": "Reel", "per_media_reach": 500},
            {"post_id": "2", "media_type": "Reel", "insights_unavailable_reason": "x"}]
    with patch.object(InstagramGraphAPIClient, "fetch_creator_snapshot", _fetch_stub(rows)):
        at = _click_fetch(AppTest.from_file(APP, default_timeout=60).run())
        [b for b in at.button if b.label.startswith("Append")][0].click().run()
    out = pd.read_csv(tmp_path / "posts.csv")
    assert list(out["post_id"]) == [1] and list(out["per_media_reach"]) == [500]
