"""The UI must never show a forecast without a trained model, and must load cleanly in that state."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[2] / "app.py")


def test_app_loads_with_four_tabs_and_no_forecast_without_model(tmp_path, monkeypatch):
    from instagram_predictor.config import settings
    monkeypatch.setattr(settings, "MODEL_PATH", tmp_path / "none.joblib")
    monkeypatch.setattr(settings, "MODEL_METADATA_PATH", tmp_path / "none.json")
    monkeypatch.setattr(settings, "POSTS_PATH", tmp_path / "posts.csv")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 4
    assert not [b for b in at.button if b.label == "Forecast"], "a Forecast button must not exist without a model"
    text = " ".join(str(x.value) for x in list(at.info) + list(at.warning))
    assert "No trained model" in text
    assert not [r for r in at.sidebar.radio], "the old mode selector must be gone"


def test_creator_search_reports_unapplied_constraints(tmp_path, monkeypatch):
    from instagram_predictor.config import settings
    monkeypatch.setattr(settings, "POSTS_PATH", tmp_path / "posts.csv")
    at = AppTest.from_file(APP, default_timeout=60).run()
    q = [t for t in at.text_input if t.label == "Query"][0]
    q.set_value("creators in india over 10m followers").run()
    assert not at.exception
    warns = " ".join(w.value for w in at.warning)
    assert "cannot be filtered" in warns or "Could not interpret" in warns
