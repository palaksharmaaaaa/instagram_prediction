"""The Meta connector may only relay values Meta returned. Nothing is defaulted, guessed or estimated."""

from unittest.mock import patch

import pytest

from instagram_predictor.integrations import InstagramGraphAPIClient, MetaGraphAPIError, mask_token, sanitize_tokens_in_text


@pytest.fixture
def client():
    return InstagramGraphAPIClient(access_token="EAAB" + "x" * 30)


def test_profile_contains_only_reported_fields(client):
    resp = {"id": "1", "username": "Some_User", "followers_count": 1234}
    with patch.object(client, "_request", return_value=resp):
        p = client.fetch_profile_data("1")
    assert p.username == "some_user" and p.total_followers == 1234
    for absent in ("total_following", "total_media_posts", "full_name", "biography", "has_bio_link", "account_category"):
        assert getattr(p, absent) is None, f"{absent} was invented"


def test_profile_without_followers_is_an_error_not_zero(client):
    with patch.object(client, "_request", return_value={"id": "1", "username": "x"}):
        with pytest.raises(MetaGraphAPIError, match="followers_count"):
            client.fetch_profile_data("1")


def _media_resp():
    return {"data": [
        {"id": "m1", "caption": "hi #a #b @c", "media_type": "VIDEO", "media_product_type": "REELS",
         "timestamp": "2026-09-18T18:30:00+0000", "like_count": 10, "comments_count": 2, "permalink": "u"},
        {"id": "m2", "media_type": "CAROUSEL_ALBUM", "timestamp": "2026-09-19T07:00:00+0000",
         "children": {"data": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}},
        {"id": "m3", "media_type": "STORY_UNKNOWN", "timestamp": "2026-09-19T07:00:00+0000"},
    ]}


def test_media_rows_omit_everything_meta_did_not_report(client):
    def fake(endpoint, params=None, **kw):
        if endpoint.endswith("/media"):
            return _media_resp()
        return {"data": [{"name": "reach", "values": [{"value": 555}]}]} if endpoint.startswith("m1") else {"data": []}
    with patch.object(client, "_request", side_effect=fake):
        rows = client.fetch_media_records("1")
    assert [r["post_id"] for r in rows] == ["m1", "m2"]            # unknown media type skipped, not mislabelled
    m1, m2 = rows
    assert m1["media_type"] == "Reel" and m1["per_media_reach"] == 555
    assert (m1["caption_length_chars"], m1["hashtags_count"], m1["mentions_count"]) == (len("hi #a #b @c"), 2, 1)
    assert (m1["posted_day_of_week"], m1["posted_hour_of_day"]) == ("Friday", 18)
    for forbidden in ("has_call_to_action", "video_duration_seconds", "per_media_impressions"):
        assert forbidden not in m1 and forbidden not in m2
    assert "per_media_reach" not in m2 and "caption_length_chars" not in m2      # no insights, no caption -> absent
    assert m2["carousel_slide_count"] == 3 and "carousel_slide_count" not in m1


def test_insight_fallback_uses_narrower_metric_sets(client):
    calls = []

    def fake(endpoint, params=None, **kw):
        calls.append(params["metric"])
        if params["metric"] != "reach":
            raise MetaGraphAPIError("unsupported metric")
        return {"data": [{"name": "reach", "values": [{"value": 7}]}]}
    with patch.object(client, "_request", side_effect=fake):
        assert client._fetch_media_insights("m", "t") == {"reach": 7}
    assert calls[-1] == "reach" and len(calls) == 3


def test_all_insights_failing_yields_empty_not_zeros(client):
    with patch.object(client, "_request", side_effect=MetaGraphAPIError("nope")):
        assert client._fetch_media_insights("m", "t") == {}


def test_snapshot_marks_current_followers_and_username(client):
    def fake(endpoint, params=None, **kw):
        if endpoint == "1":
            return {"username": "u", "followers_count": 100}
        if endpoint.endswith("/media"):
            return _media_resp()
        return {"data": []}
    with patch.object(client, "_request", side_effect=fake):
        prof, rows = client.fetch_creator_snapshot("1")
    assert all(r["username"] == "u" and r["total_followers"] == 100 and "followers_snapshot_utc" in r for r in rows)


def test_token_masking():
    assert mask_token("EAAB1234567890abcdefgh").startswith("EAAB") and "1234567890" not in mask_token("EAAB1234567890abcdefgh")
    assert "SECRET" not in sanitize_tokens_in_text("https://x?access_token=SECRET&a=1")
    assert "SECRET" not in str(MetaGraphAPIError("bad access_token=SECRET"))
