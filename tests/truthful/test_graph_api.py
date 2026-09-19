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
        p = client.fetch_profile_data("17841417366975260")
    assert p.username == "some_user" and p.total_followers == 1234
    for absent in ("total_following", "total_media_posts", "full_name", "biography", "has_bio_link", "account_category"):
        assert getattr(p, absent) is None, f"{absent} was invented"


def test_profile_without_followers_is_an_error_not_zero(client):
    with patch.object(client, "_request", return_value={"id": "1", "username": "x"}):
        with pytest.raises(MetaGraphAPIError, match="followers_count"):
            client.fetch_profile_data("17841417366975260")


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
        rows = client.fetch_media_records("17841417366975260")
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
        assert client._fetch_media_insights("m", "t") == ({"reach": 7}, None)
    assert calls[-1] == "reach" and len(calls) == 3


def test_all_insights_failing_yields_empty_not_zeros(client):
    with patch.object(client, "_request", side_effect=MetaGraphAPIError("nope")):
        metrics, reason = client._fetch_media_insights("m", "t")
    assert metrics == {} and reason == "nope"


def test_snapshot_marks_current_followers_and_username(client):
    def fake(endpoint, params=None, **kw):
        if endpoint == "17841417366975260":
            return {"username": "u", "followers_count": 100}
        if endpoint.endswith("/media"):
            return _media_resp()
        return {"data": []}
    with patch.object(client, "_request", side_effect=fake):
        prof, rows = client.fetch_creator_snapshot("17841417366975260")
    assert all(r["username"] == "u" and r["total_followers"] == 100 and "followers_snapshot_utc" in r for r in rows)


def test_token_masking():
    assert mask_token("EAAB1234567890abcdefgh").startswith("EAAB") and "1234567890" not in mask_token("EAAB1234567890abcdefgh")
    assert "SECRET" not in sanitize_tokens_in_text("https://x?access_token=SECRET&a=1")
    assert "SECRET" not in str(MetaGraphAPIError("bad access_token=SECRET"))


# ----------------------------------------------------------------------------- account-id handling / version / diagnostics
def test_api_version_is_v26(client):
    assert InstagramGraphAPIClient.DEFAULT_API_VERSION == "v26.0"
    assert client.base_url.endswith("/v26.0")


@pytest.mark.parametrize("bad", ["harshgarg_2607", "@harshgarg_2607", "", "  ", "1234", "17841abc"])
def test_username_or_malformed_id_rejected_before_any_request(client, bad):
    with patch.object(client, "_request") as req:
        for call in (client.fetch_profile_data, client.fetch_media_records, client.fetch_creator_snapshot):
            with pytest.raises(MetaGraphAPIError, match="numeric Instagram account ID"):
                call(bad)
        req.assert_not_called()


def test_numeric_id_is_accepted_and_trimmed(client):
    with patch.object(client, "_request", return_value={"username": "u", "followers_count": 5}) as req:
        client.fetch_profile_data(" 17841417366975260 ")
    assert req.call_args.args[0] == "17841417366975260"


def test_code_100_subcode_33_explains_the_id_problem_not_permissions():
    tip = InstagramGraphAPIClient._diagnose_error(
        100, 33, 400,
        "Unsupported get request. Object with ID 'x' does not exist, cannot be loaded due to missing permissions, or does not support this operation.")
    assert "numeric Instagram Business account ID" in tip and "Discover accounts" in tip
    assert "Missing required Meta permissions" not in tip


def test_expired_token_still_diagnosed_as_token_problem():
    assert "invalid or has expired" in InstagramGraphAPIClient._diagnose_error(190, None, 400, "Invalid OAuth access token")


def test_discovery_returns_ids_the_ui_can_use(client):
    resp = {"data": [{"id": "111", "name": "My Page", "instagram_business_account": {"id": "17841417366975260", "username": "me"}},
                     {"id": "222", "name": "No IG page"}]}
    with patch.object(client, "_request", return_value=resp):
        found = client.get_connected_instagram_accounts()
    assert found == [{"instagram_account_id": "17841417366975260", "username": "me", "name": "My Page",
                      "profile_picture_url": None, "page_id": "111", "page_name": "My Page"}]


def test_pre_conversion_posts_report_meta_reason_and_stop_after_one_call(client):
    calls = []

    def fake(endpoint, params=None, **kw):
        calls.append(endpoint)
        raise MetaGraphAPIError("Media posted before business account conversion", error_code=100, error_subcode=2108006)
    with patch.object(client, "_request", side_effect=fake):
        metrics, reason = client._fetch_media_insights("m", "t")
    assert metrics == {} and "converted to a Professional account" in reason
    assert len(calls) == 1, "narrower metric sets cannot help for this error"


def test_rows_carry_the_reason_no_reach_was_returned(client):
    def fake(endpoint, params=None, **kw):
        if endpoint.endswith("/media"):
            return _media_resp()
        raise MetaGraphAPIError("x", error_code=100, error_subcode=2108006)
    with patch.object(client, "_request", side_effect=fake):
        rows = client.fetch_media_records("17841417366975260")
    assert all("per_media_reach" not in r and "converted" in r["insights_unavailable_reason"] for r in rows)


def test_fetch_progress_is_monotonic_and_ends_at_100_percent(client):
    seen = []

    def fake(endpoint, params=None, **kw):
        if endpoint == "17841417366975260":
            return {"username": "u", "followers_count": 100}
        if endpoint.endswith("/media"):
            return _media_resp()
        return {"data": []}
    with patch.object(client, "_request", side_effect=fake):
        client.fetch_creator_snapshot("17841417366975260", progress=lambda f, m: seen.append((f, m)))
    fracs = [f for f, _ in seen]
    assert fracs == sorted(fracs) and fracs[0] < 0.1 and fracs[-1] == 1.0
    assert any("post 1 of" in m for _, m in seen)
