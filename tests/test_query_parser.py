from instagram_predictor.nlp import parse_query


def test_query_parser_followers_units():
    # 500k
    q_k = parse_query("Accounts above 500k followers")
    assert q_k.filters["total_followers"]["value"] == 500_000.0
    assert q_k.filters["total_followers"]["operator"] == ">"

    # 10m
    q_m = parse_query("Accounts over 10m followers")
    assert q_m.filters["total_followers"]["value"] == 10_000_000.0

    # Range: between 5m and 20m
    q_range = parse_query("Accounts between 5m and 20m followers")
    assert q_range.filters["total_followers"]["operator"] == "between"
    assert q_range.filters["total_followers"]["min"] == 5_000_000.0
    assert q_range.filters["total_followers"]["max"] == 20_000_000.0


def test_query_parser_media_and_category():
    q_reels = parse_query("Reels in Sports category with engagement above 2%")
    assert q_reels.filters["media_type"] == "Reel"
    assert q_reels.filters["category"] == "Sports"
    assert q_reels.filters["engagement_rate"]["operator"] == ">"
    assert q_reels.filters["engagement_rate"]["value"] == 0.02


def test_query_parser_prediction_intent():
    q_reach = parse_query("Predict reach for sports accounts")
    assert q_reach.predict_reach is True
    assert q_reach.predict_impressions is False

    q_both = parse_query("Forecast reach and impressions for fashion creators")
    assert q_both.predict_reach is True
    assert q_both.predict_impressions is True


def test_query_parser_country_and_top_n():
    q_in = parse_query("Top 5 accounts in India")
    assert q_in.filters["country"] == "IN"
    assert q_in.filters["_top_n"]["limit"] == 5
