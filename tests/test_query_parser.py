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


def test_nlp_01_username_extraction_robustness():
    # Preposition "for" collision with common English nouns
    q_marketing = parse_query("Show accounts for marketing campaigns")
    assert q_marketing.username is None
    assert q_marketing.filters.get("username") is None

    q_summer = parse_query("Show accounts for summer festival")
    assert q_summer.username is None
    assert q_summer.filters.get("username") is None

    q_product = parse_query("Predict reach for new product")
    assert q_product.username is None
    assert q_product.filters.get("username") is None

    # Explicit @ handle
    q_cristiano = parse_query("Show posts for @cristiano")
    assert q_cristiano.username == "cristiano"
    assert q_cristiano.filters.get("username") == "cristiano"

    # Specific handle prefixes
    q_leo = parse_query("Account of leomessi")
    assert q_leo.username == "leomessi"
    assert q_leo.filters.get("username") == "leomessi"


def test_decimal_follower_multipliers():
    # Creators with over 0.5m followers
    q_half_m = parse_query("Creators with over 0.5m followers")
    assert q_half_m.min_followers == 500_000
    assert q_half_m.filters["total_followers"]["value"] == 500_000.0
    assert q_half_m.filters["total_followers"]["operator"] == ">"

    # Creators with between 1.5m and 3m followers
    q_between = parse_query("Creators with between 1.5m and 3m followers")
    assert q_between.min_followers == 1_500_000
    assert q_between.max_followers == 3_000_000
    assert q_between.filters["total_followers"]["operator"] == "between"
    assert q_between.filters["total_followers"]["min"] == 1_500_000.0
    assert q_between.filters["total_followers"]["max"] == 3_000_000.0

    # Under 2.5k followers
    q_under_k = parse_query("Under 2.5k followers")
    assert q_under_k.max_followers == 2_500
    assert q_under_k.filters["total_followers"]["value"] == 2_500.0
    assert q_under_k.filters["total_followers"]["operator"] == "<"

    # Uppercase and billion multipliers
    q_upper = parse_query("Creators with over 0.5M followers")
    assert q_upper.min_followers == 500_000

    q_billion = parse_query("Accounts with over 1.2b followers")
    assert q_billion.min_followers == 1_200_000_000
    assert q_billion.filters["total_followers"]["value"] == 1_200_000_000.0
