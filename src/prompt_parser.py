import re

COUNTRY_MAP = {
    "united states": "US",
    "usa": "US",
    "us": "US",
    "america": "US",
    "spain": "ES",
    "es": "ES",
    "india": "IN",
    "in": "IN",
    "brazil": "BR",
    "brasil": "BR",
    "br": "BR",
    "united kingdom": "GB",
    "uk": "GB",
    "great britain": "GB",
    "england": "GB",
    "gb": "GB",
    "canada": "CA",
    "ca": "CA",
    "netherlands": "NL",
    "holland": "NL",
    "nl": "NL",
    "uruguay": "UY",
    "uy": "UY",
    "turkey": "TR",
    "tr": "TR",
    "indonesia": "ID",
    "id": "ID",
    "colombia": "CO",
    "co": "CO",
    "france": "FR",
    "fr": "FR",
    "australia": "AU",
    "au": "AU",
    "italy": "IT",
    "it": "IT",
    "united arab emirates": "AE",
    "uae": "AE",
    "dubai": "AE",
    "ae": "AE",
    "puerto rico": "PR",
    "pr": "PR",
    "switzerland": "CH",
    "ch": "CH",
    "sweden": "SE",
    "se": "SE",
    "mexico": "MX",
    "mx": "MX",
    "russia": "RU",
    "ru": "RU",
    "germany": "DE",
    "de": "DE",
    "czech republic": "CZ",
    "czechia": "CZ",
    "cz": "CZ"
}

CATEGORY_KEYWORDS = {
    "sports": "Sports",
    "sport": "Sports",
    "football": "Sports",
    "soccer": "Sports",
    "cricket": "Sports",
    "basketball": "Sports",
    "education": "Education",
    "learning": "Education",
    "music": "Music",
    "song": "Music",
    "songs": "Music",
    "singer": "Music",
    "trailers": "Trailers",
    "trailer": "Trailers",
    "gaming": "Gaming",
    "games": "Gaming",
    "game": "Gaming",
    "gamer": "Gaming",
    "apps": "Gaming & Apps",
    "nonprofits": "Nonprofits & Activism",
    "nonprofit": "Nonprofits & Activism",
    "activism": "Nonprofits & Activism",
    "entertainment": "Entertainment",
    "pets": "Pets & Animals",
    "animals": "Pets & Animals",
    "animal": "Pets & Animals",
    "film": "Film & Animation",
    "films": "Film & Animation",
    "animation": "Film & Animation",
    "movies": "Movies",
    "movie": "Movies",
    "cinema": "Movies",
    "shows": "Shows",
    "tv": "Shows",
    "autos": "Autos & Vehicles",
    "vehicles": "Autos & Vehicles",
    "cars": "Autos & Vehicles",
    "science": "Science & Technology",
    "technology": "Science & Technology",
    "tech": "Science & Technology",
    "people": "People & Blogs",
    "blogs": "People & Blogs",
    "blog": "People & Blogs",
    "lifestyle": "People & Blogs",
    "fashion": "Fashion & Beauty",
    "beauty": "Fashion & Beauty",
    "makeup": "Fashion & Beauty",
    "style": "Fashion & Beauty",
    "comedy": "Comedy",
    "humor": "Comedy",
    "funny": "Comedy",
    "comedian": "Comedy",
    "travel": "Travel & Events",
    "events": "Travel & Events",
    "tourism": "Travel & Events",
    "news": "News & Politics",
    "politics": "News & Politics"
}


def _parse_multiplier(unit_str):
    if not unit_str:
        return 1.0
    u = unit_str.lower().strip()
    if u in ["k", "kilo", "thousand"]:
        return 1_000.0
    elif u in ["m", "mil", "million", "m."]:
        return 1_000_000.0
    elif u in ["b", "bil", "billion", "b."]:
        return 1_000_000_000.0
    return 1.0


def _parse_operator_word(word):
    w = word.lower().strip()
    if w in ["above", "over", "more than", "greater than", "higher than", "exceeding", ">"]:
        return ">"
    if w in ["at least", "min", "minimum", ">="]:
        return ">="
    if w in ["below", "under", "less than", "fewer than", "lower than", "<"]:
        return "<"
    if w in ["at most", "max", "maximum", "up to", "<="]:
        return "<="
    if w in ["==", "=", "equal to", "equals", "is"]:
        return "=="
    return ">="


def parse_prompt(prompt):
    """
    Parses natural language requirements into structured filters and prediction flags.
    """
    text = prompt.strip()
    text_lower = text.lower()
    filters = {}

    # ---------------------------------------------------------
    # 1. FOLLOWERS FILTER
    # ---------------------------------------------------------
    # Pattern A: Between X and Y followers
    between_followers = re.search(
        r"(?:between|from)\s*(\d+(?:\.\d+)?)\s*(k|m|million|thousand|b|billion)?\s*(?:and|to)\s*(\d+(?:\.\d+)?)\s*(k|m|million|thousand|b|billion)?\s*followers?",
        text_lower
    )
    if between_followers:
        v1 = float(between_followers.group(1)) * _parse_multiplier(between_followers.group(2) or between_followers.group(4))
        v2 = float(between_followers.group(3)) * _parse_multiplier(between_followers.group(4))
        low, high = min(v1, v2), max(v1, v2)
        filters["Followers"] = {
            "operator": "between",
            "min": low,
            "max": high
        }
    else:
        # Pattern B: (operator) (number) (unit) followers
        # e.g., "above 5m followers", "more than 10 million followers", "under 500k followers", ">= 10m followers"
        f_match_a = re.search(
            r"(above|over|more than|greater than|higher than|exceeding|below|under|less than|fewer than|lower than|at least|at most|min|max|up to|>=|<=|>|<|==|=)\s*"
            r"(\d+(?:\.\d+)?)\s*(k|m|million|thousand|b|billion)?\s*followers?",
            text_lower
        )
        # Pattern C: followers (operator) (number) (unit)
        # e.g., "followers above 5m", "followers > 10m", "followers under 20m"
        f_match_b = re.search(
            r"followers?\s*(above|over|more than|greater than|higher than|exceeding|below|under|less than|fewer than|lower than|at least|at most|min|max|up to|>=|<=|>|<|==|=)\s*"
            r"(\d+(?:\.\d+)?)\s*(k|m|million|thousand|b|billion)?",
            text_lower
        )
        # Pattern D: (number)(unit)+ followers (e.g., "50m+ followers", "5m followers")
        f_match_c = re.search(
            r"(\d+(?:\.\d+)?)\s*(k|m|million|thousand|b|billion)\s*(\+)?\s*followers?",
            text_lower
        )

        if f_match_a:
            op_word = f_match_a.group(1)
            num_val = float(f_match_a.group(2))
            unit_val = f_match_a.group(3)
            mult = _parse_multiplier(unit_val)
            filters["Followers"] = {
                "operator": _parse_operator_word(op_word),
                "value": num_val * mult
            }
        elif f_match_b:
            op_word = f_match_b.group(1)
            num_val = float(f_match_b.group(2))
            unit_val = f_match_b.group(3)
            mult = _parse_multiplier(unit_val)
            filters["Followers"] = {
                "operator": _parse_operator_word(op_word),
                "value": num_val * mult
            }
        elif f_match_c:
            num_val = float(f_match_c.group(1))
            unit_val = f_match_c.group(2)
            has_plus = bool(f_match_c.group(3))
            mult = _parse_multiplier(unit_val)
            filters["Followers"] = {
                "operator": ">=" if has_plus else ">=",
                "value": num_val * mult
            }

    # ---------------------------------------------------------
    # 2. ENGAGEMENT RATE FILTER
    # ---------------------------------------------------------
    # Between X% and Y% engagement
    between_eng = re.search(
        r"(?:engagement|engagement rate)\s*(?:between|from)\s*(\d+(?:\.\d+)?)\s*%\s*(?:and|to)\s*(\d+(?:\.\d+)?)\s*%",
        text_lower
    )
    if between_eng:
        v1 = float(between_eng.group(1)) / 100.0
        v2 = float(between_eng.group(2)) / 100.0
        filters["Engagement Rate"] = {
            "operator": "between",
            "min": min(v1, v2),
            "max": max(v1, v2)
        }
    else:
        # Pattern A: engagement (operator) X%
        eng_match_a = re.search(
            r"(?:engagement|engagement rate)\s*(above|over|more than|greater than|higher than|below|under|less than|fewer than|lower than|at least|at most|min|max|up to|>=|<=|>|<|==|=)\s*(\d+(?:\.\d+)?)\s*%",
            text_lower
        )
        # Pattern B: (operator) X% engagement
        eng_match_b = re.search(
            r"(above|over|more than|greater than|higher than|below|under|less than|fewer than|lower than|at least|at most|min|max|up to|>=|<=|>|<|==|=)\s*(\d+(?:\.\d+)?)\s*%\s*(?:engagement|engagement rate)",
            text_lower
        )
        # Pattern C: X% (operator) engagement or just X%+ engagement
        eng_match_c = re.search(
            r"(\d+(?:\.\d+)?)\s*%\s*(\+)?\s*(?:engagement|engagement rate)",
            text_lower
        )

        if eng_match_a:
            op_word = eng_match_a.group(1)
            num_val = float(eng_match_a.group(2)) / 100.0
            filters["Engagement Rate"] = {
                "operator": _parse_operator_word(op_word),
                "value": num_val
            }
        elif eng_match_b:
            op_word = eng_match_b.group(1)
            num_val = float(eng_match_b.group(2)) / 100.0
            filters["Engagement Rate"] = {
                "operator": _parse_operator_word(op_word),
                "value": num_val
            }
        elif eng_match_c:
            num_val = float(eng_match_c.group(1)) / 100.0
            filters["Engagement Rate"] = {
                "operator": ">=",
                "value": num_val
            }

    # ---------------------------------------------------------
    # 3. CATEGORY / TOPIC FILTER
    # ---------------------------------------------------------
    # Sort category keys by length descending to match multi-word keys first
    for kw, cat_name in sorted(CATEGORY_KEYWORDS.items(), key=lambda x: -len(x[0])):
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, text_lower):
            filters["category"] = cat_name
            break

    # ---------------------------------------------------------
    # 4. COUNTRY / REGION FILTER
    # ---------------------------------------------------------
    for place_name, code in sorted(COUNTRY_MAP.items(), key=lambda x: -len(x[0])):
        # Match 'in <country>', 'from <country>', or standalone '<country> accounts'
        country_pattern = r"\b(?:in|from|country)\s+" + re.escape(place_name) + r"\b|\b" + re.escape(place_name) + r"\s+(?:accounts?|influencers?|creators?)\b"
        # Also allow exact 2-letter uppercase match like "in US" or "in IN"
        if re.search(country_pattern, text_lower):
            filters["Country"] = code
            break
        elif len(place_name) == 2 and re.search(r"\b(?:in|from)\s+" + re.escape(place_name.upper()) + r"\b", text):
            filters["Country"] = code
            break

    # ---------------------------------------------------------
    # 5. USERNAME / INFLUENCER HANDLE
    # ---------------------------------------------------------
    user_match = re.search(
        r"(?:for|account|user|influencer|creator|handle|@)\s*@?([a-zA-Z0-9_\.]{3,30})",
        text_lower
    )
    if user_match:
        candidate = user_match.group(1)
        # Avoid reserved keywords matching as username
        reserved = [
            "accounts", "followers", "reach", "impression", "impressions",
            "sports", "music", "fashion", "india", "spain", "brazil", "america",
            "engagement", "prediction", "forecast", "estimate", "performance"
        ]
        if candidate not in reserved and candidate not in CATEGORY_KEYWORDS and candidate not in COUNTRY_MAP:
            filters["Username"] = candidate

    # ---------------------------------------------------------
    # 6. TOP-N & SORTING
    # ---------------------------------------------------------
    top_match = re.search(r"\btop\s*(\d+)\b", text_lower)
    if top_match:
        limit = int(top_match.group(1))
        sort_by = "Followers"
        if "engagement" in text_lower:
            sort_by = "Engagement Rate"
        elif "views" in text_lower or "view" in text_lower:
            sort_by = "Views Avg."
        elif "likes" in text_lower or "like" in text_lower:
            sort_by = "Likes Avg."
        
        filters["_top_n"] = {
            "limit": limit,
            "sort_by": sort_by,
            "ascending": False
        }

    # ---------------------------------------------------------
    # 7. PREDICTION REQUEST DETECTION
    # ---------------------------------------------------------
    predict_keywords = ["predict", "prediction", "predicting", "estimate", "estimating", "forecast", "forecasting", "calculate"]
    has_prediction_intent = any(k in text_lower for k in predict_keywords)

    predict_reach = ("reach" in text_lower and has_prediction_intent) or ("reach" in text_lower and "what is" in text_lower)
    predict_impressions = ("impression" in text_lower and has_prediction_intent) or ("impressions" in text_lower and has_prediction_intent)

    # If generic prediction asked without explicitly mentioning reach or impressions, predict both
    if has_prediction_intent and not predict_reach and not predict_impressions:
        predict_reach = True
        predict_impressions = True

    return {
        "filters": filters,
        "predict_reach": bool(predict_reach),
        "predict_impressions": bool(predict_impressions)
    }