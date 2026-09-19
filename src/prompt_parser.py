import re


def parse_prompt(prompt):

    prompt_lower = prompt.lower()

    filters = {}

    # -------------------------
    # Followers
    # -------------------------

    million_match = re.search(
        r"(?:above|below|over|more than|greater than)\s*"
        r"(\d+(?:\.\d+)?)\s*(k|m|million|thousand)?\s*followers?",
        prompt_lower
    )

    if million_match:

        value = float(
            million_match.group(1)
        ) * 1_000_000

        if any(
            word in prompt_lower
            for word in [
                "above",
                "over",
                "more than",
                "greater than"
            ]
        ):
            operator = ">"

        elif any(
            word in prompt_lower
            for word in [
                "below",
                "under",
                "less than"
            ]
        ):
            operator = "<"

        else:
            operator = ">="

        filters["Followers"] = {
            "operator": operator,
            "value": value
        }




    # -------------------------
    # Engagement percentage
    # -------------------------

    engagement_match = re.search(
        r"(\d+(?:\.\d+)?)\s*%",
        prompt_lower
    )

    if engagement_match:

        value = float(
            engagement_match.group(1)
        ) / 100

        if any(
            word in prompt_lower
            for word in [
                "above",
                "over",
                "more than",
                "greater than"
            ]
        ):
            operator = ">"

        elif any(
            word in prompt_lower
            for word in [
                "below",
                "under",
                "less than"
            ]
        ):
            operator = "<"

        else:
            operator = ">="

        if "engagement" in prompt_lower:

            filters["Engagement Rate"] = {
                "operator": operator,
                "value": value
            }

    # -------------------------
    # Category
    # -------------------------

    categories = [
        "fashion",
        "sports",
        "technology",
        "beauty",
        "music",
        "entertainment",
        "food",
        "travel",
        "fitness"
    ]

    for category in categories:

        if category in prompt_lower:

            filters["category"] = category

            break

    # -------------------------
    # Prediction request
    # -------------------------

    predict_reach = (
        "reach" in prompt_lower
        and any(
            word in prompt_lower
            for word in [
                "predict",
                "prediction",
                "expected",
                "estimate"
            ]
        )
    )

    predict_impressions = (
        "impression" in prompt_lower
        and any(
            word in prompt_lower
            for word in [
                "predict",
                "prediction",
                "expected",
                "estimate"
            ]
        )
    )

    return {
        "filters": filters,
        "predict_reach": predict_reach,
        "predict_impressions": predict_impressions
    }


# import re


# def parse_prompt(prompt):
#     text = prompt.lower().replace(",", " ").strip()

#     filters = {}

#     # -----------------------------------------
#     # FOLLOWERS FILTER
#     # -----------------------------------------

#     # above / over / more than / greater than
#     match = re.search(
#         r"(?:above|over|more than|greater than)\s*"
#         r"(\d+(?:\.\d+)?)\s*(k|m|million|thousand)?\s*followers?",
#         text
#     )

#     if match:
#         number = float(match.group(1))
#         unit = match.group(2)

#         if unit == "k":
#             number *= 1000
#         elif unit in ["m", "million"]:
#             number *= 1000000
#         elif unit == "thousand":
#             number *= 1000

#         filters["Followers"] = {
#             "operator": ">",
#             "value": int(number)
#         }

#     # below / under / less than
#     if "Followers" not in filters:

#         match = re.search(
#             r"(?:below|under|less than)\s*"
#             r"(\d+(?:\.\d+)?)\s*(k|m|million|thousand)?\s*followers?",
#             text
#         )

#         if match:
#             number = float(match.group(1))
#             unit = match.group(2)

#             if unit == "k":
#                 number *= 1000
#             elif unit in ["m", "million"]:
#                 number *= 1000000
#             elif unit == "thousand":
#                 number *= 1000

#             filters["Followers"] = {
#                 "operator": "<",
#                 "value": int(number)
#             }

#     # -----------------------------------------
#     # PREDICTION
#     # -----------------------------------------

#     predict_reach = (
#         "reach" in text
#         and (
#             "predict" in text
#             or "prediction" in text
#             or "forecast" in text
#         )
#     )

#     predict_impressions = (
#         "impression" in text
#         and (
#             "predict" in text
#             or "prediction" in text
#             or "forecast" in text
#         )
#     )

#     return {
#         "filters": filters,
#         "predict_reach": predict_reach,
#         "predict_impressions": predict_impressions
#     }