import joblib
import pandas as pd


FEATURES = [
    "Followers",
    "Posts",
    "Likes Avg.",
    "Comments Avg.",
    "Views Avg.",
    "Boost Index",
    "Engagement Rate",
    "Engagement Rate (60 Days)",
    "Avg. 7 Day",
    "Avg. 14 Day",
    "Avg. 30 Day",
    "Country",
    "Main topic",
    "Main video category"
]

import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

REACH_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "reach_model.pkl"
)

IMPRESSION_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "impression_model.pkl"
)

# def apply_filter(df, filters):

#     result = df.copy()

#     for column, condition in filters.items():

#         if column == "category":

#             category = condition.lower()

#             result = result[
#                 (
#                     result["Main topic"]
#                     .astype(str)
#                     .str.lower()
#                     .str.contains(category)
#                 )
#                 |
#                 (
#                     result["Main video category"]
#                     .astype(str)
#                     .str.lower()
#                     .str.contains(category)
#                 )
#             ]

#             continue

#         operator = condition["operator"]
#         value = condition["value"]

#         if operator == ">":
#             result = result[
#                 result[column] > value
#             ]

#         elif operator == ">=":
#             result = result[
#                 result[column] >= value
#             ]

#         elif operator == "<":
#             result = result[
#                 result[column] < value
#             ]

#         elif operator == "<=":
#             result = result[
#                 result[column] <= value
#             ]

#         elif operator == "==":
#             result = result[
#                 result[column] == value
#             ]

#     return result


def apply_filter(df, filters):

    result = df.copy()

    print("\n========== FILTER DEBUG ==========")
    print("Total rows before filtering:", len(result))
    print("Filters received:", filters)

    for column, condition in filters.items():

        print("\nColumn:", column)
        print("Condition:", condition)

        # Category filter
        if column == "category":

            category = condition.lower()

            result = result[
                (
                    result["Main topic"]
                    .astype(str)
                    .str.lower()
                    .str.contains(category, na=False)
                )
                |
                (
                    result["Main video category"]
                    .astype(str)
                    .str.lower()
                    .str.contains(category, na=False)
                )
            ]

            print(
                "Rows after category filter:",
                len(result)
            )

            continue

        # Make sure numeric column is actually numeric
        if column in result.columns:

            result[column] = pd.to_numeric(
                result[column],
                errors="coerce"
            )

        operator = condition["operator"]
        value = condition["value"]

        print("Operator:", operator)
        print("Required value:", value)

        if operator == ">":

            result = result[
                result[column] > value
            ]

        elif operator == ">=":

            result = result[
                result[column] >= value
            ]

        elif operator == "<":

            result = result[
                result[column] < value
            ]

        elif operator == "<=":

            result = result[
                result[column] <= value
            ]

        elif operator == "==":

            result = result[
                result[column] == value
            ]

        print(
            "Rows after filter:",
            len(result)
        )

    print("\nFinal filtered rows:", len(result))
    print("=================================\n")

    return result



def predict(df):

    reach_model = joblib.load(
        "models/reach_model.pkl"
    )

    impression_model = joblib.load(
        "models/impression_model.pkl"
    )

    X = df[FEATURES]

    df = df.copy()

    df["Predicted Reach"] = reach_model.predict(X)

    df["Predicted Impressions"] = (
        impression_model.predict(X)
    )

    return df