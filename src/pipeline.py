from preprocessing import load_data
from prompt_parser import parse_prompt
from predictor import apply_filter, predict
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "top_200_instagrammers.csv"
)


def run_pipeline(prompt):

    # -------------------------
    # Load data
    # -------------------------

    df = load_data(DATA_PATH)

    # -------------------------
    # Parse prompt
    # -------------------------

    requirements = parse_prompt(prompt)

    filters = requirements["filters"]

    # -------------------------
    # Filter
    # -------------------------

    filtered_df = apply_filter(
        df,
        filters
    )

    # -------------------------
    # Prediction
    # -------------------------

    if requirements["predict_reach"] or requirements["predict_impressions"]:

        if filtered_df.empty:
            return requirements, filtered_df

        filtered_df = predict(filtered_df)

    print("FINAL COLUMNS:")
    print(filtered_df.columns.tolist())

    print("PREDICTED REACH EXISTS:")
    print("Predicted Reach" in filtered_df.columns)
    return requirements, filtered_df