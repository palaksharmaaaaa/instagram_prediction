import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from preprocessing import load_data
from synthetic_targets import create_synthetic_targets

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "top_200_instagrammers.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


def train_models():

    # -------------------------
    # Load
    # -------------------------

    df = load_data(DATA_PATH)

    # -------------------------
    # Create prototype targets
    # -------------------------

    df = create_synthetic_targets(df)

    # -------------------------
    # Features
    # -------------------------

    features = [
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

    X = df[features]

    # Targets
    y_reach = df["synthetic_reach"]
    y_impressions = df["synthetic_impressions"]

    numeric_features = [
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
        "Avg. 30 Day"
    ]

    categorical_features = [
        "Country",
        "Main topic",
        "Main video category"
    ]

    # -------------------------
    # Preprocessing
    # -------------------------

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore"
        ))
    ])

    preprocessor = ColumnTransformer([
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ])

    # -------------------------
    # Reach Model
    # -------------------------

    reach_model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                max_depth=12
            )
        )
    ])

    # -------------------------
    # Impression Model
    # -------------------------

    impression_model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                max_depth=12
            )
        )
    ])

    # -------------------------
    # Split
    # -------------------------

    X_train, X_test, y_r_train, y_r_test, y_i_train, y_i_test = train_test_split(
    X,
    y_reach,
    y_impressions,
    test_size=0.2,
    random_state=42
)

    # -------------------------
    # Train
    # -------------------------

    reach_model.fit(
        X_train,
        y_r_train
    )

    impression_model.fit(
        X_train,
        y_i_train
    )

    # -------------------------
    # Evaluation
    # -------------------------

    reach_pred = reach_model.predict(X_test)

    impression_pred = impression_model.predict(X_test)

    print(
        "Reach MAE:",
        mean_absolute_error(
            y_r_test,
            reach_pred
        )
    )

    print(
        "Reach R2:",
        r2_score(
            y_r_test,
            reach_pred
        )
    )

    print(
        "Impression MAE:",
        mean_absolute_error(
            y_i_test,
            impression_pred
        )
    )

    print(
        "Impression R2:",
        r2_score(
            y_i_test,
            impression_pred
        )
    )

    # -------------------------
    # Save
    # -------------------------

    joblib.dump(
        reach_model,
        f"{MODEL_DIR}/reach_model.pkl"
    )

    joblib.dump(
        impression_model,
        f"{MODEL_DIR}/impression_model.pkl"
    )

    print("Models saved successfully.")


if __name__ == "__main__":
    train_models()