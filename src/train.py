import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from preprocessing import load_data
from synthetic_targets import create_synthetic_targets

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "top_200_instagrammers.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

NUMERIC_FEATURES = [
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

CATEGORICAL_FEATURES = [
    "Country",
    "Main topic",
    "Main video category"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_pipeline():
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def train_models():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load data without premature imputation
    df = load_data(DATA_PATH)

    # 2. Generate synthetic prototype targets
    df = create_synthetic_targets(df)

    X = df[FEATURES]
    y_reach = df["synthetic_reach"]
    y_impressions = df["synthetic_impressions"]

    # 3. K-Fold Cross Validation Evaluation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    reach_pipeline_cv = build_pipeline()
    imp_pipeline_cv = build_pipeline()

    reach_cv_r2 = cross_val_score(reach_pipeline_cv, X, y_reach, cv=cv, scoring="r2")
    reach_cv_mae = -cross_val_score(reach_pipeline_cv, X, y_reach, cv=cv, scoring="neg_mean_absolute_error")
    imp_cv_r2 = cross_val_score(imp_pipeline_cv, X, y_impressions, cv=cv, scoring="r2")
    imp_cv_mae = -cross_val_score(imp_pipeline_cv, X, y_impressions, cv=cv, scoring="neg_mean_absolute_error")

    print(f"5-Fold CV Reach R2: {reach_cv_r2.mean():.4f} (+/- {reach_cv_r2.std():.4f})")
    print(f"5-Fold CV Reach MAE: {reach_cv_mae.mean():,.2f}")
    print(f"5-Fold CV Impression R2: {imp_cv_r2.mean():.4f} (+/- {imp_cv_r2.std():.4f})")
    print(f"5-Fold CV Impression MAE: {imp_cv_mae.mean():,.2f}")

    # 4. Train-test holdout for validation check
    X_train, X_test, y_r_train, y_r_test, y_i_train, y_i_test = train_test_split(
        X, y_reach, y_impressions, test_size=0.2, random_state=42
    )

    reach_model = build_pipeline()
    reach_model.fit(X_train, y_r_train)
    r_preds = reach_model.predict(X_test)
    print(f"\nHoldout Reach R2: {r2_score(y_r_test, r_preds):.4f}, MAE: {mean_absolute_error(y_r_test, r_preds):,.2f}")

    impression_model = build_pipeline()
    impression_model.fit(X_train, y_i_train)
    i_preds = impression_model.predict(X_test)
    print(f"Holdout Impression R2: {r2_score(y_i_test, i_preds):.4f}, MAE: {mean_absolute_error(y_i_test, i_preds):,.2f}")

    # 5. Final fit on all data and persist artifacts
    final_reach_model = build_pipeline()
    final_reach_model.fit(X, y_reach)
    reach_path = os.path.join(MODEL_DIR, "reach_model.pkl")
    joblib.dump(final_reach_model, reach_path)

    final_impression_model = build_pipeline()
    final_impression_model.fit(X, y_impressions)
    imp_path = os.path.join(MODEL_DIR, "impression_model.pkl")
    joblib.dump(final_impression_model, imp_path)

    print(f"\nModels successfully trained and persisted to:\n - {reach_path}\n - {imp_path}")


if __name__ == "__main__":
    train_models()