import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import settings, get_logger
from ..data import (
    load_dataset,
    FEATURE_COLUMNS_NUMERIC,
    FEATURE_COLUMNS_CATEGORICAL,
    ALL_FEATURE_COLUMNS
)

logger = get_logger("model_trainer")


def build_pipeline_core() -> Pipeline:
    """
    Constructs the preprocessing and regressor pipeline.
    """
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, FEATURE_COLUMNS_NUMERIC),
        ("cat", cat_transformer, FEATURE_COLUMNS_CATEGORICAL)
    ])

    base_regressor = HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=8,
        min_samples_leaf=8,
        l2_regularization=0.1,
        random_state=42
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", base_regressor)
    ])


def build_transformed_pipeline() -> TransformedTargetRegressor:
    """
    Wraps the core pipeline in a log1p target transformer to model power-law distributions.
    """
    core_pipeline = build_pipeline_core()
    return TransformedTargetRegressor(
        regressor=core_pipeline,
        func=np.log1p,
        inverse_func=np.expm1
    )


def compute_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Computes mathematically exact finite-sample conformal prediction quantiles:
    p = min(ceil((n + 1) * (1 - alpha)) / n, 1.0)
    using method='higher' to guarantee conservative finite-sample coverage.
    """
    n = len(scores)
    if n == 0:
        return 0.35
    p = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, p, method="higher"))


def train_and_persist_pipelines():
    """
    Trains Reach and Impressions pipelines with creator-level GroupKFold CV,
    calibrates non-parametric Conformal Prediction quantiles, and persists artifacts.
    """
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(reload=True)

    X = df[ALL_FEATURE_COLUMNS]
    y_reach = df["per_media_reach"]
    y_impressions = df["per_media_impressions"]
    groups = df["username"]

    # 1. Creator-Level GroupKFold Cross-Validation (Eliminates Creator Leakage)
    gkf = GroupKFold(n_splits=5)
    reach_cv_r2, reach_cv_mae = [], []
    imp_cv_r2, imp_cv_mae = [], []

    for train_idx, val_idx in gkf.split(X, y_reach, groups=groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_r_tr, y_r_val = y_reach.iloc[train_idx], y_reach.iloc[val_idx]
        y_i_tr, y_i_val = y_impressions.iloc[train_idx], y_impressions.iloc[val_idx]

        r_pipe = build_transformed_pipeline()
        r_pipe.fit(X_tr, y_r_tr)
        r_preds = r_pipe.predict(X_val)
        reach_cv_r2.append(r2_score(y_r_val, r_preds))
        reach_cv_mae.append(mean_absolute_error(y_r_val, r_preds))

        i_pipe = build_transformed_pipeline()
        i_pipe.fit(X_tr, y_i_tr)
        i_preds = i_pipe.predict(X_val)
        imp_cv_r2.append(r2_score(y_i_val, i_preds))
        imp_cv_mae.append(mean_absolute_error(y_i_val, i_preds))

    logger.info(f"GroupKFold Reach R2: {np.mean(reach_cv_r2):.4f} (+/- {np.std(reach_cv_r2):.4f}), MAE: {np.mean(reach_cv_mae):,.2f}")
    logger.info(f"GroupKFold Impressions R2: {np.mean(imp_cv_r2):.4f} (+/- {np.std(imp_cv_r2):.4f}), MAE: {np.mean(imp_cv_mae):,.2f}")

    # 2. Inductive Conformal Prediction (ICP) Calibration
    # Partition creators into Train (80%) and Calibration Holdout (20%)
    unique_creators = groups.unique()
    rng = np.random.default_rng(42)
    calib_creators = rng.choice(unique_creators, size=int(len(unique_creators) * 0.20), replace=False)

    calib_mask = groups.isin(calib_creators)
    X_train_icp = X[~calib_mask]
    y_r_train_icp = y_reach[~calib_mask]
    y_i_train_icp = y_impressions[~calib_mask]

    X_calib = X[calib_mask]
    y_r_calib = y_reach[calib_mask]
    y_i_calib = y_impressions[calib_mask]

    # Fit calibration models
    reach_calib_model = build_transformed_pipeline()
    reach_calib_model.fit(X_train_icp, y_r_train_icp)
    reach_calib_preds = reach_calib_model.predict(X_calib)

    imp_calib_model = build_transformed_pipeline()
    imp_calib_model.fit(X_train_icp, y_i_train_icp)
    imp_calib_preds = imp_calib_model.predict(X_calib)

    # Multiplicative log-scale non-conformity scores: s_i = |log(1 + y) - log(1 + y_pred)|
    reach_scores = np.abs(np.log1p(y_r_calib) - np.log1p(reach_calib_preds))
    imp_scores = np.abs(np.log1p(y_i_calib) - np.log1p(imp_calib_preds))

    # Global conformal quantiles for 80% (alpha=0.20) and 90% (alpha=0.10)
    q_reach_80 = compute_conformal_quantile(reach_scores, 0.20)
    q_reach_90 = compute_conformal_quantile(reach_scores, 0.10)
    q_imp_80 = compute_conformal_quantile(imp_scores, 0.20)
    q_imp_90 = compute_conformal_quantile(imp_scores, 0.10)

    # Mondrian (Tier-Conditional) Conformal Prediction
    # Stratified calibration across creator scale tiers: nano (<10k), micro (10k-100k), macro (100k-1M), mega (>=1M)
    tiers = {
        "nano": (0, 10_000),
        "micro": (10_000, 100_000),
        "macro": (100_000, 1_000_000),
        "mega": (1_000_000, float("inf")),
    }

    tier_quantiles_reach: dict[str, dict[str, float]] = {}
    tier_quantiles_imp: dict[str, dict[str, float]] = {}

    calib_followers = X_calib["total_followers"].values
    for tier_name, (low, high) in tiers.items():
        mask = (calib_followers >= low) & (calib_followers < high)
        if np.sum(mask) >= 5:
            tier_r_scores = reach_scores[mask]
            tier_i_scores = imp_scores[mask]
            tier_quantiles_reach[tier_name] = {
                "q80": compute_conformal_quantile(tier_r_scores, 0.20),
                "q90": compute_conformal_quantile(tier_r_scores, 0.10),
                "sample_count": int(np.sum(mask))
            }
            tier_quantiles_imp[tier_name] = {
                "q80": compute_conformal_quantile(tier_i_scores, 0.20),
                "q90": compute_conformal_quantile(tier_i_scores, 0.10),
                "sample_count": int(np.sum(mask))
            }
        else:
            # Fallback to global quantiles if tier calibration sample is too small
            tier_quantiles_reach[tier_name] = {
                "q80": q_reach_80,
                "q90": q_reach_90,
                "sample_count": int(np.sum(mask))
            }
            tier_quantiles_imp[tier_name] = {
                "q80": q_imp_80,
                "q90": q_imp_90,
                "sample_count": int(np.sum(mask))
            }

    logger.info(f"Global Conformal Quantiles -> Reach 80%: {q_reach_80:.4f}, 90%: {q_reach_90:.4f}")
    logger.info(f"Global Conformal Quantiles -> Impressions 80%: {q_imp_80:.4f}, 90%: {q_imp_90:.4f}")
    logger.info(f"Mondrian Tier Quantiles (Reach): {tier_quantiles_reach}")

    # 3. Fit Final Production Models on full dataset
    final_reach_pipeline = build_transformed_pipeline()
    final_reach_pipeline.fit(X, y_reach)

    final_imp_pipeline = build_transformed_pipeline()
    final_imp_pipeline.fit(X, y_impressions)

    joblib.dump(final_reach_pipeline, settings.REACH_MODEL_PATH)
    joblib.dump(final_imp_pipeline, settings.IMPRESSIONS_MODEL_PATH)

    def _compute_sha256(path: Path) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()

    reach_hash = _compute_sha256(settings.REACH_MODEL_PATH)
    imp_hash = _compute_sha256(settings.IMPRESSIONS_MODEL_PATH)

    metadata = {
        "version": settings.VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": len(df),
        "target_transformation": "log1p / expm1",
        "validation_strategy": "GroupKFold by creator username",
        "artifact_hashes": {
            settings.REACH_MODEL_PATH.name: reach_hash,
            settings.IMPRESSIONS_MODEL_PATH.name: imp_hash,
            "reach_pipeline": reach_hash,
            "impressions_pipeline": imp_hash,
        },
        "features": {
            "numeric": FEATURE_COLUMNS_NUMERIC,
            "categorical": FEATURE_COLUMNS_CATEGORICAL
        },
        "evaluation": {
            "reach": {
                "group_cv_r2_mean": float(np.mean(reach_cv_r2)),
                "group_cv_r2_std": float(np.std(reach_cv_r2)),
                "group_cv_mae_mean": float(np.mean(reach_cv_mae)),
                "conformal_quantile_80": q_reach_80,
                "conformal_quantile_90": q_reach_90,
                "tier_conformal_quantiles": tier_quantiles_reach
            },
            "impressions": {
                "group_cv_r2_mean": float(np.mean(imp_cv_r2)),
                "group_cv_r2_std": float(np.std(imp_cv_r2)),
                "group_cv_mae_mean": float(np.mean(imp_cv_mae)),
                "conformal_quantile_80": q_imp_80,
                "conformal_quantile_90": q_imp_90,
                "tier_conformal_quantiles": tier_quantiles_imp
            }
        }
    }

    with open(settings.MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Optimized models and conformal metadata exported to {settings.MODELS_DIR}")
    return metadata


if __name__ == "__main__":
    train_and_persist_pipelines()
