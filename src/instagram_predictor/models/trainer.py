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
    ALL_FEATURE_COLUMNS,
    PRE_PUBLISH_FEATURE_COLUMNS_NUMERIC,
    PRE_PUBLISH_FEATURE_COLUMNS_CATEGORICAL,
    ALL_PRE_PUBLISH_FEATURE_COLUMNS,
    DIAGNOSTIC_FEATURE_COLUMNS_NUMERIC,
    DIAGNOSTIC_FEATURE_COLUMNS_CATEGORICAL,
    ALL_DIAGNOSTIC_FEATURE_COLUMNS,
)

logger = get_logger("model_trainer")


def build_pipeline_core(
    numeric_features: list[str] = FEATURE_COLUMNS_NUMERIC,
    categorical_features: list[str] = FEATURE_COLUMNS_CATEGORICAL,
    min_samples_leaf: int = 8,
) -> Pipeline:
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
        ("num", num_transformer, numeric_features),
        ("cat", cat_transformer, categorical_features)
    ])

    base_regressor = HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=8,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=0.1,
        random_state=42
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", base_regressor)
    ])


def build_transformed_pipeline(
    numeric_features: list[str] = FEATURE_COLUMNS_NUMERIC,
    categorical_features: list[str] = FEATURE_COLUMNS_CATEGORICAL,
    min_samples_leaf: int = 8,
) -> TransformedTargetRegressor:
    """
    Wraps the core pipeline in a log1p target transformer to model power-law distributions.
    """
    core_pipeline = build_pipeline_core(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        min_samples_leaf=min_samples_leaf
    )
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


def _train_and_calibrate_pipeline_pair(
    X: pd.DataFrame,
    y_reach: pd.Series,
    y_impressions: pd.Series,
    groups: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    min_samples_leaf: int,
    pipeline_label: str
) -> tuple[TransformedTargetRegressor, TransformedTargetRegressor, dict, dict]:
    """
    Trains Reach and Impressions pipelines with creator-level GroupKFold CV,
    calibrates Mondrian conformal quantiles across creator scale tiers,
    and returns fitted production models with evaluation metadata.
    """
    logger.info(f"--- Training {pipeline_label} Models ---")

    # 1. Creator-Level GroupKFold Cross-Validation (Eliminates Creator Leakage)
    gkf = GroupKFold(n_splits=5)
    reach_cv_r2, reach_cv_mae = [], []
    imp_cv_r2, imp_cv_mae = [], []

    for train_idx, val_idx in gkf.split(X, y_reach, groups=groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_r_tr, y_r_val = y_reach.iloc[train_idx], y_reach.iloc[val_idx]
        y_i_tr, y_i_val = y_impressions.iloc[train_idx], y_impressions.iloc[val_idx]

        r_pipe = build_transformed_pipeline(numeric_features, categorical_features, min_samples_leaf)
        r_pipe.fit(X_tr, y_r_tr)
        r_preds = r_pipe.predict(X_val)
        reach_cv_r2.append(r2_score(y_r_val, r_preds))
        reach_cv_mae.append(mean_absolute_error(y_r_val, r_preds))

        i_pipe = build_transformed_pipeline(numeric_features, categorical_features, min_samples_leaf)
        i_pipe.fit(X_tr, y_i_tr)
        i_preds = i_pipe.predict(X_val)
        imp_cv_r2.append(r2_score(y_i_val, i_preds))
        imp_cv_mae.append(mean_absolute_error(y_i_val, i_preds))

    logger.info(f"[{pipeline_label}] GroupKFold Reach R2: {np.mean(reach_cv_r2):.4f} (+/- {np.std(reach_cv_r2):.4f}), MAE: {np.mean(reach_cv_mae):,.2f}")
    logger.info(f"[{pipeline_label}] GroupKFold Impressions R2: {np.mean(imp_cv_r2):.4f} (+/- {np.std(imp_cv_r2):.4f}), MAE: {np.mean(imp_cv_mae):,.2f}")

    # 2. Inductive Conformal Prediction (ICP) Calibration
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
    reach_calib_model = build_transformed_pipeline(numeric_features, categorical_features, min_samples_leaf)
    reach_calib_model.fit(X_train_icp, y_r_train_icp)
    reach_calib_preds = reach_calib_model.predict(X_calib)

    imp_calib_model = build_transformed_pipeline(numeric_features, categorical_features, min_samples_leaf)
    imp_calib_model.fit(X_train_icp, y_i_train_icp)
    imp_calib_preds = imp_calib_model.predict(X_calib)

    reach_scores = np.abs(np.log1p(y_r_calib) - np.log1p(reach_calib_preds))
    imp_scores = np.abs(np.log1p(y_i_calib) - np.log1p(imp_calib_preds))

    q_reach_80 = compute_conformal_quantile(reach_scores, 0.20)
    q_reach_90 = compute_conformal_quantile(reach_scores, 0.10)
    q_imp_80 = compute_conformal_quantile(imp_scores, 0.20)
    q_imp_90 = compute_conformal_quantile(imp_scores, 0.10)

    # Mondrian (Tier-Conditional) Conformal Prediction
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

    logger.info(f"[{pipeline_label}] Global Conformal Quantiles -> Reach 80%: {q_reach_80:.4f}, 90%: {q_reach_90:.4f}")
    logger.info(f"[{pipeline_label}] Global Conformal Quantiles -> Impressions 80%: {q_imp_80:.4f}, 90%: {q_imp_90:.4f}")

    # 3. Fit Final Production Models on full dataset
    final_reach_pipeline = build_transformed_pipeline(numeric_features, categorical_features, min_samples_leaf)
    final_reach_pipeline.fit(X, y_reach)

    final_imp_pipeline = build_transformed_pipeline(numeric_features, categorical_features, min_samples_leaf)
    final_imp_pipeline.fit(X, y_impressions)

    reach_eval = {
        "group_cv_r2_mean": float(np.mean(reach_cv_r2)),
        "group_cv_r2_std": float(np.std(reach_cv_r2)),
        "group_cv_mae_mean": float(np.mean(reach_cv_mae)),
        "conformal_quantile_80": q_reach_80,
        "conformal_quantile_90": q_reach_90,
        "tier_conformal_quantiles": tier_quantiles_reach
    }

    imp_eval = {
        "group_cv_r2_mean": float(np.mean(imp_cv_r2)),
        "group_cv_r2_std": float(np.std(imp_cv_r2)),
        "group_cv_mae_mean": float(np.mean(imp_cv_mae)),
        "conformal_quantile_80": q_imp_80,
        "conformal_quantile_90": q_imp_90,
        "tier_conformal_quantiles": tier_quantiles_imp
    }

    return final_reach_pipeline, final_imp_pipeline, reach_eval, imp_eval


def train_and_persist_pipelines():
    """
    Trains both Pre-Publishing Forecasting and Post-Publishing Diagnostic pipelines
    with creator-level GroupKFold CV, calibrates non-parametric Conformal Prediction quantiles,
    and persists artifacts and metadata with SHA-256 hashes.
    """
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(reload=True)

    y_reach = df["per_media_reach"]
    y_impressions = df["per_media_impressions"]
    groups = df["username"]

    # 1. Train Pre-Publishing Forecasting Pipelines (Strictly Pre-Publish Features)
    X_pre = df[ALL_PRE_PUBLISH_FEATURE_COLUMNS]
    (
        pre_reach_pipeline,
        pre_imp_pipeline,
        pre_reach_eval,
        pre_imp_eval
    ) = _train_and_calibrate_pipeline_pair(
        X=X_pre,
        y_reach=y_reach,
        y_impressions=y_impressions,
        groups=groups,
        numeric_features=PRE_PUBLISH_FEATURE_COLUMNS_NUMERIC,
        categorical_features=PRE_PUBLISH_FEATURE_COLUMNS_CATEGORICAL,
        min_samples_leaf=4,
        pipeline_label="Pre-Publishing"
    )

    # 2. Train Post-Publishing Diagnostic Pipelines (Full Diagnostic Features)
    X_diag = df[ALL_DIAGNOSTIC_FEATURE_COLUMNS]
    (
        diag_reach_pipeline,
        diag_imp_pipeline,
        diag_reach_eval,
        diag_imp_eval
    ) = _train_and_calibrate_pipeline_pair(
        X=X_diag,
        y_reach=y_reach,
        y_impressions=y_impressions,
        groups=groups,
        numeric_features=DIAGNOSTIC_FEATURE_COLUMNS_NUMERIC,
        categorical_features=DIAGNOSTIC_FEATURE_COLUMNS_CATEGORICAL,
        min_samples_leaf=8,
        pipeline_label="Post-Publishing Diagnostic"
    )

    # 3. Persist Models
    joblib.dump(pre_reach_pipeline, settings.PRE_PUBLISH_REACH_MODEL_PATH)
    joblib.dump(pre_imp_pipeline, settings.PRE_PUBLISH_IMPRESSIONS_MODEL_PATH)
    joblib.dump(diag_reach_pipeline, settings.REACH_MODEL_PATH)
    joblib.dump(diag_imp_pipeline, settings.IMPRESSIONS_MODEL_PATH)

    def _compute_sha256(path: Path) -> str:
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()

    diag_reach_hash = _compute_sha256(settings.REACH_MODEL_PATH)
    diag_imp_hash = _compute_sha256(settings.IMPRESSIONS_MODEL_PATH)
    pre_reach_hash = _compute_sha256(settings.PRE_PUBLISH_REACH_MODEL_PATH)
    pre_imp_hash = _compute_sha256(settings.PRE_PUBLISH_IMPRESSIONS_MODEL_PATH)

    metadata = {
        "version": settings.VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": len(df),
        "target_transformation": "log1p / expm1",
        "validation_strategy": "GroupKFold by creator username",
        "artifact_hashes": {
            settings.REACH_MODEL_PATH.name: diag_reach_hash,
            settings.IMPRESSIONS_MODEL_PATH.name: diag_imp_hash,
            "reach_pipeline": diag_reach_hash,
            "impressions_pipeline": diag_imp_hash,
            settings.PRE_PUBLISH_REACH_MODEL_PATH.name: pre_reach_hash,
            settings.PRE_PUBLISH_IMPRESSIONS_MODEL_PATH.name: pre_imp_hash,
            "pre_publish_reach_pipeline": pre_reach_hash,
            "pre_publish_impressions_pipeline": pre_imp_hash,
        },
        "features": {
            "numeric": FEATURE_COLUMNS_NUMERIC,
            "categorical": FEATURE_COLUMNS_CATEGORICAL,
            "diagnostic_numeric": DIAGNOSTIC_FEATURE_COLUMNS_NUMERIC,
            "diagnostic_categorical": DIAGNOSTIC_FEATURE_COLUMNS_CATEGORICAL,
            "pre_publish_numeric": PRE_PUBLISH_FEATURE_COLUMNS_NUMERIC,
            "pre_publish_categorical": PRE_PUBLISH_FEATURE_COLUMNS_CATEGORICAL,
        },
        "evaluation": {
            "reach": diag_reach_eval,
            "impressions": diag_imp_eval,
            "diagnostic_reach": diag_reach_eval,
            "diagnostic_impressions": diag_imp_eval,
            "pre_publish_reach": pre_reach_eval,
            "pre_publish_impressions": pre_imp_eval,
        }
    }

    with open(settings.MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dual-model pipelines and conformal metadata exported to {settings.MODELS_DIR}")
    return metadata


if __name__ == "__main__":
    train_and_persist_pipelines()
