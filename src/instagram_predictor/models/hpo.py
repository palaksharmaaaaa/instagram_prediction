"""
Bayesian Hyperparameter Optimization (HPO) for Instagram Prediction Pipelines.
Employs creator-level GroupKFold cross-validation to tune gradient boosted decision trees
without data leakage, minimizing Out-Of-Fold (OOF) Mean Absolute Error.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

from ..config import get_logger
from ..data import (
    load_dataset,
    ALL_FEATURE_COLUMNS,
    FEATURE_COLUMNS_NUMERIC,
    FEATURE_COLUMNS_CATEGORICAL,
)
from .trainer import build_pipeline_core

logger = get_logger("model_hpo")


def run_bayesian_hpo(
    target_column: str = "per_media_reach",
    n_trials: int = 20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes Bayesian Hyperparameter Optimization using Optuna (if installed)
    or scikit-learn GroupKFold randomized search.

    Tunes:
      - learning_rate
      - max_iter
      - max_depth
      - min_samples_leaf
      - l2_regularization
    """
    df = load_dataset()
    X = df[ALL_FEATURE_COLUMNS]
    y = df[target_column]
    groups = df["username"]

    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            lr = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            max_iter = trial.suggest_int("max_iter", 100, 300, step=50)
            max_depth = trial.suggest_int("max_depth", 4, 12)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 4, 32)
            l2_reg = trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True)

            gkf = GroupKFold(n_splits=3)
            fold_maes = []

            for train_idx, val_idx in gkf.split(X, y, groups=groups):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                base_pipe = build_pipeline_core()
                base_pipe.set_params(
                    regressor__learning_rate=lr,
                    regressor__max_iter=max_iter,
                    regressor__max_depth=max_depth,
                    regressor__min_samples_leaf=min_samples_leaf,
                    regressor__l2_regularization=l2_reg,
                    regressor__random_state=random_state,
                )

                model = TransformedTargetRegressor(
                    regressor=base_pipe,
                    func=np.log1p,
                    inverse_func=np.expm1,
                )
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                loss = mean_absolute_error(
                    np.log1p(np.maximum(y_val, 0)),
                    np.log1p(np.maximum(preds, 0))
                )
                fold_maes.append(loss)

            return float(np.mean(fold_maes))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        best_score = study.best_value
        logger.info(f"Optuna HPO Complete for {target_column}. Best CV Log-Scale MAE: {best_score:,.4f}")
        return {"best_params": best_params, "best_cv_mae": best_score, "method": "Optuna TPE"}

    except ImportError:
        logger.warning("Optuna not installed; running deterministic cross-validated grid search.")
        param_grid = [
            {"learning_rate": 0.05, "max_iter": 150, "max_depth": 6, "min_samples_leaf": 10, "l2_regularization": 0.1},
            {"learning_rate": 0.1, "max_iter": 200, "max_depth": 8, "min_samples_leaf": 8, "l2_regularization": 0.1},
            {"learning_rate": 0.08, "max_iter": 250, "max_depth": 10, "min_samples_leaf": 6, "l2_regularization": 0.5},
        ]
        gkf = GroupKFold(n_splits=3)
        best_score = float("inf")
        best_params = param_grid[1]

        for p in param_grid:
            fold_maes = []
            for train_idx, val_idx in gkf.split(X, y, groups=groups):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                base_pipe = build_pipeline_core()
                base_pipe.set_params(
                    regressor__learning_rate=p["learning_rate"],
                    regressor__max_iter=p["max_iter"],
                    regressor__max_depth=p["max_depth"],
                    regressor__min_samples_leaf=p["min_samples_leaf"],
                    regressor__l2_regularization=p["l2_regularization"],
                    regressor__random_state=random_state,
                )
                model = TransformedTargetRegressor(
                    regressor=base_pipe,
                    func=np.log1p,
                    inverse_func=np.expm1,
                )
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)
                loss = mean_absolute_error(
                    np.log1p(np.maximum(y_val, 0)),
                    np.log1p(np.maximum(preds, 0))
                )
                fold_maes.append(loss)

            mean_mae = float(np.mean(fold_maes))
            if mean_mae < best_score:
                best_score = mean_mae
                best_params = p

        logger.info(f"Grid HPO Complete for {target_column}. Best CV Log-Scale MAE: {best_score:,.4f}")
        return {"best_params": best_params, "best_cv_mae": best_score, "method": "Deterministic GroupKFold Search"}


if __name__ == "__main__":
    reach_results = run_bayesian_hpo("per_media_reach", n_trials=5)
    print("Reach HPO Results:", reach_results)
