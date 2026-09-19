"""
Honest model training.

What this module guarantees:
  * It trains ONLY on rows supplied by the caller (real, observed post-level data). No data is generated.
  * It refuses to train if there is too little evidence (settings.MIN_TRAIN_POSTS / MIN_TRAIN_CREATORS).
  * Every candidate is scored on held-out CREATORS (grouped CV, repeated) and compared with a
    follower-proportional baseline. A richer model is shipped only if it clearly beats that baseline;
    otherwise the baseline itself is shipped and labelled as such.
  * Prediction intervals come from cross-conformal residuals (out-of-fold, grouped by creator). Their
    real coverage is then measured on held-out creators and stored, whatever it turns out to be.
  * Everything reported in the metadata is measured from the supplied data.
"""

import hashlib
import json
import math
import platform as _platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import settings, get_logger
from ..data import IMPRESSIONS_COLUMN, build_features, select_feature_spec, load_posts, load_profiles, NoDataError

logger = get_logger("model_trainer")

BASELINE = "follower_proportional_baseline"
TIERS = {"nano": (0, 1e4), "micro": (1e4, 1e5), "macro": (1e5, 1e6), "mega": (1e6, float("inf"))}
MIN_TIER_RESIDUALS = 30
MIN_TIER_CREATORS = 10
MIN_IMPRESSION_COVERAGE = 0.80


class _Stage:
    """Reports fractional progress of one training stage to an optional callback(fraction, message)."""

    def __init__(self, cb, lo: float, hi: float, total: int, msg: str):
        self.cb, self.lo, self.hi, self.total, self.msg, self.n = cb, lo, hi, max(total, 1), msg, 0

    def tick(self) -> None:
        self.n += 1
        if self.cb:
            self.cb(self.lo + (self.hi - self.lo) * min(self.n / self.total, 1.0), self.msg)


class InsufficientDataError(ValueError):
    """Raised when the supplied data cannot support a trustworthy model."""


# ----------------------------------------------------------------------------- estimators
class ConstantMedian:
    """Predicts the training median of the target."""

    def fit(self, X, y):
        self.value_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)

    def get_params(self, deep=True):
        return {}


class ResidualRegressor:
    """
    Learns deviations from proportionality with followers:  y = log_followers + f(X).
    With f = constant this is 'reach is a fixed fraction of followers'; richer f can only
    learn how that fraction changes with the inputs. It also makes predictions scale correctly
    with follower count, so tree models do not plateau outside the training range.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, np.asarray(y) - X["log_followers"].to_numpy())
        return self

    def predict(self, X):
        return X["log_followers"].to_numpy() + self.estimator.predict(X)


def _ridge(spec: Dict) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), spec["numeric"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), spec["categorical"]),
    ])
    return Pipeline([("pre", pre), ("m", Ridge(alpha=10.0))])


def _hgb(spec: Dict, n_rows: int, seed: int) -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", spec["numeric"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), spec["categorical"]),
    ])
    return Pipeline([("pre", pre), ("m", HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=200, max_depth=3,
        min_samples_leaf=max(10, n_rows // 25), l2_regularization=1.0, random_state=seed,
    ))])


def _reach_candidates(spec: Dict, n_rows: int, seed: int) -> Dict[str, Callable[[], Any]]:
    return {
        BASELINE: lambda: ResidualRegressor(ConstantMedian()),
        "ridge_residual": lambda: ResidualRegressor(_ridge(spec)),
        "gradient_boosting_residual": lambda: ResidualRegressor(_hgb(spec, n_rows, seed)),
    }


def _ratio_candidates(spec: Dict, n_rows: int, seed: int) -> Dict[str, Callable[[], Any]]:
    return {
        "constant_median_ratio": lambda: ConstantMedian(),
        "ridge": lambda: _ridge(spec),
        "gradient_boosting": lambda: _hgb(spec, n_rows, seed),
    }


# ----------------------------------------------------------------------------- statistics
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Finite-sample split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score.
    Raises if n is too small for that order statistic to exist (no invented fallback value).
    """
    n = len(scores)
    k = math.ceil((n + 1) * (1.0 - alpha))
    if n == 0 or k > n:
        raise InsufficientDataError(
            f"{n} calibration residuals cannot support a {1 - alpha:.0%} interval (need at least {math.ceil(1 / alpha) - 1})"
        )
    return float(np.sort(np.asarray(scores))[k - 1])


def follower_tier(followers: float) -> str:
    for name, (lo, hi) in TIERS.items():
        if lo <= followers < hi:
            return name
    return "mega"


def _fold_ids(groups: pd.Series, k: int, rng: np.random.Generator) -> np.ndarray:
    uniq = groups.unique()
    perm = rng.permutation(len(uniq))
    fold_of = {g: int(perm[i] % k) for i, g in enumerate(uniq)}
    return groups.map(fold_of).to_numpy()


def _metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    err = pred - y
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "log_mae": float(np.mean(np.abs(err))),
        "log_rmse": float(np.sqrt(np.mean(err ** 2))),
        "median_ratio_error": float(np.median(np.expm1(np.abs(err)))),   # 0.25 = typically off by x1.25
        "bias_log": float(np.mean(err)),
        "r2_log": float(1.0 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
    }


def _grouped_cv(
    factories: Dict[str, Callable[[], Any]],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
    seed: int,
    reps: int,
    k: int,
    tick: Optional[Callable[[], None]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, np.ndarray]]:
    """Repeated creator-grouped CV. Returns per-candidate summaries and the OOF predictions of rep 0."""
    fold_mae: Dict[str, List[float]] = {n: [] for n in factories}
    rep_metrics: Dict[str, List[Dict[str, float]]] = {n: [] for n in factories}
    oof0: Dict[str, np.ndarray] = {}
    for rep in range(reps):
        folds = _fold_ids(groups, k, np.random.default_rng(seed + rep))
        oof = {n: np.zeros(len(y)) for n in factories}
        for f in range(k):
            tr, va = folds != f, folds == f
            for name, make in factories.items():
                model = make().fit(X[tr], y[tr])
                pred = model.predict(X[va])
                oof[name][va] = pred
                fold_mae[name].append(float(np.mean(np.abs(pred - y[va]))))
            if tick:
                tick()
        for name in factories:
            rep_metrics[name].append(_metrics(y, oof[name]))
        if rep == 0:
            oof0 = oof
    base_name = next(iter(factories))
    base_folds = np.array(fold_mae[base_name])
    summary: Dict[str, Dict[str, Any]] = {}
    for name in factories:
        m = {key: float(np.mean([r[key] for r in rep_metrics[name]])) for key in rep_metrics[name][0]}
        folds_arr = np.array(fold_mae[name])
        m["fold_log_mae_std"] = float(folds_arr.std())
        m["n_folds_scored"] = int(len(folds_arr))
        m["fold_win_fraction_vs_baseline"] = None if name == base_name else float(np.mean(folds_arr < base_folds))
        summary[name] = m
    return summary, oof0


def _select(summary: Dict[str, Dict[str, Any]], baseline: str) -> str:
    base = summary[baseline]["log_mae"]
    best, best_mae = baseline, base
    for name, m in summary.items():
        if name == baseline:
            continue
        clearly_better = m["log_mae"] <= (1.0 - settings.MIN_RELATIVE_IMPROVEMENT) * base
        consistent = m["fold_win_fraction_vs_baseline"] >= settings.MIN_FOLD_WIN_FRACTION
        if clearly_better and consistent and m["log_mae"] < best_mae:
            best, best_mae = name, m["log_mae"]
    return best


def _calibration(scores: np.ndarray, followers: np.ndarray, creators: np.ndarray) -> Dict[str, Any]:
    """Global and follower-tier (Mondrian) conformal quantiles. A tier is used only with enough evidence."""
    out = {
        "n_residuals": int(len(scores)),
        "global": {"q80": conformal_quantile(scores, 0.20), "q90": conformal_quantile(scores, 0.10)},
        "tiers": {},
    }
    tiers = np.array([follower_tier(f) for f in followers])
    for name in TIERS:
        m = tiers == name
        n, nc = int(m.sum()), int(len(set(creators[m])))
        entry: Dict[str, Any] = {"n_residuals": n, "n_creators": nc, "used": False}
        if n >= MIN_TIER_RESIDUALS and nc >= MIN_TIER_CREATORS:
            entry.update(used=True, q80=conformal_quantile(scores[m], 0.20), q90=conformal_quantile(scores[m], 0.10))
        out["tiers"][name] = entry
    return out


def _empirical_coverage(
    make: Callable[[], Any], X: pd.DataFrame, y: np.ndarray, groups: pd.Series, followers: np.ndarray,
    seed: int, reps: int, tick: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """
    Measures the real coverage of the interval procedure on creators never used for fitting OR calibration:
    fit on 60% of creators, calibrate on 20%, evaluate on the remaining 20%, repeated.
    """
    creators = groups.unique()
    rng = np.random.default_rng(seed + 1000)
    c80, c90, per_tier = [], [], {t: [] for t in TIERS}
    for _ in range(reps):
        if tick:
            tick()
        perm = rng.permutation(len(creators))
        n_tr, n_ca = int(0.6 * len(creators)), int(0.2 * len(creators))
        tr_c, ca_c, te_c = (set(creators[perm[:n_tr]]), set(creators[perm[n_tr:n_tr + n_ca]]), set(creators[perm[n_tr + n_ca:]]))
        tr, ca, te = (groups.isin(s).to_numpy() for s in (tr_c, ca_c, te_c))
        if min(tr.sum(), ca.sum(), te.sum()) < 10:
            continue
        model = make().fit(X[tr], y[tr])
        s_ca = np.abs(model.predict(X[ca]) - y[ca])
        s_te = np.abs(model.predict(X[te]) - y[te])
        try:
            q80, q90 = conformal_quantile(s_ca, 0.20), conformal_quantile(s_ca, 0.10)
        except InsufficientDataError:
            continue
        c80.append(float(np.mean(s_te <= q80)))
        c90.append(float(np.mean(s_te <= q90)))
        tiers_te = np.array([follower_tier(f) for f in followers[te]])
        for t in TIERS:
            mt = tiers_te == t
            if mt.sum() >= 5:
                per_tier[t].append(float(np.mean(s_te[mt] <= q80)))
    if not c80:
        return {"available": False}
    return {
        "available": True,
        "repeats": len(c80),
        "nominal_80": 0.80, "measured_80_mean": float(np.mean(c80)), "measured_80_std": float(np.std(c80)), "measured_80_min": float(np.min(c80)),
        "nominal_90": 0.90, "measured_90_mean": float(np.mean(c90)), "measured_90_std": float(np.std(c90)), "measured_90_min": float(np.min(c90)),
        "measured_80_by_tier_global_interval": {t: (float(np.mean(v)) if v else None) for t, v in per_tier.items()},
        "intervals_reliable": bool(np.mean(c80) >= 0.75 and np.mean(c90) >= 0.85),
        "note": "Fit on 60% of creators, calibrated on 20%, scored on the other 20%. The shipped intervals use more calibration data, so this is a conservative estimate.",
    }


# ----------------------------------------------------------------------------- orchestration
def _support(posts: pd.DataFrame, spec: Dict) -> Dict[str, Any]:
    ranges = {}
    for col in spec["optional_used"]:
        vals = pd.to_numeric(posts[col], errors="coerce").dropna()
        ranges[col] = [float(vals.min()), float(vals.max())]
    return {
        "followers_min": int(posts["total_followers"].min()),
        "followers_max": int(posts["total_followers"].max()),
        "platforms": sorted(posts["platform"].unique().tolist()),
        "media_types": sorted(posts["media_type"].unique().tolist()),
        "optional_ranges": ranges,
    }


def _fit_track(
    factories: Dict[str, Callable[[], Any]], baseline: str,
    X: pd.DataFrame, y: np.ndarray, groups: pd.Series, seed: int, reps: int, k: int, tick: Optional[Callable[[], None]] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, np.ndarray]]:
    summary, oof0 = _grouped_cv(factories, X, y, groups, seed, reps, k, tick=tick)
    chosen = _select(summary, baseline)
    base_mae, chosen_mae = summary[baseline]["log_mae"], summary[chosen]["log_mae"]
    return chosen, {
        "candidates": summary,
        "baseline": baseline,
        "selected": chosen,
        "selected_is_baseline": chosen == baseline,
        "relative_improvement_vs_baseline": float(1.0 - chosen_mae / base_mae) if base_mae > 0 else 0.0,
        "selection_rule": f"a candidate must lower held-out log-MAE by >= {settings.MIN_RELATIVE_IMPROVEMENT:.0%} "
                          f"and win in >= {settings.MIN_FOLD_WIN_FRACTION:.0%} of creator-held-out folds; otherwise the baseline is shipped",
    }, oof0


def train_forecaster(
    posts: pd.DataFrame,
    *,
    seed: int = 0,
    cv_repeats: int = 3,
    coverage_repeats: int = 20,
    min_posts: Optional[int] = None,
    min_creators: Optional[int] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    min_posts = settings.MIN_TRAIN_POSTS if min_posts is None else min_posts
    min_creators = settings.MIN_TRAIN_CREATORS if min_creators is None else min_creators
    n_posts, n_creators = len(posts), int(posts["username"].nunique())
    if n_posts < min_posts or n_creators < min_creators:
        raise InsufficientDataError(
            f"Not enough real data to train a trustworthy model: have {n_posts} posts from {n_creators} creators; "
            f"need at least {min_posts} posts from {min_creators} creators."
        )

    posts = posts.reset_index(drop=True)
    spec = select_feature_spec(posts)
    X = build_features(posts, spec)
    y = np.log1p(posts["per_media_reach"].to_numpy(dtype=float))
    groups = posts["username"]
    followers = posts["total_followers"].to_numpy(dtype=float)
    k = int(max(2, min(5, n_creators // 6)))
    logger.info("Training on %d posts / %d creators; features=%s", n_posts, n_creators, spec["numeric"] + spec["categorical"])

    # ---- reach
    if progress:
        progress(0.03, "checking the data")
    cv_stage = _Stage(progress, 0.05, 0.55, cv_repeats * k, "cross-validating reach models on held-out creators")
    chosen, reach_eval, oof0 = _fit_track(_reach_candidates(spec, n_posts, seed), BASELINE, X, y, groups, seed, cv_repeats, k,
                                          tick=cv_stage.tick)
    reach_scores = np.abs(oof0[chosen] - y)
    reach_cal = _calibration(reach_scores, followers, groups.to_numpy())
    reach_eval["calibration"] = reach_cal
    cov_stage = _Stage(progress, 0.55, 0.78, coverage_repeats, "measuring interval coverage on unseen creators")
    reach_eval["interval_validation"] = _empirical_coverage(
        _reach_candidates(spec, n_posts, seed)[chosen], X, y, groups, followers, seed, coverage_repeats, tick=cov_stage.tick)
    if progress:
        progress(0.78, "fitting the final reach model")
    reach_model = _reach_candidates(spec, n_posts, seed)[chosen]().fit(X, y)

    # ---- impressions (only if genuinely observed for enough rows)
    imp_eval: Dict[str, Any] = {"modelled": False}
    ratio_model = None
    has_imp = posts[IMPRESSIONS_COLUMN].notna() if IMPRESSIONS_COLUMN in posts.columns else pd.Series(False, index=posts.index)
    n_imp, n_imp_creators = int(has_imp.sum()), int(posts.loc[has_imp, "username"].nunique())
    if has_imp.mean() >= MIN_IMPRESSION_COVERAGE and n_imp >= min_posts and n_imp_creators >= min_creators:
        sub = posts[has_imp].reset_index(drop=True)
        Xi = X[has_imp.to_numpy()].reset_index(drop=True)
        y_reach_i = np.log1p(sub["per_media_reach"].to_numpy(dtype=float))
        y_ratio = np.log1p(sub[IMPRESSIONS_COLUMN].to_numpy(dtype=float)) - y_reach_i
        gi = sub["username"]
        ki = int(max(2, min(5, n_imp_creators // 6)))
        imp_stage = _Stage(progress, 0.78, 0.93, cv_repeats * ki, "cross-validating the impressions model")
        ratio_choice, imp_eval, oof_r0 = _fit_track(_ratio_candidates(spec, n_imp, seed), "constant_median_ratio",
                                                   Xi, y_ratio, gi, seed, cv_repeats, ki, tick=imp_stage.tick)
        # impressions = predicted reach x predicted frequency; OOF composition for honest calibration
        reach_oof_sub = _grouped_cv({chosen: _reach_candidates(spec, n_imp, seed)[chosen]}, Xi, y_reach_i, gi, seed, 1, ki)[1][chosen]
        composed = reach_oof_sub + np.maximum(oof_r0[ratio_choice], 0.0)
        imp_scores = np.abs(np.log1p(sub[IMPRESSIONS_COLUMN].to_numpy(dtype=float)) - composed)
        imp_eval["calibration"] = _calibration(imp_scores, sub["total_followers"].to_numpy(dtype=float), gi.to_numpy())
        imp_eval.update(modelled=True, n_rows=n_imp, n_creators=n_imp_creators,
                        composed_log_mae=float(np.mean(imp_scores)),
                        composed_median_ratio_error=float(np.median(np.expm1(imp_scores))))
        ratio_model = _ratio_candidates(spec, n_imp, seed)[ratio_choice]().fit(Xi, y_ratio)
    else:
        imp_eval["reason"] = (
            f"impressions not modelled: {n_imp} rows from {n_imp_creators} creators have observed impressions "
            f"(need >= {MIN_IMPRESSION_COVERAGE:.0%} of rows, {min_posts} rows and {min_creators} creators)"
        )

    if progress:
        progress(0.95, "writing the model report")
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(posts, index=False).to_numpy().tobytes()).hexdigest()
    metadata: Dict[str, Any] = {
        "schema_version": 3,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "software": {"project": settings.VERSION, "python": sys.version.split()[0], "scikit_learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "platform": _platform.platform()},
        "data": {"n_posts": n_posts, "n_creators": n_creators, "training_frame_sha256": fingerprint,
                 "posts_per_creator_min": int(posts.groupby("username").size().min()),
                 "posts_per_creator_median": float(posts.groupby("username").size().median()),
                 "cv_folds": k, "cv_repeats": cv_repeats, "seed": seed},
        "features": spec,
        "support": _support(posts, spec),
        "reach": reach_eval,
        "impressions": imp_eval,
        "limits": [
            "Forecasts describe statistical association in the supplied data, not causal effects of creative choices.",
            "Accuracy is only known for creators similar to those in the training data.",
            "Follower counts used for training are whatever the data file contains (often a snapshot, not the count at post time).",
        ],
    }
    bundle = {"spec": spec, "reach_model": reach_model, "ratio_model": ratio_model,
              "reach_model_name": chosen, "ratio_model_name": (imp_eval.get("selected") if ratio_model is not None else None)}
    return bundle, metadata


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def save_model(bundle: Dict[str, Any], metadata: Dict[str, Any],
               model_path: Optional[Path] = None, meta_path: Optional[Path] = None) -> None:
    model_path = Path(model_path or settings.MODEL_PATH)
    meta_path = Path(meta_path or settings.MODEL_METADATA_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    metadata = dict(metadata)
    metadata["artifact_sha256"] = _sha256(model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def train_and_persist(posts_source=None, profiles_source=None, **kwargs) -> Dict[str, Any]:
    """Loads the real posts file, validates it, trains, and saves the model. Raises rather than fabricates."""
    from .registry import clear_registry_cache

    try:
        profiles, _ = load_profiles(profiles_source)
    except NoDataError:
        profiles = None
    progress = kwargs.get("progress")
    if progress:
        progress(0.01, "reading and validating posts.csv")
    posts, report = load_posts(posts_source, profiles=profiles)
    bundle, metadata = train_forecaster(posts, **kwargs)
    metadata["data"]["validation"] = {"rows_read": report.rows_read, "rows_used": report.rows_used,
                                       "rows_rejected": report.rows_rejected, "notes": report.notes}
    if progress:
        progress(0.98, "saving the model")
    save_model(bundle, metadata)
    clear_registry_cache()
    if progress:
        progress(1.0, "done")
    return metadata


if __name__ == "__main__":
    md = train_and_persist()
    print(json.dumps({k: md[k] for k in ("data", "reach")}, indent=2)[:4000])
